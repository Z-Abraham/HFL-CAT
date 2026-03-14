import torch
from torch import nn
from torchvision import ops
from networks.backbone_share import FPN
from networks.hand_head import hand_Encoder, hand_regHead
from networks.object_head import obj_regHead, Pose2DLayer
from networks.mano_head import mano_regHead
from networks.CR import Transformer
from networks.loss import Joint2DLoss, ManoLoss, ObjectLoss
from networks.KGC import KnowledgeGuidedModule
from networks.CAT import CrossAttention


# ========== 新增：轻量2D姿态辅助分支 ==========
class Lightweight2DPoseBranch(nn.Module):
    def __init__(self, in_channels=256, joint_nb=21, hidden_dim=128):
        super().__init__()
        self.joint_nb = joint_nb
        # 分支结构：1x1降维 → 3x3卷积 → 1x1升维到关节数 + Sigmoid
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, joint_nb, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self.conv1.apply(weights_init_kaiming)
        self.conv2.apply(weights_init_kaiming)
        self.conv3.apply(init_weights)

    def forward(self, x):
        # x: 输入特征图 (B, 256, H/4, W/4)
        x = self.relu(self.conv1(x))  # 1x1降维到128
        x = self.relu(self.conv2(x))  # 3x3保持尺寸
        heatmap = self.sigmoid(self.conv3(x))  # 1x1升维到21关节，Sigmoid归一化到[0,1]
        return heatmap


# ========== 新增：从热图提取2D关节坐标的辅助函数 ==========
def heatmap_to_2d_joints(heatmap, img_size=(512, 512)):
    """
    从热图提取2D关节坐标（batch级）
    Args:
        heatmap: (B, 21, H, W) 2D姿态热图
        img_size: (width, height) 输入图像尺寸，用于还原坐标
    Returns:
        joints_2d: (B, 21, 2) 2D关节坐标（像素值）
    """
    B, J, H, W = heatmap.shape
    # 1. 找到每个关节热图的峰值坐标 (B, J)
    heatmap_flat = heatmap.reshape(B, J, -1)
    max_idx = torch.argmax(heatmap_flat, dim=-1)  # (B, J)
    # 2. 转换为(x,y)坐标
    x = (max_idx % W) * (img_size[0] / W)  # 还原到图像宽度
    y = (max_idx // W) * (img_size[1] / H)  # 还原到图像高度
    # 3. 拼接为(B, 21, 2)
    joints_2d = torch.stack([x, y], dim=-1)
    return joints_2d


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


class HONet(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512], coord_change_mat=None,
                 reg_object=True, pretrained=True):

        super(HONet, self).__init__()

        self.out_res = roi_res
        self.joint_nb = joint_nb  # 21个手部关节
        self.inp_channels = channels  # 256（FPN输出通道）

        # 新增模块初始化
        self.KGC = KnowledgeGuidedModule(2, 1024)

        # 新增CAT模块初始化
        self.CAT = CrossAttention(256, 21)

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # ========== 新增：初始化2D姿态辅助分支 ==========
        self.pose2d_branch = Lightweight2DPoseBranch(
            in_channels=self.inp_channels,
            joint_nb=self.joint_nb,
            hidden_dim=128  # 极轻量，仅128维中间层
        )

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels // 2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.transformer_obj = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth,
                                           num_heads=transformer_head)
        self.transformer_hand = Transformer(inp_res=roi_res, dim=channels * 2, depth=transformer_depth,
                                            num_heads=transformer_head)

        self.hand_head.apply(init_weights)
        self.hand_encoder.apply(init_weights)
        self.mano_branch.apply(init_weights)
        self.transformer_obj.apply(init_weights)
        self.transformer_hand.apply(init_weights)
        self.obj_head.apply(init_weights)
        # self.KGC.apply(init_weights)  # KGC分支初始化
        self.pose2d_branch.apply(init_weights)  # 新增分支初始化

    def net_forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, roots3d=None):
        batch = self.new_method(imgs)

        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight - inter_topLeft > 0).sum(dim=1)) == 2

        # ========== 核心修改：前向传播时运行2D姿态辅助分支 ==========
        # 1. 原有FPN前向，输出手部分支特征P2_h（对应架构图的Ph）、物体分支特征P2_o
        P2_h, P2_o = self.base_net(imgs)
        # 2. 运行2D姿态辅助分支，输出21个关节的热图 (B,21,H/4,W/4)
        pose2d_heatmap = self.pose2d_branch(P2_h)
        # 3. 从热图提取2D关节坐标（用于送给KGC模块）
        # 注意：img_size需和你的输入图像尺寸匹配（HO3D是512，Dex-YCB是640x480，可传参）
        img_size = (imgs.shape[3], imgs.shape[2])  # (width, height)
        pred_joints2d = heatmap_to_2d_joints(pose2d_heatmap, img_size=img_size)

        # 原有逻辑保持不变（仅补充：pred_joints2d可传给KGC模块）
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        x_hand = ops.roi_align(P2_h, roi_boxes_hand, output_size=(self.out_res, self.out_res),
                               spatial_scale=1.0 / 4.0,
                               sampling_ratio=-1)
        x_obj = ops.roi_align(P2_o, roi_boxes_hand, output_size=(self.out_res, self.out_res),
                              spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)

        # ========== 核心修改1：区分训练/测试，获取KGC的输入2D关节 ==========
        if self.training and joints_uv is not None:
            # 训练阶段：用GT的2D关节坐标（已归一化到像素值）
            kgc_input_joints = joints_uv  # (B, 21, 2)
        else:
            # 测试阶段：用2D分支预测的关节坐标
            kgc_input_joints = pred_joints2d  # (B, 21, 2)

        # ========== 核心修改2：调用KGC模块，生成拓扑先验特征Fp ==========
        # KGC输入：2D关节坐标 (B,21,2) → 输出：拓扑先验特征Fp (B,1024,32,32)（需匹配x_hand尺寸）
        Fp = self.KGC(kgc_input_joints)  # KGC输出需reshape到(B, C, H, W)，若输出是(B,1024)，需reshape为(B,1024,1,1)后上采样到32x32
        # 补充：若KGC输出是一维特征，需上采样匹配x_hand尺寸（32x32）
        if len(Fp.shape) == 2:  # 若Fp是(B,1024)
            Fp = Fp.unsqueeze(-1).unsqueeze(-1)  # (B,1024,1,1)
            Fp = nn.functional.interpolate(Fp, size=(self.out_res, self.out_res), mode='bilinear',
                                           align_corners=False)  # (B,1024,32,32)

        # ========== 核心修改3：拼接Fh+Fho，与Fp一起送入CAT模块 ==========
        # 拼接Fh(256) + Fho(256) → Fh_ho(512)（沿通道维度）
        Fh_ho = torch.cat([x_hand, x_obj], dim=1)  # (B, 512, 32, 32)
        # CAT模块融合：Fh_ho(512) + Fp(1024) → 输出融合特征F_cat (B,256,32,32)（匹配原有hand输入通道）
        F_cat = self.CAT(Fh_ho, Fp)  # CAT需适配输入通道，输出256维（与原有hand通道一致）

        # 物体分支前向（原有逻辑）
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)
            y = ops.roi_align(P2_o, roi_boxes_obj, output_size=(self.out_res, self.out_res),
                              spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)
            z_x = ops.roi_align(P2_h, roi_boxes_inter, output_size=(self.out_res, self.out_res),
                                spatial_scale=1.0 / 4.0,
                                sampling_ratio=-1)
            z_x = msk_inter[:, None, None, None] * z_x
            hand_obj = torch.cat([x_hand, x_obj.detach()], dim=1)
            hand_obj = self.transformer_hand(hand_obj, hand_obj)
            y = self.transformer_obj(y, z_x.detach())
            out_fm = self.obj_head(y)
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None

        # ========== 核心修改4：用CAT融合特征替换原有hand输入 ==========
        # 原逻辑：hand = hand_obj[:, 0:256, :, :]
        # 新逻辑：用CAT融合后的特征F_cat作为手部分支输入
        hand = F_cat  # (B,256,32,32)

        # 手部分支前向（原有逻辑，输入替换为F_cat）
        out_hm, encoding, preds_joints = self.hand_head(hand)
        mano_encoding = self.hand_encoder(out_hm, encoding)
        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params,
                                                              roots3d=roots3d)

        # ========== 返回值补充：2D热图和预测的2D关节坐标 ==========
        return preds_joints, pred_mano_results, gt_mano_results, preds_obj, pose2d_heatmap, pred_joints2d

    def new_method(self, imgs):
        batch = imgs.shape[0]
        return batch

    def forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, roots3d=None, joints_uv=None):  # 新增：joints_uv
        if self.training:
            # 训练时传入GT的2D关节坐标joints_uv给net_forward
            preds_joints, pred_mano_results, gt_mano_results, preds_obj, pose2d_heatmap, pred_joints2d = self.net_forward(
                imgs, bbox_hand, bbox_obj, mano_params=mano_params, joints_uv=joints_uv  # 新增传递
            )
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj, pose2d_heatmap, pred_joints2d
        else:
            # 测试时无需传joints_uv，用预测的2D关节
            preds_joints, pred_mano_results, _, preds_obj, _, pred_joints2d = self.net_forward(
                imgs, bbox_hand, bbox_obj, roots3d=roots3d, joints_uv=None  # 测试时为None
            )
            return preds_joints, pred_mano_results, preds_obj, pred_joints2d


# ========== 修改HOModel类，添加2D热图损失 ==========
class HOModel(nn.Module):
    def __init__(self, honet, mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None,
                 lambda_pose2d_heatmap=1.0):  # 新增：2D热图损失权重

        super(HOModel, self).__init__()
        self.honet = honet
        # 原有损失保持不变
        self.mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        self.mano_joint_loss = ManoLoss(lambda_joints3d=lambda_joints3d,
                                        lambda_regulshape=mano_lambda_regulshape,
                                        lambda_regulpose=mano_lambda_regulpose)
        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

        # ========== 新增：2D热图损失（L2损失，也可用MSE） ==========
        self.lambda_pose2d_heatmap = lambda_pose2d_heatmap
        self.pose2d_heatmap_loss = nn.MSELoss()

    def forward(self, imgs, bbox_hand, bbox_obj,
                joints_uv=None, joints_xyz=None, mano_params=None, roots3d=None,
                obj_p2d_gt=None, obj_mask=None, obj_lossmask=None,
                pose2d_heatmap_gt=None):  # 新增：2D热图GT

        if self.training:
            losses = {}
            total_loss = 0
            # 前向传播时，把GT的joints_uv传给honet
            preds_joints2d, pred_mano_results, gt_mano_results, preds_obj, pose2d_heatmap, _ = self.honet(
                imgs, bbox_hand, bbox_obj, mano_params=mano_params, joints_uv=joints_uv  # 新增传递
            )

            # 原有损失计算（保持不变）
            if mano_params is not None:
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(pred_mano_results, gt_mano_results)
                total_loss += mano_total_loss
                for key, val in mano_losses.items():
                    losses[key] = val
            if joints_uv is not None:
                joint2d_loss, joint2d_losses = self.joint2d_loss.compute_loss(preds_joints2d, joints_uv)
                for key, val in joint2d_losses.items():
                    losses[key] = val
                total_loss += joint2d_loss
            if preds_obj is not None:
                obj_total_loss, obj_losses = self.object_loss.compute_loss(obj_p2d_gt, obj_mask, preds_obj,
                                                                           obj_lossmask=obj_lossmask)
                for key, val in obj_losses.items():
                    losses[key] = val
                total_loss += obj_total_loss

            # ========== 新增：2D热图损失计算 ==========
            if pose2d_heatmap_gt is not None:
                # 计算预测热图和GT热图的MSE损失
                pose2d_loss = self.pose2d_heatmap_loss(pose2d_heatmap, pose2d_heatmap_gt)
                pose2d_loss = pose2d_loss * self.lambda_pose2d_heatmap
                losses["pose2d_heatmap_loss"] = pose2d_loss.detach().cpu()
                total_loss += pose2d_loss

            losses["total_loss"] = total_loss.detach().cpu() if total_loss != 0 else 0
            return total_loss, losses
        else:
            # 测试时honet自动用预测的2D关节
            preds_joints, pred_mano_results, preds_obj, pred_joints2d = self.honet(
                imgs, bbox_hand, bbox_obj, roots3d=roots3d, joints_uv=None
            )
            return preds_joints, pred_mano_results, preds_obj, pred_joints2d
