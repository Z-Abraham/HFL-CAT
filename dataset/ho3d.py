from logging import root
import os
from torchvision.transforms import functional
from torch.utils import data
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import json
from dataset import ho3d_util
from dataset import dataset_util
from pycocotools.coco import COCO


class HO3D(data.Dataset):
    def __init__(self, dataset_root, obj_model_root,
                 train_label_root="/data_ssd/yaohuan/Semi-Hand-Object-master/assets/data/ho3d_v2/train.txt",
                 mode="evaluation", inp_res=512,
                 max_rot=np.pi, scale_jittering=0.2, center_jittering=0.1,
                 hue=0.15, saturation=0.5, contrast=0.5, brightness=0.5, blur_radius=0.5):
        # Dataset attributes
        self.root = dataset_root
        self.mode = mode
        self.inp_res = inp_res
        self.joint_root_id = 0
        self.jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                      1, 2, 3, 17,
                                      4, 5, 6, 18,
                                      10, 11, 12, 19,
                                      7, 8, 9, 20]
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        # object informations
        self.obj_mesh = ho3d_util.load_objects_HO3D(obj_model_root)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)

        # 手部关节 用于测试
        self.pred_joints_coord_img = json.load(open("/root/autodl-tmp/HFL-Net-main/dataset/kypt_eval.json", 'r', encoding='utf8'))

        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot

            self.train_seg_root = os.path.join(train_label_root, "train_segLabel")

            self.mano_params = []
            self.joints_uv = []
            self.obj_p2ds = []
            self.K = []
            self.set_list = []
            # data_ho3d = []
            with open('/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/ho3d_train_data.json', 'r') as f:
                data_ho3d = json.load(f)

            for data in data_ho3d:
                self.set_list.append(data['seqName_id'])
                K = np.array(data['K'], dtype=np.float32)
                self.K.append(K)
                self.joints_uv.append(ho3d_util.projectPoints(np.array(data['joints_3d'], dtype=np.float32), K))
                self.mano_params.append(np.array(data['mano_params'], dtype=np.float32))
                self.obj_p2ds.append(np.array(data['obj_p2ds'], dtype=np.float32))

        else:
            self.set_list = ho3d_util.load_names(os.path.join(self.root, "evaluation.txt"))

    # zzq新增处理kgc的输入h
    def data_aug(self, img, mano_param, joints_uv, K, gray, p2d):
        # （你的原有代码：计算裁剪框、抖动中心/尺度、生成affinetrans）
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        center = center + center_offsets
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        scale = scale * scale_jittering
        rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        # 你的代码已生成affinetrans（关键：和图像裁剪用同一个矩阵）
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res], rot=rot, K=K
        )

        # （你的原有代码：处理mano_param、joints_uv、p2d）
        mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=self.coord_change_mat)
        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)  # 原始关节点→裁剪后图像（像素坐标）
        K = post_rot_trans.dot(K)
        p2d = dataset_util.transform_coords(p2d, affinetrans)
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv_norm = dataset_util.normalize_joints(joints_uv, bbox_hand)  # 你的原有归一化（基于hand bbox）
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # -------------------- 新增：HandGCAT的joints_img处理逻辑 --------------------
        # 1. 基于输入尺寸（inp_res）归一化到[0,1]（HandGCAT核心：替代你的bbox归一化）
        joints_img = joints_uv.copy()  # 用裁剪后的关节点像素坐标
        joints_img[:, 0] /= self.inp_res  # 宽度方向归一化（如512）
        joints_img[:, 1] /= self.inp_res  # 高度方向归一化（输入是正方形，宽高一致）

        # 2. 训练时添加随机噪声（HandGCAT：均值0，标准差0.01，50%概率）
        if np.random.rand() < 0.5:
            noise = np.random.normal(0., 0.01, joints_img.shape)
            joints_img = np.clip(joints_img + noise, 0., 1.)  # 防止超出[0,1]
        # --------------------------------------------------------------------------

        # （你的原有代码：处理图像、灰度图、生成obj_mask）
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        blur_radius = random.random() * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = dataset_util.color_jitter(img, brightness=self.brightness, saturation=self.saturation, hue=self.hue,
                                        contrast=self.contrast)
        gray = dataset_util.transform_img(gray, affinetrans, [self.inp_res, self.inp_res])
        gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        # 新增：把joints_img加入返回值（给__getitem__用）
        return img, mano_param, K, obj_mask, p2d, joints_uv_norm, bbox_hand, bbox_obj, joints_img

    def data_aug_old(self, img, mano_param, joints_uv, K, gray, p2d):
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)

        # Randomly jitter center
        center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        center = center + center_offsets

        # Scale jittering
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        scale = scale * scale_jittering

        rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(center, scale,
                                                                                 [self.inp_res, self.inp_res], rot=rot,
                                                                                 K=K)
        # Change mano from openGL coordinates to normal coordinates
        mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=self.coord_change_mat)

        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans

        # K0=K

        K = post_rot_trans.dot(K)

        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # P3D=np.array([0.2,0.4,0.4])
        # P2D=K0.dot(P3D.transpose()).transpose()[:2]/(K0.dot(P3D.transpose()).transpose()[-1])

        # D2affine=affinetrans
        # Kaffine=post_rot_trans
        ##D3affine=rot_mat
        # P2Daffine=dataset_util.transform_coords(P2D.reshape(1,2),D2affine)
        # K1=Kaffine.dot(K0)

        ## P3Daffine=D3affine.dot(P3D.transpose())
        # P2D1=K1.dot(P3Daffine).transpose()[:2]/K1.dot(P3Daffine).transpose()[-1]

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = random.random() * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = dataset_util.color_jitter(img, brightness=self.brightness,
                                        saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_util.transform_img(gray, affinetrans, [self.inp_res, self.inp_res])
        gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj

    # zzq新家s
    def data_crop(self, img, K, bbox_hand, p2d, pred_joints_uv):
        # （你的原有代码：计算裁剪框、生成affinetrans）
        crop_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.1)
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        # 你的代码已生成affinetrans（和图像裁剪一致）
        affinetrans, _ = dataset_util.get_affine_transform(center, scale, [self.inp_res, self.inp_res])
        bbox_hand = dataset_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()

        # -------------------- 新增：HandGCAT的joints_img处理逻辑（测试模式） --------------------
        # 1. 加载预预测的2D关节点（HandGCAT用pred_joints_coord_img，你需准备类似文件）
        # 假设你的预测文件是：./pred_joints_test.json（格式：list，每个元素是(21,2)的关节点）
        # 按索引取当前样本的预测关节点（注意：关节点顺序要和训练时一致，用你的jointsMapManoToSimple调整）

        pred_joints_uv = pred_joints_uv[self.jointsMapSimpleToMano]  # 适配你的关节点顺序

        # 2. 仿射变换：映射到裁剪后的图像（和图像用同一个affinetrans）
        joints_img = dataset_util.transform_coords(pred_joints_uv, affinetrans)

        # 3. 基于输入尺寸归一化到[0,1]（无噪声，测试时不增强）
        joints_img[:, 0] /= self.inp_res
        joints_img[:, 1] /= self.inp_res
        # --------------------------------------------------------------------------

        # （你的原有代码：处理图像、K）
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)

        # 新增：把joints_img加入返回值
        return img, K, bbox_hand, bbox_obj, joints_img

    def data_crop_old(self, img, K, bbox_hand, p2d):
        crop_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.1)
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(center, scale, [self.inp_res, self.inp_res])
        bbox_hand = dataset_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()
        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)
        return img, K, bbox_hand, bbox_obj

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, idx):
        # from PIL import ImageDraw # 构造函数中，im为一个Image对象

        sample = {}
        seqName, id = self.set_list[idx].split("/")
        img = ho3d_util.read_RGB_img(self.root, seqName, id, self.mode)

        if self.mode == 'train':
            # （你的原有代码：加载K、joints_uv、mano_param等）
            K = self.K[idx]
            joints_uv = self.joints_uv[idx]
            mano_param = self.mano_params[idx]
            gray = ho3d_util.read_gray_img(self.train_seg_root, seqName, id)
            p2d = self.obj_p2ds[idx]

            # 调用修改后的data_aug，接收新增的joints_img
            img, mano_param, K, obj_mask, p2d, joints_uv_norm, bbox_hand, bbox_obj, joints_img = self.data_aug(
                img, mano_param, joints_uv, K, gray, p2d
            )

            # （你的原有sample内容）
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["mano_param"] = mano_param
            sample["cam_intr"] = K
            sample["joints2d"] = joints_uv_norm  # 你的原有归一化结果（可保留）
            sample["obj_p2d"] = p2d
            sample["obj_mask"] = obj_mask
            sample["hand_type"] = 'right'

            # 新增：保存HandGCAT格式的joints_img到sample
            sample["joints_img"] = torch.from_numpy(joints_img.astype(np.float32))  # 转Tensor供模型用

        else:
            # （你的原有代码：加载annotations、K、p2d等）
            annotations = np.load(os.path.join(os.path.join(self.root, self.mode), seqName, 'meta', id + '.pkl'),
                                  allow_pickle=True)
            K = np.array(annotations['camMat'], dtype=np.float32)
            sample["obj_cls"] = annotations['objName']
            sample["obj_bbox3d"] = self.obj_bbox3d[sample["obj_cls"]]
            sample["obj_diameter"] = self.obj_diameters[sample["obj_cls"]]
            obj_pose = ho3d_util.pose_from_RT(annotations['objRot'].reshape((3,)), annotations['objTrans'])
            p2d = ho3d_util.projectPoints(sample["obj_bbox3d"], K, rt=obj_pose)
            sample["obj_pose"] = obj_pose
            bbox_hand = np.array(annotations['handBoundingBox'], dtype=np.float32)
            root_joint = np.array(annotations['handJoints3D'], dtype=np.float32)
            root_joint = root_joint.dot(self.coord_change_mat.T)
            sample["root_joint"] = root_joint

            pred_joints_uv = np.array(self.pred_joints_coord_img[idx], dtype=np.float32)  # idx是__getitem__的索引

            # 调用修改后的data_crop，传入idx（用于取预测关节点），接收joints_img
            img, K, bbox_hand, bbox_obj, joints_img = self.data_crop(img, K, bbox_hand, p2d, pred_joints_uv)

            # （你的原有sample内容）
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["cam_intr"] = K
            sample["hand_type"] = 'right'

            # 新增：保存HandGCAT格式的joints_img到sample
            sample["joints_img"] = torch.from_numpy(joints_img.astype(np.float32))

        return sample


if __name__ == "__main__":
    dataset = HO3D(dataset_root="/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data",
                   obj_model_root="assets/object_models",
                   train_label_root="/data1/zhifeng/ho3d-process",
                   mode="train",
                   inp_res=256)
