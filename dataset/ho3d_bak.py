import copy
from logging import root
import torch
import json
from dataset import ho3d_util
from dataset import dataset_util
from pycocotools.coco import COCO
import os
import os.path as osp
from torchvision.transforms import functional
from torch.utils import data
from torchvision.transforms import transforms
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
from ho3d_util import cam2pixel, process_bbox, get_bbox
from utils.preprocessing import augmentation, load_img
from utils.config import cfg

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HO3D(data.Dataset):
    def __init__(self, dataset_root, obj_model_root,
                 train_label_root="/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/train.txt",
                 mode="evaluation", inp_res=512,
                 max_rot=np.pi, scale_jittering=0.2, center_jittering=0.1,
                 hue=0.15, saturation=0.5, contrast=0.5, brightness=0.5, blur_radius=0.5):
        # Dataset attributes
        self.transform = transforms.ToTensor()
        self.pre_path = "/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/kypt_eval.json"
        self.pred_joints_coord_img = json.load(open(self.pre_path, 'r', encoding='utf8'))

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
            self.hand_gcat_datalist = self.load_handgcat_data()
            # data_ho3d = []
            # with open("/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/HO3D_train_data.json", 'r') as f:
            #     data_ho3d = json.load(f)

            with open("/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/ho3d_train_data_hfl_net.json", 'r') as f:
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

    def data_aug(self, img, mano_param, joints_uv, K, gray, p2d):
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

    def load_handgcat_data(self):
        if self.mode == "train":
            path = "/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/HO3D_train_data.json"
        else:
            path = "/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/HO3D_evaluation_data.json"
        db = COCO(path)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join("/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data",
                                self.mode, img['file_name'])
            img_shape = (img['height'], img['width'])
            if self.mode == 'train':
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                joints_coord_img = cam2pixel(joints_coord_cam, cam_param['focal'], cam_param['princpt'])
                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
                if bbox is None:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam,
                        "joints_coord_img": joints_coord_img,
                        "bbox": bbox, "cam_param": cam_param, "mano_pose": mano_pose, "mano_shape": mano_shape}
            else:
                root_joint_cam = np.array(ann['root_joint_cam'], dtype=np.float32)
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                bbox = np.array(ann['bbox'], dtype=np.float32)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.5)

                data = {"img_path": img_path, "img_shape": img_shape, "root_joint_cam": root_joint_cam,
                        "bbox": bbox, "cam_param": cam_param}

            datalist.append(data)

        return datalist

    def data_crop(self, img, K, bbox_hand, p2d):
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
        data = copy.deepcopy(self.hand_gcat_datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        sample = {}
        seqName, id = self.set_list[idx].split("/")
        # img = ho3d_util.read_RGB_img(self.root, seqName, id, self.mode)
        img = load_img(img_path)
        # img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.mode, do_flip=False)
        # cat_img = self.transform(cat_img.astype(np.float32))/255.
        img = Image.fromarray(img.astype(np.uint8))
        if self.mode == 'train':

            joints_img = data['joints_coord_img']
            # joints_img_xy1 = np.concatenate((joints_img[:,:2], np.ones_like(joints_img[:,:1])),1)
            # joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            # # normalize to [0,1]
            # joints_img[:,0] /= cfg.input_img_shape[1]
            # joints_img[:,1] /= cfg.input_img_shape[0]
            #
            # if np.random.rand()<0.5:
            #     noise=np.random.normal(0.,0.01,joints_img.shape)
            #     joints_img += noise

            K = self.K[idx]
            # hand information
            joints_uv = self.joints_uv[idx]
            mano_param = self.mano_params[idx]
            # object information
            gray = ho3d_util.read_gray_img(self.train_seg_root, seqName, id)

            p2d = self.obj_p2ds[idx]
            # data augmentation
            img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj = self.data_aug(img, mano_param,
                                                                                              joints_uv, K, gray, p2d)
            sample["img"] = functional.to_tensor(img)
            sample["joints_img"] = joints_img
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["mano_param"] = mano_param
            sample["cam_intr"] = K
            sample["joints2d"] = joints_uv
            sample["obj_p2d"] = p2d
            sample["obj_mask"] = obj_mask
            sample["hand_type"] = 'right'
        else:
            annotations = np.load(os.path.join(os.path.join(self.root, self.mode), seqName, 'meta', id + '.pkl'),
                                  allow_pickle=True)
            K = np.array(annotations['camMat'], dtype=np.float32)
            # object
            sample["obj_cls"] = np.array(self.pred_joints_coord_img[idx], dtype=np.float32)
            sample["joints_img"] = data['joints_coord_img']
            sample["obj_bbox3d"] = self.obj_bbox3d[sample["obj_cls"]]
            sample["obj_diameter"] = self.obj_diameters[sample["obj_cls"]]
            obj_pose = ho3d_util.pose_from_RT(annotations['objRot'].reshape((3,)), annotations['objTrans'])
            p2d = ho3d_util.projectPoints(sample["obj_bbox3d"], K, rt=obj_pose)
            sample["obj_pose"] = obj_pose

            # hand
            bbox_hand = np.array(annotations['handBoundingBox'], dtype=np.float32)
            root_joint = np.array(annotations['handJoints3D'], dtype=np.float32)
            root_joint = root_joint.dot(self.coord_change_mat.T)
            sample["root_joint"] = root_joint

            # drawer = ImageDraw.Draw(img)
            # for point in bbox_hand.reshape(2,2):
            #     drawer.point(point, fill='black')
            # img.save("1.jpg")

            img, K, bbox_hand, bbox_obj = self.data_crop(img, K, bbox_hand, p2d)
            sample["img"] = functional.to_tensor(img)

            # for point in bbox_hand.reshape(2,2):
            #         drawer.point(point, fill='black')
            # img.save("2.jpg")

            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["cam_intr"] = K
            sample["hand_type"] = 'right'

        return sample

# 手部关节连接顺序（根据MANO模型的21个关节）
HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
    [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
    [0, 17], [17, 18], [18, 19], [19, 20]   # 小指
]

def visualize_sample(sample):
    # 1. 处理图像张量 (3, H, W) -> (H, W, 3) 并转换为RGB
    img_tensor = sample['img']  # 归一化的张量 (3, H, W)
    img_np = img_tensor.permute(1, 2, 0).numpy()  # 转换为 (H, W, 3)
    img_np = (img_np * 255).astype(np.uint8)  # 从[0,1]映射到[0,255]
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 转换为BGR用于OpenCV显示
    h, w = img.shape[:2]

    # 2. 绘制手部边界框 (x1, y1, x2, y2)
    bbox_hand = sample['bbox_hand'].astype(int)
    cv2.rectangle(img, (bbox_hand[0], bbox_hand[1]),
                 (bbox_hand[2], bbox_hand[3]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'Hand', (bbox_hand[0], bbox_hand[1]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 3. 绘制物体边界框
    bbox_obj = sample['bbox_obj'].astype(int)
    cv2.rectangle(img, (bbox_obj[0], bbox_obj[1]),
                 (bbox_obj[2], bbox_obj[3]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Object', (bbox_obj[0], bbox_obj[1]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. 绘制原始手部关节点（joints_img）
    joints_img = sample['joints_img'][:, :2].astype(int)  # 取x,y坐标
    for i, (x, y) in enumerate(joints_img):
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1, cv2.LINE_AA)  # 蓝色点
        # 绘制骨骼连接
        for (p1, p2) in HAND_CONNECTIONS:
            if p1 == i:
                x2, y2 = joints_img[p2]
                cv2.line(img, (x, y), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)

    # 5. 恢复归一化关节点（joints2d）并绘制（验证用）
    joints2d = sample['joints2d']
    joints2d_recovered = recover_joints(joints2d, bbox_hand)  # 从归一化坐标恢复
    for (x, y) in joints2d_recovered.astype(int):
        cv2.circle(img, (x, y), 2, (255, 255, 0), -1, cv2.LINE_AA)  # 黄色点（与原始关节点对比）

    # 6. 恢复物体关键点（obj_p2d）并绘制
    obj_p2d = sample['obj_p2d']
    obj_p2d_recovered = recover_joints(obj_p2d, bbox_obj)  # 从归一化坐标恢复
    for (x, y) in obj_p2d_recovered.astype(int):
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)  # 青色点

    # 7. 处理物体掩码（32x32 -> 图像尺寸）
    obj_mask = sample['obj_mask'].numpy()
    mask_resized = cv2.resize(obj_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_color = np.zeros_like(img)
    mask_color[mask_resized == 1] = [255, 0, 255]  # 紫色掩码
    img = cv2.addWeighted(img, 0.8, mask_color, 0.2, 0)  # 叠加掩码

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('HO3D Sample Visualization')
    plt.show()



if __name__ == "__main__":
    import numpy as np
    import cv2
    import torch
    import matplotlib.pyplot as plt
    from dataset.dataset_util import recover_joints  # 导入数据集工具函数

    HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
        [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
        [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
    ]

    dataset = HO3D(dataset_root="/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data",
                   obj_model_root="/root/autodl-tmp/HFL-Net-main/assets/object_models",
                   train_label_root="/data1/zhifeng/ho3d-process", mode="train")

    data = dataset.hand_gcat_datalist[00000]
    # 1. 加载原始图像
    img = cv2.imread(data['img_path'])
    if img is None:
        raise FileNotFoundError(f"图像路径不存在: {data['img_path']}")


    joints_2d = data['joints_coord_img'][:, :2].astype(np.int32)

    # 4. 绘制骨骼线（蓝色）
    line_color = (255, 0, 0)  # BGR格式，蓝色
    line_thickness = 2
    for (i, j) in HAND_CONNECTIONS:
        x1, y1 = joints_2d[i]
        x2, y2 = joints_2d[j]
        cv2.line(img, (x1, y1), (x2, y2), line_color, line_thickness)

    # 5. 绘制关节点（红色，覆盖在线上更清晰）
    point_color = (0, 0, 255)  # 红色
    point_radius = 5
    point_thickness = -1  # 填充圆
    for (x, y) in joints_2d:
        cv2.circle(img, (x, y), point_radius, point_color, point_thickness)

    # 6. 保存或显示结果
    save_path = 'hand_skeleton_visualization.png'
    cv2.imwrite(save_path, img)
    print(f"带骨骼的可视化结果已保存至: {save_path}")

    # 可选：直接显示图像（如果环境支持GUI）
    # cv2.imshow('Joints Visualization', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # sample = dataset.__getitem__(00000)
    # visualize_sample(sample)
    # print(sample)


