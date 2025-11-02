import numpy as np
import json
import os
import cv2
from PIL import Image
from utils.config import cfg

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def load_names(image_path):
    with open(image_path) as f:
        return [line.strip() for line in f]


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def read_RGB_img(base_dir, seq_name, file_id, split):
    img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')
    img = Image.open(img_filename).convert("RGB")
    return img


def read_gray_img(base_dir, seq_name, file_id):
    img_filename = os.path.join(base_dir, seq_name, file_id + '.png')
    img = Image.open(img_filename)
    return img


def pose_from_RT(R, T):
    pose = np.zeros((4, 4))
    pose[:3, 3] = T
    pose[3, 3] = 1
    R33, _ = cv2.Rodrigues(R)
    pose[:3, :3] = R33
    # change from OpenGL coord to normal coord
    pose[1, :] = -pose[1, :]
    pose[2, :] = -pose[2, :]
    return pose


def projectPoints(xyz, K, rt=None):
    xyz = np.array(xyz)
    K = np.array(K)
    if rt is not None:
        uv = np.matmul(K, np.matmul(rt[:3, :3], xyz.T) + rt[:3, 3].reshape(-1, 1)).T
    else:
        uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def get_bbox(joint_img, joint_valid, expansion_factor=1.0):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1];
    y_img = y_img[joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = (xmax - xmin) * expansion_factor;
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width

    y_center = (ymin + ymax) / 2.;
    height = (ymax - ymin) * expansion_factor;
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def process_bbox(bbox, img_width, img_height, expansion_factor=1.25):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = cfg.input_img_shape[1] / cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * expansion_factor
    bbox[3] = h * expansion_factor
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


def load_objects_HO3D(obj_root):
    import trimesh
    object_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
                    '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors', '004_sugar_box']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, 'points.xyz')
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = np.array(mesh.vertices)
    return all_models


def filter_test_object_ho3d(mesh_dict, diameter_dict):
    # filter objects in the evaluation dataset
    mesh_dict_out, diameter_dict_out = {}, {}
    for k in mesh_dict.keys():
        # print(k)
        # if k in range(1,22):
        if k in ['021_bleach_cleanser', '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can']:
            mesh_dict_out[k] = mesh_dict[k]
            diameter_dict_out[k] = diameter_dict[k]
    return mesh_dict_out, diameter_dict_out


def filter_test_object_dexycb(mesh_dict, diameter_dict):
    # filter objects in the evaluation dataset
    mesh_dict_out, diameter_dict_out = {}, {}
    for k in mesh_dict.keys():
        # print(k)
        # if k in range(1,22):
        # if k in ['021_bleach_cleanser','006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can']:
        mesh_dict_out[k] = mesh_dict[k]
        diameter_dict_out[k] = diameter_dict[k]
    return mesh_dict_out, diameter_dict_out


def get_unseen_test_object():
    # unseen objects appear in evaluation set but not in training set
    return ['019_pitcher_base']
