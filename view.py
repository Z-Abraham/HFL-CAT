import os
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from manopth.manolayer import ManoLayer

# ------------------------------
# 辅助函数：3D到2D投影（修正版）
# ------------------------------
def projectPoints(points_3d, K):
    """将3D点投影到2D图像平面（修复矩阵维度不匹配问题）"""
    points_3d = np.array(points_3d)  # 确保输入为numpy数组
    K = np.array(K)                  # 确保内参为numpy数组
    
    # 关键修正：使用3D坐标直接计算（不添加齐次坐标的第四维）
    # 矩阵乘法维度：(3x3) @ (3xN) = (3xN)
    points_proj = (K @ points_3d.T).T  # 矩阵乘法
    
    # 透视除法（使用z坐标）
    points_2d = points_proj[:, :2] / points_proj[:, 2:]
    return points_2d.astype(np.int32)

# ------------------------------
# 2D可视化函数
# ------------------------------
def showHandJoints(imgInOrg, gtIn, filename=None):
    """在图像上绘制手部2D关节点"""
    img = imgInOrg.copy()
    # 手部关节点连接关系（MANO模型标准连接）
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
        [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
        [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20]   # 小指
    ]
    # 绘制关节点
    for (x, y) in gtIn:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 绿色点
    # 绘制连接线
    for (i, j) in connections:
        x1, y1 = gtIn[i]
        x2, y2 = gtIn[j]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 保存图像
    if filename:
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img

def showObjJoints(imgIn, gtIn, estIn, filename=None):
    """在图像上绘制物体2D关节点（真实值vs预测值）"""
    img = imgIn.copy()
    # 绘制真实关节点（绿色）
    for (x, y) in gtIn:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    # 绘制预测关节点（红色）
    for (x, y) in estIn:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    if filename:
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img

# ------------------------------
# 3D可视化函数
# ------------------------------
def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """调整3D视角，确保各轴比例一致"""
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # 翻转y和z轴以匹配视觉习惯
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def display_hand(hand_info, mano_faces, ax=None, alpha=0.2, batch_idx=0, show=True, save_path=None):
    """可视化3D手部网格和关节点"""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    verts = hand_info['verts'][batch_idx].numpy()
    joints = hand_info['joints'][batch_idx].numpy()
    
    # 绘制网格
    if mano_faces is not None:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)  # 浅蓝色
        edge_color = (50 / 255, 50 / 255, 50 / 255)     # 深灰色
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    # 绘制关节点（红色）
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r', s=50)
    
    cam_equal_aspect_3d(ax, verts)
    ax.set_title("3D Hand Visualization")
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    return ax

def plot3dVisualize(ax, m, c="b", alpha=0.5, camPose=None):
    """在3D空间中绘制物体或手部模型"""
    verts = m['v'].copy()
    # 应用姿态变换（如果提供）
    if camPose is not None:
        R = camPose[:3, :3]
        t = camPose[:3, 3:]
        verts = (R @ verts.T + t).T
    # 绘制网格
    if 'f' in m:
        faces = m['f']
        mesh = Poly3DCollection(verts[faces], alpha=alpha, color=c)
        ax.add_collection3d(mesh)
    # 调整视角
    cam_equal_aspect_3d(ax, verts)
    return ax

# ------------------------------
# 主功能：批量可视化
# ------------------------------
def visualize_results(pred_path, img_dir, mano_model_path, save_vis_dir, num_samples=10):
    """
    批量可视化预测结果
    pred_path: 预测结果JSON文件路径
    img_dir: 原始图像目录
    mano_model_path: MANO模型文件目录
    save_vis_dir: 可视化结果保存目录
    num_samples: 可视化样本数量
    """
    os.makedirs(save_vis_dir, exist_ok=True)
    os.makedirs(os.path.join(save_vis_dir, "2d"), exist_ok=True)
    os.makedirs(os.path.join(save_vis_dir, "3d"), exist_ok=True)
    
    # 加载预测结果
    with open(pred_path, 'r') as f:
        xyz_pred_list, verts_pred_list = json.load(f)
    
    # 加载MANO模型（用于获取面信息）
    mano_layer = ManoLayer(mano_root=mano_model_path, use_pca=False)
    mano_faces = mano_layer.th_faces.numpy()
    
    # 相机内参（根据实际数据集调整，这里使用HO3D常见内参）
    K = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1]])  # 3x3内参矩阵
    
    # 处理前N个样本
    for i in range(min(num_samples, len(xyz_pred_list))):
        # 加载数据
        pred_joints3d = np.array(xyz_pred_list[i])  # 形状应为(N, 3)
        pred_verts3d = np.array(verts_pred_list[i])
        img_name = f"{i:04d}.png"  # 假设图像命名格式为00000.jpg, 00001.jpg等
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"图像文件不存在: {img_path}，跳过该样本")
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2D可视化（使用修正后的投影函数）
        pred_joints2d = projectPoints(pred_joints3d, K)
        vis_2d_path = os.path.join(save_vis_dir, "2d", f"hand_2d_{i}.png")
        showHandJoints(img, pred_joints2d, filename=vis_2d_path)
        
        # 3D可视化
        hand_info = {
            'verts': torch.tensor(pred_verts3d[None, ...]),  # 增加batch维度
            'joints': torch.tensor(pred_joints3d[None, ...])
        }
        vis_3d_path = os.path.join(save_vis_dir, "3d", f"hand_3d_{i}.png")
        display_hand(hand_info, mano_faces, show=False, save_path=vis_3d_path)
        plt.close()  # 关闭当前图像，避免内存占用
        
        print(f"已处理样本 {i+1}/{num_samples}，结果保存至 {save_vis_dir}")

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 配置路径（请根据实际情况修改）
    PRED_PATH = "/root/autodl-tmp/HFL-Net-main/host_folder/ho3d/pred_epoch_40.json"    # 预测结果JSON路径
    IMG_DIR = "/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data/evaluation/AP10/rgb"         # 原始图像目录
    MANO_MODEL_PATH = "/root/autodl-tmp/HFL-Net-main/manopth/mano/models"                      # MANO模型目录
    SAVE_VIS_DIR = "/root/autodl-tmp/HFL-Net-main"               # 可视化结果保存目录
    
    # 执行可视化
    visualize_results(
        pred_path=PRED_PATH,
        img_dir=IMG_DIR,
        mano_model_path=MANO_MODEL_PATH,
        save_vis_dir=SAVE_VIS_DIR,
        num_samples=10  # 可视化前10个样本
    )