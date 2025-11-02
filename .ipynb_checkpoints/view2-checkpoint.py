# visualize_results.py
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from utils.manolayer_ho3d import ManoLayer

# --------------------------
# 1. 可视化训练损失曲线
# --------------------------
def plot_loss_curve(log_path="exp_results/train.txt", save_path=None):
    """从训练日志绘制损失曲线"""
    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        return

    with open(log_path, "r") as f:
        lines = f.readlines()

    # 提取epoch和主要损失（可根据日志内容调整关键词）
    epochs = []
    total_losses = []
    for line in lines:
        if "epoch:" in line and "total_loss" in line:
            epoch = int(re.findall(r"epoch: (\d+)", line)[0])
            total_loss = float(re.findall(r"total_loss:([\d.]+)", line)[0])
            epochs.append(epoch)
            total_losses.append(total_loss)

    # 绘制曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, total_losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"损失曲线已保存至: {save_path}")
    else:
        plt.show()


# --------------------------
# 2. 可视化预测的3D手模型
# --------------------------
def visualize_predicted_hand(pred_json_path="exp_results/pred.json", save_img=None):
    """从pred.json可视化预测的手模型（需ManoLayer的面索引）"""
    if not os.path.exists(pred_json_path):
        print(f"预测文件不存在: {pred_json_path}")
        return

    # 初始化ManoLayer获取面索引（与训练时参数一致）
    mano_layer = ManoLayer(
        center_idx=0,
        flat_hand_mean=True,
        ncomps=45,
        side='right',
        mano_root='assets/mano_models',
        use_pca=True
    )
    faces = mano_layer.th_faces.detach().cpu().numpy()

    # 读取预测结果（取第一个样本）
    with open(pred_json_path, "r") as f:
        xyz_pred_list, verts_pred_list = json.load(f)  # 仓库中pred.json的格式
    joints = np.array(xyz_pred_list[0])  # 3D关节
    verts = np.array(verts_pred_list[0])  # 3D网格顶点

    # 创建Open3D几何对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色网格

    joints_pcd = o3d.geometry.PointCloud()
    joints_pcd.points = o3d.utility.Vector3dVector(joints)
    joints_pcd.paint_uniform_color([1, 0, 0])  # 红色关节

    # Jupyter中启用Web可视化
    o3d.visualization.webrtc_server.enable_webrtc()

    # 显示或保存
    if save_img:
        # 渲染为图片
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)
        vis.add_geometry(joints_pcd)
        vis.update_renderer()
        img = vis.capture_screen_float_buffer()
        vis.destroy_window()
        plt.imsave(save_img, np.asarray(img))
        print(f"手模型已保存至: {save_img}")
    else:
        o3d.visualization.draw([mesh, joints_pcd], window_name="Predicted Hand")


# --------------------------
# 3. 一键可视化入口
# --------------------------
if __name__ == "__main__":
    # 可视化损失曲线
    plot_loss_curve(
        log_path="/root/autodl-tmp/HFL-Net-main/host_folder/ho3d/train.txt",
        save_path="exp_results/loss_curve.png"  # 可选：保存图片
    )

    # 可视化预测的手模型（需先运行测试生成pred.json）
    visualize_predicted_hand(
        pred_json_path="/root/autodl-tmp/HFL-Net-main/host_folder/ho3d/pred_40.json",
        save_img="exp_results/predicted_hand.png"  # 可选：保存图片
    )