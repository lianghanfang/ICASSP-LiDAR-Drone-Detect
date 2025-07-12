import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import os


class Visualizer:
    @staticmethod
    def save_colored_ply(output_filepath, cluster_info):
        total_scores = [info["score"]["total_score"] for label, info in cluster_info.items()]

        if not total_scores:
            return

        min_score = min(total_scores)
        max_score = max(total_scores)

        all_points = []
        for label, info in cluster_info.items():
            score = info["score"]["total_score"]
            color = Visualizer.score_to_color(score, min_score, max_score)

            for point in info["merged_points"]:
                all_points.append((point[0], point[1], point[2], *color))

        vertex = np.array(all_points,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(output_filepath)

    @staticmethod
    def save_to_txt_with_fixed_color(output_filepath, cluster_info):
        all_points = []
        fixed_color = (100, 100, 100)  # 固定颜色为100

        for label, info in cluster_info.items():
            for point in info["merged_points"]:
                all_points.append((point[0], point[1], point[2], *fixed_color))

        # 将数据保存为 .txt 文件，并格式化到小数点后6位
        with open(output_filepath, 'w') as f:
            for x, y, z, r, g, b in all_points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

    @staticmethod
    def score_to_color(score, min_score, max_score):
        if max_score == min_score:
            norm_score = 0
        else:
            norm_score = (score - min_score) / (max_score - min_score)

        tanh_curve = np.tanh(2 * (norm_score - 0.5))
        blue = int((tanh_curve + 1) / 2 * 255)
        red = int((1 - (tanh_curve + 1) / 2) * 255)
        green = 0

        return red, green, blue

    @staticmethod
    def plot_cluster_metrics(i, output_filepath, cluster_info, gt_folder):
        labels = list(cluster_info.keys())
        counts = [info["counts"] for info in cluster_info.values()]
        volumes = [info["volume"] for info in cluster_info.values()]
        relative_densities = [info["score"]["relative_density_score"] for info in cluster_info.values()]
        voxel_rois = [info["score"]["voxel_roi_score"] for info in cluster_info.values()]
        path_lengths = [info["score"]["path_length_score"] for info in cluster_info.values()]  # 获取路径长度得分

        fig, ax = plt.subplots(figsize=(12, 8))

        width = 0.2
        x = np.arange(len(labels))

        ax.bar(x - 1.5 * width, counts, width, label='Counts')
        ax.bar(x - 0.5 * width, volumes, width, label='Volume')
        ax.bar(x + 0.5 * width, relative_densities, width, label='Relative Density')
        ax.bar(x + 1.5 * width, voxel_rois, width, label='Voxel RoI')
        ax.bar(x + 2 * width, path_lengths, width, label='Path Length')  # 添加路径长度得分

        # for idx, val in enumerate(path_lengths):
        #     ax.text(x[idx] + 2 * width, val, f'{val:.2f}', ha='center', va='bottom')

        ax.set_yscale('log')
        ax.set_xlabel('Cluster Labels')
        ax.set_ylabel('Metrics (Log Scale)')
        ax.set_title(f'Cluster Metrics for Param Set {i}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # 加载 gt_xyz 数据并找到正确的类标签
        gt_points = Visualizer.load_gt_xyz(gt_folder)
        correct_labels = Visualizer.find_correct_labels(cluster_info, gt_points)

        # 标注正确的类标签
        for label in correct_labels:
            if label in labels:
                idx = labels.index(label)
                ax.text(x[idx], max(counts[idx], volumes[idx], relative_densities[idx], voxel_rois[idx]),
                        f'Correct: {label}', ha='center', va='bottom', fontsize=8, color='red')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.close()

    """
    在图中标出真值的类
    """
    @staticmethod
    def load_gt_xyz(gt_folder):
        """加载 gt_xyz 文件夹中的点云数据"""
        gt_points = []
        for filename in os.listdir(gt_folder):
            if filename.endswith('.xyz'):
                filepath = os.path.join(gt_folder, filename)
                with open(filepath, 'r') as file:
                    points = [list(map(float, line.split())) for line in file if line.strip()]
                    gt_points.extend(points)
        return np.array(gt_points)

    @staticmethod
    def find_correct_labels(cluster_info, gt_points):
        """找到正确的聚类标签，假设正确的聚类是与 gt_xyz 数据匹配得分最高的"""
        correct_labels = []
        for label, info in cluster_info.items():
            cluster_points = np.array(info['merged_points'])
            if cluster_points.size == 0:
                continue

            # 计算匹配得分，简单示例为计算最近点的平均距离
            distances = np.linalg.norm(cluster_points[:, None, :] - gt_points[None, :, :], axis=2)
            mean_distance = np.mean(np.min(distances, axis=1))

            # 假设距离越小越匹配
            if mean_distance < 5.0:  # 这里的 1.0 是一个阈值，可以根据需要调整
                correct_labels.append(label)

        return correct_labels
