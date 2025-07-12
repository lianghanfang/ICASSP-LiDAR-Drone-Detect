from point_loader import PointLoader
from voxel_calculator import VoxelCalculator
from dbscan_cluster import DBSCANClusterer
from cluster_evaluator import ClusterEvaluator
from visualizer import Visualizer
from cluster_analyzer import ClusterAnalyzer
import os
import json
from collections import defaultdict


class ClusterPipeline:
    def __init__(self, param_sets, seq):
        self.param_sets = param_sets

    def save_all_clusters(self, cluster_info, result_path, file_prefix):
        """
        将所有聚类信息保存为一个 .json 文件，包括筛选的聚类。
        """
        # 将 NumPy 的 int64 类型转换为 Python 的 int 类型
        converted_cluster_info = {str(label): info for label, info in cluster_info.items()}

        output_file = os.path.join(result_path, f"{file_prefix}_all_clusters.json")
        with open(output_file, 'w') as f:
            json.dump(converted_cluster_info, f, indent=4)
        print(f"All clusters info saved to {output_file}")

    def save_selected_cluster(self, selected_cluster, result_path, file_prefix):
        """
        将筛选出来的聚类单独保存为一个 .json 标签文件。
        """
        selected_label = list(selected_cluster.keys())[0]  # 假设只有一个筛选出的聚类
        output_file = os.path.join(result_path, f"{file_prefix}_selected_cluster.json")
        with open(output_file, 'w') as f:
            json.dump(selected_cluster[selected_label], f, indent=4)
        print(f"Selected cluster info saved to {output_file}")

    def save_clusters_by_source(self, cluster_info, output_folder):
        """
        根据来源文件路径将每个聚类中的点拆分并保存为单独的 .xyz 文件。
        :param cluster_info: 聚类信息字典
        :param output_folder: 保存 .xyz 文件的目标文件夹
        """
        os.makedirs(output_folder, exist_ok=True)

        # 聚合所有点，根据来源文件分类
        points_by_file = defaultdict(list)

        for label, info in cluster_info.items():
            # 遍历聚类中的所有点和来源文件路径
            for point, file_path in zip(info["merged_points"], info["file_paths"]):
                file_name = os.path.basename(file_path)  # 提取文件名
                points_by_file[file_name].append(point)

        # 保存每个文件的点云
        for file_name, points in points_by_file.items():
            output_file = os.path.join(output_folder, file_name)
            try:
                with open(output_file, 'w') as f:  # 每个文件一次性写入，避免追加多次
                    for point in points:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")
                print(f"Saved {len(points)} points to {output_file}")
            except IOError as e:
                print(f"Error saving points to {output_file}: {e}")

    def run(self, lidar_folder, result_path, seq):
        previous_scores = None

        for i, params in enumerate(self.param_sets):
            print(f"Running clustering with parameters: {params}")

            point_loader = PointLoader()
            if not point_loader.merge_xyz_files(lidar_folder):
                continue

            dbscan_clusterer = DBSCANClusterer(params)
            cluster_info = dbscan_clusterer.perform_clustering(point_loader.get_merged_points())

            analyzer = ClusterAnalyzer(cluster_info, params)
            analyzer.calculate_segment_info()
            analyzer.calculate_relative_density()
            analyzer.calculate_roi_union()
            analyzer.calculate_cluster_path_lengths()

            # 计算特征、评分等
            evaluator = ClusterEvaluator()
            for label in cluster_info.keys():
                evaluator.calculate_score(cluster_info, label)

            # 保存每个聚类为单独的 .json 文件
            self.save_all_clusters(cluster_info, result_path, f"params_{i}")

            # 筛选得分最高的聚类，例如选择前 3 个得分最高的聚类
            top_clusters = evaluator.filter_top_clusters(cluster_info, top_n=1)

            self.save_selected_cluster(top_clusters, result_path, f"params_{i}")

            if i == 0:
                self.save_clusters_by_source(top_clusters, os.path.join(result_path, "drone_output"))

            # 保存和可视化筛选后的结果
            Visualizer.save_colored_ply(os.path.join(result_path, f"output_{i}_top3.ply"), top_clusters)

            # save dataset for mmaud
            # if i == 0:
            #     tmp_path = os.path.join(r"E:\CCFA\project\model\2024.11.11\MMAUD", "seq" + str(seq),
            #                             "ALlPoints", "Annotations")
            #
            #     os.makedirs(tmp_path, exist_ok=True)
            #
            #     Visualizer.save_to_txt_with_fixed_color(os.path.join(tmp_path, f"drone.txt"), top_clusters)

            Visualizer.plot_cluster_metrics(i, os.path.join(result_path, f"metrics_{i}_top3.png"), top_clusters,
                                            os.path.join(result_path, f"gt_xyz"))

            # 保存和可视化结果
            Visualizer.save_colored_ply(os.path.join(result_path, f"output_{i}.ply"), cluster_info)
            Visualizer.plot_cluster_metrics(i, os.path.join(result_path, f"metrics_{i}.png"), cluster_info,
                                            os.path.join(result_path, f"gt_xyz"))

            # 比较和筛选
            current_scores = {label: cluster_info[label]["score"] for label in cluster_info.keys()}
            if previous_scores is not None:
                filtered_classes = evaluator.compare_and_filter(previous_scores, current_scores)

                # 处理筛选结果...
                filtered_info = {label: cluster_info[label] for label in filtered_classes}

                # 保存和可视化筛选后的结果
                Visualizer.save_colored_ply(os.path.join(result_path, f"output_{i}_filtered.ply"), filtered_info)
                Visualizer.plot_cluster_metrics(i, os.path.join(result_path, f"metrics_{i}_filtered.png"),
                                                filtered_info,
                                                os.path.join(result_path, f"gt_xyz"))

            previous_scores = current_scores

        print("Clustering and evaluation completed for all parameter sets.")
