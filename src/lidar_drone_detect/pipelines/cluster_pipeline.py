# -*- coding: utf-8 -*-
"""
功能: 串联点云加载、聚类、分析、评分与可视化输出。
输入: 点云目录、输出目录、配置参数。
输出: 聚类结果、指标图与筛选结果文件。
运行: 由 scripts/run_pipeline.py 调用。
"""

import json
import os
from collections import defaultdict

from lidar_drone_detect.io.point_loader import PointLoader
from lidar_drone_detect.clustering.dbscan_cluster import DBSCANClusterer
from lidar_drone_detect.clustering.cluster_evaluator import ClusterEvaluator
from lidar_drone_detect.clustering.cluster_analyzer import ClusterAnalyzer
from lidar_drone_detect.visualization.visualizer import Visualizer


class ClusterPipeline:
    def __init__(self, param_sets, clustering_cfg, pipeline_cfg, visualization_cfg, output_subfolders):
        self.param_sets = param_sets
        self.clustering_cfg = clustering_cfg
        self.pipeline_cfg = pipeline_cfg
        self.visualization_cfg = visualization_cfg
        self.output_subfolders = output_subfolders
        self.top_n = clustering_cfg["top_n"]

    def _format_output_name(self, template, index):
        return template.format(index=index, top_n=self.top_n)

    def save_all_clusters(self, cluster_info, output_file):
        converted_cluster_info = {str(label): info for label, info in cluster_info.items()}

        with open(output_file, "w") as f:
            json.dump(converted_cluster_info, f, indent=4)
        print(f"All clusters info saved to {output_file}")

    def save_selected_cluster(self, selected_cluster, output_file):
        selected_label = list(selected_cluster.keys())[0]
        with open(output_file, "w") as f:
            json.dump(selected_cluster[selected_label], f, indent=4)
        print(f"Selected cluster info saved to {output_file}")

    def save_clusters_by_source(self, cluster_info, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        points_by_file = defaultdict(list)

        for label, info in cluster_info.items():
            for point, file_path in zip(info["merged_points"], info["file_paths"]):
                file_name = os.path.basename(file_path)
                points_by_file[file_name].append(point)

        for file_name, points in points_by_file.items():
            output_file = os.path.join(output_folder, file_name)
            try:
                with open(output_file, "w") as f:
                    for point in points:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")
                print(f"Saved {len(points)} points to {output_file}")
            except IOError as e:
                print(f"Error saving points to {output_file}: {e}")

    def run(self, lidar_folder, result_path, gt_folder):
        previous_scores = None

        for i, params in enumerate(self.param_sets):
            print(f"Running clustering with parameters: {params}")

            point_loader = PointLoader()
            if not point_loader.merge_xyz_files(lidar_folder):
                continue

            dbscan_clusterer = DBSCANClusterer(
                params,
                voxel_size=self.clustering_cfg["voxel_size"],
                min_centroid_z=self.clustering_cfg["min_centroid_z"],
            )
            cluster_info = dbscan_clusterer.perform_clustering(point_loader.get_merged_points())

            analyzer = ClusterAnalyzer(
                cluster_info,
                params,
                voxel_size=self.clustering_cfg["voxel_size"],
                frames_num_fallback_ratio=self.clustering_cfg["frames_num_fallback_ratio"],
            )
            analyzer.calculate_segment_info()
            analyzer.calculate_relative_density()
            analyzer.calculate_roi_union(
                use_zero_for_empty_segments=self.clustering_cfg["use_zero_for_empty_segments"]
            )
            analyzer.calculate_cluster_path_lengths()

            evaluator = ClusterEvaluator()
            for label in cluster_info.keys():
                evaluator.calculate_score(cluster_info, label)

            if self.visualization_cfg["save_all_clusters"]:
                output_all_clusters = os.path.join(
                    result_path,
                    self._format_output_name(self.pipeline_cfg["output_filenames"]["all_clusters"], i),
                )
                self.save_all_clusters(cluster_info, output_all_clusters)

            top_clusters = evaluator.filter_top_clusters(cluster_info, top_n=self.top_n)

            if self.visualization_cfg["save_top_clusters"]:
                output_selected = os.path.join(
                    result_path,
                    self._format_output_name(self.pipeline_cfg["output_filenames"]["selected_cluster"], i),
                )
                self.save_selected_cluster(top_clusters, output_selected)

            if i == 0 and self.pipeline_cfg["save_clusters_by_source_first_param_only"]:
                self.save_clusters_by_source(
                    top_clusters,
                    os.path.join(result_path, self.output_subfolders["drone_output"]),
                )

            if self.visualization_cfg["save_colored_ply"]:
                output_top_ply = os.path.join(
                    result_path,
                    self._format_output_name(self.pipeline_cfg["output_filenames"]["output_top_ply"], i),
                )
                Visualizer.save_colored_ply(output_top_ply, top_clusters)

            if self.visualization_cfg["save_metrics_plot"]:
                output_top_metrics = os.path.join(
                    result_path,
                    self._format_output_name(self.pipeline_cfg["output_filenames"]["metrics_top_plot"], i),
                )
                Visualizer.plot_cluster_metrics(
                    i,
                    output_top_metrics,
                    top_clusters,
                    gt_folder,
                    self.visualization_cfg["plot"],
                    self.visualization_cfg["correct_label_distance_threshold"],
                )

            if self.visualization_cfg["save_colored_ply"]:
                output_ply = os.path.join(
                    result_path,
                    self._format_output_name(self.pipeline_cfg["output_filenames"]["output_ply"], i),
                )
                Visualizer.save_colored_ply(output_ply, cluster_info)

            if self.visualization_cfg["save_metrics_plot"]:
                output_metrics = os.path.join(
                    result_path,
                    self._format_output_name(self.pipeline_cfg["output_filenames"]["metrics_plot"], i),
                )
                Visualizer.plot_cluster_metrics(
                    i,
                    output_metrics,
                    cluster_info,
                    gt_folder,
                    self.visualization_cfg["plot"],
                    self.visualization_cfg["correct_label_distance_threshold"],
                )

            current_scores = {label: cluster_info[label]["score"] for label in cluster_info.keys()}
            if previous_scores is not None:
                filtered_classes = evaluator.compare_and_filter(previous_scores, current_scores)

                filtered_info = {label: cluster_info[label] for label in filtered_classes}

                if self.visualization_cfg["save_colored_ply"]:
                    output_filtered = os.path.join(
                        result_path,
                        self._format_output_name(self.pipeline_cfg["output_filenames"]["output_filtered_ply"], i),
                    )
                    Visualizer.save_colored_ply(output_filtered, filtered_info)

                if self.visualization_cfg["save_metrics_plot"]:
                    output_filtered_metrics = os.path.join(
                        result_path,
                        self._format_output_name(self.pipeline_cfg["output_filenames"]["metrics_filtered_plot"], i),
                    )
                    Visualizer.plot_cluster_metrics(
                        i,
                        output_filtered_metrics,
                        filtered_info,
                        gt_folder,
                        self.visualization_cfg["plot"],
                        self.visualization_cfg["correct_label_distance_threshold"],
                    )

            previous_scores = current_scores

        print("Clustering and evaluation completed for all parameter sets.")

