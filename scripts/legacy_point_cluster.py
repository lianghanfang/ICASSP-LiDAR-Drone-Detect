# -*- coding: utf-8 -*-
"""
功能: 旧版单文件聚类流程（保留原始逻辑，便于对比）。
输入: 配置文件 tools.legacy_point_cluster 部分。
输出: 聚类结果 .ply 与指标图。
运行: python scripts/legacy_point_cluster.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lidar_drone_detect.config import load_config


class PointLoader:
    def __init__(self):
        self.point_info = {
            "merged_points": [],
            "file_paths": []
        }

    def merge_xyz_files(self, folder_path):
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath) and filename.endswith(".xyz"):
                with open(filepath, "r") as file:
                    for line in file:
                        if line.strip():
                            point = list(map(float, line.strip().split()))
                            self.point_info["merged_points"].append({"point": point, "file": filename})
                            self.point_info["file_paths"].append(filepath)

        if len(self.point_info["merged_points"]) == 0:
            print(f"Warning: No points found in directory {folder_path}. Skipping processing.")
            return False

        return True

    def get_merged_points(self):
        return self.point_info["merged_points"]

    def get_file_paths(self):
        return self.point_info["file_paths"]


class VoxelCalculator:
    @staticmethod
    def calculate_voxel_volume(cluster_points, voxel_size):
        if len(cluster_points) == 0:
            return 0

        cluster_points = np.array(cluster_points)
        voxel_indices = np.floor(cluster_points / voxel_size).astype(int)
        unique_voxels = np.unique(voxel_indices, axis=0)
        volume = len(unique_voxels) * (voxel_size ** 3)

        return volume

    @staticmethod
    def calculate_overlap_volume(points1, points2, voxel_size):
        if len(points1) == 0 or len(points2) == 0:
            return 0

        points1 = np.array(points1)
        points2 = np.array(points2)

        voxel_indices1 = np.floor(points1 / voxel_size).astype(int)
        voxel_indices2 = np.floor(points2 / voxel_size).astype(int)

        unique_voxels1 = set(map(tuple, np.unique(voxel_indices1, axis=0)))
        unique_voxels2 = set(map(tuple, np.unique(voxel_indices2, axis=0)))

        overlap_voxels = unique_voxels1.intersection(unique_voxels2)
        overlap_volume = len(overlap_voxels) * (voxel_size ** 3)

        return overlap_volume


class DBSCANClusterer:
    def __init__(self, params, voxel_size):
        self.params = params
        self.voxel_size = voxel_size
        self.cluster_info = defaultdict(lambda: {
            "merged_points": [],
            "file_paths": [],
            "counts": 0.0,
            "volume": 0.0,
            "global_density": 0.0,
            "relative_density": 0.0,
            "voxel_RoI": 0.0,
            "segments": [],
            "segments_info": {},
            "score": {
                "relative_density_score": 0.0,
                "voxel_roi_score": 0.0,
                "total_score": 0.0
            }
        })

    def perform_clustering(self, points):
        dbscan = DBSCAN(eps=self.params["eps"], min_samples=self.params["min_samples"])
        tmp_points = [point_info["point"] for point_info in points]
        labels = dbscan.fit_predict(tmp_points)
        unique_labels, _ = np.unique(labels, return_counts=True)

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_points = [points[i] for i in cluster_indices]
            cluster_volume = VoxelCalculator.calculate_voxel_volume(
                [point["point"] for point in cluster_points],
                self.voxel_size,
            )
            cluster_count = len(cluster_indices)

            if cluster_count > self.params["max_points_per_cluster"] or \
                    cluster_count < self.params["min_points_per_cluster"] or \
                    cluster_volume < self.params["cluster_volume"]:
                continue

            self.cluster_info[label]["merged_points"] = [point["point"] for point in cluster_points]
            self.cluster_info[label]["file_paths"] = [point["file"] for point in cluster_points]
            self.cluster_info[label]["counts"] = cluster_count
            self.cluster_info[label]["volume"] = cluster_volume
            self.cluster_info[label]["global_density"] = cluster_count / cluster_volume

        return self.cluster_info


class ClusterEvaluator:
    @staticmethod
    def calculate_score(cluster_info, label):
        info = cluster_info[label]
        relative_density_value = info["relative_density"]
        voxel_roi_value = info["voxel_RoI"]
        cluster_info[label]["score"]["relative_density_score"] = relative_density_value
        cluster_info[label]["score"]["voxel_roi_score"] = voxel_roi_value

    @staticmethod
    def compare_and_filter(prev_scores, current_scores):
        filtered_classes = {}
        for label, current_score in current_scores.items():
            if label in prev_scores:
                prev_score = prev_scores[label]
                if (current_score["relative_density_score"] > prev_score["relative_density_score"] and
                        current_score["voxel_roi_score"] < prev_score["voxel_roi_score"]):
                    filtered_classes[label] = label
        return filtered_classes


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

        vertex = np.array(
            all_points,
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        el = PlyElement.describe(vertex, "vertex")
        PlyData([el], text=True).write(output_filepath)

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
    def plot_cluster_metrics(i, output_filepath, cluster_info, bar_width, log_scale):
        labels = list(cluster_info.keys())
        counts = [info["counts"] for info in cluster_info.values()]
        volumes = [info["volume"] for info in cluster_info.values()]
        relative_densities = [info["score"]["relative_density_score"] for info in cluster_info.values()]
        voxel_rois = [info["score"]["voxel_roi_score"] for info in cluster_info.values()]

        fig, ax = plt.subplots()

        width = bar_width
        x = np.arange(len(labels))

        ax.bar(x - 1.5 * width, counts, width, label="Counts")
        ax.bar(x - 0.5 * width, volumes, width, label="Volume")
        ax.bar(x + 0.5 * width, relative_densities, width, label="Relative Density")
        ax.bar(x + 1.5 * width, voxel_rois, width, label="Voxel RoI")

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Cluster Labels")
        ax.set_ylabel("Metrics")
        ax.set_title(f"Cluster Metrics for Param Set {i}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_filepath)


class ClusterPipeline:
    def __init__(self, param_sets, voxel_size, bar_width, log_scale, output_filenames):
        self.param_sets = param_sets
        self.voxel_size = voxel_size
        self.bar_width = bar_width
        self.log_scale = log_scale
        self.output_filenames = output_filenames

    def _format_name(self, template, index):
        return template.format(index=index)

    def run(self, lidar_folder, result_path):
        previous_scores = None

        for i, params in enumerate(self.param_sets):
            print(f"Running clustering with parameters: {params}")

            point_loader = PointLoader()
            if not point_loader.merge_xyz_files(lidar_folder):
                continue

            dbscan_clusterer = DBSCANClusterer(params, self.voxel_size)
            cluster_info = dbscan_clusterer.perform_clustering(point_loader.get_merged_points())

            evaluator = ClusterEvaluator()
            for label in cluster_info.keys():
                evaluator.calculate_score(cluster_info, label)

            Visualizer.save_colored_ply(
                os.path.join(result_path, self._format_name(self.output_filenames["output_ply"], i)),
                cluster_info,
            )
            Visualizer.plot_cluster_metrics(
                i,
                os.path.join(result_path, self._format_name(self.output_filenames["metrics_plot"], i)),
                cluster_info,
                self.bar_width,
                self.log_scale,
            )

            current_scores = {label: cluster_info[label]["score"] for label in cluster_info.keys()}
            if previous_scores is not None:
                filtered_classes = evaluator.compare_and_filter(previous_scores, current_scores)
                self.save_filtered_classes(
                    filtered_classes,
                    cluster_info,
                    os.path.join(result_path, self._format_name(self.output_filenames["filtered_ply"], i)),
                )

            previous_scores = current_scores

        print("Clustering and evaluation completed for all parameter sets.")

    def save_filtered_classes(self, filtered_classes, cluster_info, output_filepath):
        filtered_points = []
        for label in filtered_classes:
            filtered_points.extend(cluster_info[label]["merged_points"])

        if not filtered_points:
            print("No filtered points to save.")
            return

        vertex = np.array(filtered_points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        el = PlyElement.describe(vertex, "vertex")
        PlyData([el], text=True).write(output_filepath)

        print(f"Filtered points saved to {output_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Legacy point cluster pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    legacy_cfg = config["tools"]["legacy_point_cluster"]

    if not legacy_cfg["enabled"]:
        print("legacy_point_cluster is disabled in config.")
        return

    pipeline = ClusterPipeline(
        param_sets=legacy_cfg["param_sets"],
        voxel_size=legacy_cfg["voxel_size"],
        bar_width=config["visualization"]["plot"]["bar_width"],
        log_scale=config["visualization"]["plot"]["log_scale"],
        output_filenames=legacy_cfg["output_filenames"],
    )
    pipeline.run(legacy_cfg["lidar_folder"], legacy_cfg["result_path"])


if __name__ == "__main__":
    main()
