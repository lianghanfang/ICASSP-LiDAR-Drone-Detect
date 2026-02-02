# -*- coding: utf-8 -*-
"""
功能: 基于 DBSCAN 的点云聚类，并按条件过滤聚类结果。
输入: 点云列表与聚类参数、体素大小与过滤阈值。
输出: 聚类信息字典（点、体积、密度等）。
运行: 由聚类流水线调用。
"""

import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

from lidar_drone_detect.utils.voxel_calculator import VoxelCalculator


class DBSCANClusterer:
    def __init__(self, params, voxel_size, min_centroid_z):
        self.params = params
        self.voxel_size = voxel_size
        self.min_centroid_z = min_centroid_z
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
                "path_length_score": 0.0,
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
            _, centroid_z = VoxelCalculator.calculate_centroid_magnitude(
                [point["point"] for point in cluster_points]
            )
            cluster_count = len(cluster_indices)

            if cluster_count > self.params["max_points_per_cluster"] or \
                    cluster_count < self.params["min_points_per_cluster"] or \
                    centroid_z < self.min_centroid_z or \
                    cluster_volume < self.params["cluster_volume"]:
                continue

            self.cluster_info[label]["merged_points"] = [point["point"] for point in cluster_points]
            self.cluster_info[label]["file_paths"] = [point["file"] for point in cluster_points]
            self.cluster_info[label]["counts"] = cluster_count
            self.cluster_info[label]["volume"] = cluster_volume
            self.cluster_info[label]["global_density"] = cluster_count / cluster_volume

        return self.cluster_info

