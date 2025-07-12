import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from voxel_calculator import VoxelCalculator


class DBSCANClusterer:
    def __init__(self, params):
        self.params = params
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
        unique_labels, counts = np.unique(labels, return_counts=True)

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_points = [points[i] for i in cluster_indices]
            cluster_volume = VoxelCalculator.calculate_voxel_volume([point["point"] for point in cluster_points])
            centroid_magnitude, centroid_z = VoxelCalculator.calculate_centroid_magnitude(
                [point["point"] for point in cluster_points])
            cluster_count = len(cluster_indices)

            if cluster_count > self.params["max_points_per_cluster"] or \
                    cluster_count < self.params["min_points_per_cluster"] or \
                    centroid_z < 6 or \
                    cluster_volume < self.params["cluster_volume"]:
                continue

            self.cluster_info[label]["merged_points"] = [point["point"] for point in cluster_points]
            self.cluster_info[label]["file_paths"] = [point["file"] for point in cluster_points]
            self.cluster_info[label]["counts"] = cluster_count
            self.cluster_info[label]["volume"] = cluster_volume
            self.cluster_info[label]["global_density"] = cluster_count / cluster_volume

        return self.cluster_info
