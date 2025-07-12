import numpy as np


class VoxelCalculator:
    @staticmethod
    def calculate_voxel_volume(cluster_points, voxel_size=0.2):
        if len(cluster_points) == 0:
            return 0

        cluster_points = np.array(cluster_points)
        voxel_indices = np.floor(cluster_points / voxel_size).astype(int)
        unique_voxels = np.unique(voxel_indices, axis=0)
        volume = len(unique_voxels) * (voxel_size ** 3)

        return volume

    @staticmethod
    def calculate_overlap_volume(points1, points2, voxel_size=0.2):
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

    @staticmethod
    def calculate_centroid_magnitude(cluster_points):
        if len(cluster_points) == 0:
            return 0, 0

        # 计算质心
        centroid = np.mean(cluster_points, axis=0)
        # 计算质心的模
        centroid_magnitude = np.linalg.norm(centroid)

        centroid_z = centroid[2]

        return centroid_magnitude, centroid_z
