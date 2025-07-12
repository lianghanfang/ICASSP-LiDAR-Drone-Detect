from point_loader import PointLoader
from voxel_calculator import VoxelCalculator
from dbscan_cluster import DBSCANClusterer
from cluster_evaluator import ClusterEvaluator
from visualizer import Visualizer
from data_converter import XYZConverter
from cluster_pipeline import ClusterPipeline
from avia_cluster import AviaPointCluster
import os


def main(seq):
    sequence = seq
    root_path = r"I:\MMAUDv1\seq" + str(sequence)
    output_file_path = r"I:\MMAUDv1\filtered"
    result_path = os.path.join(output_file_path, str(sequence))

    converter = XYZConverter(root_path=root_path, output_file_path=output_file_path, sequence=sequence)
    # converter.convert_ground_truth()
    # converter.convert_lidar_360()
    # converter.convert_livox_avia()
    lidar_360_filtered = converter.filter_lidar_360(height_threshold=3.5)

    # clusterer = AviaPointCluster(eps=2, min_samples=5)
    # clusterer.cluster_folder(os.path.join(result_path, "livox_avia_xyz"),
    #                          os.path.join(result_path, "lidar_360_filtered"))

    # param_sets = [
    #     {"eps": 2, "min_samples": 10, "max_points_per_cluster": 10000, "min_points_per_cluster": 30, "file_counts": 100,
    #      "cluster_volume": 1, "frames_num": 2},
    #     {"eps": 2, "min_samples": 10, "max_points_per_cluster": 10000, "min_points_per_cluster": 30, "file_counts": 100,
    #      "cluster_volume": 1, "frames_num": 10},
    # ]

    param_sets = [
        {"eps": 2, "min_samples": 10, "max_points_per_cluster": 10000, "min_points_per_cluster": 100,
         "file_counts": 100,
         "cluster_volume": 0.3, "frames_num": 2},
        {"eps": 2, "min_samples": 10, "max_points_per_cluster": 10000, "min_points_per_cluster": 100,
         "file_counts": 100,
         "cluster_volume": 0.3, "frames_num": 5},
        {"eps": 2, "min_samples": 10, "max_points_per_cluster": 10000, "min_points_per_cluster": 100,
         "file_counts": 100,
         "cluster_volume": 0.3},
    ]

    pipeline = ClusterPipeline(param_sets, seq)
    pipeline.run(lidar_360_filtered, result_path, seq)


if __name__ == "__main__":
    # for i in [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 25, 26, 27, 28, 29, 30, 32, 35, 37, 38,
    #           39, 40, 41, 42, 43, 44, 46, 52, 53, 54, 55, 56, 57, 61, 62, 63, 64, 66, 67, 71, 72, 87, 89, 90, 91]:
    for i in [1, 2, 3]:
        main(i)
