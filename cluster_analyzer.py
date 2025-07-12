from voxel_calculator import VoxelCalculator
import math
from scipy.spatial import KDTree
import numpy as np


class ClusterAnalyzer:
    def __init__(self, cluster_info, params):
        self.cluster_info = cluster_info
        self.params = params

        total_files_count = len(set(self.cluster_info["file_paths"]))  # 动态计算frames num 对于voxel roi 这个越大越好
        """动态计算的是每一个聚类的类的文件数来计算 而不是参数集的传入文件数"""
        self.frames_num = params.get('frames_num', math.ceil(total_files_count / 2))
        print(f"Using frames_num: {self.frames_num} (calculated as half of total files count: {total_files_count})")

    def calculate_segment_info(self):
        print("Calculating segment info...")
        for label, info in self.cluster_info.items():
            segments = self.segment_file_paths(info["file_paths"])
            total_segments = len(segments)

            # 将生成的 segments 存储到 info 中
            info["segments"] = segments
            print(f"Processing label {label} with {len(info['merged_points'])} points...")

            for idx, segment in enumerate(segments):
                segment_points = []
                for file_path in segment:
                    segment_points.extend([point for i, point in enumerate(info["merged_points"])
                                           if info["file_paths"][i] == file_path])

                if segment_points:
                    volume = VoxelCalculator.calculate_voxel_volume(segment_points)
                    if volume > 0:

                        # 这里可能有点问题 权重应该是 （分段文件数）/（总文件数）
                        # 分段内的点数与整个聚类的点数之比
                        # density = (len(segment_points) / volume) * (len(segment_points) / len(info["merged_points"]))
                        density = (len(segment_points) / volume)
                    else:
                        density = 0

                    info["segments_info"][idx] = {
                        "points": segment_points,
                        "volume": volume,
                        "counts": len(segment_points),
                        "density": density,
                        "roi": 0,
                        "union": 0
                    }

                print(
                    f"Processed segment {idx + 1}/{total_segments} for label {label}, "
                    f"segment size: {len(segment_points)}")
        print("Calculate segment info done")

    def calculate_relative_density(self):
        print("Calculating Relative Density...")
        for label, info in self.cluster_info.items():
            total_volume = VoxelCalculator.calculate_voxel_volume(info["merged_points"])

            if total_volume > 0:
                class_density = len(info["merged_points"]) / total_volume
            else:
                class_density = 0
                print(f"Class {label} has zero volume, skipping density calculation.")
                continue

            total_density_difference = 0
            valid_segments = 0

            for i, segment_info in info["segments_info"].items():
                segment_points = segment_info["points"]
                segment_volume = segment_info["volume"]

                if len(segment_points) > 0 and segment_volume > 0:
                    segment_density = len(segment_points) / segment_volume
                    density_difference = abs(segment_density - class_density) / class_density  # 相对差异 相对差异越小越好
                    total_density_difference += density_difference
                    valid_segments += 1
                else:
                    print(f"Skipping segment {i} for label {label} due to empty segment points or zero volume.")

            if valid_segments > 0:
                average_density_difference = total_density_difference / float(valid_segments)
                info["relative_density"] = average_density_difference
            else:
                info["relative_density"] = 0

            print(f"Relative density for label {label}: {info['relative_density']}")

        print("Calculate Relative Density done")

    """
    为什么这个voxel roi没有得分显示呢？
    中间什么地方出了问题 还是画图的时候没有画？
    """
    def calculate_roi_union(self, use_zero_for_empty_segments=True):
        print("Calculating RoI Union...")
        for label, info in self.cluster_info.items():
            score = 0

            total_segments = len(info["segments"])
            valid_segments = 0
            total_files_count = len(set(info["file_paths"]))

            # 计算这个类的总文件数，去重后计算
            # print(f"Total files count for label {label}: {total_files_count}")
            # print(f"Total files segments for label {label}: {total_segments}")

            for i in range(total_segments - 1):

                print(f"Processing segment pair {i} and {i + 1} for label {label}...")

                if i >= len(info["segments"]) - 1:  # 确保不会超出范围
                    print(f"Skipping segment {i} for label {label} due to out of range.")
                    continue

                segment_files = set(info["segments"][i] + info["segments"][i + 1])
                segment_files_count = len(segment_files)
                print(f"Segment files count: {segment_files_count}")

                segment1_points = info["segments_info"].get(i, {}).get("points", [])
                segment2_points = info["segments_info"].get(i + 1, {}).get("points", [])

                if len(segment1_points) == 0 or len(segment2_points) == 0:
                    if use_zero_for_empty_segments:  # 如果文件中有的segments中不包含点 就取0 然后加权
                        print(f"Segment {i} for label {label} has empty points. Assigning score of 0.")
                        overlap_volume = 0
                        union_volume = 1  # 为了避免除零错误，将联合体积设为 1
                        tmp_score = 0
                    else:
                        print(f"Skipping segment {i} for label {label} due to empty segment points.")
                        continue
                else:
                    overlap_volume = VoxelCalculator.calculate_overlap_volume(segment1_points, segment2_points)
                    union_volume = VoxelCalculator.calculate_voxel_volume(segment1_points + segment2_points)
                    tmp_score = overlap_volume / union_volume if union_volume > 0 else 0

                print(
                    f"Segment {i} - overlap_volume: {overlap_volume}, "
                    f"union_volume: {union_volume}, tmp_score: {tmp_score}")

                # 计算文件数的权重
                segment_files_count = len(set(info["segments"][i] + info["segments"][i + 1]))
                weighted_score = tmp_score * (segment_files_count / total_files_count)

                print(f"Weighted score for segment {i}: {weighted_score}")

                score += weighted_score
                valid_segments += 1
                print("tmp_score:", tmp_score, "weighted_score:", weighted_score)

                info["segments_info"][i]["roi"] = overlap_volume
                info["segments_info"][i]["union"] = union_volume

                print(
                    f"Processed segment {i} for label {label}: overlap_volume={overlap_volume}, "
                    f"union_volume={union_volume}")

            if valid_segments > 0:
                info["voxel_RoI"] = score / valid_segments
            else:
                info["voxel_RoI"] = 0

            print(f"Final score for label {label}: {info['voxel_RoI']}")

        print("Calculate RoI Union done")

    def segment_file_paths(self, file_paths):
        """
        根据每个类的文件路径进行分段，以确保每个段中都有点。

        :param file_paths: 该类点来自的文件路径列表
        :return: 分段后的文件路径列表
        """
        print("Segmenting file paths based on cluster data...")
        segments = []
        unique_files = list(set(file_paths))
        unique_files.sort()

        # 使用类属性 frames_num
        frames_num = self.frames_num

        # 分段，每段中包含的文件数等于 frames_num
        for i in range(0, len(unique_files), frames_num):
            segments.append(unique_files[i:i + frames_num])

        print(f"Segment file paths done, total segments: {len(segments)}")
        return segments

    def calculate_path_length(self, points):
        kdtree = KDTree(points)
        total_length = 0
        for i, point in enumerate(points):
            distances, indices = kdtree.query(point, k=2)
            nearest_point = points[indices[1]]
            total_length += np.linalg.norm(point - nearest_point)
        return total_length

    # 计算每个聚类的路径长度
    def calculate_cluster_path_lengths(self):
        print("Calculating Path Lengths for each cluster...")
        for label, info in self.cluster_info.items():
            points = np.array(info["merged_points"])
            if len(points) > 1:
                path_length = self.calculate_path_length(points)
                info["path_length"] = path_length
                print(f"Path length for label {label}: {path_length}")
            else:
                info["path_length"] = 0
                print(f"Skipping label {label} due to insufficient points.")
        print("Path length calculation done")

