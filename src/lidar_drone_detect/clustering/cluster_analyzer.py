# -*- coding: utf-8 -*-
"""
功能: 对聚类结果做分段、密度、RoI 与路径长度等分析。
输入: 聚类信息字典、参数与体素配置。
输出: 在聚类信息中补充分析指标。
运行: 由聚类流水线调用。
"""

import math
import numpy as np
from scipy.spatial import KDTree

from lidar_drone_detect.utils.voxel_calculator import VoxelCalculator


class ClusterAnalyzer:
    def __init__(self, cluster_info, params, voxel_size, frames_num_fallback_ratio):
        self.cluster_info = cluster_info
        self.params = params
        self.voxel_size = voxel_size

        total_files_count = len(set(self.cluster_info["file_paths"]))
        fallback_frames = math.ceil(total_files_count * frames_num_fallback_ratio)
        self.frames_num = params.get("frames_num", fallback_frames)
        print(
            "Using frames_num: {} (calculated as {} * total files count: {})".format(
                self.frames_num,
                frames_num_fallback_ratio,
                total_files_count,
            )
        )

    def calculate_segment_info(self):
        print("Calculating segment info...")
        for label, info in self.cluster_info.items():
            segments = self.segment_file_paths(info["file_paths"])
            total_segments = len(segments)

            info["segments"] = segments
            print(f"Processing label {label} with {len(info['merged_points'])} points...")

            for idx, segment in enumerate(segments):
                segment_points = []
                for file_path in segment:
                    segment_points.extend(
                        [point for i, point in enumerate(info["merged_points"]) if info["file_paths"][i] == file_path]
                    )

                if segment_points:
                    volume = VoxelCalculator.calculate_voxel_volume(segment_points, self.voxel_size)
                    if volume > 0:
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
                    f"segment size: {len(segment_points)}"
                )
        print("Calculate segment info done")

    def calculate_relative_density(self):
        print("Calculating Relative Density...")
        for label, info in self.cluster_info.items():
            total_volume = VoxelCalculator.calculate_voxel_volume(info["merged_points"], self.voxel_size)

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
                    density_difference = abs(segment_density - class_density) / class_density
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

    def calculate_roi_union(self, use_zero_for_empty_segments):
        print("Calculating RoI Union...")
        for label, info in self.cluster_info.items():
            score = 0

            total_segments = len(info["segments"])
            valid_segments = 0
            total_files_count = len(set(info["file_paths"]))

            for i in range(total_segments - 1):
                print(f"Processing segment pair {i} and {i + 1} for label {label}...")

                if i >= len(info["segments"]) - 1:
                    print(f"Skipping segment {i} for label {label} due to out of range.")
                    continue

                segment_files = set(info["segments"][i] + info["segments"][i + 1])
                segment_files_count = len(segment_files)
                print(f"Segment files count: {segment_files_count}")

                segment1_points = info["segments_info"].get(i, {}).get("points", [])
                segment2_points = info["segments_info"].get(i + 1, {}).get("points", [])

                if len(segment1_points) == 0 or len(segment2_points) == 0:
                    if use_zero_for_empty_segments:
                        print(f"Segment {i} for label {label} has empty points. Assigning score of 0.")
                        overlap_volume = 0
                        union_volume = 1
                        tmp_score = 0
                    else:
                        print(f"Skipping segment {i} for label {label} due to empty segment points.")
                        continue
                else:
                    overlap_volume = VoxelCalculator.calculate_overlap_volume(
                        segment1_points,
                        segment2_points,
                        self.voxel_size,
                    )
                    union_volume = VoxelCalculator.calculate_voxel_volume(
                        segment1_points + segment2_points,
                        self.voxel_size,
                    )
                    tmp_score = overlap_volume / union_volume if union_volume > 0 else 0

                print(
                    f"Segment {i} - overlap_volume: {overlap_volume}, "
                    f"union_volume: {union_volume}, tmp_score: {tmp_score}"
                )

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
                    f"union_volume={union_volume}"
                )

            if valid_segments > 0:
                info["voxel_RoI"] = score / valid_segments
            else:
                info["voxel_RoI"] = 0

            print(f"Final score for label {label}: {info['voxel_RoI']}")

        print("Calculate RoI Union done")

    def segment_file_paths(self, file_paths):
        print("Segmenting file paths based on cluster data...")
        segments = []
        unique_files = list(set(file_paths))
        unique_files.sort()

        frames_num = self.frames_num

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

