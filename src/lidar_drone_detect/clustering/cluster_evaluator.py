# -*- coding: utf-8 -*-
"""
功能: 计算聚类评分并进行筛选。
输入: 聚类信息字典与标签。
输出: 更新后的评分字段与筛选结果。
运行: 由聚类流水线调用。
"""


class ClusterEvaluator:
    @staticmethod
    def calculate_score(cluster_info, label):
        info = cluster_info[label]
        path_length_value = info.get("path_length", 0)
        relative_density_value = info["relative_density"]
        voxel_roi_value = info["voxel_RoI"]
        cluster_info[label]["score"]["relative_density_score"] = relative_density_value
        cluster_info[label]["score"]["voxel_roi_score"] = voxel_roi_value
        cluster_info[label]["score"]["path_length_score"] = path_length_value

    @staticmethod
    def compare_and_filter(prev_scores, current_scores):
        filtered_classes = {}
        for label, current_score in current_scores.items():
            if label in prev_scores:
                prev_score = prev_scores[label]
                if (current_score["relative_density_score"] < prev_score["relative_density_score"] and
                        current_score["voxel_roi_score"] > prev_score["voxel_roi_score"]):
                    filtered_classes[label] = label
        return filtered_classes

    @staticmethod
    def calculate_total_score(cluster_info):
        print("Calculating Total Score...")
        for label, info in cluster_info.items():
            relative_density_score = info.get("relative_density", 0)
            voxel_roi_score = info.get("voxel_RoI", 0)

            info["score"] = {
                "total_score": 0.5 * relative_density_score + 0.5 * voxel_roi_score
            }

            print(f"Total score for label {label}: {info['score']['total_score']}")

        print("Calculate Total Score done")

    @staticmethod
    def filter_top_clusters(cluster_info, top_n):
        sorted_clusters = sorted(cluster_info.items(), key=lambda item: item[1]["voxel_RoI"], reverse=True)
        top_clusters = dict(sorted_clusters[:top_n])
        return top_clusters

