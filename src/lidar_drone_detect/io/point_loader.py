# -*- coding: utf-8 -*-
"""
功能: 合并目录中的 .xyz 点云文件并提供读取接口。
输入: .xyz 文件夹路径。
输出: 合并点云列表与文件路径列表。
运行: 供聚类流水线调用。
"""

import os


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

