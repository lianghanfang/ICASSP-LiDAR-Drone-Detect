# -*- coding: utf-8 -*-
"""
功能: 将点云数据集中的 .npy 转为 .xyz，并支持高度阈值过滤。
输入: 数据集根目录、输出目录、序列号及子目录配置。
输出: 生成的 .xyz 文件与过滤结果目录。
运行: 由流水线脚本调用。
"""

import os
import numpy as np


class XYZConverter:
    def __init__(self, root_path, output_root, sequence, input_subfolders, output_subfolders):
        self.root_path = root_path
        self.output_root = output_root
        self.sequence = sequence
        self.input_subfolders = input_subfolders
        self.output_subfolders = output_subfolders

    def _build_output_folder(self, subfolder_name):
        if subfolder_name:
            return os.path.join(self.output_root, str(self.sequence), subfolder_name)
        return os.path.join(self.output_root, str(self.sequence))

    def _write_xyz_file(self, data_array, output_file):
        if data_array.size == 0 or np.all(data_array == 0):
            print(f"Skipping saving {output_file} as it contains no valid points.")
            return

        with open(output_file, "w") as f:
            if data_array.ndim == 1 and len(data_array) == 3:
                x, y, z = data_array
                if not np.all([x, y, z] == 0):
                    f.write(f"{x} {y} {z}\n")
            elif data_array.ndim == 2:
                valid_points = 0
                for point in data_array:
                    if len(point) == 3 and not np.all(point == 0):
                        x, y, z = point
                        f.write(f"{x} {y} {z}\n")
                        valid_points += 1
                if valid_points == 0:
                    print(f"No valid points in {output_file}. File will not be saved.")
                    f.close()
                    os.remove(output_file)
            else:
                print(f"Unexpected data format: {data_array}")

    def _convert_to_xyz(self, folder_name, subfolder_name):
        folder_path = os.path.join(self.root_path, folder_name)
        file_list = os.listdir(folder_path)
        output_folder = self._build_output_folder(subfolder_name)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in file_list:
            data_array = np.load(os.path.join(folder_path, file))
            if data_array.size == 0 or np.all(data_array == 0):
                print(f"Skipping {file} as it contains no valid points.")
                continue

            filename = os.path.splitext(file)[0] + ".xyz"
            output_file = os.path.join(output_folder, filename)
            self._write_xyz_file(data_array, output_file)

        print(f"Sequence {self.sequence}: {folder_name} to {subfolder_name} xyz file conversion complete.")

    def convert_ground_truth(self):
        self._convert_to_xyz(self.input_subfolders["ground_truth"], self.output_subfolders["gt_xyz"])

    def convert_lidar_360(self):
        self._convert_to_xyz(self.input_subfolders["lidar_360"], self.output_subfolders["lidar_360_xyz"])

    def convert_livox_avia(self):
        self._convert_to_xyz(self.input_subfolders["livox_avia"], self.output_subfolders["livox_avia_xyz"])

    def convert_custom(self, input_subfolder, output_subfolder):
        self._convert_to_xyz(input_subfolder, output_subfolder)

    def filter_lidar_360(self, height_threshold, lidar_subfolder, output_subfolder):
        l360_path = os.path.join(self.root_path, lidar_subfolder)
        file_list = os.listdir(l360_path)
        output_folder = self._build_output_folder(output_subfolder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in file_list:
            data_array = np.load(os.path.join(l360_path, file))
            filtered_points = [
                point
                for point in data_array
                if len(point) == 3 and not np.all(point == 0) and point[2] > height_threshold
            ]

            if len(filtered_points) == 0:
                print(f"No valid points met the threshold in {file}, skipping file creation.")
                continue

            filename = os.path.splitext(file)[0] + ".xyz"
            output_file = os.path.join(output_folder, filename)

            self._write_xyz_file(np.array(filtered_points), output_file)

        print(f"Sequence {self.sequence}: Lidar 360 xyz file filtering complete.")
        return output_folder

