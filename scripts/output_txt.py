# -*- coding: utf-8 -*-
"""
功能: 各类输出格式整理工具（合并 XYZ、重命名、过滤点、修正 PKL）。
输入: 配置文件 tools.output_txt 部分。
输出: 指定的 .txt/.pkl 结果文件或目录。
运行: python scripts/output_txt.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lidar_drone_detect.config import load_config


def process_xyz_files_to_single_txt(input_dirs, output_filepath, fixed_color):
    with open(output_filepath, "w") as output_file:
        for input_dir in input_dirs:
            for filename in os.listdir(input_dir):
                if filename.endswith(".xyz"):
                    input_filepath = os.path.join(input_dir, filename)
                    points = np.loadtxt(input_filepath, dtype=np.float32, ndmin=2)

                    for point in points:
                        if len(point) < 3:
                            continue
                        x, y, z = point[:3]
                        output_file.write(
                            f"{x:.6f} {y:.6f} {z:.6f} "
                            f"{fixed_color[0]:.6f} {fixed_color[1]:.6f} {fixed_color[2]:.6f}\n"
                        )

                    print(f"Processed {input_filepath} into {output_filepath}")


def rename_folders_to_area_format(directory):
    folders = sorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])
    for i, folder in enumerate(folders, start=1):
        old_path = os.path.join(directory, folder)
        new_folder_name = f"Area_{i}"
        new_path = os.path.join(directory, new_folder_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{folder}' to '{new_folder_name}'")


def load_points_from_txt(file_path):
    return np.loadtxt(file_path)


def save_points_to_txt(file_path, points):
    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np.savetxt(file_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")


def remove_points(a_file, b_file, c_file):
    points_a = load_points_from_txt(a_file)
    points_b = load_points_from_txt(b_file)

    mask = ~np.in1d(points_a.view([('', points_a.dtype)] * points_a.shape[1]),
                    points_b.view([('', points_b.dtype)] * points_b.shape[1]))
    filtered_points = points_a[mask]

    save_points_to_txt(c_file, filtered_points)
    print(f"Processed {a_file}: saved result to {c_file}")


def strip_singleton_list_in_pkl(pkl_input, pkl_output):
    import mmengine
    data = mmengine.load(pkl_input)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    mmengine.dump(data, pkl_output)
    print("Removed outer list wrapper and saved updated PKL.")


def main():
    parser = argparse.ArgumentParser(description="Output TXT/PKL tools")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tool_cfg = config["tools"]["output_txt"]

    if not tool_cfg["enabled"]:
        print("output_txt tool is disabled in config.")
        return

    mode = tool_cfg["mode"]

    if mode == "merge_xyz":
        process_xyz_files_to_single_txt(
            tool_cfg["merge_xyz"]["input_dirs"],
            tool_cfg["merge_xyz"]["output_file"],
            tool_cfg["fixed_color"],
        )
    elif mode == "rename_folders":
        rename_folders_to_area_format(tool_cfg["rename_folders"]["directory"])
    elif mode == "remove_points":
        remove_points(
            tool_cfg["remove_points"]["a_file"],
            tool_cfg["remove_points"]["b_file"],
            tool_cfg["remove_points"]["c_file"],
        )
    elif mode == "pkl_strip_list":
        strip_singleton_list_in_pkl(tool_cfg["pkl_input"], tool_cfg["pkl_output"])
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
