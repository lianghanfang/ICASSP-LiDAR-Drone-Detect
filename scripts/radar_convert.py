# -*- coding: utf-8 -*-
"""
功能: 将雷达增强点云 .npy 转为 .xyz，并可选执行高度过滤。
输入: 配置文件 tools.radar 部分。
输出: 雷达点云的 .xyz 文件与可选过滤结果。
运行: python scripts/radar_convert.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lidar_drone_detect.config import load_config
from lidar_drone_detect.io.data_converter import XYZConverter


def main():
    parser = argparse.ArgumentParser(description="Radar point cloud conversion")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    radar_cfg = config["tools"]["radar"]

    sequence = radar_cfg["sequence"]
    seq_dir = config["paths"]["seq_dir_pattern"].format(sequence=sequence)
    root_path = os.path.join(radar_cfg["dataset_root"], seq_dir)

    converter = XYZConverter(
        root_path=root_path,
        output_root=radar_cfg["output_root"],
        sequence=sequence,
        input_subfolders=config["paths"]["subfolders"],
        output_subfolders=config["paths"]["output_subfolders"],
    )

    converter.convert_custom(radar_cfg["input_subfolder"], radar_cfg["output_subfolder"])

    if radar_cfg["run_lidar_360_filter"]:
        converter.filter_lidar_360(
            height_threshold=radar_cfg["lidar_360_height_threshold"],
            lidar_subfolder=radar_cfg["lidar_360_subfolder"],
            output_subfolder=config["paths"]["output_subfolders"]["lidar_360_filtered"],
        )


if __name__ == "__main__":
    main()
