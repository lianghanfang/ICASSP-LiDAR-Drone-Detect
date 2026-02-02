# -*- coding: utf-8 -*-
"""
功能: 执行点云数据转换、过滤与聚类流水线。
输入: 配置文件与序列号列表。
输出: 聚类结果文件与可视化图。
运行: python scripts/run_pipeline.py --config configs/config.yaml
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

from lidar_drone_detect.config import load_config, resolve_sequence_paths
from lidar_drone_detect.io.data_converter import XYZConverter
from lidar_drone_detect.clustering.avia_cluster import AviaPointCluster
from lidar_drone_detect.pipelines.cluster_pipeline import ClusterPipeline


def run_sequence(sequence, config):
    paths = resolve_sequence_paths(config, sequence)
    os.makedirs(paths["result_root"], exist_ok=True)

    converter = XYZConverter(
        root_path=str(paths["sequence_root"]),
        output_root=str(Path(config["paths"]["output_root"])),
        sequence=sequence,
        input_subfolders=config["paths"]["subfolders"],
        output_subfolders=config["paths"]["output_subfolders"],
    )

    steps = config["run"]["steps"]

    if steps["convert_ground_truth"]:
        converter.convert_ground_truth()
    if steps["convert_lidar_360"]:
        converter.convert_lidar_360()
    if steps["convert_livox_avia"]:
        converter.convert_livox_avia()

    if steps["filter_lidar_360"]:
        lidar_360_filtered = converter.filter_lidar_360(
            height_threshold=config["data_converter"]["lidar_360_height_threshold"],
            lidar_subfolder=config["paths"]["subfolders"]["lidar_360"],
            output_subfolder=config["paths"]["output_subfolders"]["lidar_360_filtered"],
        )
    else:
        lidar_360_filtered = str(paths["output_subfolders"]["lidar_360_filtered"])

    if steps["run_avia_cluster"]:
        avia_cfg = config["avia_cluster"]
        clusterer = AviaPointCluster(
            eps=avia_cfg["eps"],
            min_samples=avia_cfg["min_samples"],
            merge_distance_threshold=avia_cfg["merge_distance_threshold"],
        )
        clusterer.cluster_folder(
            str(paths["output_subfolders"]["livox_avia_xyz"]),
            lidar_360_filtered,
        )

    if steps["run_cluster_pipeline"]:
        pipeline = ClusterPipeline(
            param_sets=config["clustering"]["param_sets"],
            clustering_cfg=config["clustering"],
            pipeline_cfg=config["pipeline"],
            visualization_cfg=config["visualization"],
            output_subfolders=config["paths"]["output_subfolders"],
        )
        pipeline.run(
            lidar_folder=lidar_360_filtered,
            result_path=str(paths["result_root"]),
            gt_folder=str(paths["output_subfolders"]["gt_xyz"]),
        )


def discover_sequences(config):
    run_cfg = config["run"]
    if not run_cfg["process_all_sequences"]:
        return run_cfg["sequences"]

    dataset_root = Path(config["paths"]["dataset_root"])
    if not dataset_root.exists():
        print("Dataset root not found, fallback to run.sequences.")
        return run_cfg["sequences"]

    pattern = config["paths"]["seq_dir_pattern"]
    if "{sequence}" in pattern:
        prefix, suffix = pattern.split("{sequence}")
    else:
        prefix, suffix = pattern, ""

    sequences = []
    for entry in dataset_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith(prefix) or (suffix and not name.endswith(suffix)):
            continue
        middle = name[len(prefix):]
        if suffix:
            middle = middle[: -len(suffix)]
        if middle.isdigit():
            sequences.append(int(middle))

    if not sequences:
        print("No sequences discovered, fallback to run.sequences.")
        return run_cfg["sequences"]

    return sorted(sequences)


def main():
    parser = argparse.ArgumentParser(description="Run LiDAR clustering pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    for sequence in discover_sequences(config):
        run_sequence(sequence, config)


if __name__ == "__main__":
    main()
