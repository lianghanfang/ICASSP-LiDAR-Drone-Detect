# -*- coding: utf-8 -*-
"""
功能: 读取 YAML 配置并提供路径解析工具。
输入: YAML 配置文件路径（可选）。
输出: 配置字典与序列相关路径字典。
运行: 在脚本中调用 load_config(...) / resolve_sequence_paths(...).
"""

from pathlib import Path
import os
from typing import Dict, Any, Optional

import yaml


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.environ.get("ICASSPLIDAR_CONFIG")
    if config_path is None:
        config_path = get_repo_root() / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_sequence_paths(config: Dict[str, Any], sequence: int) -> Dict[str, Any]:
    paths_cfg = config["paths"]

    dataset_root = Path(paths_cfg["dataset_root"])
    output_root = Path(paths_cfg["output_root"])
    seq_dir = paths_cfg["seq_dir_pattern"].format(sequence=sequence)
    output_seq_dir = paths_cfg["output_seq_dir_pattern"].format(sequence=sequence)

    sequence_root = dataset_root / seq_dir
    result_root = output_root / output_seq_dir

    input_subfolders = {
        name: sequence_root / subfolder
        for name, subfolder in paths_cfg["subfolders"].items()
    }
    output_subfolders = {
        name: result_root / subfolder
        for name, subfolder in paths_cfg["output_subfolders"].items()
    }

    return {
        "sequence": sequence,
        "sequence_root": sequence_root,
        "result_root": result_root,
        "input_subfolders": input_subfolders,
        "output_subfolders": output_subfolders,
    }

