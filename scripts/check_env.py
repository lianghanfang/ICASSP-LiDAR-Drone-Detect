# -*- coding: utf-8 -*-
"""
功能: 检查 PyTorch/Mamba 运行环境与设备信息。
输入: 配置文件 tools.env_check 参数。
输出: 控制台环境信息与一次前向推理结果。
运行: python scripts/check_env.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from mamba_ssm import Mamba

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lidar_drone_detect.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Environment check")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config["tools"]["env_check"]

    if torch.cuda.is_available():
        device = "cuda"
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    else:
        device = "cpu"
        cuda_version = "N/A"
        cudnn_version = "N/A"

    python_version = sys.version
    conda_env_path = os.environ.get("CONDA_PREFIX", "Not in a Conda environment")

    print("使用的设备: {}".format(device))
    print("CUDA 版本: {}".format(cuda_version))
    print("cuDNN 版本: {}".format(cudnn_version))
    print("Python 版本: {}".format(python_version))
    print("当前环境路径: {}".format(conda_env_path))

    batch, length, dim = env_cfg["batch"], env_cfg["length"], env_cfg["dim"]
    x = torch.randn(batch, length, dim).to(device)
    model = Mamba(
        d_model=dim,
        d_state=env_cfg["d_state"],
        d_conv=env_cfg["d_conv"],
        expand=env_cfg["expand"],
    ).to(device)
    y = model(x)
    assert y.shape == x.shape
    print("成功运行，模型输出维度为: {}".format(y.shape))


if __name__ == "__main__":
    main()
