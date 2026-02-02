# ICASSP LiDAR Drone Detect

本项目对点云无人机检测流程进行工程化整理，所有路径/参数/开关统一放在 `configs/config.yaml` 中管理。

## 环境依赖

- Python 3.8+
- 必需依赖
  - numpy
  - scipy
  - scikit-learn
  - matplotlib
  - plyfile
  - pyyaml
- 视具体脚本可选
  - torch, mamba-ssm（`scripts/check_env.py`）
  - mmengine（`scripts/output_txt.py` 中的 pkl 工具）

建议用虚拟环境安装依赖。

## 配置方式

1. 打开 `configs/config.yaml`。
2. 在 `paths` / `run` / `clustering` / `visualization` / `tools` 中填写实际路径与参数。
3. 所有脚本均通过 `--config` 指定配置文件（不指定时默认读取 `configs/config.yaml`）。

## 脚本说明与运行方式

- 聚类流水线
  - `scripts/run_pipeline.py`
  - 作用: 数据转换/过滤 + 聚类/分析/可视化
  - 运行: `python scripts/run_pipeline.py --config configs/config.yaml`

- 雷达点云转换
  - `scripts/radar_convert.py`
  - 作用: 雷达增强点云 `.npy` -> `.xyz`（可选过滤）
  - 运行: `python scripts/radar_convert.py --config configs/config.yaml`

- 输出格式整理工具
  - `scripts/output_txt.py`
  - 作用: 合并 XYZ、重命名目录、移除点、修正 PKL
  - 运行: `python scripts/output_txt.py --config configs/config.yaml`
  - 说明: 由 `tools.output_txt.enabled` 与 `tools.output_txt.mode` 控制

- 环境检查
  - `scripts/check_env.py`
  - 作用: 检测 CUDA/CPU 并执行一次 Mamba 前向
  - 运行: `python scripts/check_env.py --config configs/config.yaml`

- 旧版单文件流程（保留）
  - `scripts/legacy_point_cluster.py`
  - 作用: 旧版聚类流程，不建议日常使用
  - 运行: `python scripts/legacy_point_cluster.py --config configs/config.yaml`

## 典型使用流程

1. 在 `configs/config.yaml` 设置数据根目录与输出目录。
2. 设置 `run.sequences` 与 `run.steps`。
3. 执行 `scripts/run_pipeline.py`。
4. 在输出目录查看 `.ply` / `.png` / `.json` 结果。

## 推荐目录结构与职责说明

```
ICASSP-LiDAR-Drone-Detect-main/
├─ configs/
│  └─ config.yaml               # 全局配置（路径/参数/开关）
├─ scripts/
│  ├─ run_pipeline.py           # 主聚类流水线入口
│  ├─ radar_convert.py          # 雷达点云转换
│  ├─ output_txt.py             # 输出/格式整理工具
│  ├─ check_env.py              # 运行环境检查
│  └─ legacy_point_cluster.py   # 旧版单文件流程（保留）
├─ src/
│  └─ lidar_drone_detect/
│     ├─ config.py              # 配置加载与路径解析
│     ├─ io/
│     │  ├─ data_converter.py   # 数据转换/过滤
│     │  ├─ point_loader.py     # 点云加载与合并
│     │  └─ data_loader.py      # PyTorch 数据集/加载器
│     ├─ clustering/
│     │  ├─ dbscan_cluster.py   # DBSCAN 聚类
│     │  ├─ cluster_analyzer.py # 分段/密度/RoI/路径分析
│     │  ├─ cluster_evaluator.py# 聚类评分/筛选
│     │  └─ avia_cluster.py     # Avia 点云聚类
│     ├─ pipelines/
│     │  └─ cluster_pipeline.py # 聚类完整流程编排
│     ├─ visualization/
│     │  └─ visualizer.py       # 可视化与导出
│     └─ utils/
│        └─ voxel_calculator.py # 体素几何计算
└─ README.md
```

如需新增功能，优先在 `src/` 中补模块，再在 `scripts/` 增加入口脚本，并统一走 `configs/config.yaml`。
