import torch
import sys
import os
from mamba_ssm import Mamba

# 检测 CUDA 是否可用
if torch.cuda.is_available():
    device = "cuda"
    cuda_version = torch.version.cuda  # CUDA 版本
    cudnn_version = torch.backends.cudnn.version()  # cuDNN 版本
else:
    device = "cpu"
    cuda_version = "N/A"
    cudnn_version = "N/A"

# 获取当前 Python 版本
python_version = sys.version

# 获取当前环境路径
conda_env_path = os.environ.get("CONDA_PREFIX", "Not in a Conda environment")

# 打印信息
print("使用的是：{}".format(device))
print("CUDA 版本：{}".format(cuda_version))
print("cuDNN 版本：{}".format(cudnn_version))
print("Python 版本：{}".format(python_version))
print("当前环境路径：{}".format(conda_env_path))

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to(device)
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim,  # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,  # Local convolution width
    expand=2,  # Block expansion factor
).to(device)
y = model(x)
assert y.shape == x.shape
print("成功运行，模型输出维度为：{}".format(y.shape))