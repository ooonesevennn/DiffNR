# DiffNR 环境配置指南

## 1. 安装 Miniconda

如果系统没有 conda，先安装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3.sh
bash /tmp/Miniconda3.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 2. 创建 conda 环境

```bash
conda create -n diffnr python=3.10 -y
conda activate diffnr
```

## 3. 安装 CUDA Toolkit

编译 CUDA 扩展需要 `nvcc`。通过 conda 安装 CUDA Toolkit（版本需与 PyTorch CUDA 版本一致）：

```bash
conda install -c nvidia/label/cuda-13.0.0 cuda-toolkit -y
```

> 安装完成后验证：
> ```bash
> which nvcc
> nvcc --version
> ```

## 4. 安装 Python 依赖

```bash
cd DiffNR_open
pip install -r requirements.txt
```

## 5. 编译安装 CUDA 子模块

必须先设置 `CUDA_HOME` 环境变量，然后逐个编译安装三个子模块：

```bash
export CUDA_HOME=$CONDA_PREFIX
```

### 5.1 simple-knn

**注意：** 使用非 editable 模式安装，editable 模式（`-e`）可能导致模块无法被正确导入。

```bash
pip install --no-build-isolation r2_gaussian/submodules/simple-knn
```

### 5.2 fused-ssim

```bash
pip install --no-build-isolation -e r2_gaussian/submodules/fused-ssim
```

### 5.3 xray-gaussian-rasterization-voxelization

编译前需要修复两个头文件的 C++ 兼容性问题（新版 CUDA/GCC 不再隐式包含 `<cstdint>`）：

在以下两个文件的 `#include` 区域顶部添加 `#include <cstdint>`：

- `r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/cuda_rasterizer/rasterizer_impl.h`
- `r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/cuda_voxelizer/voxelizer_impl.h`

修改示例（以 `rasterizer_impl.h` 为例）：

```cpp
#pragma once

#include <cstdint>    // <-- 添加这一行
#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
```

然后编译安装：

```bash
pip install --no-build-isolation -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
```

> 编译过程中会输出大量 GLM 相关的 warning，属于正常现象，可以忽略。

## 6. 验证安装

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
from simple_knn._C import distCUDA2
print('simple_knn: OK')
import fused_ssim
print('fused_ssim: OK')
import xray_gaussian_rasterization_voxelization
print('xray_gaussian_rasterization_voxelization: OK')
"
```

## 7. 运行训练

```bash
conda activate diffnr
cd DiffNR_open

python scripts/run_DiffNR.py train \
  --source data/<your_case_folder> \
  --output output/<experiment_name> \
  --config configs/luna16.yaml \
  --slicefixer-model-path model/model_32001.pkl \
  --organ-type Chest \
  --gpu 0
```

## 环境信息参考

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.11.0+cu130 |
| CUDA Toolkit | 13.0 |
| NVIDIA Driver | 580.65.06 |
| Conda | 26.1.1 |
