#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image



def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_3dwindow(window_size, channel):
    # 创建3D高斯核
    _1D_window = torch.tensor([[(i - window_size // 2) ** 2 for i in range(window_size)]])
    _1D_window = torch.exp(-0.5 * _1D_window / (window_size ** 2))
    _1D_window /= _1D_window.sum()  # 归一化
    _3D_window = _1D_window[:, :, None] * _1D_window[:, None, :] * _1D_window[None, :, :]
    window = _3D_window.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1, 1)
    return window

def ssim3d(volume1, volume2, window_size=11, size_average=True):
    channel = volume1.size(-4)  # 取通道维度
    window = create_3dwindow(window_size, channel)
    
    if volume1.is_cuda:
        window = window.cuda(volume1.get_device())
    window = window.type_as(volume1)

    return _ssim3d(volume1, volume2, window, window_size, channel, size_average)

def _ssim3d(volume1, volume2, window, window_size, channel, size_average=True):
    # 3D卷积计算均值
    mu1 = F.conv3d(volume1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(volume2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差
    sigma1_sq = F.conv3d(volume1 * volume1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(volume2 * volume2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(volume1 * volume2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)


def metric_vol_loss(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1.copy()).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy()).float()

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
        return psnr_out, None  # 返回张量而非 item()
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None])  # 返回张量
                    count += 1
                else:
                    result = torch.tensor(0.0, device=img1.device)
                results.append(result)
                
                """
                # 保存切片图像
                # 使用 matplotlib 或 PIL 保存
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(slice1.detach().cpu().numpy(), cmap='gray')
                ax[0].set_title(f"Slice {i} - img1")
                ax[1].imshow(slice2.detach().cpu().numpy(), cmap='gray')
                ax[1].set_title(f"Slice {i} - img2\nSSIM: {result.item():.4f}")
                
                # 保存每个切片的图像
                plt.tight_layout()
                plt.savefig(os.path.join("output/check_enhanced_slices_ssim", f"axis_{axis}_slice_{i:03d}.png"))
                plt.close(fig)
                """
                
            results = torch.stack(results)  # 改为 stack
            mean_results = torch.sum(results) / count
            ssims.append(mean_results)
        return torch.mean(torch.stack(ssims)), ssims  # 返回张量

