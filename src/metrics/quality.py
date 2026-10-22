"""HDR 评测指标（Kalantari 口径）。

μ 律色调映射（Drago 2003，DeepHDR 系列论文通用）：
  T(x) = log(1 + mu*x) / log(1 + mu)，mu=5000
线性 HDR 域的巨大数值会让 PSNR 失真，所有论文都在 μ 映射域算
PSNR-μ / SSIM-μ。我们的融合输出本身就是显示域图像，与 μ(GT) 直接比。
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def mu_tonemap(hdr, mu=5000.0):
    hdr = np.clip(hdr.astype(np.float64), 0, None)
    return (np.log1p(mu * hdr) / np.log(1 + mu)).astype(np.float32)


def psnr(a, b, data_range=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / mse))


def ssim(a, b):
    a = a if a.ndim == 2 else a.mean(axis=2)
    b = b if b.ndim == 2 else b.mean(axis=2)
    return float(structural_similarity(a, b, data_range=1.0))
