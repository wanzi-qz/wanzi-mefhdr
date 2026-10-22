"""Mertens 2007 三权重图。

每张曝光图算三个分数（都归一化到 0~1）：
  C  对比度：灰度拉普拉斯绝对值，梯度大 = 细节多 = 分高
  S  饱和度：三通道标准差，通道差异大 = 颜色鲜艳，但过大说明偏色
  E  适度曝光：亮度与 mu(0.5) 的高斯距离，越近分越高

总分 W = C^wc * S^ws * E^we，然后跨曝光归一化（每像素除以 N 张图的 W 之和），
保证所有图在每个像素上的权重加起来等于 1。
"""
import cv2
import numpy as np


def compute_weights(imgs, w_contrast=1.0, w_saturated=1.0, w_exposed=1.0,
                    mu=0.5, sigma=0.2, return_maps=False):
    """imgs: list[BGR uint8]。返回归一化后的权重图列表（float32, 同尺寸单通道）。

    return_maps=True 时同时返回 (W_list, {'C':…, 'S':…, 'E':…}) 便于可视化。
    """
    W, maps = [], {"C": [], "S": [], "E": []}
    for img in imgs:
        f = img.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 对比度：拉普拉斯响应绝对值，再做轻度动态范围压缩
        C = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        C = C / (C.max() + 1e-12)

        # 饱和度：通道标准差
        S = f.std(axis=2)

        # 适度曝光：高斯环绕
        E = np.exp(-((gray - mu) ** 2) / (2 * sigma ** 2))

        maps["C"].append(C); maps["S"].append(S); maps["E"].append(E)
        W.append((C ** w_contrast) * (S ** w_saturated) * (E ** w_exposed) + 1e-12)

    total = sum(W)
    Wn = [w / total for w in W]
    return (Wn, maps) if return_maps else Wn
