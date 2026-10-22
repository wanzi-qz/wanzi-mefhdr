"""拉普拉斯金字塔多频段融合。

为什么不用"逐像素直接加权"：权重在物体边界处突变，直接加权会在边界产生
亮度台阶（光晕感）。金字塔融合把每张图拆成低频(高斯塔) + 各级高频(拉普拉斯塔)，
每个频段用该频段下采样后的权重单独加权，再逐级还原——大结构由低频权重决定，
细节由高频权重决定，边界过渡自然。

流程：
  1) 每张图建拉普拉斯塔（levels 层）；每张权重图建高斯塔
  2) 每层：R_l = Σ_i W_i^l * L_i^l
  3) 从顶层开始上采样累加还原
"""
import cv2
import numpy as np


def _down(img):
    return cv2.pyrDown(img)


def _up(img, shape):
    return cv2.pyrUp(img, dstsize=(shape[1], shape[0]))


def laplacian_blend(imgs, weights, levels=6):
    """imgs: list[BGR float32 0~1]; weights: list[float32 单通道 0~1，和为 1]。
    返回融合图 float32 BGR 0~1。"""
    h, w = imgs[0].shape[:2]
    levels = min(levels, int(np.log2(min(h, w))) - 1)  # 层数受尺寸限制
    assert levels >= 2, "图像太小，金字塔至少 2 层"

    # 1) 建塔
    lp, gp = [], []
    for img in imgs:
        g = [img.astype(np.float32)]
        for _ in range(levels - 1):
            g.append(_down(g[-1]))
        l = [g[k] - _up(g[k + 1], g[k].shape) for k in range(levels - 1)] + [g[-1]]
        lp.append(l); gp.append(g)
    wgp = []
    for wmap in weights:
        g = [wmap.astype(np.float32)]
        for _ in range(levels - 1):
            g.append(_down(g[-1]))
        wgp.append(g)

    # 2) 每层加权求和
    out_levels = []
    for k in range(levels):
        acc = sum(wgp[i][k][..., None] * lp[i][k] for i in range(len(imgs)))
        out_levels.append(acc)

    # 3) 还原
    out = out_levels[-1]
    for k in range(levels - 2, -1, -1):
        out = _up(out, out_levels[k].shape) + out_levels[k]
    return np.clip(out, 0, 1)
