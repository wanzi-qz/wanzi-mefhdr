"""把一组曝光对齐到中间那张（基准 = EV0）。

统一入口：align_images(imgs, method) → (对齐后的图列表, 每张的对齐信息)。
基准图本身原样返回（info 为 None）。
"""
import cv2
import numpy as np

from .mtb import mtb_align
from .flow import flow_align
from .mtb_ecc import mtb_ecc_align
from .feature import orb_ecc_align, orb_flow_align


def align_images(imgs, method="mtb", **kwargs):
    """method: 'mtb' | 'flow' | 'mtb_ecc'。返回 (imgs_aligned, infos)。"""
    if method not in ("mtb", "flow", "mtb_ecc", "orb_ecc", "orb_flow"):
        raise ValueError(f"未知对齐方法: {method}")
    ref_idx = len(imgs) // 2
    ref = imgs[ref_idx]
    out = [None] * len(imgs)
    infos = [None] * len(imgs)
    out[ref_idx] = ref.copy()
    for i, img in enumerate(imgs):
        if i == ref_idx:
            continue
        if method == "mtb":
            warped, (dx, dy) = mtb_align(ref, img, **kwargs)
            infos[i] = {"method": "mtb", "dx": int(dx), "dy": int(dy)}
        elif method == "mtb_ecc":
            warped, info = mtb_ecc_align(ref, img, **kwargs)
            infos[i] = info
        elif method == "orb_ecc":
            warped, info = orb_ecc_align(ref, img, **kwargs)
            infos[i] = info
        elif method == "orb_flow":
            warped, info = orb_flow_align(ref, img, **kwargs)
            infos[i] = info
        else:
            warped, flow = flow_align(ref, img, **kwargs)
            med = np.median(flow.reshape(-1, 2), axis=0)
            infos[i] = {"method": "flow", "median_dx": float(med[0]),
                        "median_dy": float(med[1])}
        out[i] = warped
    return out, infos
