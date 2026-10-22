"""过曝 / 欠曝统计（8bit 口径，0~255）。"""
import numpy as np


def exposure_stats(img_u8, over=250, under=5):
    """img: uint8 灰度或 BGR。返回占比（百分比）。"""
    if img_u8.ndim == 3:
        g = img_u8.mean(axis=2)  # 用亮度近似即可，统计用途足够
    else:
        g = img_u8
    g = g.astype(np.float32)
    return {
        "over_pct": round(float((g > over).mean() * 100), 3),
        "under_pct": round(float((g < under).mean() * 100), 3),
        "mean": round(float(g.mean()), 2),
        "p1": round(float(np.percentile(g, 1)), 1),
        "p99": round(float(np.percentile(g, 99)), 1),
    }
