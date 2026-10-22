"""pipeline：整条链路的编排入口。

M0 版本：读图 → 权重 → 金字塔融合 → 输出 + 统计。对齐模块 M2 接入，
预留 align=False 开关。
"""
from pathlib import Path
import cv2
import numpy as np
import yaml

from .io import load_bracket, load_bracket_raw, Timer, save_timing, exposure_stats
from .io.reader import uimwrite
from .fusion import compute_weights, laplacian_blend
from .align import align_images

DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(input_dir, out_dir, cfg=None, align=False, raw=False):
    """跑通一个场景。raw=True 时读 DNG（RAW 路径）。返回结果摘要 dict。"""
    cfg = cfg or load_config()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    timer = Timer()

    # 1) 读图
    with timer.stage("load"):
        loader = load_bracket_raw if raw else load_bracket
        imgs, paths = loader(input_dir, sort=cfg["input"]["sort"])

    # 2) 对齐（基准 = 中间那张曝光）
    align_info = None
    if align:
        method = align if isinstance(align, str) else "mtb"
        with timer.stage("align_" + method):
            imgs, align_info = align_images(imgs, method)

    # 3) 权重
    with timer.stage("weights"):
        w = cfg["weights"]
        Wn, maps = compute_weights(
            imgs, w["w_contrast"], w["w_saturated"], w["w_exposed"],
            w["mu"], w["sigma"], return_maps=True)

    # 4) 金字塔融合
    with timer.stage("fusion"):
        floats = [i.astype(np.float32) / 255.0 for i in imgs]
        fused = laplacian_blend(floats, Wn, levels=cfg["pyramid"]["levels"])

    # 5) 输出
    with timer.stage("save"):
        result_u8 = (fused * 255).astype(np.uint8)
        uimwrite(str(out / "fused.jpg"), result_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])
        uimwrite(str(out / "fused.png"), result_u8)
        for k, img in enumerate(imgs):  # 输入留档
            uimwrite(str(out / f"input_{k}_{k+1}of{len(imgs)}.jpg"), img)
        # 权重图可视化
        for name in ("C", "S", "E"):
            for i, m in enumerate(maps[name]):
                uimwrite(str(out / f"wmap_{name}_{i}.png"), (m * 255).astype(np.uint8))
        uimwrite(str(out / "weight_sum_preview.png"),
                    (np.stack(Wn, axis=2) * 255).astype(np.uint8))

    # 6) 统计
    s_cfg = cfg["stats"]
    stats = {"inputs": [exposure_stats(i, s_cfg["over_threshold"], s_cfg["under_threshold"])
                        for i in imgs],
             "fused": exposure_stats(result_u8, s_cfg["over_threshold"], s_cfg["under_threshold"]),
             "inputs_paths": paths, "config": cfg, "align_info": align_info}
    payload = save_timing(timer, out, extra=stats)

    import json
    with open(out / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
