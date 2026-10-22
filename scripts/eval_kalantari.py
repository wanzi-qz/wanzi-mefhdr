"""M4 评测：Kalantari SIG17 测试集（15 场景，带 GT HDR）上跑本项目 pipeline。

口径说明（防"对比不公平"质疑）：
  - 输入：数据集 3 张 16bit LDR TIFF，按 exposure.txt 排序（短→长）
  - 我们的方法：MTB 对齐 + Mertens 权重 + 拉普拉斯金字塔（与自采数据完全
    同一套参数，configs/default.yaml，零调参）
  - 指标：融合结果（显示域 0~1）与 μ 律色调映射后的 GT 比，算 PSNR-μ / SSIM-μ
  - 深度方法的数字来自其论文报告值，口径同为 PSNR-μ / SSIM-μ，但注意其
    融合结果是网络直接输出 μ 域图像，我们的结果是另一条 tone-mapping 路径，
    这个系统性差异在报告中明确说明
运行：python scripts/eval_kalantari.py --data data/external/SIG17/Test --out runs/m4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io.reader import uimwrite
from src.metrics import mu_tonemap, psnr, ssim
from src.align import align_images
from src.fusion import compute_weights, laplacian_blend
from src.pipeline import load_config


def read_expos(scene_dir):
    vals = [float(x) for x in
            (Path(scene_dir) / "exposure.txt").read_text().split()]
    return vals


def load_scene(scene_dir):
    """返回按曝光升序的 3 张 LDR（float 0~1 BGR）与 GT HDR。"""
    scene_dir = Path(scene_dir)
    tifs = sorted(scene_dir.glob("*.tif"))
    assert len(tifs) == 3, f"{scene_dir} 应有 3 张 tif，实际 {len(tifs)}"
    imgs = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in tifs]
    imgs = [i.astype(np.float32) / 65535.0 if i.dtype == np.uint16
            else i.astype(np.float32) / 255.0 for i in imgs]
    expos = read_expos(scene_dir)
    order = np.argsort(expos)
    imgs = [imgs[k] for k in order]
    gt = cv2.imread(str(scene_dir / "HDRImg.hdr"), cv2.IMREAD_UNCHANGED)
    gt = gt.astype(np.float32)
    return imgs, [tifs[k].name for k in order], gt


def fuse(imgs, cfg, align_method="mtb"):
    """本项目 pipeline 的融合（与自采数据同一套代码与参数）。"""
    u8s = [(np.clip(i, 0, 1) * 255).astype(np.uint8) for i in imgs]
    if align_method:
        u8s, infos = align_images(u8s, align_method)
    else:
        infos = None
    floats = [i.astype(np.float32) / 255.0 for i in u8s]
    w = cfg["weights"]
    Wn = compute_weights(floats, w["w_contrast"], w["w_saturated"],
                         w["w_exposed"], w["mu"], w["sigma"])
    fused = laplacian_blend(floats, Wn, levels=cfg["pyramid"]["levels"])
    return fused, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/external/SIG17/Test")
    ap.add_argument("--out", default="runs/m4_kalantari")
    ap.add_argument("--align", default="mtb")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    # 场景目录 = 含 HDRImg.hdr 的目录（Test 下分 EXTRA/PAPER 两级，不能只看一层）
    scenes = sorted(p for p in Path(args.data).rglob("HDRImg.hdr"))
    scenes = [p.parent for p in scenes]
    print(f"测试场景数: {len(scenes)}")
    rows = []
    for sc in scenes:
        t0 = time.perf_counter()
        imgs, names, gt = load_scene(sc)
        fused, infos = fuse(imgs, cfg, args.align)
        elapsed = time.perf_counter() - t0
        fused_u8 = (np.clip(fused, 0, 1) * 255).astype(np.uint8)
        gt_mu = mu_tonemap(gt)
        gt_mu_u8 = (np.clip(gt_mu, 0, 1) * 255).astype(np.uint8)
        p_mu = psnr(gt_mu, np.clip(fused, 0, 1))
        s_mu = ssim(gt_mu, np.clip(fused, 0, 1))
        # 诊断：最优全局增益后的 PSNR-μ（分离"色调曲线差异"与"结构误差"）
        f32 = np.clip(fused, 0, 1).astype(np.float64)
        g64 = gt_mu.astype(np.float64)
        gain = float((f32 * g64).sum() / ((f32 * f32).sum() + 1e-12))
        p_gain = psnr(np.clip(f32 * gain, 0, 1), g64)
        # 对照：单张中间曝光 vs GT（传统方法必须赢过"什么都不做"）
        mid = mu_tonemap(imgs[len(imgs)//2] * float(2 ** np.median(read_expos(sc))))
        p_mid = psnr(np.clip(mid, 0, 1), gt_mu)
        rows.append({"scene": sc.name, "psnr_mu": round(p_mu, 2),
                     "ssim_mu": round(s_mu, 4),
                     "psnr_mu_optgain": round(p_gain, 2),
                     "opt_gain": round(gain, 3),
                     "psnr_mu_single_mid": round(p_mid, 2),
                     "time_s": round(elapsed, 1),
                     "align": infos})
        uimwrite(out / f"{sc.name}_fused.jpg", fused_u8,
                 [cv2.IMWRITE_JPEG_QUALITY, 95])
        uimwrite(out / f"{sc.name}_gt_mu.jpg", gt_mu_u8,
                 [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {sc.name}: PSNR-μ={p_mu:.2f} SSIM-μ={s_mu:.4f} "
              f"(单张中间曝光 {p_mid:.2f}) {elapsed:.1f}s")

    with open(out / "m4_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    avg_p = float(np.mean([r["psnr_mu"] for r in rows]))
    avg_s = float(np.mean([r["ssim_mu"] for r in rows]))
    print(f"== 平均: PSNR-μ={avg_p:.2f}  SSIM-μ={avg_s:.4f}")
    with open(out / "m4_summary.json", "w", encoding="utf-8") as f:
        json.dump({"psnr_mu_avg": round(avg_p, 2),
                   "ssim_mu_avg": round(avg_s, 4)}, f, indent=2)


if __name__ == "__main__":
    main()
