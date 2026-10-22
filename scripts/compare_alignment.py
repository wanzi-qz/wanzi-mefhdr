"""M2 受控实验：MTB vs 光流对齐，鲁棒性与耗时对比。

实验设计（两个实验都要跑）：
  实验 A · 已知位移反算：
    以 scene01 的 EV0 为底图，把它随机平移已知像素(3/5/10/20px，8个方向平均)，
    让两种对齐器估计位移，报告 |估计-真值| 误差与耗时。
  实验 B · 前景运动鬼影：
    在底图上抠一块矩形单独平移（模拟运动物体），对齐后再统计
    "鬼影像素占比"（|对齐图-底图|>25 的像素比例，排除运动区域）。
  实验 C · 真实场景端到端：
    scene01 三档曝光，三种模式（不对齐/MTB/光流）各跑一次完整 pipeline，
    记录耗时与指标。

运行：python scripts/compare_alignment.py --out runs/m2_alignment
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io.reader import load_bracket, uimread, uimwrite
from src.align.mtb import mtb_align, mtb_shift
from src.align.flow import flow_align
from src.pipeline import run_pipeline

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "self_capture" / "scene01"
SHIFTS = [3, 5, 10, 20]
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]


def shift_img(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def est_global_shift_flow(ref, mov, flow):
    """用光流中位数当全局平移估计（mov 相对 ref 的位移）。"""
    med = np.median(flow.reshape(-1, 2), axis=0)
    return float(med[0]), float(med[1])


def exp_a(ref):
    """已知位移反算。"""
    rows = []
    for s in SHIFTS:
        for name, fn in [("mtb", None), ("flow", None)]:
            errs, ts = [], []
            for dx, dy in DIRS:
                mov = shift_img(ref, s * dx, s * dy)
                t0 = time.perf_counter()
                if name == "mtb":
                    ex, ey = mtb_shift(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY))
                    # mtb_shift 返回"校正量"(mov 需平移多少回到 ref)，
                    # 统一口径为"mov 相对 ref 的位移" = -校正量
                    pred = (-ex, -ey)
                else:
                    _, flow = flow_align(ref, mov)
                    mx, my = est_global_shift_flow(ref, mov, flow)
                    pred = (mx, my)
                ts.append(time.perf_counter() - t0)
                # 真值：mov 内容相对 ref 位移 (s*dx, s*dy)
                tx, ty = s * dx, s * dy
                errs.append(np.hypot(pred[0] - tx, pred[1] - ty))
            rows.append({"exp": "A_shift", "shift_px": s, "method": name,
                         "mean_err_px": round(float(np.mean(errs)), 3),
                         "max_err_px": round(float(np.max(errs)), 3),
                         "time_s": round(float(np.mean(ts)), 3)})
    return rows


def exp_b(ref):
    """前景运动鬼影：中间 1/4 区域单独平移 15px。"""
    h, w = ref.shape[:2]
    y0, y1, x0, x1 = h // 4, h // 2, w // 4, w // 2
    mov = ref.copy()
    mov[y0:y1, x0:x1] = shift_img(ref[y0:y1, x0:x1], 15, 0)
    # 全局对齐器会试图用"整体平移"硬凑，正确答案是 (0,0)
    rows = []
    t0 = time.perf_counter()
    wx, wy = mtb_shift(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY))
    t_mtb = time.perf_counter() - t0
    w_mtb = shift_img(mov, wx, wy)
    t0 = time.perf_counter()
    w_flow, flow = flow_align(ref, mov)
    t_flow = time.perf_counter() - t0

    def ghost(img):
        d = np.abs(img.astype(np.int32) - ref.astype(np.int32)).mean(axis=2)
        d[y0:y1, x0:x1] = 0  # 运动区域排除（它本来就不一样）
        return round(float((d > 25).mean() * 100), 3)

    rows.append({"exp": "B_ghost", "method": "no_align",
                 "ghost_pct": ghost(mov), "time_s": 0.0})
    rows.append({"exp": "B_ghost", "method": "mtb",
                 "ghost_pct": ghost(w_mtb), "time_s": round(t_mtb, 3),
                 "note": f"估计平移=({wx},{wy})，真值(0,0)"})
    rows.append({"exp": "B_ghost", "method": "flow",
                 "ghost_pct": ghost(w_flow), "time_s": round(t_flow, 3)})
    uimwrite("runs/m2_alignment/exp_b_mtb.png", w_mtb)
    uimwrite("runs/m2_alignment/exp_b_flow.png", w_flow)
    return rows


def exp_c(out):
    rows = []
    for label, align in [("no_align", False), ("mtb", "mtb"), ("flow", "flow")]:
        r = run_pipeline(str(DATA), str(Path(out) / f"scene01_{label}"),
                         align=align)
        f = r["fused"]
        rows.append({"exp": "C_pipeline", "method": label,
                     "total_s": r["total_sec"],
                     "fusion_s": r["timing_sec"]["fusion"],
                     "align_s": r["timing_sec"].get("align_mtb",
                                r["timing_sec"].get("align_flow", 0.0)),
                     "over_pct": f["over_pct"], "under_pct": f["under_pct"],
                     "align_info": r["align_info"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/m2_alignment")
    ap.add_argument("--skip-c", action="store_true", help="跳过耗时的实验 C")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    imgs, paths = load_bracket(str(DATA))
    ref = imgs[len(imgs) // 2]  # EV0
    print("基准图:", paths[len(imgs) // 2])

    results = []
    print("== 实验 A：已知位移反算 ==")
    ra = exp_a(ref); results += ra
    for r in ra:
        print(f"  shift={r['shift_px']:>2}px {r['method']:>4}: "
              f"err={r['mean_err_px']:.2f}px (max {r['max_err_px']:.2f}) "
              f"{r['time_s']*1000:.0f}ms")
    print("== 实验 B：前景运动鬼影 ==")
    rb = exp_b(ref); results += rb
    for r in rb:
        print(f"  {r['method']:>8}: ghost={r['ghost_pct']:.2f}% {r['time_s']*1000:.0f}ms")
    if not args.skip_c:
        print("== 实验 C：scene01 端到端 ==")
        rc = exp_c(out); results += rc
        for r in rc:
            ai = r["align_info"] or ""
            print(f"  {r['method']:>8}: total={r['total_s']:.2f}s "
                  f"align={r['align_s']:.2f}s over={r['over_pct']}% "
                  f"under={r['under_pct']}% {ai}")

    with open(out / "m2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("结果已写入", out / "m2_results.json")


if __name__ == "__main__":
    main()
