"""可运行入口：python scripts/run_pipeline.py --input <场景文件夹> --out runs/<名字>

可选参数：
  --config configs/default.yaml   换配置
  --align                          开启对齐（M2 前不可用）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline, load_config


def main():
    ap = argparse.ArgumentParser(description="多曝光融合 HDR pipeline")
    ap.add_argument("--input", required=True, help="连拍文件夹（同一场景多张曝光）")
    ap.add_argument("--out", required=True, help="输出目录，如 runs/smoke")
    ap.add_argument("--config", default=None, help="配置文件路径")
    ap.add_argument("--align", nargs="?", const="mtb", default=None,
                    choices=["mtb", "flow", "mtb_ecc", "orb_ecc", "orb_flow"],
                    help="对齐方法：mtb(默认) | mtb_ecc(平移+旋转精修) | flow")
    ap.add_argument("--raw", action="store_true", help="读 DNG RAW 而不是 JPG")
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = run_pipeline(args.input, args.out, cfg, align=args.align, raw=args.raw)
    print("== 各阶段耗时(秒) ==")
    for k, v in r["timing_sec"].items():
        print(f"  {k:10s} {v:.3f}")
    print(f"  {'total':10s} {r['total_sec']:.3f}")
    f = r["fused"]
    print(f"== 融合结果: mean={f['mean']} over>{cfg['stats']['over_threshold']}:"
          f"{f['over_pct']}% under<{cfg['stats']['under_threshold']}:{f['under_pct']}%")
    print(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()
