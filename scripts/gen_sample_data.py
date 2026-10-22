"""生成合成样例数据（M0 冒烟测试用，不依赖外部数据）。

原理：造一张有亮区（灯）、暗区（阴影细节）、中间调的"场景"，然后按
EV-2 / 0 / +2 模拟三档曝光：
  - 模拟曝光 = 线性域亮度 × 2^EV，再 gamma 编码（sRGB）
  - 暗曝图加噪声（真实相机的短曝噪声更大）
输出 data/sample/ 下三张 JPG。
"""
from pathlib import Path
import sys

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io.reader import uimwrite
import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"


def make_scene(w=960, h=640):
    """三块区域：渐变天空(亮)、棋盘暗室(暗)、纹理中间调。"""
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = np.zeros((h, w, 3), np.float32)
    # 上半：亮渐变天空（模拟过曝区）
    sky = 0.85 + 0.15 * xx
    img[: h // 2] = sky[: h // 2, :, None] * np.array([1.0, 0.92, 0.8], np.float32)
    # 左下：暗室 + 低对比棋盘（考验暗部细节）
    dark = 0.02 + 0.03 * (((np.indices((h, w))[0] // 40 + np.indices((h, w))[1] // 40) % 2)[h // 2:, :w // 2])
    img[h // 2:, :w // 2, 0] = dark
    img[h // 2:, :w // 2, 1] = dark * 0.9
    img[h // 2:, :w // 2, 2] = dark * 0.8
    # 右下：中间调随机纹理（考验噪声与细节）
    rng = np.random.default_rng(7)
    tex = 0.45 + 0.1 * rng.standard_normal((h // 2, w // 2))
    img[h // 2:, w // 2:] = np.clip(tex, 0, 1)[..., None] * np.array([0.9, 0.95, 1.0], np.float32)
    return np.clip(img, 0, 1)


def srgb_encode(lin):
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def shoot(lin_scene, ev, noise_sigma=0.01):
    """线性域按 EV 缩放曝光 → sRGB 编码 → 8bit；短曝(EV<0)加更大噪声。"""
    exposed = np.clip(lin_scene * (2.0 ** ev), 0, None)
    rng = np.random.default_rng(100 + int(ev * 10))
    sigma = noise_sigma * (2.0 ** -ev)  # 越暗噪声相对越大
    exposed = np.clip(exposed + rng.standard_normal(exposed.shape) * sigma, 0, 1)
    return (srgb_encode(exposed) * 255).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = make_scene()
    for name, ev in [("ev-2", -2), ("ev0", 0), ("ev+2", 2)]:
        uimwrite(str(OUT / f"{name}.jpg"), cv2.cvtColor(shoot(scene, ev), cv2.COLOR_RGB2BGR))
    print(f"样例数据已生成: {OUT}")


if __name__ == "__main__":
    main()
