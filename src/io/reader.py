"""读取一组连拍曝光。

约定：输入是一个文件夹，里面放着同一场景的 JPG/PNG 连拍（3~N 张）。
排序方式由参数决定：
  - brightness：按灰度均值升序，暗的在前（= EV 低 → EV 高）。
    这是默认值，因为文件名顺序不保证对应曝光顺序。
  - filename：按文件名排序（当你确定命名即曝光顺序时用）。

注意：OpenCV 的 imread/imwrite 在 Windows 上不支持中文路径，
本项目统一使用 imdecode/imencode 包装版 uimread/uimwrite。
"""
from pathlib import Path
import cv2
import numpy as np


def uimread(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的 imread。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def uimwrite(path, img, params=None):
    """支持中文路径的 imwrite，返回是否成功。"""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def load_bracket(folder, sort="brightness", exts=("jpg", "jpeg", "png", "bmp", "tif", "tiff")):
    """返回 (list[BGR uint8 ndarray], list[路径])，按曝光从暗到亮排列。"""
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower().lstrip(".") in exts)
    if len(paths) < 2:
        raise FileNotFoundError(f"{folder} 下只找到 {len(paths)} 张图，至少需要 2 张")

    imgs = [uimread(p) for p in paths]
    if any(i is None for i in imgs):
        raise IOError(f"有图片读不出来，检查文件是否完整: {paths}")

    if sort == "brightness":
        order = np.argsort([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).mean() for i in imgs])
        imgs = [imgs[k] for k in order]
        paths = [paths[k] for k in order]
    return imgs, [str(p) for p in paths]


def load_bracket_raw(folder, sort="brightness"):
    """读取 DNG 连拍（RAW 主路径）。

    关键点（见 docs/failures.md F2）：
      - no_auto_bright=True：不许库偷偷拉伸亮度，否则曝光差被抹平
      - output_bps=16：先取 16bit 线性域，再统一做 sRGB gamma 编码到 8bit。
        直接 /257 转 8bit 会让夜景全黑（线性域太暗），gamma 编码是相机 JPG
        直出时做的事情，属于合理显影而非"美化"。
    排序同样按亮度（DNG 文件名顺序不保证对应曝光）。
    """
    import rawpy
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() == ".dng")
    if len(paths) < 2:
        raise FileNotFoundError(f"{folder} 下只找到 {len(paths)} 张 DNG")

    imgs = []
    for p in paths:
        with rawpy.imread(str(p)) as raw:
            lin = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                                  output_bps=16).astype(np.float32) / 65535.0
        srgb = np.where(lin <= 0.0031308, lin * 12.92,
                        1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
        imgs.append((np.clip(srgb, 0, 1) * 255).astype(np.uint8))

    if sort == "brightness":
        order = np.argsort([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).mean() for i in imgs])
        imgs = [imgs[k] for k in order]
        paths = [paths[k] for k in order]
    return imgs, [str(p) for p in paths]
