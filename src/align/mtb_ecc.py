"""MTB + ECC 欧氏精修对齐。

为什么需要这一级（M3 scene03 教训）：
  手持连拍如果不是"端着不动连拍"而是"拍一张重新构图拍一张"，会有 0.3°~1°
  的旋转。MTB 的纯平移模型救不了旋转，融合图出现边缘双影。
  对策：MTB 平移做初值（大范围、曝光鲁棒），ECC 在归一化灰度上精修
  旋转+平移（MOTION_EUCLIDEAN，2 自由度）。ECC 对初值敏感，单独用会
  陷在局部最优，所以两者是接力关系而非竞争关系。
"""
import cv2
import numpy as np

from .mtb import mtb_shift


def _norm_gray(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX)


def mtb_ecc_align(ref_bgr, mov_bgr, ecc_iters=200, gauss_filt_size=5):
    """返回 (对齐后 BGR, info dict)。"""
    dx, dy = mtb_shift(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY))
    h, w = ref_bgr.shape[:2]
    warp = np.float32([[1, 0, dx], [0, 1, dy]])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, 1e-6)
    try:
        ret = cv2.findTransformECC(_norm_gray(ref_bgr), _norm_gray(mov_bgr),
                                   warp, cv2.MOTION_EUCLIDEAN, crit, None,
                                   gauss_filt_size)
        cc, M = ret if isinstance(ret[0], float) else (ret[1], ret[0])
        ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        info = {"method": "mtb_ecc", "mtb": [int(dx), int(dy)],
                "angle_deg": round(ang, 3),
                "tx": round(float(M[0, 2]), 1), "ty": round(float(M[1, 2]), 1),
                "ecc_cc": round(float(cc), 4)}
    except cv2.error as e:  # ECC 不收敛时退回纯 MTB
        info = {"method": "mtb_ecc_fallback_mtb", "mtb": [int(dx), int(dy)],
                "error": str(e)[:120]}
        M = np.float32([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(mov_bgr, M, (w, h),
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
    return warped, info
