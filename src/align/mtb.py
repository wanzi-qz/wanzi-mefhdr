"""MTB 对齐（Median Threshold Bitmap, Ward 2003）。

原理：
  1) 灰度图按中值二值化 → MTB 位图（比中值亮=1，暗=0）。
     曝光变了中值也跟着变，位图结构基本不变 → 对曝光差鲁棒。
  2) 过亮/过暗像素（离中值 <4/255）不可靠，放入 exclusion 掩膜排除。
  3) 图像金字塔从粗到细：最粗层搜索 [-shift_bits, shift_bits]^2 的平移，
     之后每层只精修 [-1,1]，最终得到全局平移 (dx, dy)。

局限：只估计全局平移。旋转/缩放/视差/前景运动都不在能力范围内。
"""
import cv2
import numpy as np


def _pyr_down_mask(m, levels):
    out = [m]
    for _ in range(levels - 1):
        out.append(cv2.pyrDown((out[-1] * 255).astype(np.uint8)).astype(bool))
    return out


def mtb_shift(ref_gray, mov_gray, levels=5, shift_bits=4):
    """返回 mov 相对 ref 的平移 (dx, dy)：mov 平移 (dx,dy) 后与 ref 对齐。
    即 warp 时用 M=[[1,0,dx],[0,1,dy]]。"""
    ref_gray = ref_gray.astype(np.float32)
    mov_gray = mov_gray.astype(np.float32)

    def make_tb(g):
        med = np.median(g)
        tb = (g > med).astype(np.uint8)
        mask = (np.abs(g - med) > 4).astype(np.uint8)  # 排除过亮/过暗
        return tb, mask

    # 建金字塔
    ref_t, ref_m, mov_t, mov_m = [], [], [], []
    r, m = make_tb(ref_gray);  rt = [r]; rm = [m]
    t, k = make_tb(mov_gray);  mt = [t]; mm = [k]
    for _ in range(levels - 1):
        rt.append(cv2.pyrDown(rt[-1].astype(np.float32)) > 0.5)
        rm.append(cv2.pyrDown((rm[-1] * 255).astype(np.uint8)) > 127)
        mt.append(cv2.pyrDown(mt[-1].astype(np.float32)) > 0.5)
        mm.append(cv2.pyrDown((mm[-1] * 255).astype(np.uint8)) > 127)
    rt = [x.astype(np.uint8) for x in rt]; rm = [x.astype(np.uint8) for x in rm]
    mt = [x.astype(np.uint8) for x in mt]; mm = [x.astype(np.uint8) for x in mm]

    def xor_score(lv, dx, dy):
        h, w = rt[lv].shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(mt[lv], M, (w, h))
        diff = cv2.bitwise_xor(rt[lv], warped)
        diff = cv2.bitwise_and(diff, rm[lv])
        diff = cv2.bitwise_and(diff, mm[lv])
        return cv2.countNonZero(diff)

    # 最粗层大范围搜索
    cur = [0, 0]
    lv = levels - 1
    best = min(((xor_score(lv, dx, dy), dx, dy)
                for dx in range(-shift_bits, shift_bits + 1)
                for dy in range(-shift_bits, shift_bits + 1)))
    cur = [best[1], best[2]]
    # 逐层精修 ±1
    for lv in range(levels - 2, -1, -1):
        cur = [c * 2 for c in cur]
        best = min(((xor_score(lv, cur[0] + dx, cur[1] + dy), cur[0] + dx, cur[1] + dy)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
        cur = [best[1], best[2]]
    return cur[0], cur[1]


def mtb_align(ref_bgr, mov_bgr, levels=5, shift_bits=4):
    """把 mov 平移对齐到 ref。返回 (对齐后 BGR, (dx,dy))。"""
    ref_g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    mov_g = cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY)
    dx, dy = mtb_shift(ref_g, mov_g, levels, shift_bits)
    h, w = ref_g.shape
    warped = cv2.warpAffine(mov_bgr, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
    return warped, (dx, dy)
