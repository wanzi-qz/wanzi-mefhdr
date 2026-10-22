"""特征点单应对齐：ORB 初值 + ECC 单应精修（场景03/04 的解法）。

演进史（对应 docs/failures.md F5）：
  MTB 平移 → 手持重新构图后重影（旋转/缩放没建模）
  MTB+ECC 欧氏 → 只补了旋转，仍有重影（缩放/透视没建模）
  ORB 单应 → 模型对了但精度只有几 px（ORB 整数精度）
  ORB 单应 + ECC 精修 → 模型对 + 亚像素精度，最终方案

ORB 对曝光差相对鲁棒（基于局部模式而非绝对灰度），ECC 精修在归一化
灰度上进行，从单应初值出发不会陷进坏局部最优。
"""
import cv2
import numpy as np


def _norm_gray(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX)


def orb_homography(ref_bgr, mov_bgr, nfeatures=8000, ratio=0.75, ransac=4.0):
    """ORB 匹配 + RANSAC 单应。返回 (H 3x3, 内点数)。"""
    g1 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=10)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        raise RuntimeError("ORB 未检测到特征点")
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
            if m.distance < ratio * n.distance]
    if len(good) < 20:
        raise RuntimeError(f"ORB 有效匹配过少: {len(good)}")
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    return H, int(mask.sum()), len(good)


def orb_ecc_align(ref_bgr, mov_bgr, ecc_iters=100, use_ecc=True):
    """返回 (对齐后 BGR, info)。ECC 失败时退回纯 ORB 结果。"""
    H, inliers, n_good = orb_homography(ref_bgr, mov_bgr)
    info = {"method": "orb_ecc", "orb_inliers": inliers, "orb_matches": n_good}
    if not use_ecc:
        M = H
    else:
        try:
            warp = H[:2].copy()  # ECC 接受 2x3
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    ecc_iters, 1e-6)
            ret = cv2.findTransformECC(_norm_gray(ref_bgr), _norm_gray(mov_bgr),
                                       warp, cv2.MOTION_AFFINE, crit, None, 5)
            cc, M2 = ret if isinstance(ret[0], float) else (ret[1], ret[0])
            M = np.vstack([M2, [0, 0, 1]])
            info.update({"ecc_cc": round(float(cc), 4),
                         "angle_deg": round(float(np.degrees(
                             np.arctan2(M2[1, 0], M2[0, 0]))), 3)})
        except cv2.error as e:
            info["ecc_error"] = str(e)[:120]
            M = H
    h, w = ref_bgr.shape[:2]
    warped = cv2.warpPerspective(mov_bgr, M, (w, h),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
    info["H"] = np.round(M, 4).tolist()
    return warped, info


def _residual_flow_warp(ref_bgr, warped, winsize=31):
    """全局对齐后的逐像素残差校正（视差/局部运动专用）。

    在归一化灰度上算 ref->warped 的稠密残差流，流场做高斯平滑抑制噪声，
    再 remap 修正。平滑核越大越保守（只修低频错位），太小会引入扭曲。
    """
    flow = cv2.calcOpticalFlowFarneback(_norm_gray(ref_bgr), _norm_gray(warped),
                                        None, 0.5, 4, winsize, 5, 7, 1.5, 0)
    h, w = ref_bgr.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    fx = xx + cv2.GaussianBlur(flow[..., 0], (winsize, winsize), 0)
    fy = yy + cv2.GaussianBlur(flow[..., 1], (winsize, winsize), 0)
    return cv2.remap(warped, fx, fy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def orb_flow_align(ref_bgr, mov_bgr, winsize=31):
    """ORB 单应（全局模型）+ 光流残差（局部视差）。最终推荐方案。"""
    warped, info = orb_ecc_align(ref_bgr, mov_bgr, use_ecc=False)
    refined = _residual_flow_warp(ref_bgr, warped, winsize)
    info["method"] = "orb_flow"
    info["flow_winsize"] = winsize
    return refined, info
