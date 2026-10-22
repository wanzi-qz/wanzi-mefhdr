"""光流对齐（Farnebäck 稠密光流）。

关键预处理——为什么不能直接算：
  Farnebäck 假设"同一像素在两帧间灰度不变"。EV±2 下同一物体亮度差好几倍，
  梯度方向和幅值全变了，光流会跟着曝光跑而不是跟着运动跑。
  对策：把两张灰度图各自按均值归一化（再线性域近似拉伸），削弱全局曝光差；
  残余的局部差异交给平滑参数处理。

输出：参考坐标系下的整幅 remap 结果 + 光流场（M2 实验里用光流中位数
当"全局平移估计"与已知位移对比）。
"""
import cv2
import numpy as np


def _normalize(gray):
    g = gray.astype(np.float32)
    g = g / (g.mean() + 1e-6)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def flow_align(ref_bgr, mov_bgr, pyr_scale=0.5, levels=4, winsize=21,
               iterations=5, poly_n=7, poly_sigma=1.5):
    """把 mov warp 到 ref 坐标系。返回 (对齐后 BGR, flow(dx,dy float32 HxWx2))。"""
    ref_g = _normalize(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
    mov_g = _normalize(cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY))
    # flow: ref 中每个像素对应 mov 中的位置偏移
    flow = cv2.calcOpticalFlowFarneback(
        ref_g, mov_g, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
    h, w = ref_g.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    warped = cv2.remap(mov_bgr, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped, flow
