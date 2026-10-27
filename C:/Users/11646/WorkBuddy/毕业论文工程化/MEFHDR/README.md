# 多曝光融合 HDR 成像链路（MEF-HDR Pipeline）

![status](https://img.shields.io/badge/status-M5%E6%94%B6%E5%AE%98-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

## 介绍

夜景、逆光、窗边人像这类大光比场景，单张照片顾此失彼：亮部过曝或暗部死黑。多曝光融合
是经典解决路线——用同一场景不同曝光的照片（EV-2 / 0 / +2 三档）合成一张亮部不过曝、
暗部有细节的 HDR 照片，不需要训练模型、可在端侧 CPU 上实时跑，应用方向包括无人机
多目测量/融合、监控宽动态（Local Tone-mapping）前级。工程上要回答两个问题，而不是
刷指标：

1. **对齐方案怎么选**：光流对齐和 MTB 中值阈值位图对齐，哪个更抗鬼影、
   各自耗时多少？（M2 给出受控对比）
2. **传统方法的边界在哪**：和深度学习方法（DeepU-Fuse 等）相比，什么场景
   传统够用、什么场景传统会输？（M4 定量对比 + 失败案例分析）

整条链路**不训练任何模型**，全部由经典算法组成：

```
连拍输入(JPG/RAW) → 曝光对齐(抗鬼影) → 三权重图(对比度/饱和度/适度曝光)
                  → 拉普拉斯金字塔融合 → HDR 输出 + 过曝/欠曝统计
```

结果：**MTB 对齐**在 3~20px 全局平移上误差 **0.00px**、单场景 **0.6s**（光流对齐
≥10px 后失效、耗时 4.4s 且 warp 引入暗边），默认选型 MTB（限固定机位连拍）；在
Kalantari 测试集 15 场景上**零调参**评测 PSNR-μ **17.43 dB** / SSIM-μ **0.7525**，
CPU 单场景 ~0.8s（1000×1500）；并定量划出传统方法的边界：静态场景够用
（PSNR-μ 18~21.5），前景运动场景必须深度去鬼影（SSIM-μ 掉到 0.54~0.75）。

## 项目贡献

- **对齐选型靠受控对比不靠肉眼**：对齐质量用"合成已知偏移的图对 + 反算位移误差"量化；
  重新构图式素材的四种补救方案均受视差限制，如实记录（
  [m3 发现 2](reports/m3_self_capture_report.md) / [failures.md F5](docs/failures.md)）。
- **适用边界有数字**：Kalantari 15 场景零调参评测 + 失败案例单独开档，写清
  "输入长什么样、为什么失败"，见 [docs/failures.md](docs/failures.md)。
- **自采数据拍摄规范**：固定机位连拍 / 避开近景前景层 / 夜景大光比开 RAW——全部来自
  实拍踩坑（见下方"拍摄规范"）。
- **全链路可复现**：参数集中 YAML 管理、一个脚本干一件事、每次实验的产物
  （图 + JSON 指标 + 耗时）落 `runs/`，目录名即实验名。

## 数据集

- **Kalantari HDR 测试集**：15 个场景（含不同曝光的 LDR 序列与对应 HDR 真值），
  M4 零调参评测用。
- **自采连拍**（`data/raw/self_capture`）：按"拍摄规范"采集，夜景/大光比开 RAW。
- **自带合成样例**（`data/sample`）：最小 pipeline 不依赖任何外部数据，装完就能跑。

## 实验环境

纯 CPU 实现：单场景 ~0.8s（1000×1500）。报告所有耗时统一注明硬件与图像分辨率，
不混口径。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 项目骨架 + 合成样例数据 + 最小 pipeline 跑通 | ✅ → [快速开始](#快速开始) |
| M1 | 三权重图 + 金字塔融合完整实现，参数说明文档 | ✅（M0 一并实现）→ [configs/](configs/) |
| M2 | MTB vs 光流对齐：鲁棒性 & 耗时受控对比 | ✅（选 MTB）→ [reports/m2_alignment_report.md](reports/m2_alignment_report.md) |
| M3 | 自采数据完整测试 + 过曝/欠曝统计 + 各阶段耗时 | ✅（4 场景 + 2 条发现）→ [reports/m3_self_capture_report.md](reports/m3_self_capture_report.md) |
| M4 | 公开数据集评测（Kalantari 15 场景），与深度方法对比 | ✅（17.43/19.38 dB）→ [reports/m4_kalantari_report.md](reports/m4_kalantari_report.md) |
| M5 | 技术总结：参数调优经验 + 失败案例 + 适用边界 | ✅ → [docs/technical_summary.md](docs/technical_summary.md) / [docs/failures.md](docs/failures.md) |

## 实测记录

### M2 · 对齐方案对比

| 方案 | 全局平移精度(3~20px) | 耗时 | 真实场景表现 |
|---|---|---|---|
| **MTB** | **0.00px 误差** | **0.6s** | 正确估出手持 60px 抖动，重影消除 |
| 光流 | ≥10px 后失效 | 4.4s | 全局估计不可靠，warp 引入暗边 |

默认方案 **MTB**（限固定机位连拍）；重新构图式素材的四种补救方案均受视差
限制（详见 [m3 发现 2](reports/m3_self_capture_report.md) 与 [F5](docs/failures.md)）。

### M4 · Kalantari 测试集评测（15 场景，零调参）

| 指标 | 数值 |
|---|---|
| PSNR-μ | 17.43 dB（最优增益诊断口径 19.38 dB） |
| SSIM-μ | 0.7525 |
| 单场景耗时 | ~0.8s（CPU，1000×1500） |

传统方法边界：静态场景够用（PSNR-μ 18~21.5），前景运动场景必须深度去鬼影
（SSIM-μ 掉到 0.54~0.75）。详见 [m4 报告](reports/m4_kalantari_report.md)。

技术总结（调优经验 + 5 个失败案例 + 适用边界）见 [docs/technical_summary.md](docs/technical_summary.md)。

## 拍摄规范（自采数据必读）

1. **固定机位连拍**：靠稳/三脚架/2 秒定时，拍三张之间不要重新构图
   （移动拍摄点会产生视差，任何全局对齐模型都救不了前景双影）
2. **避开近景前景层**：主体是平面立面/远景最安全
3. **夜景/大光比必须开 RAW**：手机 JPG 直出会把 EV 间隔压没（D6）
4. **当场自检**：EV-2 高光有层次、EV+2 暗部有纹理、三张构图一致

## 项目结构

```
MEFHDR/
├── src/
│   ├── io/         输入输出：连拍读取、RAW(可选)、HDR 输出与直方图统计
│   ├── align/      曝光对齐：MTB 位图对齐、光流对齐（含质量评估）
│   ├── fusion/     权重图计算 + 拉普拉斯金字塔融合（Mertens 路线）
│   └── metrics/    MEF-SSIM、PSNR、过曝/欠曝占比
├── configs/        参数集中管理（YAML），一个场景一个配置
├── data/           自采数据(data/raw/self_capture)、外部数据集(data/external)
├── scripts/        可运行入口（一个脚本干一件事）
├── runs/           每次实验的产物（图 + JSON 指标 + 耗时），目录名即实验名
├── reports/        汇总报告与对比图
└── docs/           教学笔记、参数调优记录、失败案例
```

## 快速开始

```bash
# 1. 装依赖
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows

# 2. 跑通最小样例（用自带的合成测试数据，不需要任何外部数据）
python scripts/run_pipeline.py --input data/sample --out runs/smoke

# 3. 跑你自己的连拍
python scripts/run_pipeline.py --input data/raw/self_capture/场景名 --out runs/my_scene
```

## 诚实性设计（和低光项目同一套规则）

- 每个里程碑先写**验收标准**，跑完数字再对照，负结果如实入册
- 对齐质量用"合成已知偏移的图对 + 反算位移误差"来量化，不靠肉眼
- 报告所有耗时统一注明硬件与图像分辨率，不混口径
- 失败案例单独开档（[docs/failures.md](docs/failures.md)），写清"输入长什么样、为什么失败"
