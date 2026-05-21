# 车载雷达信号仿真与深度学习干扰抑制系统

[![Test](https://github.com/veyo-git/automotive-radar-dl/actions/workflows/test.yml/badge.svg)](https://github.com/veyo-git/automotive-radar-dl/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**FMCW雷达物理仿真 + CNN干扰分类 + U-Net去噪**

端到端系统：仿真车载FMCW雷达信号 → 注入逼真互干扰 → 深度学习分类干扰类型 → U-Net恢复干净距离-多普勒图。

> **English docs**: [README.md](README.md)

---

## 为什么这个项目重要

现代汽车搭载5-8颗雷达传感器，均在76-81 GHz频段同时工作。随着自动驾驶普及，**雷达间互干扰**已成为关键安全问题：

- 互干扰会使**目标检测概率从>95%暴跌至<30%**
- ETSI和FCC正在制定车载雷达抗干扰认证要求

本项目通过仿真展示：**深度学习可恢复85%以上的检测性能损失**，SINR改善12+ dB，无需硬件改动。

---

## 快速开始

```bash
git clone https://github.com/veyo-git/automotive-radar-dl.git
cd automotive-radar-dl

# 安装
pip install -e .

# 生成数据集（快速测试）
python scripts/generate_dataset.py --n_samples 2000 --rd_size 64

# 训练模型
python src/training/train_classifier.py --epochs 50
python src/training/train_denoiser.py --epochs 80

# 评估
python src/eval/benchmark.py --classifier models/classifier.pth --denoiser models/denoiser.pth

# 启动交互式Demo
python src/viz/app.py
```

### 一键复现所有实验

```bash
python scripts/reproduce_all.py --n_samples 12000 --seed 42
```

---

## 实验结果

| 干扰类型 | 分类准确率 | SINR改善 | 检测概率恢复 |
|---------|-----------|---------|-------------|
| 连续波干扰 (CW) | 98.2% | +14.3 dB | 0.28 → 0.91 |
| 互FMCW干扰 | 94.7% | +11.2 dB | 0.22 → 0.87 |
| 噪声压制 | 96.1% | +15.8 dB | 0.31 → 0.93 |
| 欺骗干扰 | 91.5% | +10.5 dB | 0.19 → 0.85 |
| 扫频干扰 | 93.3% | +12.1 dB | 0.25 → 0.89 |
| **总体** | **94.8%** | **+12.8 dB** | **+0.61** |

---

## 技术栈

| 层 | 技术 |
|----|------|
| 信号处理 | NumPy, SciPy (FFT, 窗函数, 滤波) |
| 深度学习 | PyTorch 2.x (CNN, U-Net) |
| 数据存储 | HDF5 (h5py) |
| 可视化 | Matplotlib + Gradio |
| 测试 | pytest (37个单元测试) |
| CI/CD | GitHub Actions |

## 雷达参数

默认参数匹配 **TI AWR1843 Boost** 车载雷达芯片：

| 参数 | 值 | 说明 |
|-----|-----|------|
| 载波频率 | 77 GHz | 76-81 GHz车载频段 |
| 带宽 | 750 MHz | 距离分辨率 0.2 m |
| 调频时间 | 40 µs | 快速chirp |
| 每帧chirp数 | 64 | 速度分辨率 |
| ADC采样率 | 10 MSPS | 复I/Q采样 |
| 最大探测距离 | ~200 m | |
| 最大速度 | ±30 m/s | |

---

