# AREF-Net 核心代码

仅包含完整模型的核心模块、模型配置、三种边缘数据配置及训练/评估入口。
不包含权重、消融配置、数据、实验结果或完整 Ultralytics 框架。

## 模块

| 论文名称 | 文件 | 对应原实现 |
|---|---|---|
| MDCB | `aref_net/mdcb.py` | `C3k2_DWR` |
| ESRB | `aref_net/esrb.py` | `C3k2_RVB_SE` |
| BCAF | `aref_net/bcaf.py` | `CGAFusion` |

`configs/aref-net.yaml` 只使用论文模块名称，保留原完整模型的 n 规模和连接关系。
模块重命名不改变原计算或参数键；原始实现来源及许可在代码中保留。

## 环境

依赖父目录 `4.27/ultralytics` 中的原始 YOLOMM 框架，不能用官方普通
Ultralytics 包直接替代。`aref_net/runtime.py` 只在当前进程注册模块，不修改原工程。
目录移动后，使用 `AREF_FRAMEWORK_ROOT` 指定原框架位置。

本机可用环境为 `conda activate detr`，包含 Python 3.10、
PyTorch 2.4.1+cu118、timm 0.9.8、einops 0.8.1、PyYAML 6.0.2 和 OpenCV 4.11.0。

## 运行

在本目录执行：

```bash
python train.py --edge canny
python train.py --edge sobel
python train.py --edge log
```

默认训练设置按已保存记录采用 `epochs=300, batch=8, imgsz=640, seed=0,
optimizer=auto`；原日志中自动优化器实际选择 AdamW，初始学习率为 0.00125。
论文当前写的 batch=4 与这些记录不一致，未据此改动训练设置。
数据 YAML 使用现有 MDMCS 数据路径及已生成边缘图；更换数据位置时修改 `path`。

评估时直接指定原位置的权重，无需复制：

```bash
python evaluate.py --edge canny --weights ../HRCDS/train/full_canny/weights/best.pt
python evaluate.py --edge sobel --weights ../HRCDS/train/dwr_rvbse_cgafusion/weights/best.pt
python evaluate.py --edge log --weights ../HRCDS/train/full_log/weights/best.pt
```

默认评估 test 集，沿用原 COCO 评估器及 `conf=0.05`，不以论文数值替代实测结果。
原始记录尚未与论文 80.7%/80.4% 对齐；本目录整理不代表这些指标已经复现。
