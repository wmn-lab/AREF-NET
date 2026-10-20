# YOLOMM / RTDETRMM 可视化 vis 使用说明（简版）

本文仅说明 vis 的参数层级与用途，并提供双模态示例（YOLOMM 与 RTDETRMM）。

## 参数层级与用途
- 一级参数（vis 通用）
  - `rgb_source`：RGB 输入。仅支持图片“文件路径”或“目录路径”。目录模式下将按文件名去后缀（stem）与 `x_source` 目录一一配对；若只提供 RGB 目录则视为单侧输入（需结合 `modality` 做消融）。
  - `x_source`：X 模态输入。仅支持图片“文件路径”或“目录路径”。目录模式需与 `rgb_source` 同为目录并按同名文件配对；若只提供 X 目录则视为单侧输入（需结合 `modality` 做消融）。
  - `method`：可视化方法选择，取 `heat|heatmap` 或 `feature|feature_map`。
  - `layers`：待可视化的层索引列表（必填，整数，从 0 起）。
  - `modality`：`auto/dual/rgb/x`；双模态可强制消融为 `rgb` 或 `x`。
  - `save`：是否保存渲染结果到磁盘。
  - `project` / `name`：输出目录控制，推荐使用；`out_dir` 为兼容参数（已废弃）。
  - `device`：期望运行设备；与模型当前设备不一致将报错（不自动迁移）。

- 二级参数（方法专属，仅在对应方法中生效）
  - 热力图（`method='heat'|'heatmap'`）
    - `overlay`：热图叠加底图，`'rgb'|'x'|'dual'`。
    - `alg`：热力图算法（如 `'gradcam'`）。
    - `blend_alpha`：叠加透明度，默认 0.5（0~1）。
    - `colormap`：颜色映射，`'jet'|'viridis'|'inferno'|...`。
  - 特征图（`method='feature'|'feature_map'`）
    - `top_k`：按通道评分选取的可视化通道数（默认 8）。
    - `metric`：通道评分方式（`'sum'|'var'`）。
    - `normalize`：归一化方式（基础版 `'minmax'`）。
    - `colormap`：网格渲染色图（如 `'gray'|'jet'`）。
    - `align_base`：双模态对齐基准（`'rgb'|'x'`）。
    - `split`：是否额外导出单通道小图（布尔）。
    - `ablation_fill`：做消融时的填充值（`'zeros'|'mean'`）。

提示：若参数与输入条件不匹配（如缺少 `layers`、设备不一致、叠加底图与输入不符等），会直接报错；不做任何自动降级。

## 双模态示例（仅此项）

> 将示例路径替换为你本机的图片/权重路径。

### YOLOMM 热力图（双模态）
```python
from ultralytics import YOLOMM

model = YOLOMM('path/to/yolomm/best.pt')
model.vis(
    rgb_source='path/to/rgb.png',
    x_source='path/to/x.png',
    method='heat',
    layers=[7, 15, 18, 29],
    overlay='rgb',        # 二级参数：叠加在 RGB 底图
    alg='gradcam',        # 二级参数：热图算法
    save=True,
    project='runs/visualize/yolomm',
    name='exp_yolo_dual',
)
```

### RTDETRMM 热力图（双模态）
```python
from ultralytics import RTDETRMM

model = RTDETRMM('path/to/rtdetrmm/best.pt')
model.vis(
    rgb_source='path/to/rgb.png',
    x_source='path/to/x.png',
    method='heatmap',
    layers=[2, 4, 6, 18, 28],
    overlay='rgb',        # 二级参数：叠加在 RGB 底图
    alg='gradcam',        # 二级参数：热图算法
    save=True,
    project='runs/visualize/rtdetr',
    name='exp_rtdetr_dual',
)
```
