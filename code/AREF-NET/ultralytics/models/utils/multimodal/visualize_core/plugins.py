"""Visualization method plugin stubs (to be implemented in later steps)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import torch

from .types import CoreVisualizationResult
from .registry import REGISTRY
from .preprocessor import Preprocessor
from .renderer import Renderer


class MethodPlugin:
    """Base interface for visualization method plugins."""

    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        raise NotImplementedError


class HeatmapPlugin(MethodPlugin):
    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        # Import existing heatmap visualizer (reused implementation)
        from ultralytics.models.yolo.multimodal.visualize.heatmap import HeatmapVisualizer

        # Originals for overlay (保持原始图，不做伪彩)
        originals: Dict[str, np.ndarray] | np.ndarray
        if 'rgb' in inputs and 'x' in inputs:
            originals = {'rgb': inputs['rgb'], 'x': inputs['x']}
            inferred_modality = 'dual'
        elif 'rgb' in inputs:
            originals = inputs['rgb']
            inferred_modality = 'rgb'
        else:
            originals = inputs['x']
            inferred_modality = 'x'

        # 统一前处理（letterbox + 归一化），生成模型输入张量（NCHW, float32, [0,1]）
        pre = Preprocessor.prepare_inputs(inputs, model)  # HWC, float32, [0,1]
        if pre.ndim == 3:
            input_tensor = torch.from_numpy(np.transpose(pre, (2, 0, 1))).unsqueeze(0).float()
        else:
            raise ValueError("预处理后的热力图输入维度异常，应为 HWC")

        # Heatmap visualizer 接收预处理后的张量以避免二次 letterbox，同时使用 originals 做叠加参考
        heat_layers = [str(i) for i in layers]
        alg = kwargs.get('alg', 'gradcam')

        vis = HeatmapVisualizer(model)
        results = vis.visualize(
            images=originals,
            layers=heat_layers,
            alg=alg,
            preprocessed_input=input_tensor,
            original_images=originals,
            **{k: v for k, v in kwargs.items() if k != 'alg'}
        )

        out: List[CoreVisualizationResult] = []
        blend_alpha = float(kwargs.get('blend_alpha', 0.5))
        cmap = kwargs.get('colormap', 'jet')
        cmap_map = {
            'jet': cv2.COLORMAP_JET,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'plasma': cv2.COLORMAP_PLASMA,
        }
        cmap_cv2 = cmap_map.get(str(cmap).lower(), cv2.COLORMAP_JET)

        # 叠加底图控制：overlay ∈ {rgb, x, dual}，默认 rgb；若仅提供 X，则自动转为 x
        overlay_req = kwargs.get('overlay', None)
        has_rgb = 'rgb' in inputs
        has_x = 'x' in inputs
        overlay_base: str
        if overlay_req is None:
            overlay_base = 'rgb' if has_rgb else ('x' if has_x else 'rgb')
        else:
            overlay_base = str(overlay_req).lower().strip()
            if overlay_base not in {'rgb', 'x', 'dual'}:
                raise ValueError(f"overlay 参数非法：{overlay_req}，可选：rgb|x|dual")
            # 自动调整：若用户仅传 X，但要求 rgb，则调整为 x（按用户要求允许）
            if overlay_base == 'rgb' and (not has_rgb) and has_x:
                overlay_base = 'x'
        # 对于 overlay='x' 但无 X，或 overlay='dual' 但非双模态，Fail-Fast
        if overlay_base == 'x' and not has_x:
            raise ValueError("overlay='x' 需要提供 X 模态输入")
        if overlay_base == 'dual' and not (has_rgb and has_x):
            raise ValueError("overlay='dual' 需要同时提供 RGB 与 X 输入")

        for li, r in zip(layers, results):
            # 尝试统一叠加风格：优先使用原图 + 本次渲染参数重新叠加
            data = None
            try:
                originals = getattr(r, 'original_image', None)
                heatmaps = getattr(r, 'heatmap', None)
                if isinstance(originals, dict) and isinstance(heatmaps, dict):
                    if overlay_base == 'dual':
                        data = Renderer.heat_overlay_multimodal(originals, heatmaps, alpha=blend_alpha, colormap=cmap_cv2)
                    else:
                        if overlay_base not in originals or overlay_base not in heatmaps:
                            raise ValueError(f"叠加底图 {overlay_base} 在输入中不可用")
                        data = Renderer.heat_overlay(originals[overlay_base], heatmaps[overlay_base], alpha=blend_alpha, colormap=cmap_cv2)
                elif isinstance(originals, np.ndarray) and isinstance(heatmaps, np.ndarray):
                    data = Renderer.heat_overlay(originals, heatmaps, alpha=blend_alpha, colormap=cmap_cv2)
            except Exception:
                # 若统一叠加异常，回退到可视化器的既有结果（仍为热图叠加图像）
                data = getattr(r, 'overlay', getattr(r, 'data', None))
                # 颜色空间统一：overlay 多为 OpenCV 生成（BGR），统一转为 RGB 再保存/显示
                try:
                    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[2] == 3:
                        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    elif isinstance(data, dict):
                        converted = {}
                        for dk, dv in data.items():
                            if isinstance(dv, np.ndarray) and dv.ndim == 3 and dv.shape[2] == 3:
                                try:
                                    converted[dk] = cv2.cvtColor(dv, cv2.COLOR_BGR2RGB)
                                except Exception:
                                    converted[dk] = dv
                            else:
                                converted[dk] = dv
                        data = converted
                except Exception:
                    pass

            meta = {
                'method': 'heat',
                'layer_idx': li,
                # 文件命名采用叠加底图语义（rgb/x/dual）
                'modality': overlay_base,
                'family': family,
                'algorithm': alg,
                'alpha': blend_alpha,
                'colormap': cmap,
            }
            out.append(CoreVisualizationResult(type='heat', data=data, meta=meta))
        return out


class FeatureMapPlugin(MethodPlugin):
    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        """
        基础版特征图可视化（与热力图对齐语义）：
        - 自动识别单/双模态：仅一侧时自动做消融（另一侧默认补 0），两侧都有则按双模态。
        - 统一 letterbox（可选对齐基准），拼接为 HWC[3+Xch]，再转 NCHW。
        - 逐层：通道打分（sum/var）→ 选 top_k → 网格渲染（默认灰度）。
        - 返回每层一张网格图，保存命名与热力图风格保持一致（feature_layer{idx}_...）。
        """

        # -----------------
        # 参数与前置校验
        # -----------------
        align_base = str(kwargs.get('align_base', 'rgb')).lower()
        metric = str(kwargs.get('metric', 'sum')).lower()  # 'sum' | 'var'
        top_k = int(kwargs.get('top_k', 8))
        normalize = str(kwargs.get('normalize', 'minmax')).lower()  # 仅实现 minmax
        colormap = str(kwargs.get('colormap', 'gray')).lower()
        split = bool(kwargs.get('split', False))

        # 自动推断/执行：若只传一侧则自动消融填充，默认 zeros（不新增控制参数）
        has_rgb = 'rgb' in inputs
        has_x = 'x' in inputs
        if modality is None or str(modality).lower() == 'auto':
            m = ('dual' if (has_rgb and has_x) else ('rgb' if has_rgb else 'x'))
        else:
            m = str(modality).lower()
        ablation_fill = str(kwargs.get('ablation_fill', 'zeros')).lower()

        # -----------------
        # 预处理：对齐式 letterbox + 显式消融
        # -----------------
        size = Preprocessor.model_input_size(model)
        # 推断模型期望的总输入通道数（优先使用首个 Conv 的 in_channels）
        def _infer_in_channels(mod: Any) -> int:
            try:
                import torch.nn as nn  # noqa: F401
                for m_ in mod.modules():
                    if hasattr(m_, 'in_channels') and hasattr(m_, 'out_channels'):
                        ic = int(getattr(m_, 'in_channels'))
                        if ic > 0:
                            return ic
            except Exception:
                pass
            return 6  # 常见多模态默认

        if has_rgb and has_x:
            hwc = Preprocessor.letterbox_dual_aligned(inputs['rgb'], inputs['x'], size=size, align_base=align_base)
            if hwc.ndim != 3 or hwc.shape[2] < 4:
                raise ValueError(f"预处理后的形状异常：{hwc.shape}，期望通道数 ≥4（RGB3 + Xch≥1）")
            rgb_hwc = hwc[:, :, :3]
            x_hwc = hwc[:, :, 3:]
        elif has_rgb and not has_x:
            rgb_hwc = Preprocessor.letterbox_single(inputs['rgb'], size)
            if rgb_hwc.ndim == 2:
                rgb_hwc = rgb_hwc[:, :, None]
            if rgb_hwc.shape[2] == 1:
                rgb_hwc = np.repeat(rgb_hwc, 3, axis=2)
            elif rgb_hwc.shape[2] > 3:
                rgb_hwc = rgb_hwc[:, :, :3]
            total_ch = _infer_in_channels(model)
            x_expect = max(total_ch - 3, 0)
            fill_val = 0.0 if ablation_fill == 'zeros' else 0.5
            x_hwc = np.full((rgb_hwc.shape[0], rgb_hwc.shape[1], x_expect), fill_val, dtype=np.float32) if x_expect > 0 else np.zeros((rgb_hwc.shape[0], rgb_hwc.shape[1], 0), dtype=np.float32)
        elif has_x and not has_rgb:
            x_hwc = Preprocessor.letterbox_single(inputs['x'], size)
            if x_hwc.ndim == 2:
                x_hwc = x_hwc[:, :, None]
            total_ch = _infer_in_channels(model)
            x_expect = max(total_ch - 3, 0)
            if x_hwc.shape[2] == 1 and x_expect > 1:
                x_hwc = np.repeat(x_hwc, x_expect, axis=2)
            elif x_hwc.shape[2] < x_expect:
                reps = int(np.ceil(x_expect / x_hwc.shape[2]))
                x_hwc = np.concatenate([x_hwc] * reps, axis=2)[:, :, :x_expect]
            elif x_hwc.shape[2] > x_expect:
                x_hwc = x_hwc[:, :, :x_expect]
            rgb_hwc = np.zeros((x_hwc.shape[0], x_hwc.shape[1], 3), dtype=np.float32)
        else:
            raise ValueError("未检测到有效的 RGB/X 输入。")

        hwc = np.concatenate([rgb_hwc, x_hwc], axis=2)

        # NCHW + batch
        nchw = np.transpose(hwc, (2, 0, 1))[None, ...].astype(np.float32)
        inp = torch.from_numpy(nchw)

        # -----------------
        # 前向 hook 捕获指定层输出（逐层）
        # -----------------
        feats: Dict[int, torch.Tensor] = {}
        handles = []

        def _first_tensor(x: Any) -> torch.Tensor | None:
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, (list, tuple)):
                for t in x:
                    if isinstance(t, torch.Tensor):
                        return t
            return None

        for li in layers:
            if not hasattr(model, 'model') or li < 0 or li >= len(model.model):
                raise ValueError(f"层索引越界：{li}")
            mod = model.model[li]

            def _hook_closure(idx: int):
                def _hook(module, inputs_, output):
                    t = _first_tensor(output)
                    if t is not None:
                        feats[idx] = t.detach()
                return _hook

            handles.append(mod.register_forward_hook(_hook_closure(li)))

        # 前向执行
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        inp = inp.to(device)
        with torch.no_grad():
            _ = model(inp)

        # 清理 hook
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

        # -----------------
        # 打分与选通道 → 网格渲染（逐层输出）
        # -----------------
        def _score_channel_map(t: torch.Tensor) -> torch.Tensor:
            # t: [C,H,W]
            if metric == 'sum':
                return t.abs().sum(dim=(1, 2))
            elif metric == 'var':
                return t.var(dim=(1, 2))
            else:
                raise ValueError(f"不支持的 metric: {metric}（仅 'sum'|'var'）")

        def _norm_uint8(arr: np.ndarray) -> np.ndarray:
            if normalize == 'minmax':
                a_min, a_max = float(arr.min()), float(arr.max())
                if a_max > a_min:
                    out = (arr - a_min) / (a_max - a_min)
                else:
                    out = np.zeros_like(arr)
                return (out * 255.0).astype(np.uint8)
            else:
                raise ValueError(f"不支持的 normalize: {normalize}（基础版仅 'minmax'）")

        def _cv2_colormap(name: str) -> int | None:
            name = str(name).lower()
            if name in {'gray', 'grey', 'grayscale', 'none'}:
                return None
            m = {
                'jet': cv2.COLORMAP_JET,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'inferno': cv2.COLORMAP_INFERNO,
                'magma': cv2.COLORMAP_MAGMA,
                'plasma': cv2.COLORMAP_PLASMA,
            }
            return m.get(name, cv2.COLORMAP_VIRIDIS)

        def _render_grid(feat_list: List[tuple[int, float, np.ndarray]], cell: tuple[int, int] = (128, 128)) -> np.ndarray:
            # feat_list: list of (ch_idx, score, 2D np.array)
            if not feat_list:
                return np.zeros((256, 256, 3), dtype=np.uint8)
            n = len(feat_list)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            h, w = cell
            pad = 5
            label_h = 22
            canvas = np.ones((rows * (h + label_h + pad) + pad, cols * (w + pad) + pad, 3), dtype=np.uint8) * 255

            cmap = _cv2_colormap(colormap)

            for i, (ch, sc, fm) in enumerate(feat_list):
                r, c = i // cols, i % cols
                y = pad + r * (h + label_h + pad)
                x = pad + c * (w + pad)
                u8 = _norm_uint8(fm)
                u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA)
                if cmap is None:
                    rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.applyColorMap(u8, cmap)
                    try:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                canvas[y:y + h, x:x + w] = rgb
                label = f"ch:{ch} score:{sc:.1f}"
                cv2.putText(canvas, label, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            return canvas

        modality_meta = {
            ('rgb', True): 'rgb_ablate',
            ('x', True): 'x_ablate',
        }.get((m, not (has_rgb and has_x)), m)

        results: List[CoreVisualizationResult] = []
        for li in layers:
            t = feats.get(li, None)
            if t is None:
                # 指定层可能无张量输出（例如返回 tuple 无张量）
                continue
            if t.dim() == 4:
                # [B,C,H,W] → 仅支持逐样本；基础版默认 B=1
                b, c, h_, w_ = t.shape
                # 逐样本处理（基础版仅取第 0 个）
                tt = t[0]
            elif t.dim() == 3:
                # [C,H,W]
                tt = t
                c = tt.shape[0]
            else:
                # 非 3/4 维数据不处理
                continue

            scores = _score_channel_map(tt)
            k = int(min(max(1, top_k), int(scores.numel())))
            top_idx = torch.topk(scores, k).indices.cpu().numpy().tolist()

            feat_list: List[tuple[int, float, np.ndarray]] = []
            tiles_imgs: List[np.ndarray] = []
            tiles_channels: List[int] = []
            tiles_scores: List[float] = []

            def _render_tile(fm: np.ndarray, cell: tuple[int, int] = (128, 128)) -> np.ndarray:
                u8 = _norm_uint8(fm)
                u8 = cv2.resize(u8, (cell[1], cell[0]), interpolation=cv2.INTER_AREA)
                cmap_code = _cv2_colormap(colormap)
                if cmap_code is None:
                    return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.applyColorMap(u8, cmap_code)
                    try:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                    return rgb

            for ch in top_idx:
                fmap = tt[ch].detach().cpu().float().numpy()
                sc = float(scores[ch].detach().cpu().item())
                feat_list.append((int(ch), sc, fmap))
                if split:
                    tiles_imgs.append(_render_tile(fmap))
                    tiles_channels.append(int(ch))
                    tiles_scores.append(sc)

            grid = _render_grid(feat_list)
            meta = {
                'method': 'feature',
                'layer_idx': li,
                'modality': modality_meta,
                'family': family,
                'metric': metric,
                'top_k': top_k,
                'normalize': normalize,
                'align_base': align_base,
            }
            # 附带 img_key（目录/批量模式用于分目录保存）
            img_key = kwargs.get('img_key', inputs.get('img_key') if isinstance(inputs, dict) else None)
            if img_key is not None:
                meta['img_key'] = img_key
            results.append(CoreVisualizationResult(type='feature', data=grid, meta=meta))

            if split and tiles_imgs:
                tiles_meta = {
                    **meta,
                    'channels': tiles_channels,
                    'scores': tiles_scores,
                    'subdir': f'layer{li}',
                }
                if img_key is not None:
                    tiles_meta['img_key'] = img_key
                results.append(CoreVisualizationResult(type='feature_tiles', data=tiles_imgs, meta=tiles_meta))

        return results


# Register plugins
REGISTRY.register('heat', HeatmapPlugin)
REGISTRY.register('feature', FeatureMapPlugin)
