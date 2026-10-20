"""Unified Renderer for visualize_core.

Provides consistent rendering helpers for heatmap overlays and (future) feature
grids. For heatmaps, applies a consistent alpha blending and optional colormap.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import cv2


class Renderer:
    @staticmethod
    def identity(x: Any) -> Any:
        return x

    @staticmethod
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[:, :, :3]
        raise ValueError(f"Unsupported image shape for RGB conversion: {img.shape}")

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        x = img
        if x.max() <= 1.0:
            x = (x * 255.0).clip(0, 255)
        return x.astype(np.uint8)

    @staticmethod
    def heat_overlay(
        original: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap onto original RGB image with consistent alpha.

        - If heatmap is single-channel float, apply colormap first.
        - If heatmap is already color (H,W,3), blend directly.
        - Original/X 保持真实通道；只对可见化渲染阶段进行 RGB 叠加。
        """
        ori = Renderer._to_uint8(Renderer._ensure_rgb(original))
        hm = heatmap
        # 尺寸对齐：若热图大小与原图不同，先缩放到原图尺寸
        if hm.ndim == 2:
            if hm.shape[0] != ori.shape[0] or hm.shape[1] != ori.shape[1]:
                hm = cv2.resize(hm, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_CUBIC)
        elif hm.ndim == 3 and hm.shape[2] in (1, 3):
            if hm.shape[0] != ori.shape[0] or hm.shape[1] != ori.shape[1]:
                hm = cv2.resize(hm, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_CUBIC)
        if hm.ndim == 2:
            hm = Renderer._to_uint8(hm)
            hm = cv2.applyColorMap(hm, colormap)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        elif hm.ndim == 3 and hm.shape[2] == 3:
            hm = Renderer._to_uint8(hm)
            # 若热图来源于 cv2.applyColorMap，则为 BGR；统一转为 RGB 再叠加
            try:
                hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            except Exception:
                pass
        else:
            raise ValueError(f"Unsupported heatmap shape: {heatmap.shape}")
        return cv2.addWeighted(ori, 1 - alpha, hm, alpha, 0)

    @staticmethod
    def heat_overlay_multimodal(
        originals: Dict[str, np.ndarray],
        heatmaps: Dict[str, np.ndarray],
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k in originals.keys():
            if k in heatmaps:
                out[k] = Renderer.heat_overlay(originals[k], heatmaps[k], alpha=alpha, colormap=colormap)
        return out
