"""Unified saver for visualization results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .types import CoreVisualizationResult


class Saver:
    @staticmethod
    def _ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _save_array(img: np.ndarray, path: Path) -> None:
        # Assume RGB input; convert to BGR for OpenCV
        if img.ndim == 3 and img.shape[2] == 3:
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), img)

    @classmethod
    def save(cls, *, results: List[CoreVisualizationResult], out_dir: Path, method: str) -> List[str]:
        cls._ensure_dir(out_dir)
        saved: List[str] = []
        for r in results:
            layer_idx = r.meta.get("layer_idx", "na")
            modality = r.meta.get("modality", "auto")
            # feature map may be a grid image, heat is overlay or colormap
            base = f"{method}_layer{layer_idx}_{modality}"
            # Determine destination directory (support per-sample subdir via img_key)
            base_dir = out_dir
            img_key = r.meta.get('img_key', None)
            if isinstance(img_key, str) and len(img_key) > 0:
                base_dir = out_dir / img_key
                cls._ensure_dir(base_dir)
            if r.type == 'feature_tiles':
                # Save per-channel tiles under sub-directory per layer
                subdir = r.meta.get('subdir', None)
                dst_dir = base_dir / subdir if subdir else base_dir
                cls._ensure_dir(dst_dir)
                channels = r.meta.get('channels', None)
                if isinstance(r.data, list):
                    for i, v in enumerate(r.data):
                        if isinstance(v, np.ndarray):
                            if channels and i < len(channels):
                                fname = f"{base}_ch{int(channels[i])}.png"
                            else:
                                fname = f"{base}_{i:03d}.png"
                            path = dst_dir / fname
                            cls._save_array(v, path)
                            saved.append(str(path))
                # Proceed next result
                continue
            # Default naming for non-tiles
            if method == 'feature':
                base = f"{base}_grid"

            if isinstance(r.data, dict):
                for k, v in r.data.items():
                    if isinstance(v, np.ndarray):
                        path = base_dir / f"{base}_{k}.png"
                        cls._save_array(v, path)
                        saved.append(str(path))
            elif isinstance(r.data, list):
                for i, v in enumerate(r.data):
                    if isinstance(v, np.ndarray):
                        path = base_dir / f"{base}_{i:03d}.png"
                        cls._save_array(v, path)
                        saved.append(str(path))
            elif isinstance(r.data, np.ndarray):
                path = base_dir / f"{base}.png"
                cls._save_array(r.data, path)
                saved.append(str(path))
        return saved
