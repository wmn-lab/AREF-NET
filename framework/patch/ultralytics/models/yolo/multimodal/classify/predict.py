# Ultralytics YOLO, AGPL-3.0 license

"""
多模态分类预测器

本模块提供多模态分类任务的预测器，支持 RGB+X 模态的图像分类推理。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import classify_transforms
from ultralytics.data.multimodal.image_io import MultiModalImageIOMixin
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class MultiModalClassificationPredictor(BasePredictor, MultiModalImageIOMixin):
    """
    多模态分类预测器

    继承 BasePredictor 和 MultiModalImageIOMixin，
    提供多模态分类任务的推理功能。

    Attributes:
        args: 配置参数
        transforms: 图像变换

    Methods:
        preprocess: 预处理输入图像
        postprocess: 后处理预测结果

    Examples:
        >>> from ultralytics.models.yolo.multimodal.classify import MultiModalClassificationPredictor
        >>> predictor = MultiModalClassificationPredictor(overrides={"model": "yolomm-cls.pt"})
        >>> results = predictor.predict(["rgb.jpg", "depth.png"])
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[Dict[str, Any]] = None,
        _callbacks=None
    ):
        """
        初始化多模态分类预测器

        Args:
            cfg: 默认配置
            overrides: 配置覆盖
            _callbacks: 回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"

        # 推理模态控制（与 MultiModalDetectionPredictor 保持一致）
        self.modality = getattr(self.args, "modality", None)  # None=双模态；'rgb'/'x'/...
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        self._dual_input_detected = False

        # 多模态元信息（以 router 为准；args 仅作兜底）
        self.x_modality = getattr(self.args, "x_modality", "depth")
        self.x_modality_dir = getattr(self.args, "x_modality_dir", f"images_{self.x_modality}")
        self.expected_xch = int(getattr(self.args, "Xch", 3))

    # -----------------------------
    # Helper methods for MM routing
    # -----------------------------
    def _get_mm_router(self):
        """获取有效的 MultiModalRouter（兼容 AutoBackend 与不同命名）。"""
        m = getattr(self, "model", None)
        if m is None:
            return None

        # 1) 直连 PyTorch 模型：优先 mm_router，其次 multimodal_router
        for key in ("mm_router", "multimodal_router"):
            if hasattr(m, key):
                obj = getattr(m, key)
                if obj is not None:
                    return obj

        # 2) AutoBackend(PyTorch) 场景：实际模型在 m.model
        if hasattr(m, "pt") and getattr(m, "pt", False) and hasattr(m, "model"):
            inner = getattr(m, "model", None)
            if inner is None:
                return None
            for key in ("mm_router", "multimodal_router"):
                if hasattr(inner, key):
                    obj = getattr(inner, key)
                    if obj is not None:
                        return obj
        return None

    def _set_runtime_modality_for_router(self):
        """
        根据当前输入模式设置 router 的运行时参数。
        - 双模态（或明确检测到双模态输入）：runtime_modality=None
        - 单模态：runtime_modality=<modality>（'rgb' 或 'x' 等）
        """
        mm_router = self._get_mm_router()
        if not mm_router:
            return
        if self._dual_input_detected:
            mm_router.set_runtime_params(None)
            return
        mm_router.set_runtime_params(
            self.modality,
            strategy=getattr(self.args, "ablation_strategy", None),
            seed=getattr(self.args, "seed", None),
        )

    def _get_dual_channels(self) -> int:
        """读取路由器配置的 Dual 通道数(3+Xch)。若不可用则回退为6。"""
        mm_router = self._get_mm_router()
        try:
            if mm_router and hasattr(mm_router, "INPUT_SOURCES"):
                return int(mm_router.INPUT_SOURCES.get("Dual", 6))
        except Exception:
            pass
        return 6

    def _get_xch(self) -> int:
        """读取路由器配置的 X 通道数(Xch)。若不可用则回退到 args.Xch。"""
        mm_router = self._get_mm_router()
        try:
            if mm_router and hasattr(mm_router, "INPUT_SOURCES"):
                return int(mm_router.INPUT_SOURCES.get("X", self.expected_xch))
        except Exception:
            pass
        return int(self.expected_xch)

    def setup_source(self, source):
        """设置数据源和推理模式"""
        super().setup_source(source)

        # 更新 transforms
        updated = (
            self.model.model.transforms.transforms[0].size != max(self.imgsz)
            if hasattr(self.model.model, "transforms") and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        self.transforms = (
            classify_transforms(self.imgsz) if updated or not self.model.pt else self.model.model.transforms
        )

    def preprocess(self, img):
        """
        预处理输入图像

        支持两种输入模式：
        1. 单张图像：仅 RGB 推理
        2. 图像列表/元组：[RGB, X] 多模态推理

        Args:
            img: 输入图像（BGR numpy 数组或列表）

        Returns:
            预处理后的张量
        """
        if not isinstance(img, torch.Tensor):
            # 处理输入格式（与 MM OBB/Detect 的 [rgb, x] 约定对齐）
            if isinstance(img, (list, tuple)) and len(img) == 2 and self.is_dual_modal:
                # 多模态输入：[RGB, X]（注意：此处的元素通常是 BGR numpy）
                self._dual_input_detected = True
                self._set_runtime_modality_for_router()

                rgb_imgs, x_imgs = img
                if not isinstance(rgb_imgs, list):
                    rgb_imgs = [rgb_imgs]
                if not isinstance(x_imgs, list):
                    x_imgs = [x_imgs]
                if len(rgb_imgs) != len(x_imgs):
                    raise ValueError(f"双模态批大小不一致: RGB={len(rgb_imgs)}, X={len(x_imgs)}")

                tensors = []
                for rgb_img, x_img in zip(rgb_imgs, x_imgs):
                    tensors.append(self._preprocess_multimodal(rgb_img, x_img))
                img = torch.stack(tensors, dim=0)
            else:
                # 单模态输入：由 router 负责填充另一模态（需要设置 runtime_modality）
                self._dual_input_detected = False
                self._set_runtime_modality_for_router()

                if not isinstance(img, list):
                    img = [img]
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img],
                    dim=0,
                )

        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()

    def _preprocess_multimodal(
        self,
        rgb_img: np.ndarray,
        x_img: np.ndarray
    ) -> torch.Tensor:
        """
        预处理多模态图像对

        Args:
            rgb_img: RGB 图像（BGR numpy 数组）
            x_img: X 模态图像

        Returns:
            多模态张量 (C, H, W)
        """
        # 转换 RGB 为 PIL Image
        rgb_pil = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

        # 应用 transforms
        rgb_tensor = self.transforms(rgb_pil)  # (3, H, W)

        # 处理 X 模态
        # 对齐到 RGB 尺寸
        x_img = self.align_x_to_rgb(x_img, rgb_img.shape[:2])

        # 校验通道数（以 router 的 Xch 为准）
        expected_xch = self._get_xch()
        x_img = self.validate_x_channels(x_img, expected_xch)

        # Resize 到与 RGB 相同尺寸
        target_size = rgb_tensor.shape[1:]  # (H, W)
        x_resized = cv2.resize(x_img, (target_size[1], target_size[0]))

        # 确保 3D
        if len(x_resized.shape) == 2:
            x_resized = x_resized[:, :, np.newaxis]

        # 归一化并转为张量（与训练/数据集保持一致）
        if x_resized.dtype == np.uint8:
            x_tensor = torch.from_numpy(x_resized.astype(np.float32) / 255.0)
        elif x_resized.dtype == np.uint16:
            x_tensor = torch.from_numpy(x_resized.astype(np.float32) / 65535.0)
        elif np.issubdtype(x_resized.dtype, np.floating):
            x_f = x_resized.astype(np.float32)
            if x_f.min() < 0.0 or x_f.max() > 1.0:
                raise ValueError(
                    f"X模态为浮点类型但不在[0,1]范围内: min={x_f.min():.4g}, max={x_f.max():.4g}"
                )
            x_tensor = torch.from_numpy(x_f)
        else:
            raise TypeError(f"不支持的X模态数据类型: {x_resized.dtype}")

        # (H, W, C) -> (C, H, W)
        x_tensor = x_tensor.permute(2, 0, 1)

        # 拼接
        return torch.cat([rgb_tensor, x_tensor], dim=0)

    def postprocess(self, preds, img, orig_imgs):
        """
        后处理预测结果

        Args:
            preds: 模型原始输出
            img: 预处理后的图像
            orig_imgs: 原始图像

        Returns:
            Results 对象列表
        """
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds

        # Dual 模态：原始输入可能是 2 张（RGB/X），但实际结果可能只有 1 个（配对成一条样本）
        paths = self.batch[0]
        if (
            isinstance(paths, (list, tuple))
            and len(paths) == 2
            and isinstance(orig_imgs, (list, tuple))
            and len(orig_imgs) == 2
            and hasattr(preds, "shape")
            and int(preds.shape[0]) == 1
        ):
            paths = [paths[0]]
            orig_imgs = [orig_imgs[0]]
        elif not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, paths)
        ]

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        多模态分类推理流

        关键点：
        - 双模态输入([rgb, x])在 dataloader 层面会表现为两张图；但预处理会将其合成为 1 条样本；
          因此需按 results 数量而不是 im0s 数量来写结果/计时。
        - 多模态模型 warmup 需要使用 Dual 通道数(3+Xch)。
        """
        if self.args.verbose:
            LOGGER.info("")

        if not self.model:
            self.setup_model(model)

        with self._lock:
            self.setup_source(source if source is not None else self.args.source)

            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            if not self.done_warmup:
                display_ch = self._get_dual_channels()
                self.model.warmup(
                    imgsz=(
                        1 if getattr(self.model, "pt", False) or getattr(self.model, "triton", False) else self.dataset.bs,
                        display_ch,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (ops.Profile(device=self.device), ops.Profile(device=self.device), ops.Profile(device=self.device))
            self.run_callbacks("on_predict_start")

            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                with profilers[0]:
                    im = self.preprocess(im0s)

                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds
                        continue

                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                n_in = len(im0s) if isinstance(im0s, (list, tuple)) else 1
                n_out = len(self.results)
                if n_out != n_in:
                    LOGGER.debug(f"多模态分类推理: 输入{n_in}张图像，生成{n_out}个结果")

                for i in range(n_out):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n_out,
                        "inference": profilers[1].dt * 1e3 / n_out,
                        "postprocess": profilers[2].dt * 1e3 / n_out,
                    }

                    if n_out < (len(paths) if isinstance(paths, (list, tuple)) else 1):
                        result_path = Path(paths[0])
                        result_string = s[0] if s else ""
                        if isinstance(paths, (list, tuple)) and len(paths) > 1:
                            modality_info = f"({len(paths)}模态输入)"
                            result_string = f"{result_string} {modality_info}" if result_string else modality_info
                    else:
                        result_path = Path(paths[i])
                        result_string = s[i] if i < len(s) else ""

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        result_string += self.write_results(i, result_path, im, result_string)

                    if i < len(s):
                        s[i] = result_string
                    elif len(s) == 0:
                        s = [result_string]

                if self.args.verbose:
                    valid_strings = [s_item for s_item in s[:n_out] if s_item]
                    if valid_strings:
                        LOGGER.info("\n".join(valid_strings))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results
