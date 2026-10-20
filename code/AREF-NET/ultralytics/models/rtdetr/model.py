# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
# 说明：可视化异常与工具已迁移至 visualize_core，此处无需引入旧路径
# from ultralytics.models.yolo.multimodal.modal_filling import generate_modality_filling
# 说明：模态填充与消融已统一下沉至 MultiModalRouter（ultralytics/nn/mm/router.py）。
# 这里不再直接调用 generate_modality_filling，旧实现保留为注释以便回溯。

from .cocoval import RTDETRMMCOCOValidator
from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware
    query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """
        Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """
        Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }

    def cocoval(self, **kwargs):
        """
        Run COCO evaluation using RTDETRMMCOCOValidator.

        Args:
            **kwargs: Additional arguments to pass to the validator.

        Returns:
            dict: Validation metrics from the COCO evaluation.
        """
        from .cocoval import RTDETRMMCOCOValidator
        
        self._check_is_pytorch_model()
        
        args = {**self.overrides, **{"rect": True, "conf": 0.05}, **kwargs, **{"mode": "val"}}
        # 在验证前统一输出GFLOPs（架构级 + 路由感知）
        try:
            from ultralytics.utils.torch_utils import compute_model_gflops
            imgsz = int(args.get("imgsz", 640))
            modality = args.get("modality", None)
            arch_gflops = compute_model_gflops(self.model, imgsz=imgsz, modality=None, route_aware=False)
            route_gflops = compute_model_gflops(self.model, imgsz=imgsz, modality=modality, route_aware=True)
            from ultralytics.utils import LOGGER as _LOGGER
            _LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[{modality or 'dual'}]): {route_gflops:.2f}")
        except Exception:
            pass

        validator = RTDETRMMCOCOValidator(
            dataloader=None,
            save_dir=None,
            pbar=None,
            args=args,
            _callbacks=self.callbacks
        )
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics


class RTDETRMM(RTDETR):
    """
    RT-DETR MultiModal (RTDETRMM) object detection model.

    RTDETRMM extends the RT-DETR architecture to support multi-modal input (RGB + X modality) for enhanced
    object detection performance. It supports flexible channel configurations and automatic modality
    routing for RGB, X, and Dual modality inputs.

    Attributes:
        model: The loaded RTDETRMM model instance.
        task: The task type (detect).
        overrides: Configuration overrides for the model.
        input_channels: Number of input channels (3 for RGB-only, 6 for RGB+X).
        modality_config: Configuration for supported modalities.
        is_multimodal: Whether the model is configured for multi-modal operation.

    Methods:
        __init__: Initialize RTDETRMM model with multi-modal configuration.
        task_map: Map tasks to their corresponding multi-modal model, trainer, validator, and predictor classes.
        validate_input_channels: Validate input channels against model configuration.
        get_modality_info: Get information about supported modalities.

    Examples:
        Load a RTDETRMM detection model
        >>> model = RTDETRMM("rtdetr-r18-mm.yaml")

        Load with specific channel configuration
        >>> model = RTDETRMM("rtdetr-r34-mm.yaml", ch=6)  # RGB+X modality

        RGB-only mode
        >>> model = RTDETRMM("rtdetr-r50-mm.yaml", ch=3)  # RGB-only
    """

    def __init__(self, model: Union[str, Path] = "rtdetr-r18-mm.pt", ch: Optional[int] = None, 
                 verbose: bool = False) -> None:
        """
        Initialize RTDETRMM multi-modal model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'rtdetr-r18-mm.yaml', 'rtdetr-r50-mm.pt'.
            ch (int, optional): Number of input channels. If None, auto-detected from model config.
                Supported values: 3 (RGB-only), 6 (RGB+X).
            verbose (bool): Display model info on load.

        Examples:
            >>> model = RTDETRMM("rtdetr-r18-mm.yaml")  # Auto-detect channels
            >>> model = RTDETRMM("rtdetr-r34-mm.yaml", ch=6)  # RGB+X modality
            >>> model = RTDETRMM("rtdetr-r50-mm.yaml", ch=3)  # RGB-only mode
        """
        # Store multi-modal specific attributes
        self.input_channels = ch
        self.modality_config = {}
        # 强制启用多模态模式 - RTDETRMM设计为专用多模态模型
        self.is_multimodal = True
        
        # Check model file name for warning (but don't affect is_multimodal)
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-mm" not in path.stem:
            if verbose:
                print(f"⚠️ 警告: 模型文件 '{path.stem}' 不包含 '-mm' 后缀，但RTDETRMM已强制启用多模态模式")
                print(f"   请确保使用的是多模态训练的权重文件，否则可能出现维度不匹配")
        elif verbose:
            print(f"✅ 检测到多模态RTDETR模型: {path.stem}")
        
        # Initialize base RTDETR model
        super().__init__(model=model)
        
        # Configure multi-modal settings (always configure for RTDETRMM)
        self._configure_multimodal_settings(verbose)
        
        # Ensure mm_router exists for multimodal operation
        self._ensure_mm_router(verbose)
    
    def _configure_multimodal_settings(self, verbose: bool = False) -> None:
        """
        Configure multi-modal settings based on model configuration.

        Args:
            verbose (bool): Display configuration info.
        """
        try:
            # Get model configuration
            if hasattr(self.model, 'yaml') and self.model.yaml:
                model_yaml = self.model.yaml
                
                # Check for multimodal layers in configuration
                has_multimodal_layers = self._detect_multimodal_layers(model_yaml)
                
                # Determine input channels from model configuration
                model_channels = model_yaml.get('ch', model_yaml.get('channels', 3))
                
                # If multimodal layers detected, determine channel count
                if has_multimodal_layers:
                    # Check for Dual modality layers (6 channels)
                    has_dual_layers = self._has_dual_modality_layers(model_yaml)
                    if has_dual_layers:
                        model_channels = 6
                    else:
                        model_channels = 3  # RGB or X only
                
                # Validate input channels
                if self.input_channels is None:
                    self.input_channels = model_channels
                    if verbose:
                        print(f"Auto-detected input channels: {self.input_channels}")
                elif self.input_channels != model_channels:
                    if verbose:
                        print(f"Warning: Specified channels ({self.input_channels}) differ from model config ({model_channels})")
                
                # Validate channel configuration
                self.validate_input_channels()
                
                # Configure modality information based on detected multimodal layers
                if has_multimodal_layers:
                    if self.input_channels == 6:
                        self.modality_config.update({
                            'rgb_channels': [0, 1, 2],
                            'x_channels': [3, 4, 5],
                            'supported_modalities': ['RGB', 'X', 'Dual'],
                            'default_modality': 'Dual'
                        })
                    else:
                        self.modality_config.update({
                            'rgb_channels': [0, 1, 2],
                            'x_channels': [3, 4, 5],
                            'supported_modalities': ['RGB', 'X'],
                            'default_modality': 'RGB'
                        })
                else:
                    self.modality_config.update({
                        'rgb_channels': [0, 1, 2],
                        'x_channels': [],
                        'supported_modalities': ['RGB'],
                        'default_modality': 'RGB'
                    })
                
                if verbose and self.modality_config:
                    print(f"RTDETRMM configured: {self.input_channels} channels, "
                          f"modalities: {self.modality_config.get('supported_modalities', [])}")
                    
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to configure multi-modal settings: {e}")
            # Set default configuration
            self.input_channels = self.input_channels or 3
            self.modality_config = {
                'supported_modalities': ['RGB'],
                'default_modality': 'RGB'
            }
    
    def _detect_multimodal_layers(self, model_yaml: dict) -> bool:
        """
        Detect if the model configuration contains multimodal layers.

        Args:
            model_yaml (dict): Model YAML configuration

        Returns:
            bool: True if multimodal layers detected
        """
        all_layers = model_yaml.get('backbone', []) + model_yaml.get('head', [])
        
        for layer_config in all_layers:
            if len(layer_config) >= 5:
                input_source = layer_config[4]
                if input_source in ['RGB', 'X', 'Dual']:
                    return True
        return False
    
    def _has_dual_modality_layers(self, model_yaml: dict) -> bool:
        """
        Check if the model configuration has Dual modality layers.

        Args:
            model_yaml (dict): Model YAML configuration

        Returns:
            bool: True if Dual modality layers found
        """
        all_layers = model_yaml.get('backbone', []) + model_yaml.get('head', [])
        
        for layer_config in all_layers:
            if len(layer_config) >= 5:
                input_source = layer_config[4]
                if input_source == 'Dual':
                    return True
        return False
    
    def validate_input_channels(self) -> None:
        """
        Validate input channels against supported configurations.

        Raises:
            ValueError: If input channels are not supported.
        """
        supported_channels = [3, 6]
        if self.input_channels not in supported_channels:
            raise ValueError(
                f"Unsupported input channels: {self.input_channels}. "
                f"Supported channels: {supported_channels} "
                f"(3=RGB-only, 6=RGB+X)"
            )
    
    def _ensure_mm_router(self, verbose: bool = False) -> None:
        """
        Ensure the model has mm_router for multimodal operation.
        
        This is necessary because models loaded from weights may not have mm_router initialized.
        
        Args:
            verbose (bool): Display debug information.
        """
        if hasattr(self, 'model'):
            # Check if the RTDETRDetectionModel has mm_router
            if not hasattr(self.model, 'mm_router') or self.model.mm_router is None:
                try:
                    from ultralytics.nn.mm import MultiModalRouter
                    # Create mm_router with model configuration
                    config_dict = getattr(self.model, 'yaml', None)
                    self.model.mm_router = MultiModalRouter(config_dict, verbose=verbose)
                    if verbose:
                        print(f"✅ RTDETRMM: 为RTDETRDetectionModel创建了MultiModalRouter")
                except Exception as e:
                    if verbose:
                        print(f"⚠️ RTDETRMM: 无法创建MultiModalRouter: {e}")
                        print(f"   这可能会影响多模态推理功能")
    
    def get_modality_info(self) -> Dict[str, Any]:
        """
        Get information about supported modalities and configuration.

        Returns:
            dict: Modality configuration information.
        """
        return {
            'input_channels': self.input_channels,
            'modality_config': self.modality_config.copy(),
            'model_type': 'RTDETRMM',
            'task': getattr(self, 'task', 'detect'),
            'is_multimodal': self.is_multimodal
        }
    
    @property
    def task_map(self) -> dict:
        """
        Return a task map for RTDETRMM, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RTDETRMM model.
        """
        if self.is_multimodal:
            try:
                # Import multi-modal components (only if available)
                from ultralytics.models.rtdetr.multimodal import (
                    RTDETRMMTrainer,
                    RTDETRMMValidator,
                    RTDETRMMPredictor
                )
                
                return {
                    "detect": {
                        "predictor": RTDETRMMPredictor,
                        "validator": RTDETRMMValidator,
                        "trainer": RTDETRMMTrainer,
                        "model": RTDETRDetectionModel,  # Use standard model with multi-modal routing
                    }
                }
            except ImportError as e:
                # If multi-modal components are not available, fall back to standard RTDETR
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"Warning: Multi-modal components not available ({e}), using standard RTDETR components")
                return super().task_map
        else:
            # For non-multimodal models, use parent's task_map
            return super().task_map
    
    def vis(self,
            rgb_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
            x_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
            method: str = 'heat',
            layers: Optional[List[int]] = None,
            modality: Optional[str] = None,
            save: bool = True,
            overlay: Optional[str] = None,
            project: Optional[Union[str, Path]] = None,
            name: Optional[str] = None,
            out_dir: Optional[Union[str, Path]] = None,  # 已废弃：请使用 project/name
            device: Optional[str] = None,
            **kwargs) -> Union['VisualizationResult', List['VisualizationResult']]:
        """
        可视化入口（重构版）：委托到家族 Runner（RTDETRMMVisualizationRunner）。

        参数说明：
        - rgb_source/x_source：输入源（路径/np.ndarray/列表），位置参数依次为 RGB、X；可只传其中之一
        - method：'heat'|'heatmap' 或 'feature'|'feature_map'
        - layers：待可视化的层索引列表（必填）
        - modality：模态消融控制；
            * 单模态输入下强制消融（保持单侧真实，另一侧由 Router 填充）
            * 双模态输入下默认不消融；若显式为 'rgb' 或 'x' 则强制消融
        - overlay：热图叠加底图控制 'rgb'|'x'|'dual'；默认叠加到 RGB，仅传 X 时自动改为 X
        - project/name：保存目录控制（推荐）；out_dir 已废弃，仅兼容

        注意：严格 Fail‑Fast，无未授权的自动降级。
        """
        from ultralytics.models.rtdetr.visualize.runner import RTDETRMMVisualizationRunner

        return RTDETRMMVisualizationRunner.run(
            model=self.model,
            rgb_source=rgb_source,
            x_source=x_source,
            method=method,
            layers=layers,
            modality=modality,
            save=save,
            overlay=overlay,
            project=str(project) if project is not None else None,
            name=str(name) if name is not None else None,
            out_dir=str(out_dir) if out_dir is not None else None,
            device=device,
            **kwargs,
        )

    # --- 便捷封装：仅设置 method 并转发 ---
    def vis_heat(
        self,
        rgb_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        x_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        layers: Optional[List[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        overlay: Optional[str] = None,
        project: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        out_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        return self.vis(
            rgb_source=rgb_source,
            x_source=x_source,
            method='heat',
            layers=layers,
            modality=modality,
            save=save,
            overlay=overlay,
            project=project,
            name=name,
            out_dir=out_dir,
            device=device,
            **kwargs,
        )

    def vis_feature(
        self,
        rgb_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        x_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        layers: Optional[List[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        overlay: Optional[str] = None,
        project: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        out_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        return self.vis(
            rgb_source=rgb_source,
            x_source=x_source,
            method='feature',
            layers=layers,
            modality=modality,
            save=save,
            overlay=overlay,
            project=project,
            name=name,
            out_dir=out_dir,
            device=device,
            **kwargs,
        )

# ------------------------------
# Legacy vis implementation (commented for refactor history)
# ------------------------------
# The previous implementation of RTDETRMM.vis performed:
# - method alias mapping {'heat'|'heatmap','feature'|'feature_map'}
# - strict layers validation with custom exceptions
# - modality auto-inference/conflict checks (dual/rgb/x)
# - device consistency check without auto-switch
# - delegation to VisualizationPipeline(self.model) with project/name dispatch to runs/visualize/rtdetr
# - alg forwarding for heatmap
# Kept here as high-level reference to avoid hard deletion per refactor guideline.
