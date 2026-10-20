# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal trainer module.

This module provides the RTDETRMMTrainer class for training multi-modal RT-DETR models
with support for RGB+X modality inputs.
"""

import torch
from copy import copy
from pathlib import Path
from typing import Optional

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel, compute_model_gflops
from ultralytics.utils.patches import torch_load


class RTDETRMMTrainer(DetectionTrainer):
    """
    A trainer class for RT-DETR MultiModal (RTDETRMM) object detection models.

    This class extends DetectionTrainer to support multi-modal inputs (RGB + X modality)
    during the training process. It integrates seamlessly with YOLOMM's data pipeline
    while maintaining RT-DETR's training strategies.

    Key Features:
        - Supports RGB+X dual-modal training with 6-channel input
        - Automatic detection of multi-modal configuration (-mm suffix)
        - Integration with YOLOMultiModalImageDataset
        - Support for single-modal ablation training via modality parameter
        - Graceful degradation to standard RT-DETR when multi-modal unavailable

    Attributes:
        args: Training arguments and hyperparameters.
        model: The RTDETRMM model being trained.
        modality: Specific modality for single-modal training (None for dual-modal).
        is_multimodal: Whether the model is configured for multi-modal operation.
        multimodal_config: Configuration for multi-modal training.

    Methods:
        get_dataset: Build multi-modal dataset using YOLOMultiModalImageDataset.
        get_model: Initialize RT-DETR model with multi-modal support.
        preprocess_batch: Preprocess batch data for multi-modal inputs.
        get_validator: Return multi-modal compatible validator.

    Examples:
        >>> # Dual-modal training
        >>> trainer = RTDETRMMTrainer(overrides={'model': 'rtdetr-r18-mm.yaml', 'data': 'multimodal-dataset.yaml'})
        >>> trainer.train()
        
        >>> # Single-modal ablation
        >>> trainer = RTDETRMMTrainer(overrides={'model': 'rtdetr-r18-mm.yaml', 'modality': 'thermal'})
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize RTDETRMMTrainer for multi-modal training.

        Args:
            cfg (str | DictConfig, optional): Configuration file path or configuration dictionary.
            overrides (dict, optional): Dictionary to override default training arguments.
            _callbacks (list, optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect"  # Ensure task type is correct
        
        # Pre-initialize multi-modal attributes before parent init
        # This is necessary because parent init calls get_dataset
        model_name = str(overrides.get('model', cfg.get('model', '')))
        self.is_multimodal = '-mm' in Path(model_name).stem
        self.modality = overrides.get('modality', None)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        self.multimodal_config = None
        
        # Initialize parent trainer
        super().__init__(cfg, overrides, _callbacks)
        
        # Update modality from args after parent init
        self.modality = getattr(self.args, 'modality', None)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        
        # Log initialization with modality information
        if self.is_multimodal:
            if self.modality:
                LOGGER.info(f"Initializing RTDETRMMTrainer - Single-modal training mode: {self.modality}-only")
            else:
                LOGGER.info("Initializing RTDETRMMTrainer - Dual-modal training mode (RGB+X)")
        else:
            LOGGER.info("Initializing RTDETRMMTrainer - Standard RT-DETR mode")
    
    def _check_multimodal_model(self):
        """
        Check if the model is configured for multi-modal operation.
        
        Returns:
            bool: True if model name contains '-mm' suffix
        """
        model_name = str(self.args.model)
        return '-mm' in Path(model_name).stem
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build multi-modal dataset using YOLOMultiModalImageDataset.

        This method follows YOLOMM's successful pattern by overriding build_dataset
        instead of get_dataset, ensuring proper initialization timing and data access.

        Args:
            img_path (str): Path to images
            mode (str): Dataset mode ('train', 'val', 'test')
            batch (int, optional): Batch size for rectangle training

        Returns:
            Dataset: YOLOMultiModalImageDataset for multi-modal, standard dataset otherwise
        """
        if not self.is_multimodal:
            raise ValueError(
                "RTDETRMMTrainer is designed for multi-modal RT-DETR models only. "
                "For standard RT-DETR training, use RTDETRTrainer instead. "
                "Multi-modal models should have '-mm' in the model name."
            )

        # Get model stride parameter (consistent with DetectionTrainer)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        # Lazy loading: parse multi-modal configuration on demand
        if not hasattr(self, 'multimodal_config') or self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态配置解析完成 - 模态: {self.multimodal_config['models']}")

        # Use parsed modality configuration
        modalities = self.multimodal_config['models']
        
        # 🔑 关键修复：明确提取X模态信息（与YOLOMM保持一致）
        x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
        x_modality_dir = self.multimodal_config['modalities'][x_modality]

        LOGGER.info(f"构建多模态数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")

        # If single-modal training is enabled, log modality padding info and validate compatibility
        if self.modality:
            self._validate_modality_compatibility()
            LOGGER.info(f"启用单模态训练: {self.modality}-only，将应用智能模态填充")

        # Call build_yolo_dataset with multi_modal_image=True to enable multi-modal dataset
        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=mode == "val",  # ✅ 修复：统一rect参数处理逻辑
            stride=gs,
            multi_modal_image=True,  # Key parameter: enable YOLOMultiModalImageDataset
            x_modality=x_modality,  # ✅ 修复：传递模态类型
            x_modality_dir=x_modality_dir,  # ✅ 修复：传递模态目录
            enable_self_modal_generation=getattr(self.args, 'enable_self_modal_generation', False)  # ✅ 修复：添加自动模态生成支持
        )
    
    
    def get_model(self, cfg: Optional[dict] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Initialize and return RT-DETR model with multi-modal support.
        
        This method creates RTDETRDetectionModel with correct channel configuration,
        following the same pattern as YOLOMM for consistency.
        
        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.
            
        Returns:
            RTDETRDetectionModel: Initialized model ready for multi-modal routing.
        """
        from ultralytics.utils import RANK
        
        # 计算正确的输入通道数 (参考YOLOMM的逻辑)
        if self.is_dual_modal:
            # 双模态训练：从data配置中读取Xch
            x_channels = self.data.get('Xch', 3)
            channels = 3 + x_channels  # RGB(3) + X(Xch)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态RT-DETR模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch总输入")
        else:
            # 单模态训练：始终使用3通道
            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态RT-DETR模型初始化: {self.modality or 'RGB'}(3ch)")
        
        # 🔧 关键修复：更新data配置以修复验证器通道数不匹配问题
        self.data["channels"] = channels
        if verbose and RANK in {-1, 0}:
            LOGGER.info(f"已更新data[\"channels\"]为{channels}，确保验证器兼容性")
        
        # Create RT-DETR model（在构建前将dataset_config注入yaml以在解析期识别x_modality）
        cfg_in = cfg or self.args.model
        cfg_dict = None
        if isinstance(cfg_in, str):
            try:
                from ultralytics.nn.tasks import yaml_model_load
                cfg_dict = yaml_model_load(cfg_in)
            except Exception:
                cfg_dict = None
        elif isinstance(cfg_in, dict):
            from copy import deepcopy
            cfg_dict = deepcopy(cfg_in)

        if cfg_dict is not None:
            cfg_dict['dataset_config'] = dict(self.data)
            model = RTDETRDetectionModel(
                cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1
            )
        else:
            model = RTDETRDetectionModel(
                cfg_in, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1
            )
        
        # 加载权重（如果提供）
        if weights:
            model.load(weights)
        # 同步多模态路由配置（与 YOLO 路径保持一致）
        try:
            if hasattr(model, 'multimodal_router') and model.multimodal_router:
                model.multimodal_router.update_dataset_config(self.data)
        except Exception as e:
            LOGGER.warning(f"RTDETRMMTrainer: update_dataset_config failed: {e}")

        # 统一打印GFLOPs（架构级 + 路由感知），确保训练前即可见
        try:
            imgsz = int(getattr(self.args, 'imgsz', 640))
            arch_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=False)
            if self.modality:
                route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=self.modality, route_aware=True)
                LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[{self.modality}]): {route_gflops:.2f}")
            else:
                route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=True)
                LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[dual]): {route_gflops:.2f}")
        except Exception as e:
            LOGGER.warning(f"GFLOPs 统计失败（可忽略，不影响训练）：{e}")

        return model
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch data for multi-modal inputs.

        Handles both standard (3-channel) and multi-modal (6-channel) inputs,
        with support for single-modal ablation training.
        
        Args:
            batch (dict): Batch data dictionary containing images and labels.
            
        Returns:
            dict: Preprocessed batch ready for training.
        """
        # Standard preprocessing
        batch = super().preprocess_batch(batch)
        
        if not self.is_multimodal:
            return batch
        
        # Multi-modal specific preprocessing
        # 移除本地通道置零的单模态消融，统一交由 MultiModalRouter 处理
        # 若设置了单模态训练参数，这里仅通过路由器的运行时参数进行数据路由与填充
        try:
            if self.modality and hasattr(self, 'model') and self.model is not None:
                # 训练期间，DetectionTrainer 会持有实际的底层模型在 self.model
                if hasattr(self.model, 'mm_router') and self.model.mm_router:
                    self.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, 'ablation_strategy', None),
                        seed=getattr(self.args, 'seed', None),
                    )
        except Exception:
            # 保持训练健壮性：如无路由器则不做本地置零
            pass

        return batch
    
    def get_validator(self):
        """
        Return validator compatible with multi-modal RT-DETR.
        
        Returns:
            RTDETRMMValidator: Multi-modal validator instance.
        """
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"  # RT-DETR specific losses
        
        # Import RTDETRMMValidator from multimodal subpackage
        from ultralytics.models.rtdetr.multimodal.val import RTDETRMMValidator
        
        # 创建验证器并传递更新后的data配置
        validator = RTDETRMMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args)
        )
        
        # 🔧 关键修复：传递trainer的data配置给验证器
        # 确保验证器使用包含更新后channels信息的data配置
        validator.data = self.data
        
        return validator
    
    def _parse_multimodal_config(self):
        """
        Parse and validate multi-modal configuration from data.yaml.
        
        Adapted from YOLOMM's implementation to maintain consistency.
        
        Returns:
            dict: Parsed multi-modal configuration.
        """
        # Single-modal training configuration
        if self.modality:
            if self.modality == 'rgb':
                x_modality = self._determine_x_modality_from_data()
                config = {
                    'models': ['rgb', x_modality],
                    'modalities': {
                        'rgb': 'images',
                        x_modality: f'images_{x_modality}'
                    }
                }
                LOGGER.info(f"RGB single-modal training, X modality: {x_modality}")
            else:
                # 处理 'X' 特殊标记（大小写不敏感）
                if self.modality.upper() == 'X':
                    # 'X' 是特殊标记，需要解析为实际的X模态
                    actual_x_modality = self._determine_x_modality_from_data()
                    # 从data.yaml获取实际的路径映射
                    x_modality_path = self._get_x_modality_path(actual_x_modality)
                    
                    config = {
                        'models': ['rgb', actual_x_modality],
                        'modalities': {
                            'rgb': 'images',
                            actual_x_modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态训练: {actual_x_modality}-only (从'X'解析)")
                else:
                    # 用户指定了具体的模态名称（如 'depth', 'thermal', 'ir' 等）
                    x_modality_path = self._get_x_modality_path(self.modality)
                    
                    config = {
                        'models': ['rgb', self.modality],
                        'modalities': {
                            'rgb': 'images',
                            self.modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态训练: {self.modality}-only")
            return config
        
        # Dual-modal training configuration
        return self._get_default_multimodal_config()
    
    def _get_default_multimodal_config(self):
        """
        Get default multi-modal configuration from data.yaml.

        Returns:
            dict: Default multi-modal configuration.
        """
        # Check if data is available
        data = getattr(self, 'data', None)

        # Priority 1: modality_used field
        if data and 'modality_used' in data:
            modality_used = data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                config = {
                    'models': modality_used,
                    'modalities': {}
                }

                # Get path mappings from modality field
                if 'modality' in data and isinstance(data['modality'], dict):
                    modality_paths = data['modality']
                    for mod in modality_used:
                        config['modalities'][mod] = modality_paths.get(
                            mod, 'images' if mod == 'rgb' else f'images_{mod}'
                        )
                else:
                    # Generate default paths
                    for mod in modality_used:
                        config['modalities'][mod] = 'images' if mod == 'rgb' else f'images_{mod}'

                LOGGER.info(f"Loaded multi-modal config: {modality_used}")
                return config

        # Priority 2: models field (backward compatibility)
        if data and 'models' in data:
            models = data['models']
            if isinstance(models, list) and len(models) >= 2:
                config = {'models': models, 'modalities': {}}
                for modality in models:
                    config['modalities'][modality] = 'images' if modality == 'rgb' else f'images_{modality}'
                return config

        # Determine default configuration
        x_modality = self._determine_x_modality_from_data()
        config = {
            'models': ['rgb', x_modality],
            'modalities': {
                'rgb': 'images',
                x_modality: f'images_{x_modality}'
            }
        }
        LOGGER.info(f"Using default multi-modal config: rgb+{x_modality}")
        return config
    
    def _get_x_modality_path(self, modality_name):
        """
        获取指定模态的实际路径。
        
        优先从data.yaml的modality字段读取，
        如果不存在则使用默认格式 'images_{modality_name}'。
        
        Args:
            modality_name (str): 模态名称（如 'ir', 'depth', 'thermal'）
            
        Returns:
            str: 模态对应的目录路径
        """
        # 优先从data.yaml的modality字段读取
        if self.data and 'modality' in self.data:
            modality_paths = self.data['modality']
            if isinstance(modality_paths, dict) and modality_name in modality_paths:
                return modality_paths[modality_name]
        
        # 向后兼容：检查modalities字段
        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict) and modality_name in modalities:
                return modalities[modality_name]
        
        # 如果没有配置，使用默认格式
        return f'images_{modality_name}'
    
    def _determine_x_modality_from_data(self):
        """
        Intelligently determine X modality type from data configuration.

        Returns:
            str: X modality identifier (e.g., 'depth', 'thermal', 'ir').
        """
        # Safe data access - use getattr to avoid AttributeError during initialization
        data = getattr(self, 'data', None)

        # Check data.yaml for modality information
        if data:
            # Check modality_used
            if 'modality_used' in data:
                modality_used = data['modality_used']
                if isinstance(modality_used, list):
                    x_modalities = [m for m in modality_used if m != 'rgb']
                    if x_modalities:
                        return x_modalities[0]

            # Check models field
            if 'models' in data:
                models = data['models']
                if isinstance(models, list):
                    x_modalities = [m for m in models if m != 'rgb']
                    if x_modalities:
                        return x_modalities[0]
        
        # Use depth as default when undetermined
        LOGGER.warning("Cannot determine X modality type, using default: depth")
        return 'depth'
    
    def _validate_modality_compatibility(self):
        """
        Validate that specified modality is compatible with available data.
        
        Raises:
            ValueError: When modality parameter is incompatible with data configuration.
        """
        if not self.modality:
            return
        
        # Get available modalities
        available_modalities = []
        if self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        
        # Validate modality compatibility
        if available_modalities:
            # 处理 'X' 特殊标记的验证
            if self.modality.upper() == 'X':
                # 'X' 是特殊标记，检查是否有非RGB的X模态
                x_modalities = [m for m in available_modalities if m != 'rgb']
                if x_modalities:
                    LOGGER.info(f"✅ 模态兼容性验证通过: '{self.modality}' 映射到 {x_modalities[0]}")
                else:
                    raise ValueError(
                        f"指定的modality '{self.modality}' 无法映射到有效的X模态。"
                        f"可用模态列表: {available_modalities}，但没有找到非RGB的X模态。"
                    )
            else:
                # 标准模态验证
                if self.modality not in available_modalities:
                    raise ValueError(
                        f"Specified modality '{self.modality}' not in available modalities: {available_modalities}. "
                        f"Please check data configuration or modality parameter."
                    )
                LOGGER.info(f"✅ Modality compatibility validated: {self.modality} in {available_modalities}")
        else:
            # 如果无法获取可用模态，仅给出警告
            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")
    
    def save_model(self):
        """
        Save model with multi-modal configuration information.
        
        Extends parent's save_model to include multi-modal metadata.
        """
        # Call parent's model saving
        super().save_model()
        
        # Save multi-modal configuration to checkpoint
        if self.is_multimodal and hasattr(self, 'multimodal_config'):
            # Update last checkpoint
            ckpt = torch_load(self.last, map_location='cpu')
            ckpt['multimodal_config'] = self.multimodal_config
            ckpt['modality'] = self.modality
            ckpt['is_multimodal'] = True
            torch.save(ckpt, self.last)
            
            # Update best checkpoint if exists
            if self.best.exists():
                ckpt_best = torch_load(self.best, map_location='cpu')
                ckpt_best['multimodal_config'] = self.multimodal_config
                ckpt_best['modality'] = self.modality
                ckpt_best['is_multimodal'] = True
                torch.save(ckpt_best, self.best)
    
    def final_eval(self):
        """
        Perform final evaluation with multi-modal specific logging.
        """
        # Execute standard evaluation
        super().final_eval()
        
        # Log multi-modal specific information
        if self.is_multimodal and hasattr(self, 'multimodal_config') and self.multimodal_config:
            x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
            if self.modality:
                LOGGER.info(f"Final evaluation complete - Single-modal training: {self.modality}-only")
            else:
                LOGGER.info(f"Final evaluation complete - Dual-modal training: RGB+{x_modality}")
        else:
            LOGGER.info("Final evaluation complete - Standard RT-DETR training")
