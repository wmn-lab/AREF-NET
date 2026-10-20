# Ultralytics YOLO, AGPL-3.0 license

"""
Multi-Modal Pose Predictor.

Provides MultiModalPosePredictor for RGB+X pose estimation inference.
"""

from ultralytics.models.yolo.multimodal.mm_predictor import YOLOMMPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class MultiModalPosePredictor(YOLOMMPredictor):
    """
    Multi-modal pose predictor with keypoint detection support.

    Extends YOLOMMPredictor to handle pose keypoints alongside bounding boxes.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize MultiModalPosePredictor for pose estimation tasks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"

        # MPS device warning for pose models
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct result with keypoints from prediction.

        Args:
            pred: Predicted boxes, scores, and keypoints.
            img: Processed input image tensor.
            orig_img: Original unprocessed image.
            img_path: Path to original image.

        Returns:
            Results object with bounding boxes and keypoints.
        """
        result = super().construct_result(pred, img, orig_img, img_path)

        # Extract and scale keypoints
        pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape)
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)

        return result
