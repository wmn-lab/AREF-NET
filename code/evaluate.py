"""Evaluate an existing full-model checkpoint with the original COCO evaluator."""

import argparse
import json
from pathlib import Path

from aref_net.runtime import CODE_ROOT, build_model


def encode_mask(mask):
    """Uncompressed COCO RLE without a Python object for every image pixel."""
    import numpy as np

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}")
    flat = (mask.astype(np.uint8) > 0).reshape(-1, order="F")
    if not flat.size:
        counts = [0]
    else:
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        counts = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
        if flat[0]:
            counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--edge", choices=("canny", "sobel", "log"), default="canny")
    parser.add_argument("--data", help="Override the selected dataset YAML.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(CODE_ROOT / "runs" / "evaluate"))
    args = parser.parse_args()
    data = args.data or str(CODE_ROOT / "configs" / "data" / f"mdmcs-{args.edge}.yaml")
    model = build_model(weights=args.weights)
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.multimodal.segment import cocoval
    cocoval.rle_encode = encode_mask

    class SegCOCOValidator(cocoval.MultiModalSegmentationCOCOValidator):
        def __setattr__(self, name, value):
            if name == "args" and isinstance(value, dict):
                value = get_cfg(overrides=value)
            super().__setattr__(name, value)

    save_dir = Path(args.project) / f"aref_{args.edge}_{args.split}"
    validator = SegCOCOValidator(save_dir=save_dir)
    results = model.cocoval(
        validator=validator, data=data, split=args.split, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, conf=args.conf, device=args.device,
        save_json=False, plots=False, project=args.project,
        name=f"aref_{args.edge}_{args.split}",
    )
    output = Path(validator.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8",
    )
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
