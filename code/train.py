"""Train the full AREF-Net using the existing paired RGB/edge dataset."""

import argparse

from aref_net.runtime import CODE_ROOT, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", choices=("canny", "sobel", "log"), default="canny")
    parser.add_argument("--data", help="Override the selected dataset YAML.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(CODE_ROOT / "runs" / "train"))
    args = parser.parse_args()
    data = args.data or str(CODE_ROOT / "configs" / "data" / f"mdmcs-{args.edge}.yaml")
    model = build_model()
    model.train(
        data=data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, cache=True, seed=0, deterministic=True,
        optimizer="auto", project=args.project, name=f"aref_{args.edge}",
    )


if __name__ == "__main__":
    main()
