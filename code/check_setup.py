"""Check the external framework, Python dependencies, and dataset paths."""

import argparse
import os
from pathlib import Path

import yaml

from aref_net.runtime import CODE_ROOT, register_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", help="Directory containing ultralytics/nn/tasks.py.")
    parser.add_argument("--data", help="Dataset YAML to validate.")
    args = parser.parse_args()
    if args.framework_root:
        os.environ["AREF_FRAMEWORK_ROOT"] = str(Path(args.framework_root).expanduser())

    register_modules()
    data_file = Path(args.data or CODE_ROOT / "configs" / "data" / "mdmcs-canny.yaml")
    cfg = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    dataset_root = Path(os.path.expandvars(os.path.expanduser(str(cfg["path"]))))
    if not dataset_root.is_absolute():
        dataset_root = (data_file.parent / dataset_root).resolve()
    missing = []
    for key in ("train", "val", "test"):
        split = Path(str(cfg[key]))
        split = split if split.is_absolute() else dataset_root / split
        if not split.exists():
            missing.append(f"{key}: {split}")
    print("YOLOMM framework: OK")
    print(f"Dataset root: {dataset_root}")
    if missing:
        print("Dataset paths missing:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(2)
    print("Dataset paths: OK")


if __name__ == "__main__":
    main()
