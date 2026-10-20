"""Overlay the AREF-NET YOLOMM patch onto an Ultralytics checkout."""

import argparse
import shutil
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parent / "patch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", required=True)
    args = parser.parse_args()

    framework = Path(args.framework_root).expanduser().resolve()
    target = framework / "ultralytics"
    if not (target / "nn" / "tasks.py").is_file():
        raise SystemExit(f"Not an Ultralytics checkout: {target}")
    for source in PATCH_ROOT.rglob("*"):
        if source.is_file():
            destination = framework / source.relative_to(PATCH_ROOT.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print(f"Applied AREF-NET YOLOMM patch to {framework}")


if __name__ == "__main__":
    main()
