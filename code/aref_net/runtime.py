"""Register paper module names without editing the existing framework files."""

import os
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = CODE_ROOT / "configs" / "aref-net.yaml"


def register_modules():
    framework = Path(os.environ.get("AREF_FRAMEWORK_ROOT", CODE_ROOT.parent)).resolve()
    if not (framework / "ultralytics" / "nn" / "tasks.py").is_file():
        raise FileNotFoundError(
            "Set AREF_FRAMEWORK_ROOT to the original YOLOMM project directory."
        )
    sys.path.insert(0, str(framework))
    import ultralytics
    from ultralytics.nn import tasks
    from .mdcb import MDCB
    from .esrb import ESRB
    from .bcaf import BCAF

    if Path(ultralytics.__file__).resolve().parent != framework / "ultralytics":
        raise RuntimeError("A different Ultralytics package was already imported.")
    if not hasattr(tasks, "C3K2_CLASS"):
        raise RuntimeError("This integration requires the original YOLOMM parser.")
    tasks.MDCB = MDCB
    tasks.ESRB = ESRB
    tasks.BCAF = BCAF
    tasks.C3K2_CLASS = tuple(dict.fromkeys((*tasks.C3K2_CLASS, MDCB, ESRB)))
    # Reuse the parser's dual-input fusion branch in this process only.
    tasks.CGAFusion = BCAF
    return ultralytics.YOLOMM


def build_model(weights=None, verbose=False):
    YOLOMM = register_modules()
    model = YOLOMM(str(MODEL_YAML), task="segment", verbose=verbose)
    if weights is not None:
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        from ultralytics.nn.tasks import attempt_load_one_weight
        original, _ = attempt_load_one_weight(str(weights), fuse=False)
        model.model.load_state_dict(original.float().state_dict(), strict=True)
        model.model.names = original.names
    else:
        model.model.names = {
            0: "corrosion", 1: "crack", 2: "exposed rebar", 3: "spalling",
        }
    return model
