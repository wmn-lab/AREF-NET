# AREF-NET Core Code

This directory contains the core modules of the full model, the model configuration, dataset configurations for three edge representations, and training and evaluation scripts.
Weights, ablation configurations, datasets, experimental results, and the complete Ultralytics framework are not included.

## Modules

| Paper Module | File | Original Implementation |
|---|---|---|
| MDCB | `aref_net/mdcb.py` | `C3k2_DWR` |
| ESRB | `aref_net/esrb.py` | `C3k2_RVB_SE` |
| BCAF | `aref_net/bcaf.py` | `CGAFusion` |

`configs/aref-net.yaml` uses the module names from the paper while preserving the `n` scale and connectivity of the original full model.
Renaming the modules does not change their computations or parameter keys. Attribution and license notices for the original implementations are retained in the code.

## Environment

The original YOLOMM framework, located at `4.27/ultralytics` in the source environment, is required. The standard official Ultralytics package cannot serve as a direct replacement.
`aref_net/runtime.py` registers modules within the current process without modifying the original project.
After moving this directory, set `AREF_FRAMEWORK_ROOT` to the original framework's project directory.

The existing local environment is activated with `conda activate detr` and includes Python 3.10, PyTorch 2.4.1+cu118, timm 0.9.8, einops 0.8.1, PyYAML 6.0.2, and OpenCV 4.11.0.

## Usage

Run the following commands from this directory:

```bash
python train.py --edge canny
python train.py --edge sobel
python train.py --edge log
```

The default training settings follow the saved records: `epochs=300, batch=8, imgsz=640, seed=0, optimizer=auto`.
In the original logs, automatic optimizer selection chose AdamW with an initial learning rate of 0.00125.
The paper currently specifies `batch=4`, which differs from these records; the training settings have not been changed to match that value.
The dataset YAML files reference the existing MDMCS dataset location and precomputed edge images. Update `path` when using a different dataset location.

For evaluation, reference the weights at their original locations without copying them:

```bash
python evaluate.py --edge canny --weights ../HRCDS/train/full_canny/weights/best.pt
python evaluate.py --edge sobel --weights ../HRCDS/train/dwr_rvbse_cgafusion/weights/best.pt
python evaluate.py --edge log --weights ../HRCDS/train/full_log/weights/best.pt
```

Evaluation uses the test split by default, with the original COCO evaluator and `conf=0.05`. Reported results come from evaluation rather than being replaced with values from the paper.
The original records have not yet been reconciled with the paper's 80.7%/80.4% figures. Organizing this code does not establish that those metrics have been reproduced.
