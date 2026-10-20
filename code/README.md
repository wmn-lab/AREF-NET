# AREF-NET Core Code

This directory contains the AREF-NET modules, the full-model YAML, dataset
configuration files, and small training/evaluation entry points. It deliberately
does not include model weights, datasets, ablation settings, experiment logs, or
the full Ultralytics source tree.

## 1. Obtain the required framework

AREF-NET uses the YOLOMM-enabled Ultralytics source used by this project. The
ordinary `pip install ultralytics` package is not a replacement because the
multimodal `YOLOMM` model and paired RGB/edge dataloader are not part of the
standard package.

Obtain that YOLOMM framework separately from the project/source distribution and
make sure the directory contains:

```text
<framework-root>/ultralytics/nn/tasks.py
<framework-root>/ultralytics/models/yolo/multimodal/
```

Point this repository to it with:

```bash
export AREF_FRAMEWORK_ROOT=/path/to/YOLOMM-project
```

On Windows PowerShell, use:

```powershell
$env:AREF_FRAMEWORK_ROOT = "D:\path\to\YOLOMM-project"
```

When this repository is placed next to the original project as `4.27/code`, the
framework is detected automatically. No framework files are copied into this
repository.

## 2. Install Python dependencies

Create an environment with a compatible PyTorch build first, then install the
small dependency list:

```bash
python -m pip install -r ../requirements.txt
```

Install PyTorch separately according to the CUDA or CPU configuration of the
machine. The reference environment used for verification was Python 3.10,
PyTorch 2.4.1+cu118, `timm==0.9.8`, `einops==0.8.1`, `PyYAML==6.0.2`, and
OpenCV 4.11.0.

## 3. Download and prepare the dataset

Download and extract the MDMCS/HRCDS dataset from the link in
[`data/README.md`](../data/README.md). The dataset must provide paired RGB and
edge images plus YOLO-segmentation labels.

Edit the `path` value in the selected YAML file if the extracted dataset is not
at `/home/wmn/code/data/seg/MDMCS/HRCDS`:

```text
configs/data/mdmcs-canny.yaml
configs/data/mdmcs-sobel.yaml
configs/data/mdmcs-log.yaml
```

Check the installation before training:

```bash
python check_setup.py --data configs/data/mdmcs-canny.yaml
```

## 4. Train

Run from this directory:

```bash
python train.py --edge canny
python train.py --edge sobel
python train.py --edge log
```

The defaults are `epochs=300`, `batch=8`, `imgsz=640`, `seed=0`, and
`optimizer=auto`. Override them when needed:

```bash
python train.py --edge canny --epochs 1 --batch 1 --device cpu
```

Training outputs are written below `runs/`, which is ignored by Git.

## 5. Evaluate

Provide a checkpoint produced by training or downloaded separately:

```bash
python evaluate.py --edge canny --weights /path/to/best.pt
```

The default split is `test`, with `conf=0.05`. Use `--split val` for validation
and `--data` to select a different dataset YAML.

## Modules

| Paper module | File | Source implementation |
|---|---|---|
| MDCB | `aref_net/mdcb.py` | `C3k2_DWR` |
| ESRB | `aref_net/esrb.py` | `C3k2_RVB_SE` |
| BCAF | `aref_net/bcaf.py` | `CGAFusion` |

The paper names in `configs/aref-net.yaml` preserve the original full-model
connectivity and computation. The module renaming does not alter the parameter
keys or arithmetic.

## Reproducibility note

The original training records use automatic AdamW selection with an initial
learning rate of 0.00125. The paper text reports a different batch size, so the
entry point keeps the recorded settings and does not silently replace them.
Reported metrics must be generated with the supplied evaluation script; this
repository does not claim that stored paper values have been reproduced.
