# AREF-NET

Official repository for AREF-NET for concrete defect instance segmentation.

This repository contains the AREF-NET code and a small patch for the YOLOMM
multimodal framework. It does not include the complete Ultralytics source,
datasets, checkpoints, or generated training outputs.

## Quick start

The commands below assume Linux or WSL. The same Python commands work on
Windows PowerShell after replacing paths with Windows paths.

### 1. Clone AREF-NET

```bash
git clone https://github.com/wmn-lab/AREF-NET.git
cd AREF-NET
```

### 2. Download the base Ultralytics source

Download the pinned upstream version used as the patch base:

```bash
git clone --branch v8.3.163 --depth 1 \
  https://github.com/ultralytics/ultralytics.git ultralytics-8.3.163
```

The ordinary `pip install ultralytics` package is not sufficient by itself:
AREF-NET needs the YOLOMM multimodal model and paired RGB/edge dataloader.

Apply the AREF-NET patch:

```bash
python framework/apply_patch.py \
  --framework-root "$PWD/ultralytics-8.3.163"
```

Set the framework location for the current terminal:

```bash
export AREF_FRAMEWORK_ROOT="$PWD/ultralytics-8.3.163"
```

For Windows PowerShell:

```powershell
$env:AREF_FRAMEWORK_ROOT = "$PWD\ultralytics-8.3.163"
```

The patch only copies the custom files supplied in `framework/patch/`; it does
not replace the complete upstream checkout.

### 3. Create the Python environment

Create an environment with Python 3.10 or newer. Install PyTorch separately
using the command appropriate for the machine's CUDA or CPU configuration.
Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The reference environment used for the project included Python 3.10,
PyTorch 2.4.1+cu118, `timm==0.9.8`, `einops==0.8.1`, `PyYAML==6.0.2`, and
OpenCV 4.11.0.

### 4. Download and prepare the dataset

Download the MDMCS/HRCDS dataset using the link and extraction instructions in
[`data/README.md`](data/README.md). The extracted dataset must contain paired
RGB and edge images and YOLO-segmentation labels.

Edit the `path` field in the selected file if the dataset is stored elsewhere:

```text
code/configs/data/mdmcs-canny.yaml
code/configs/data/mdmcs-sobel.yaml
code/configs/data/mdmcs-log.yaml
```

### 5. Check the installation

```bash
cd code
python check_setup.py --data configs/data/mdmcs-canny.yaml
```

The check confirms that the patched YOLOMM framework can be found and that the
dataset split directories exist.

### 6. Train

```bash
python train.py --edge canny
python train.py --edge sobel
python train.py --edge log
```

The default settings are 300 epochs, batch size 8, image size 640,
`optimizer=auto`, and device `0`. For a short local trial:

```bash
python train.py --edge canny --epochs 1 --batch 1 --device cpu
```

Training outputs are written to `code/runs/`, which is ignored by Git.

### 7. Evaluate

Use a checkpoint produced by training or obtained separately:

```bash
python evaluate.py --edge canny --weights /path/to/best.pt
```

The default split is `test` and the default confidence threshold is `0.05`.
Use `--split val` for validation or `--data` to select another dataset YAML.

## Repository structure

- [`code/`](code/): AREF-NET modules, model YAML, data YAML, training, and evaluation.
- [`framework/`](framework/): Minimal YOLOMM patch and patch-application script.
- [`data/`](data/): Dataset download and extraction information.
- [`requirements.txt`](requirements.txt): Python dependencies other than PyTorch.

## License and attribution

The Ultralytics base source and the YOLOMM patch retain their applicable
upstream licensing and attribution. See [`framework/NOTICE.md`](framework/NOTICE.md)
before redistributing the patched framework.
