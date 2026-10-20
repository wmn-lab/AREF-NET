# AREF-NET

Official repository for AREF-NET for concrete defect instance segmentation.

## Code Availability

The core source code is available in [code](code/). The repository intentionally
keeps the download small: the YOLOMM-enabled Ultralytics framework, PyTorch, the
dataset, and model weights are obtained separately. Follow the step-by-step
instructions in the [code README](code/README.md).

## Repository Structure

- [code](code/): AREF-NET core modules, model configurations, and training/evaluation scripts.
- [data](data/): Dataset download link and extraction code.
- `requirements.txt`: Small Python dependency list; install PyTorch separately for the target CUDA/CPU platform.
