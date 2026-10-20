# AREF-NET

Source code for AREF-NET, including the customized Ultralytics package, model configurations, and example training, validation, and prediction scripts.

## Contents

- `ultralytics/`: Model implementations, configurations, data processing, and training utilities.
- `trainMMseg.py`, `valMMseg.py`, and `predictMMseg.py`: Example scripts for multimodal instance segmentation.
- `wiki/`: Additional usage documentation.

## Data

See the [dataset download information](../../data/README.md) for the Baidu Netdisk link and extraction code.

## Configuration

Run scripts from this directory so that they use the included `ultralytics` package. Update the dataset, model, checkpoint, and image paths in the example scripts for your environment before running them.

Python caches, pretrained weight files, and video tutorial access files are excluded from this source upload.

## Acknowledgments

This code includes a customized version of Ultralytics. The original package names, copyright notices, and license headers are retained.
