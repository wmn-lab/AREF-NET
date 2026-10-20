# YOLOMM framework patch

This directory contains only the custom Ultralytics files needed by the AREF-NET
multimodal training and evaluation entry points. It is not a complete copy of
Ultralytics.

## Apply the patch

1. Download the upstream Ultralytics source at the pinned `v8.3.163` tag:

   ```bash
   git clone --branch v8.3.163 --depth 1 \
     https://github.com/ultralytics/ultralytics.git ultralytics-8.3.163
   ```

2. Apply this patch over that checkout:

   ```bash
   python framework/apply_patch.py --framework-root /path/to/ultralytics-8.3.163
   ```

3. Point AREF-NET to the patched checkout:

   ```bash
   export AREF_FRAMEWORK_ROOT=/path/to/ultralytics-8.3.163
   ```

The patch files are copied over matching upstream paths. The repository does not
include datasets, checkpoints, cache files, or generated experiment outputs.

The patched files originate from the YOLOMM-enabled Ultralytics source used for
the AREF-NET experiments. Keep the upstream license and attribution files with
the checkout when redistributing or modifying it.
