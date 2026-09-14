#!/usr/bin/env bash
#
# End-to-end driver for the published multimodal model.
#
# Data must already be in place (see docs/data.md) and the histogram-matching
# reference set in each script (see docs/reproduction.md). Paths are configured
# inside the scripts, not here.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Preprocessing demo (stage-by-stage figure)"
python src/common/preprocessing_demo.py

echo "==> Training: EfficientNet-B0 + MLP, FiLM fusion, 5-fold CV"
python src/multimodal/train.py

echo "==> Evaluation from saved checkpoints"
python src/multimodal/evaluate.py

echo "==> Grad-CAM"
python src/multimodal/gradcam.py

echo "==> Done. See results/ for metrics, ROC curves and Grad-CAM overlays."
