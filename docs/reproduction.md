# Reproduction notes

## Before running

1. Place the data as described in [`data.md`](data.md).
2. Set the reference radiograph. Each entry-point script defines
   `REFERENCE_DCM_PATH` as `data/images/<reference_study>.dcm`; replace the
   placeholder with a study drawn from your **training** split. Histogram matching
   is applied against this single image throughout.
3. Paths are set in the `if __name__ == '__main__'` block at the bottom of each
   script. There is no command-line interface.

## Running

```bash
python src/multimodal/train.py             # 5-fold CV, writes checkpoints + Grad-CAM
python src/multimodal/evaluate.py          # per-fold metrics from saved checkpoints
python src/multimodal/gradcam.py           # standalone Grad-CAM from checkpoints
python src/multimodal/train_single_modal.py  # image-only ablation
python src/common/preprocessing_demo.py    # stage-by-stage preprocessing figure
```

`train.py` writes:

- `checkpoints/<backbone>_<fusion>_<tabular>/fold{1..5}.pt`
- `results/efficientnet_b0_film_mlp/fold_results.jsonl` — per-fold metrics, one JSON
  object per line, followed by a summary object carrying `fold_avg`
- `results/efficientnet_b0_film_mlp/` — combined ROC curves
- `results/efficientnet_b0_film_mlp/gradcam_images/fold{1..5}/`

## Selecting a configuration

`train.py` sweeps the lists near the bottom of its main block. The published model is
the default:

```python
backbones = ['efficientnet_b0']
fusions   = ['film']
tabs      = ['mlp']
```

To reproduce the Table 4 ablation, widen them — every combination is trained with the
same folds and schedule:

```python
backbones = ['vgg19_bn', 'resnet50', 'densenet121', 'efficientnet_b0', 'convnext_base']
fusions   = ['concat', 'general_attention', 'film']
tabs      = ['mlp', 'cnn']
```

The image-only row of Table 4 comes from `train_single_modal.py` with
`USE_IMAGE_ONLY = True`.

## What is and is not deterministic

The fold split is fixed: `MultilabelStratifiedKFold(n_splits=5, shuffle=True,
random_state=42)` over the joint `(FVC_class, FEV1_class)` distribution, so the same
cohort table always yields the same folds.

Training is **not** seeded. Weight initialization of the heads, augmentation sampling
and MixUp/CutMix draws vary between runs, so per-fold metrics move by roughly the
standard deviations reported in Table 3. Reported figures are means across the five
folds of a single run.

## Cross-validation, not a held-out test set

There is no separate test split. All reported metrics are validation metrics
aggregated over the five folds, and model selection (lowest validation loss,
early stopping patience 10) happens on the same fold being scored. This is the
protocol described in the paper for a 178-patient cohort; it is optimistic relative
to a held-out test set, and external validation was not feasible — see the paper's
Limitations.

`evaluate.py` recomputes fold metrics from saved checkpoints. Its `__main__` block
rebuilds the split rather than loading the indices `train.py` used; for an exact
reproduction, persist the fold indices during training and load them here.

## Thresholds

Metrics are reported at a fixed 0.5 threshold. The `*_tuned` fields in the results
JSON use the threshold maximizing Youden's J **on the same fold**, so they are
optimistically biased and are not the headline numbers.

## Hardware

Trained on NVIDIA RTX A6000 GPUs. `train.py` sets `CUDA_VISIBLE_DEVICES = "0,1"` and
wraps the model in `nn.DataParallel`; adjust both for a different machine. Batch size
16 at 512 × 512 fits comfortably on a single 48 GB card for EfficientNet-B0, but the
ConvNeXt and VGG ablations are heavier.
