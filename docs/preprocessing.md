# Preprocessing

Every standing posteroanterior radiograph passes through the same pipeline before
it reaches the network. The stages are implemented in
[`src/multimodal/dataset.py`](../src/multimodal/dataset.py);
[`src/common/preprocessing_demo.py`](../src/common/preprocessing_demo.py) writes each
intermediate stage to disk so the pipeline can be inspected stage by stage
(Online Resource 1, Fig. S1).

## Stages

| # | Stage | Detail |
|---|---|---|
| 1 | Rescale | DICOM `RescaleSlope` / `RescaleIntercept` applied when present |
| 2 | Windowing | `WindowCenter` / `WindowWidth` from the header, falling back to the full dynamic range; `MONOCHROME1` images are inverted |
| 3 | Burned-in text removal | Top-hat detection inside an upper ROI, then Telea inpainting (two passes) |
| 4 | Histogram matching | Matched to one fixed reference radiograph drawn from the training set |
| 5 | CLAHE | `cv2.createCLAHE`, clip limit 2.0, 8×8 tiles |
| 6 | Resize | 512 × 512 |
| 7 | To tensor | Grayscale replicated to 3 channels, ImageNet mean/std normalization |

The whole thoracic cage is retained. The region of interest is deliberately **not**
cropped to the spine, because rib crowding and lung-field geometry outside the
vertebral column carry much of the signal relevant to pulmonary function. No
denoising is applied.

## Burned-in text removal

Scanner-generated annotations are burned into the pixel data of some studies, and
they sit in the same intensity range as bone. Left in place they survive histogram
matching and CLAHE, and give the network a shortcut unrelated to anatomy.

Detection is restricted to the region where such text actually appears — a top band
covering the upper 22% of the image plus the two upper corner boxes — so that bright
anatomy elsewhere is never touched. Within that ROI:

1. A multi-scale top-hat (kernels 31 and 51) enhances small bright structures.
2. A percentile threshold (98.5) selects candidate text pixels.
3. Morphological closing and dilation extend the mask over the anti-aliased fringe.
4. Connected components wider than 60% or taller than 35% of the image are dropped —
   at that size the component is anatomy, not a label.
5. The mask is inpainted (Telea, radius 6).

The sequence runs twice; the second pass removes the grey halo the first leaves behind.

## Histogram matching reference

One radiograph from the **training set** is fixed as the histogram-matching reference
and used for every image in every fold. Choosing it from the training set keeps
validation intensity statistics out of the training pipeline. The reference is
identified in the scripts as `<reference_study>.dcm`; substitute a study from your
own training split.

## Tabular features

Clinical variables are used as-is, without imputation beyond a zero fill for missing
entries. Categorical variables (sex, Lenke type) are one-hot encoded; continuous
variables (Cobb angle, thoracic kyphosis) are z-scored using **training-split
statistics only**, recomputed for each fold. See [`data.md`](data.md) for the schema.
