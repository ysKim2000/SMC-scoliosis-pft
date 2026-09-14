# Expected data layout and schemas

No patient data is included in this repository. The scripts expect the following
layout relative to the repository root:

```
data/
├── images/                      one DICOM per patient (standing PA radiograph)
│   ├── <patient_id>_....dcm
│   └── ...
├── ais_tabular_sampling.csv     encoded clinical variables (model input)
├── ais_tabular.csv              raw clinical variables (before encoding)
└── ais_labels.csv               binary PFT outcomes
```

## Patient identifier

`Patient_ID` is the join key across all three tables and is matched to the DICOM
files by the **first 8 characters of the filename**, zero-padded. A file named
`12345678_151208_5_.Seq1.Ser1.Img1.dcm` is therefore matched to `Patient_ID`
`12345678`. Patients without a matching DICOM, or without both labels, are dropped
at dataset construction and reported in `failed_patient_ids`.

## `ais_labels.csv`

| Column | Type | Description |
|---|---|---|
| `Patient_ID` | str | 8-digit zero-padded identifier |
| `FVC_class` | 0/1 | 1 if FVC < 80% of predicted |
| `FEV1_class` | 0/1 | 1 if FEV1 < 80% of predicted |

Predicted values follow the GLI-2012 reference equations for the northeast Asian
population, adjusted for age, sex and height.

## `ais_tabular.csv` — raw clinical variables

| Column | Type | Description |
|---|---|---|
| `Patient_ID` | str | identifier |
| `sex` | categorical | patient sex |
| `lenke` | 1–6 | Lenke curve type |
| `lenke2` | categorical | Lenke lumbar spine modifier |
| `cobb` | float | major Cobb angle, degrees |
| `thoracic_cobb` | float | thoracic Cobb angle, degrees |
| `t1_12_kyphosis` | float | T1–T12 kyphosis, degrees |
| `t4_12_kyphosis` | float | T4–T12 kyphosis, degrees |

## `ais_tabular_sampling.csv` — encoded model input

This is the table the training scripts actually read. Categorical variables are
one-hot encoded and one derived feature is added:

| Column | Description |
|---|---|
| `Patient_ID` | identifier |
| `sex_0`, `sex_1` | one-hot sex |
| `lenke_1` … `lenke_6` | one-hot Lenke curve type |
| `lenke2_0` … `lenke2_2` | one-hot Lenke modifier |
| `cobb` | major Cobb angle, degrees |
| `t1_12_kyphosis`, `t4_12_kyphosis` | kyphosis angles, degrees |
| `kyphosis_delta` | `t1_12_kyphosis − t4_12_kyphosis` |

Every column other than `Patient_ID`, `FVC_class` and `FEV1_class` is treated as a
model feature — the feature list is derived from the table, so adding a column adds
an input. Continuous features are z-scored per fold using training-split statistics
only; the same statistics are reapplied at evaluation time.

## Cohort

178 consecutive patients with adolescent idiopathic scoliosis, aged 10–18, who
underwent both standing PA radiography and pulmonary function testing as part of a
preoperative evaluation at a single institution.

| Outcome | Normal (≥ 80%) | Abnormal (< 80%) |
|---|---|---|
| FVC | 104 | 74 |
| FEV1 | 60 | 118 |

Both tasks are class-imbalanced, which is handled with a per-task `pos_weight` in the
loss rather than by resampling.
