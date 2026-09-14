import os
import json
import argparse
import numpy as np
from tqdm import tqdm

# =========================
# Test function
# =========================
def test_fold(fold, ckpt_path, base_dataset, train_idx, test_idx,
              device, backbone, fusion, tabular_type, val_transform):

    # test dataset
    test_dataset = AISTabularXRDataset(
        image_dir=base_dataset.image_dir,
        tabular_path=base_dataset.tabular_path,
        label_path=base_dataset.label_path,
        transform=val_transform,
        reference_dcm_path=base_dataset.reference_dcm_path,
        verbose=False,
    )
    test_dataset.df = base_dataset.df.iloc[test_idx].reset_index(drop=True)
    test_dataset.feature_cols = base_dataset.feature_cols

    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=4, collate_fn=custom_collate, pin_memory=True
    )

    # Normalize the tabular features with the training-split statistics
    feats = base_dataset.feature_cols
    tr_df = base_dataset.df.iloc[train_idx].reset_index(drop=True)
    tab_means = torch.tensor(tr_df[feats].mean().values, dtype=torch.float32, device=device)
    tab_stds  = torch.tensor(tr_df[feats].std().values,  dtype=torch.float32, device=device).clamp(min=1e-2)

    # model
    model = MultiModalMTLModel(
        num_tabular_features=len(feats),
        backbone_name=backbone,
        fusion_type=fusion,
        tabular_type=tabular_type,
        pretrained=False,
        in_chans=3,
        num_tasks=2,
    )

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model).to(device)
    model.eval()

    y_true_fvc, y_prob_fvc = [], []
    y_true_fev, y_prob_fev = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold{fold} Test", leave=False):
            imgs, tabs, labels, _ = batch
            if imgs is None:
                continue

            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)

            tabs_norm = (tabs - tab_means) / tab_stds   # same statistics as training

            y_fvc  = labels['FVC_class'].float().to(device)
            y_fev1 = labels['FEV1_class'].float().to(device)

            logits = model(imgs, tabs_norm)
            probs = torch.sigmoid(logits)

            y_true_fvc.extend(y_fvc.cpu().tolist())
            y_prob_fvc.extend(probs[:, 0].cpu().tolist())
            y_true_fev.extend(y_fev1.cpu().tolist())
            y_prob_fev.extend(probs[:, 1].cpu().tolist())

    # --- Metrics @ 0.5 ---
    fvc_acc, fvc_sens, fvc_spec, fvc_auc = binary_metrics(y_true_fvc, y_prob_fvc, thr=0.5)
    fev_acc, fev_sens, fev_spec, fev_auc = binary_metrics(y_true_fev, y_prob_fev, thr=0.5)
    macro_acc = np.mean([fvc_acc, fev_acc])
    macro_sens = np.mean([fvc_sens, fev_sens])
    macro_spec = np.mean([fvc_spec, fev_spec])
    macro_auc = np.mean([fvc_auc, fev_auc])

    # --- Metrics @ tuned threshold ---
    t_fvc = best_thresh_youden(y_true_fvc, y_prob_fvc, fallback=0.5)
    t_fev = best_thresh_youden(y_true_fev, y_prob_fev, fallback=0.5)
    fvc_acc_T, fvc_sens_T, fvc_spec_T, _ = binary_metrics(y_true_fvc, y_prob_fvc, thr=t_fvc)
    fev_acc_T, fev_sens_T, fev_spec_T, _ = binary_metrics(y_true_fev, y_prob_fev, thr=t_fev)
    macro_acc_T = np.mean([fvc_acc_T, fev_acc_T])
    macro_sens_T = np.mean([fvc_sens_T, fev_sens_T])
    macro_spec_T = np.mean([fvc_spec_T, fev_spec_T])

    result = {
        "fold": fold,
        "checkpoint": ckpt_path,
        "fvc_acc": fvc_acc, "fvc_sens": fvc_sens, "fvc_spec": fvc_spec, "fvc_auc": fvc_auc,
        "fev_acc": fev_acc, "fev_sens": fev_sens, "fev_spec": fev_spec, "fev_auc": fev_auc,
        "macro_acc": macro_acc, "macro_sens": macro_sens, "macro_spec": macro_spec, "macro_auc": macro_auc,
        "t_fvc": t_fvc, "t_fev": t_fev,
        "fvc_acc_tuned": fvc_acc_T, "fvc_sens_tuned": fvc_sens_T, "fvc_spec_tuned": fvc_spec_T,
        "fev_acc_tuned": fev_acc_T, "fev_sens_tuned": fev_sens_T, "fev_spec_tuned": fev_spec_T,
        "macro_acc_tuned": macro_acc_T, "macro_sens_tuned": macro_sens_T, "macro_spec_tuned": macro_spec_T,
    }
    return result


def print_summary_table(all_results):
    metric_cols = [
        ("macro_acc", "Macro Acc"),
        ("macro_sens", "Macro Sens"),
        ("macro_spec", "Macro Spec"),
        ("macro_auc", "Macro AUC"),
        ("fvc_acc", "FVC Acc"),
        ("fvc_auc", "FVC AUC"),
        ("fev_acc", "FEV1 Acc"),
        ("fev_auc", "FEV1 AUC"),
    ]

    print("\n=== Paper-style Summary (@ threshold 0.5) ===")
    print(
        f"{'Fold':<10}"
        f"{'Macro Acc':>10}{'Macro Sens':>12}{'Macro Spec':>12}{'Macro AUC':>12}"
        f"{'FVC Acc':>10}{'FVC AUC':>10}"
        f"{'FEV1 Acc':>11}{'FEV1 AUC':>11}"
    )
    for result in all_results:
        print(
            f"Fold {int(result['fold']):<5}"
            f"{result['macro_acc']:>10.3f}{result['macro_sens']:>12.3f}"
            f"{result['macro_spec']:>12.3f}{result['macro_auc']:>12.3f}"
            f"{result['fvc_acc']:>10.3f}{result['fvc_auc']:>10.3f}"
            f"{result['fev_acc']:>11.3f}{result['fev_auc']:>11.3f}"
        )

    summary = {}
    for key, _ in metric_cols:
        values = np.array([result[key] for result in all_results], dtype=float)
        summary[key] = (float(np.mean(values)), float(np.std(values, ddof=1)))

    def mean_sd(key):
        return f"{summary[key][0]:.3f}+-{summary[key][1]:.3f}"

    print(
        f"{'Mean+-SD':<10}"
        f"{mean_sd('macro_acc'):>13}{mean_sd('macro_sens'):>13}"
        f"{mean_sd('macro_spec'):>13}{mean_sd('macro_auc'):>13}"
        f"{mean_sd('fvc_acc'):>13}{mean_sd('fvc_auc'):>13}"
        f"{mean_sd('fev_acc'):>13}{mean_sd('fev_auc'):>13}"
    )


def load_prev_results(jsonl_path):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "fold" in row:
                rows.append(row)
    return sorted(rows, key=lambda row: int(row["fold"]))


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-prev",
        default=None,
        help="Print the previously saved fold metrics JSONL without re-running inference.",
    )
    args = parser.parse_args()

    if args.from_prev:
        all_results = load_prev_results(args.from_prev)
        print(f"[Restored] Loaded saved results from {args.from_prev}")
        print_summary_table(all_results)
        raise SystemExit(0)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from dataset_sampling import AISTabularXRDataset, custom_collate
    from model_20250826 import MultiModalMTLModel
    from train_20250826 import binary_metrics, best_thresh_youden, build_transforms

    DATA_DIR = "data"
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    TABULAR_PATH = os.path.join(DATA_DIR, "ais_tabular_sampling.csv")
    LABEL_PATH = os.path.join(DATA_DIR, "ais_labels.csv")
    REFERENCE_DCM_PATH = os.path.join(IMAGE_DIR, "<reference_study>.dcm")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_transform = build_transforms(train=False)

    # Dataset, indexed over the full cohort
    base_dataset = AISTabularXRDataset(
        image_dir=IMAGE_DIR,
        tabular_path=TABULAR_PATH,
        label_path=LABEL_PATH,
        transform=val_transform,
        reference_dcm_path=REFERENCE_DCM_PATH,
        verbose=True,
    )
    
    base_dataset.tabular_path = TABULAR_PATH
    base_dataset.label_path = LABEL_PATH
    base_dataset.reference_dcm_path = REFERENCE_DCM_PATH

    # Test split. For an exact reproduction, load the split indices that train.py
    # used rather than recomputing them here.
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    y_multi = base_dataset.df[['FVC_class','FEV1_class']].values
    skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Checkpoint directory
    ckpt_dir = "checkpoints/efficientnet_b0_film_mlp"

    all_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(base_dataset.df, y_multi), start=1):
        ckpt_path = os.path.join(ckpt_dir, f"fold{fold}.pt")
        result = test_fold(
            fold, ckpt_path, base_dataset,
            train_idx=train_idx,
            test_idx=test_idx,
            device=DEVICE,
            backbone="efficientnet_b0", fusion="film", tabular_type="mlp",
            val_transform=val_transform
        )

        all_results.append(result)
        print(f"\n[Fold{fold} Test Results]")
        for k,v in result.items():
            if k not in ["checkpoint", "fold", "t_fvc", "t_fev"]:
                print(f"  {k}: {v:.4f}")

    print("\n=== All folds done ===")
    print_summary_table(all_results)
