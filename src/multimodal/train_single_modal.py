import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm
from torchvision import transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt

from dataset import AISTabularXRDataset, custom_collate
from model import MultiModalMTLModel, SingleModalMTLModel

# =========================
# Configuration Switch
# =========================
USE_IMAGE_ONLY = False  # set True to train the image-only ablation

# =========================
# Aug / Utils
# =========================
class GrayTo3CH(object):
    def __call__(self, x):
        if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x

def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=7, translate=(0.03, 0.03),
                    scale=(0.97, 1.03), shear=(-3, 3, -3, 3)
                )
            ], p=0.7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.05)], p=0.25),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.25),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.25),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.05, brightness=0.05)], p=0.25),
            transforms.ToTensor(),
            GrayTo3CH(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            GrayTo3CH(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# =========================
# Metrics helpers
# =========================
def binary_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob > float(thr)).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, sens, spec, roc_auc

def best_thresh_youden(y_true, y_prob, fallback=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        if thr is None or len(thr) == 0:
            return float(fallback)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(thr[idx])
    except Exception:
        return float(fallback)

# =========================
# Fold training
# =========================
def train_fold(
    fold,
    train_idx, val_idx,
    base_dataset,
    device,
    epochs,
    patience,
    batch_size,
    lr,
    backbone_name,
    fusion_type,    # ignored in image-only mode
    tabular_type,   # ignored in image-only mode
    train_transform,
    val_transform,
):
    # Checkpoint Directory Setting
    CKPT_ROOT = 'checkpoints'
    
    if USE_IMAGE_ONLY:
        # Separate output directory for the image-only runs
        run_name = f"{backbone_name}_ImageOnly"
    else:
        run_name = f"{backbone_name}_{fusion_type}_{tabular_type}"
        
    ckpt_dir = os.path.join(CKPT_ROOT, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dataset Init
    train_dataset = AISTabularXRDataset(
        image_dir=base_dataset.image_dir,
        tabular_path=base_dataset.tabular_path,
        label_path=base_dataset.label_path,
        transform=train_transform,
        reference_dcm_path=base_dataset.reference_dcm_path,
        verbose=False,
    )
    val_dataset = AISTabularXRDataset(
        image_dir=base_dataset.image_dir,
        tabular_path=base_dataset.tabular_path,
        label_path=base_dataset.label_path,
        transform=val_transform,
        reference_dcm_path=base_dataset.reference_dcm_path,
        verbose=False,
    )
    
    train_dataset.df = base_dataset.df.iloc[train_idx].reset_index(drop=True)
    val_dataset.df   = base_dataset.df.iloc[val_idx].reset_index(drop=True)
    train_dataset.feature_cols = base_dataset.feature_cols
    val_dataset.feature_cols   = base_dataset.feature_cols

    # Tabular statistics. Computed even in image-only mode because the dataset
    # still returns the features; they are simply not passed to the model.
    feats  = train_dataset.feature_cols
    tr_df  = train_dataset.df
    # pos_weight calc
    def pw(name):
        n_pos = int((tr_df[name] == 1).sum())
        n_neg = int((tr_df[name] == 0).sum())
        return float(n_neg / max(n_pos, 1))
    
    pos_weights = torch.tensor([pw('FVC_class'), pw('FEV1_class')], dtype=torch.float32, device=device)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate, pin_memory=True)

    # Model Initialization Switch
    if USE_IMAGE_ONLY:
        print(f">>> [Fold {fold}] Initializing SingleModalMTLModel (Image Only)...")
        model = SingleModalMTLModel(
            backbone_name=backbone_name,
            pretrained=True,
            num_tasks=2,
            dropout=0.3
        )
    else:
        # Multimodal model
        model = MultiModalMTLModel(
            num_tabular_features=len(feats),
            backbone_name=backbone_name,
            fusion_type=fusion_type,
            tabular_type=tabular_type,
            pretrained=True,
            in_chans=3,
            num_tasks=2,
        )
        
    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='mean')

    best_loss = float('inf')
    wait = 0
    best_state = None
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        y_true_fvc_tr, y_prob_fvc_tr = [], []
        y_true_fev_tr, y_prob_fev_tr = [], []

        for batch in tqdm(train_loader, desc=f"Fold{fold} Train", leave=False):
            imgs, tabs, labels, _ = batch
            if imgs is None: continue

            imgs = imgs.to(device, non_blocking=True)
            y_fvc  = labels['FVC_class'].float().to(device)
            y_fev1 = labels['FEV1_class'].float().to(device)
            y = torch.stack([y_fvc, y_fev1], dim=1)

            # Forward pass
            if USE_IMAGE_ONLY:
                logits = model(imgs)  # image only, no tabular input
            else:
                tabs = tabs.to(device, non_blocking=True)
                logits = model(imgs, tabs)

            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            probs = torch.sigmoid(logits)

            y_true_fvc_tr.extend(y_fvc.long().cpu().tolist())
            y_prob_fvc_tr.extend(probs[:, 0].detach().cpu().tolist())
            y_true_fev_tr.extend(y_fev1.long().cpu().tolist())
            y_prob_fev_tr.extend(probs[:, 1].detach().cpu().tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        
        # --- Validation ---
        model.eval()
        val_losses = []
        y_true_fvc_val, y_prob_fvc_val = [], []
        y_true_fev_val, y_prob_fev_val = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold{fold} Val", leave=False):
                imgs, tabs, labels, _ = batch
                if imgs is None: continue
                imgs = imgs.to(device, non_blocking=True)
                y_fvc  = labels['FVC_class'].float().to(device)
                y_fev1 = labels['FEV1_class'].float().to(device)
                y = torch.stack([y_fvc, y_fev1], dim=1)

                if USE_IMAGE_ONLY:
                    logits = model(imgs)
                else:
                    tabs = tabs.to(device, non_blocking=True)
                    logits = model(imgs, tabs)
                    
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)

                y_true_fvc_val.extend(y_fvc.long().cpu().tolist())
                y_prob_fvc_val.extend(probs[:, 0].cpu().tolist())
                y_true_fev_val.extend(y_fev1.long().cpu().tolist())
                y_prob_fev_val.extend(probs[:, 1].cpu().tolist())

        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        # --- Metrics ---
        va_fvc_acc, va_fvc_sens, va_fvc_spec, va_fvc_auc = binary_metrics(y_true_fvc_val, y_prob_fvc_val, thr=0.5)
        va_fev_acc, va_fev_sens, va_fev_spec, va_fev_auc = binary_metrics(y_true_fev_val, y_prob_fev_val, thr=0.5)
        va_acc  = np.nanmean([va_fvc_acc, va_fev_acc])
        va_auc  = np.nanmean([va_fvc_auc, va_fev_auc])

        print(f"[Fold{fold}] Ep {epoch} | ValLoss={val_loss:.4f} | FVC Acc={va_fvc_acc:.4f} AUC={va_fvc_auc:.4f} | FEV1 Acc={va_fev_acc:.4f} AUC={va_fev_auc:.4f} | Macro Acc={va_acc:.4f} AUC={va_auc:.4f}")


        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = model.module.state_dict()
            best_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'fvc_auc': va_fvc_auc, 'fvc_accuracy':va_fvc_acc, 'fvc_sensitivity': va_fvc_sens, 'fvc_specificity': va_fvc_spec,
                'fev_auc': va_fev_auc, 'fev_accuracy':va_fev_acc, 'fev_sensitivity': va_fev_sens, 'fev_specificity': va_fev_spec,
                'macro_auc': va_auc, 'macro_acc': va_acc,
                'y_true_fvc': y_true_fvc_val, 'y_prob_fvc': y_prob_fvc_val,
                'y_true_fev': y_true_fev_val, 'y_prob_fev': y_prob_fev_val,
            }
        else:
            wait += 1
            if wait >= patience:
                print("  Early stopping")
                break
        
        scheduler.step(epoch)

    # Save
    ckpt_path = os.path.join(ckpt_dir, f'fold{fold}.pt')
    torch.save(best_state, ckpt_path)
    best_metrics['checkpoint'] = ckpt_path
    return best_metrics

def save_combined_roc(fold_results, save_dir, backbone, run_name):
    # ROC plotting; identical to train.py except for the output filename
    y_true_fvc_all, y_prob_fvc_all = [], []
    y_true_fev_all, y_prob_fev_all = [], []

    for fr in fold_results:
        y_true_fvc_all.extend(fr["y_true_fvc"])
        y_prob_fvc_all.extend(fr["y_prob_fvc"])
        y_true_fev_all.extend(fr["y_true_fev"])
        y_prob_fev_all.extend(fr["y_prob_fev"])

    roc_dir = os.path.join(save_dir, "20250921_roc_curves")
    os.makedirs(roc_dir, exist_ok=True)
    
    # Simple Combined ROC Plot
    plt.figure()
    if len(set(y_true_fvc_all)) == 2:
        fpr, tpr, _ = roc_curve(y_true_fvc_all, y_prob_fvc_all)
        auc = roc_auc_score(y_true_fvc_all, y_prob_fvc_all)
        plt.plot(fpr, tpr, label=f"FVC (AUC={auc:.3f})", color="blue")
    
    if len(set(y_true_fev_all)) == 2:
        fpr, tpr, _ = roc_curve(y_true_fev_all, y_prob_fev_all)
        auc = roc_auc_score(y_true_fev_all, y_prob_fev_all)
        plt.plot(fpr, tpr, label=f"FEV1 (AUC={auc:.3f})", color="green")
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC - {run_name}")
    plt.legend()
    plt.savefig(os.path.join(roc_dir, f"roc_{run_name}.png"))
    plt.close()


# =========================
# Main
# =========================
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    DATA_DIR = 'data'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images')
    TABULAR_PATH = os.path.join(DATA_DIR, 'ais_tabular_sampling.csv')
    LABEL_PATH   = os.path.join(DATA_DIR, 'ais_labels.csv')
    REFERENCE_DCM_PATH = os.path.join(IMAGE_DIR, '<reference_study>.dcm')
    
    RESULTS_DIR = 'results/image_only'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Filenames for the image-only runs
    if USE_IMAGE_ONLY:
        SUMMARY_FILE = os.path.join(RESULTS_DIR, 'rev_efficientnet_b5_film_fold_results.json')
    else:
        SUMMARY_FILE = os.path.join(RESULTS_DIR, 'rev_efficientnet_b5_film_PFT_fold_results.json')

    # Remove any stale summary files
    if os.path.exists(SUMMARY_FILE):
        os.remove(SUMMARY_FILE)

    BATCH_SIZE = 16
    EPOCHS = 100
    PATIENCE = 10
    NUM_FOLDS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = build_transforms(train=True)
    val_transform   = build_transforms(train=False)

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

    y_multi = base_dataset.df[['FVC_class','FEV1_class']].values
    skf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Backbones for the image-only ablation
    backbones = ['efficientnet_b5'] 

    # Loop
    for backbone in backbones:
        # In image-only mode the fusion/tabular loops are meaningless, so run once
        iter_fusions = ['None'] if USE_IMAGE_ONLY else ['film']
        iter_tabs    = ['None'] if USE_IMAGE_ONLY else ['mlp']
        
        for fusion in iter_fusions:
            for tabular_type in iter_tabs:
                fold_results = []
                print(f"\n>>> Start Training: {backbone} | ImageOnly={USE_IMAGE_ONLY}")
                
                for fold, (tr_idx, va_idx) in enumerate(skf.split(base_dataset.df, y_multi), start=1):
                    metrics = train_fold(
                        fold=fold,
                        train_idx=tr_idx,
                        val_idx=va_idx,
                        base_dataset=base_dataset,
                        device=DEVICE,
                        epochs=EPOCHS,
                        patience=PATIENCE,
                        batch_size=BATCH_SIZE,
                        lr=1e-4,
                        backbone_name=backbone,
                        fusion_type=fusion,
                        tabular_type=tabular_type,
                        train_transform=train_transform,
                        val_transform=val_transform,
                    )
                    
                    entry = {
                        'backbone': backbone,
                        'mode': 'ImageOnly' if USE_IMAGE_ONLY else 'Multimodal',
                        'fold': fold,
                        **metrics
                    }
                    fold_results.append(entry)
                    with open(SUMMARY_FILE, 'a') as f:
                        f.write(json.dumps(entry) + '\n')

                # Save the ROC after the folds finish
                run_name = f"{backbone}_ImageOnly" if USE_IMAGE_ONLY else f"{backbone}_{fusion}_{tabular_type}"
                save_combined_roc(fold_results, RESULTS_DIR, backbone, run_name)

    print("Training complete.")