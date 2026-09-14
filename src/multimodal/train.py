# train.py (updated)
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm
from torchvision import transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt


from dataset import AISTabularXRDataset, custom_collate
from model import MultiModalMTLModel

# =========================
# Aug / Utils
# =========================
class GrayTo3CH(object):
    def __call__(self, x):
        if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x

class YoloMixup:
    """
    YOLO Style MixUp & CutMix for Multi-modal (Image + Tabular)
    """
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob  # probability of choosing CutMix over MixUp

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, imgs, tabs, labels):
        """
        imgs: [B, C, H, W]
        tabs: [B, F]
        labels: [B, 2] (FVC, FEV1)
        """
        if np.random.rand() > self.prob:
            return imgs, tabs, labels, None  # not applied

        batch_size = imgs.size(0)
        indices = torch.randperm(batch_size).to(imgs.device)

        # Decide MixUp vs CutMix
        use_cutmix = np.random.rand() < self.switch_prob

        if use_cutmix:
            # --- CutMix ---
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
            
            # Paste the image patch
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[indices, :, bbx1:bbx2, bby1:bby2]
            
            # Recompute lambda from the exact pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
        else:
            # --- MixUp ---
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            imgs = lam * imgs + (1 - lam) * imgs[indices]

        # Mix the tabular features at the same ratio (linear interpolation)
        tabs = lam * tabs + (1 - lam) * tabs[indices]

        # Return both target sets and lambda so the mixing is applied inside the
        # loss computation rather than here.
        targets_orig = labels
        targets_shuffled = labels[indices]
        
        return imgs, tabs, (targets_orig, targets_shuffled, lam), "mix"

def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            
            # --- YOLO style Affine ---
            # YOLO combines translate, scale and degrees
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=10,             # small rotations are acceptable on radiographs
                    translate=(0.1, 0.1),   # YOLO default: 0.1
                    scale=(0.5, 1.5),       # strong scale jittering
                    shear=0.0               # shear risks distorting the anatomy
                )
            ], p=0.5),

            # --- RandomResizedCrop in place of YOLO-style mosaic ---
            # Mosaic composites across a batch; on a single image RandomResizedCrop
            # gives a comparable scale variation.
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.9, 1.1)),

            transforms.RandomHorizontalFlip(p=0.5),
            
            # --- Photometric jitter ---
            # Radiographs are grayscale replicated to 3 channels, so hue and
            # saturation have no effect; only brightness and contrast are used.
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.0,  # no effect on grayscale
                    hue=0.0          # no effect on grayscale
                )
            ], p=0.3),

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
    """
    Returns: (acc, sens, spec, auc)
    """
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
    """
    Threshold maximizing Youden's J = TPR - FPR.
    Returns `fallback` when the ROC cannot be computed (e.g. a single class present).
    """
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
    fusion_type,
    tabular_type,
    train_transform,
    val_transform,
    gradcam_root,
    gradcam_num_train=100,
    gradcam_num_test=40,
):
    """
    base_dataset: the full AISTabularXRDataset
    train_idx, val_idx: numpy arrays from StratifiedKFold
    """
    # Paths
    CKPT_ROOT = 'checkpoints'
    ckpt_dir = os.path.join(CKPT_ROOT, f"{backbone_name}_{fusion_type}_{tabular_type}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Rebuild the dataset per split so each gets its own transform
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
    # Apply the fold indices
    train_dataset.df = base_dataset.df.iloc[train_idx].reset_index(drop=True)
    val_dataset.df   = base_dataset.df.iloc[val_idx].reset_index(drop=True)
    # Share the feature columns
    train_dataset.feature_cols = base_dataset.feature_cols
    val_dataset.feature_cols   = base_dataset.feature_cols

    # Tabular statistics, computed on the training split only
    feats  = train_dataset.feature_cols
    tr_df  = train_dataset.df
    tab_means = torch.tensor(tr_df[feats].mean().values, dtype=torch.float32, device=device)
    tab_stds  = torch.tensor(tr_df[feats].std().values,  dtype=torch.float32, device=device).clamp(min=1e-2)

    # pos_weight per task
    def pw(name):
        n_pos = int((tr_df[name] == 1).sum())
        n_neg = int((tr_df[name] == 0).sum())
        return float(n_neg / max(n_pos, 1))
    w_fvc  = pw('FVC_class')
    w_fev1 = pw('FEV1_class')
    pos_weights = torch.tensor([w_fvc, w_fev1], dtype=torch.float32, device=device)

    # DataLoaders

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=custom_collate, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=custom_collate, pin_memory=True
    )

    # Model / Optim / Sched / Loss
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='mean')

    # Early stopping on validation loss
    best_loss = float('inf')
    wait = 0
    best_state = None
    best_metrics = {}
    
    mixup_fn = YoloMixup(prob=0.5, switch_prob=0.5)

    for epoch in range(1, epochs + 1):
        # -----------------
        # Train
        # -----------------
        model.train()
        train_losses = []
        y_true_fvc_tr, y_prob_fvc_tr = [], []
        y_true_fev_tr, y_prob_fev_tr = [], []

        for batch in tqdm(train_loader, desc=f"Fold{fold} Train", leave=False):
            imgs, tabs, labels, _ = batch
            if imgs is None:  # guard against a batch where every sample failed to load
                continue

            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            tabs_norm = (tabs - tab_means) / tab_stds

            y_fvc  = labels['FVC_class'].float().to(device)
            y_fev1 = labels['FEV1_class'].float().to(device)
            y = torch.stack([y_fvc, y_fev1], dim=1)  # [B,2]
            # Apply MixUp / CutMix
            imgs, tabs_norm, mix_targets, status = mixup_fn(imgs, tabs_norm, y)
            
            # 2. Forward
            logits = model(imgs, tabs_norm)          # [B,2]
            # 3. Loss Calculation (Soft Label handling)
            # ==========================================
            if status == "mix":
                y_a, y_b, lam = mix_targets
                # BCEWithLogitsLoss supports soft labels; mix the two losses linearly
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y)
                
                
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # ---- Gradient Clipping ----
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            probs = torch.sigmoid(logits)            # [B,2]

            y_true_fvc_tr.extend(y_fvc.long().cpu().tolist())
            y_prob_fvc_tr.extend(probs[:, 0].detach().cpu().tolist())
            y_true_fev_tr.extend(y_fev1.long().cpu().tolist())
            y_prob_fev_tr.extend(probs[:, 1].detach().cpu().tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        tr_fvc_acc, tr_fvc_sens, tr_fvc_spec, tr_fvc_auc = binary_metrics(y_true_fvc_tr, y_prob_fvc_tr, thr=0.5)
        tr_fev_acc, tr_fev_sens, tr_fev_spec, tr_fev_auc = binary_metrics(y_true_fev_tr, y_prob_fev_tr, thr=0.5)

        # -----------------
        # Validation
        # -----------------
        model.eval()
        val_losses = []
        y_true_fvc_val, y_prob_fvc_val = [], []
        y_true_fev_val, y_prob_fev_val = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold{fold} Val", leave=False):
                imgs, tabs, labels, _ = batch
                if imgs is None:
                    continue
                imgs = imgs.to(device, non_blocking=True)
                tabs = tabs.to(device, non_blocking=True)
                tabs_norm = (tabs - tab_means) / tab_stds

                y_fvc  = labels['FVC_class'].float().to(device)
                y_fev1 = labels['FEV1_class'].float().to(device)
                y = torch.stack([y_fvc, y_fev1], dim=1)

                logits = model(imgs, tabs_norm)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                y_true_fvc_val.extend(y_fvc.long().cpu().tolist())
                y_prob_fvc_val.extend(probs[:, 0].cpu().tolist())
                y_true_fev_val.extend(y_fev1.long().cpu().tolist())
                y_prob_fev_val.extend(probs[:, 1].cpu().tolist())

        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        # --- Metrics @ thr=0.5 ---
        va_fvc_acc, va_fvc_sens, va_fvc_spec, va_fvc_auc = binary_metrics(y_true_fvc_val, y_prob_fvc_val, thr=0.5)
        va_fev_acc, va_fev_sens, va_fev_spec, va_fev_auc = binary_metrics(y_true_fev_val, y_prob_fev_val, thr=0.5)
        va_acc  = np.nanmean([va_fvc_acc, va_fev_acc])
        va_sens = np.nanmean([va_fvc_sens, va_fev_sens])
        va_spec = np.nanmean([va_fvc_spec, va_fev_spec])
        va_auc  = np.nanmean([va_fvc_auc, va_fev_auc])

        # --- Find tuned thresholds by Youden's J (per task) ---
        t_fvc = best_thresh_youden(y_true_fvc_val, y_prob_fvc_val, fallback=0.5)
        t_fev = best_thresh_youden(y_true_fev_val, y_prob_fev_val, fallback=0.5)

        # --- Metrics @ tuned thresholds ---
        va_fvc_acc_T, va_fvc_sens_T, va_fvc_spec_T, _ = binary_metrics(y_true_fvc_val, y_prob_fvc_val, thr=t_fvc)
        va_fev_acc_T, va_fev_sens_T, va_fev_spec_T, _ = binary_metrics(y_true_fev_val, y_prob_fev_val, thr=t_fev)
        va_acc_T  = np.nanmean([va_fvc_acc_T,  va_fev_acc_T])
        va_sens_T = np.nanmean([va_fvc_sens_T, va_fev_sens_T])
        va_spec_T = np.nanmean([va_fvc_spec_T, va_fev_spec_T])

        # Logging
        print(f"[Fold{fold}] Epoch {epoch}")
        # print(f"  TrainLoss={train_loss:.4f} | "
        #       f"FVC:Acc={tr_fvc_acc:.4f}/Sens={tr_fvc_sens:.4f}/Spec={tr_fvc_spec:.4f}/AUC={tr_fvc_auc:.4f} | "
        #       f"FEV1:Acc={tr_fev_acc:.4f}/Sens={tr_fev_sens:.4f}/Spec={tr_fev_spec:.4f}/AUC={tr_fev_auc:.4f}")
        print(f"  ValLoss={val_loss:.4f} | "
              f"FVC@0.5:Acc={va_fvc_acc:.4f}/Sens={va_fvc_sens:.4f}/Spec={va_fvc_spec:.4f}/AUC={va_fvc_auc:.4f} | "
              f"FEV1@0.5:Acc={va_fev_acc:.4f}/Sens={va_fev_sens:.4f}/Spec={va_fev_spec:.4f}/AUC={va_fev_auc:.4f} | "
              f"Macro@0.5:Acc={va_acc:.4f}/Sens={va_sens:.4f}/Spec={va_spec:.4f}/AUC={va_auc:.4f}")
        # print(f"  Val(tuned):  "
        #       f"FVC@{t_fvc:.3f}:Acc={va_fvc_acc_T:.4f}/Sens={va_fvc_sens_T:.4f}/Spec={va_fvc_spec_T:.4f} | "
        #       f"FEV1@{t_fev:.3f}:Acc={va_fev_acc_T:.4f}/Sens={va_fev_sens_T:.4f}/Spec={va_fev_spec_T:.4f} | "
        #       f"Macro:Acc={va_acc_T:.4f}/Sens={va_sens_T:.4f}/Spec={va_spec_T:.4f}")

        # Early stopping on validation loss
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = model.module.state_dict()
            best_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                # Per-task metrics @0.5
                'fvc_acc': va_fvc_acc, 'fvc_sens': va_fvc_sens,
                'fvc_spec': va_fvc_spec, 'fvc_auc': va_fvc_auc,
                'fev_acc': va_fev_acc, 'fev_sens': va_fev_sens,
                'fev_spec': va_fev_spec, 'fev_auc': va_fev_auc,
                # Macro summary @0.5
                'macro_acc': va_acc, 'macro_sens': va_sens,
                'macro_spec': va_spec, 'macro_auc': va_auc,
                # Tuned thresholds & metrics
                't_fvc': t_fvc, 't_fev': t_fev,
                'fvc_acc_tuned': va_fvc_acc_T, 'fvc_sens_tuned': va_fvc_sens_T, 'fvc_spec_tuned': va_fvc_spec_T,
                'fev_acc_tuned': va_fev_acc_T, 'fev_sens_tuned': va_fev_sens_T, 'fev_spec_tuned': va_fev_spec_T,
                'macro_acc_tuned': va_acc_T, 'macro_sens_tuned': va_sens_T, 'macro_spec_tuned': va_spec_T,
                # --- Keep raw outputs for the ROC curves ---
                'y_true_fvc': y_true_fvc_val,
                'y_prob_fvc': y_prob_fvc_val,
                'y_true_fev': y_true_fev_val,
                'y_prob_fev': y_prob_fev_val,
            }
            print(f"  ** model updated - val loss {val_loss:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping @ epoch {epoch}")
                break

        # Scheduler
        scheduler.step(epoch)

    # Save ckpt
    ckpt_path = os.path.join(ckpt_dir, f'fold{fold}.pt')
    torch.save(best_state, ckpt_path)
    
    
    # =========================
    # Grad-CAM (Train no-aug / Test=val split)
    # =========================
    try:
        # Rebuild the best model on a single device (no DataParallel)
        cam_model = MultiModalMTLModel(
            num_tabular_features=len(feats),
            backbone_name=backbone_name,
            fusion_type=fusion_type,
            tabular_type=tabular_type,
            pretrained=False,   # weights are loaded from the checkpoint
            in_chans=3,
            num_tasks=2,
        ).to(device)
        cam_model.load_state_dict(best_state, strict=True)
        cam_model.eval()

        # Grad-CAM must run on un-augmented images, so use val_transform here too
        train_noaug_dataset = AISTabularXRDataset(
            image_dir=base_dataset.image_dir,
            tabular_path=base_dataset.tabular_path,
            label_path=base_dataset.label_path,
            transform=val_transform,   # <-- no-aug
            reference_dcm_path=base_dataset.reference_dcm_path,
            verbose=False,
        )
        train_noaug_dataset.df = base_dataset.df.iloc[train_idx].reset_index(drop=True)
        train_noaug_dataset.feature_cols = base_dataset.feature_cols

        # Validation split (already uses val_transform)
        test_dataset = AISTabularXRDataset(
            image_dir=base_dataset.image_dir,
            tabular_path=base_dataset.tabular_path,
            label_path=base_dataset.label_path,
            transform=val_transform,
            reference_dcm_path=base_dataset.reference_dcm_path,
            verbose=False,
        )
        test_dataset.df = base_dataset.df.iloc[val_idx].reset_index(drop=True)
        test_dataset.feature_cols = base_dataset.feature_cols

        # Output directory
        tag = f"{backbone_name}_{fusion_type}_{tabular_type}"
        fold_dir = os.path.join(gradcam_root, tag, f"fold{fold}")
        out_train = os.path.join(fold_dir, "train_noaug")
        out_test  = os.path.join(fold_dir, "test")

        save_gradcam_for_split(
            model=cam_model,
            dataset=train_noaug_dataset,
            device=device,
            tab_means=tab_means,
            tab_stds=tab_stds,
            out_dir=out_train,
            split_name="train",
            max_samples=gradcam_num_train,
        )
        save_gradcam_for_split(
            model=cam_model,
            dataset=test_dataset,
            device=device,
            tab_means=tab_means,
            tab_stds=tab_stds,
            out_dir=out_test,
            split_name="test",
            max_samples=gradcam_num_test,
        )
    except Exception as e:
        print(f"[GradCAM] skipped due to error: {e}")

    
    
    best_metrics['checkpoint'] = ckpt_path
    return best_metrics

def save_combined_roc(fold_results, save_dir, backbone, fusion, tabular_type):
    y_true_fvc_all, y_prob_fvc_all = [], []
    y_true_fev_all, y_prob_fev_all = [], []

    for fr in fold_results:
        y_true_fvc_all.extend(fr["y_true_fvc"])
        y_prob_fvc_all.extend(fr["y_prob_fvc"])
        y_true_fev_all.extend(fr["y_true_fev"])
        y_prob_fev_all.extend(fr["y_prob_fev"])

    # -----------------------
    # 1. Per-task ROC (FVC, FEV1)
    # -----------------------
    roc_dir = os.path.join(save_dir, "20250921_roc_curves")
    os.makedirs(roc_dir, exist_ok=True)

    for task, y_true, y_prob in zip(
        ["fvc", "fev1"],
        [y_true_fvc_all, y_true_fev_all],
        [y_prob_fvc_all, y_prob_fev_all]
    ):
        if len(set(y_true)) == 2:  # both classes must be present
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)

            plt.figure()
            plt.plot(fpr, tpr, label=f"{task.upper()} (AUC={auc:.3f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{task.upper()} ROC - {backbone}_{fusion}_{tabular_type}")
            plt.legend()
            save_path = os.path.join(roc_dir, f"{task}_roc_{backbone}_{fusion}_{tabular_type}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[ROC] saved: {save_path}")

    # -----------------------
    # 2. Macro-averaged ROC
    # -----------------------
    if len(set(y_true_fvc_all)) == 2 and len(set(y_true_fev_all)) == 2:
        fpr_fvc, tpr_fvc, _ = roc_curve(y_true_fvc_all, y_prob_fvc_all)
        fpr_fev, tpr_fev, _ = roc_curve(y_true_fev_all, y_prob_fev_all)

        # Interpolate onto a common grid so the curves can be averaged
        fpr_grid = np.linspace(0, 1, 100)
        tpr_fvc_interp = np.interp(fpr_grid, fpr_fvc, tpr_fvc)
        tpr_fev_interp = np.interp(fpr_grid, fpr_fev, tpr_fev)
        tpr_macro = (tpr_fvc_interp + tpr_fev_interp) / 2

        auc_macro = np.mean([
            roc_auc_score(y_true_fvc_all, y_prob_fvc_all),
            roc_auc_score(y_true_fev_all, y_prob_fev_all)
        ])

        plt.figure()
        plt.plot(fpr_grid, tpr_macro, label=f"Macro (AUC={auc_macro:.3f})", color="purple")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Macro ROC - {backbone}_{fusion}_{tabular_type}")
        plt.legend()
        save_path = os.path.join(roc_dir, f"macro_roc_{backbone}_{fusion}_{tabular_type}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[ROC] saved: {save_path}")

    # -----------------------
    # 3. Combined ROC (FVC + FEV1 + macro)
    # -----------------------
    plt.figure()
    colors = {"FVC": "blue", "FEV1": "green", "Macro": "purple"}
    for task, y_true, y_prob in zip(
        ["FVC", "FEV1"],
        [y_true_fvc_all, y_true_fev_all],
        [y_prob_fvc_all, y_prob_fev_all]
    ):
        if len(set(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            plt.plot(fpr, tpr, label=f"{task} (AUC={auc:.3f})", color=colors[task])

    if len(set(y_true_fvc_all)) == 2 and len(set(y_true_fev_all)) == 2:
        plt.plot(fpr_grid, tpr_macro, label=f"Macro (AUC={auc_macro:.3f})", color=colors["Macro"])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Combined ROC - {backbone}_{fusion}_{tabular_type}")
    plt.legend()
    save_path = os.path.join(roc_dir, f"combined_roc_{backbone}_{fusion}_{tabular_type}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[ROC] saved: {save_path}")


# =========================
# Grad-CAM helpers
# =========================
import torch.nn.functional as F

def find_last_conv_layer(module: nn.Module):
    """
    Find the last Conv2d layer in the model to use as the Grad-CAM target.
    Works for the CNN backbones (EfficientNet, ConvNeXt, ResNet, VGG).
    """
    last_name, last_layer = None, None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_name, last_layer = name, m
    return last_name, last_layer

class GradCAM:
    """
    Vanilla Grad-CAM, for both binary and multi-task heads.
    - target_layer: an nn.Conv2d layer, normally the last conv of the backbone
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, grad_input, grad_output):
            # grad_output: tuple, (dL/dout,)
            self.gradients = grad_output[0]

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        # Full backward hook (current PyTorch API)
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def compute_cam(self, upsample_size):
        """
        activations: [B,C,H,W], gradients: [B,C,H,W]
        cam: [B,1,H,W] -> upsample -> normalize(0~1)
        """
        assert self.activations is not None and self.gradients is not None, "Hooks did not capture act/grads."

        grads = self.gradients
        acts = self.activations

        # channel weights: GAP over spatial dims
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True) # [B,1,H,W]
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=upsample_size, mode="bilinear", align_corners=False)

        # normalize per-sample
        B = cam.size(0)
        cam_ = cam.view(B, -1)
        cam_min = cam_.min(dim=1)[0].view(B,1,1,1)
        cam_max = cam_.max(dim=1)[0].view(B,1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam  # [B,1,H,W], 0~1

def denormalize_img(img_3chw: torch.Tensor, mean, std):
    """
    img_3chw: [3,H,W] normalized tensor -> [H,W,3] float(0~1)
    """
    x = img_3chw.detach().cpu().clone()
    mean = torch.tensor(mean).view(3,1,1)
    std  = torch.tensor(std).view(3,1,1)
    x = x * std + mean
    x = x.clamp(0, 1)
    x = x.permute(1,2,0).numpy()  # HWC
    return x

def overlay_cam(img_hwc, cam_hw, alpha=0.35, cmap_name="jet"):
    """
    img_hwc: [H,W,3] (0~1)
    cam_hw:  [H,W]   (0~1)
    return:  overlayed [H,W,3]
    """
    cmap = plt.get_cmap(cmap_name)
    heat = cmap(cam_hw)[..., :3]  # RGB
    out = (1 - alpha) * img_hwc + alpha * heat
    out = np.clip(out, 0, 1)
    return out

def _safe_get_patient_id(df_row, fallback):
    # Column names vary between exports, so extract defensively
    for k in ["Patient_ID", "patient_id", "patientId", "id"]:
        if k in df_row.index:
            return str(df_row[k])
    return str(fallback)

def save_gradcam_for_split(
    model: nn.Module,
    dataset,
    device,
    tab_means: torch.Tensor,
    tab_stds: torch.Tensor,
    out_dir: str,
    split_name: str,
    max_samples: int = 50,
    seed: int = 42,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    For each of the train and validation splits:
    - forward the un-augmented image plus tabular features
    - compute a Grad-CAM per task (FVC, FEV1)
    - save the original image and the overlay
    """
    os.makedirs(out_dir, exist_ok=True)

    # Target layer: the last conv inside the backbone
    # (assumes MultiModalMTLModel exposes model.backbone)
    if not hasattr(model, "backbone"):
        raise AttributeError("MultiModalMTLModel has no model.backbone; the target-layer lookup needs updating.")

    last_name, target_layer = find_last_conv_layer(model.backbone)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found (a transformer backbone needs a different XAI method).")

    cam_engine = GradCAM(model, target_layer)

    # Select samples
    n = len(dataset)
    idxs = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)
    idxs = idxs[: min(max_samples, n)]

    model.eval()

    for j, i in enumerate(idxs):
        try:
            sample = dataset[i]
        except Exception:
            continue

        # Handle the dataset __getitem__ return shape defensively.
        # Expected: (img_tensor, tab_tensor, labels_dict, meta)
        img = tab = labels = meta = None
        if isinstance(sample, (list, tuple)):
            if len(sample) >= 3:
                img, tab, labels = sample[0], sample[1], sample[2]
                meta = sample[3] if len(sample) >= 4 else None
        elif isinstance(sample, dict):
            img = sample.get("image", None)
            tab = sample.get("tabular", None)
            labels = sample.get("labels", None)
            meta = sample.get("meta", None)

        if img is None or tab is None or labels is None:
            continue

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        if not isinstance(tab, torch.Tensor):
            tab = torch.tensor(tab)

        # GrayTo3CH in the transform already handles [1,H,W]; this is a safeguard
        if img.ndim == 3 and img.size(0) == 1:
            img = img.repeat(3,1,1)

        # labels
        # Labels arrive as a dict (custom_collate preserves the dict)
        y_fvc  = labels["FVC_class"]
        y_fev1 = labels["FEV1_class"]
        if isinstance(y_fvc, torch.Tensor):  y_fvc = int(y_fvc.item())
        if isinstance(y_fev1, torch.Tensor): y_fev1 = int(y_fev1.item())

        # tab norm
        tab = tab.to(device).float()
        tab_norm = (tab - tab_means) / tab_stds

        # forward
        img_in = img.unsqueeze(0).to(device).float()      # [1,3,H,W]
        tab_in = tab_norm.unsqueeze(0)                    # [1,F]

        # The forward hook has stored the activations
        with torch.enable_grad():
            logits = model(img_in, tab_in)                # [1,2]
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]  # [2]

            H, W = img.shape[1], img.shape[2]
            img_vis = denormalize_img(img, mean, std)     # [H,W,3]

            cams = []
            for task_idx, task_name in [(0, "FVC"), (1, "FEV1")]:
                model.zero_grad(set_to_none=True)
                score = logits[:, task_idx].sum()
                # The last task does not need to retain the graph
                score.backward(retain_graph=(task_idx == 0))

                cam = cam_engine.compute_cam(upsample_size=(H, W))   # [1,1,H,W]
                cam_hw = cam[0,0].detach().cpu().numpy()            # [H,W]
                overlay = overlay_cam(img_vis, cam_hw, alpha=0.35, cmap_name="jet")
                cams.append((task_name, cam_hw, overlay))

        # Filename, including the Patient_ID when available
        df_row = dataset.df.iloc[i] if hasattr(dataset, "df") else None
        pid = _safe_get_patient_id(df_row, fallback=i) if df_row is not None else str(i)

        fn = (f"{split_name}_pid{pid}_idx{i}"
              f"_yFVC{y_fvc}_yFEV{y_fev1}"
              f"_pFVC{probs[0]:.3f}_pFEV{probs[1]:.3f}.png")
        save_path = os.path.join(out_dir, fn)

        # Save the original image and the overlays
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_vis)
        plt.title("Original (no-aug)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cams[0][2])
        plt.title(f"Grad-CAM {cams[0][0]}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cams[1][2])
        plt.title(f"Grad-CAM {cams[1][0]}")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    cam_engine.remove_hooks()
    print(f"[GradCAM] saved {split_name} cams -> {out_dir}")



# =========================
# Main
# =========================
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Settings
    DATA_DIR = 'data'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images')
    TABULAR_PATH = os.path.join(DATA_DIR, 'ais_tabular_sampling.csv')  # NEW TABULAR
    LABEL_PATH   = os.path.join(DATA_DIR, 'ais_labels.csv')
    REFERENCE_DCM_PATH = os.path.join(IMAGE_DIR, '<reference_study>.dcm')

    RESULTS_DIR = 'results/efficientnet_b0_film_mlp'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    SUMMARY_FILE = os.path.join(RESULTS_DIR, 'fold_results.jsonl')
    METHOD_SUMMARY_FILE = os.path.join(RESULTS_DIR, 'fold_results.jsonl')

    # Remove existing summary files to avoid appending stale results
    for f in [SUMMARY_FILE, METHOD_SUMMARY_FILE]:
        if os.path.exists(f):
            os.remove(f)

    BATCH_SIZE = 16
    EPOCHS = 100
    PATIENCE = 10
    NUM_FOLDS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    GRADCAM_ROOT = os.path.join(RESULTS_DIR, "gradcam_images")
    GRADCAM_NUM_TRAIN = 100
    GRADCAM_NUM_TEST  = 60

    # Build transforms
    train_transform = build_transforms(train=True)
    val_transform   = build_transforms(train=False)

    # Load base dataset
    base_dataset = AISTabularXRDataset(
        image_dir=IMAGE_DIR,
        tabular_path=TABULAR_PATH,
        label_path=LABEL_PATH,
        transform=val_transform,  # [3,H,W]; augmentation is applied in the per-fold train dataset
        reference_dcm_path=REFERENCE_DCM_PATH,
        verbose=True,
    )
    # Record the source paths for the per-fold datasets
    base_dataset.tabular_path = TABULAR_PATH
    base_dataset.label_path = LABEL_PATH
    base_dataset.reference_dcm_path = REFERENCE_DCM_PATH

    # Stratify
    y_multi = base_dataset.df[['FVC_class','FEV1_class']].values
    skf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # backbones = ['convnext_tiny','resnet18', 'resnet34','vggnet']
    backbones = [
        # 'convnext_tiny', 'convnext_base',
        # 'resnet18', 'resnet34',
        # 'vgg11_bn', 'vgg16_bn', 'vgg19_bn',
        # 'mobilenetv2_100', 'mobilenetv3_small_100', 'mobilenetv3_large_100',
        'efficientnet_b0'
    ]

    fusions   = ['film']
    tabs      = ['mlp']

    for backbone in backbones:
        for fusion in fusions:
            for tabular_type in tabs:
                fold_results = []
                for fold,(tr_idx,va_idx) in enumerate(skf.split(base_dataset.df, y_multi), start=1):
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
                        gradcam_root=GRADCAM_ROOT,
                        gradcam_num_train=GRADCAM_NUM_TRAIN,
                        gradcam_num_test=GRADCAM_NUM_TEST,
                    )
                    entry = {
                        'backbone': backbone,
                        'fusion': fusion,
                        'tabular_type': tabular_type,
                        'fold': fold,
                        **metrics
                    }
                    fold_results.append(entry)
                    with open(SUMMARY_FILE, 'a') as f:
                        f.write(json.dumps(entry) + '\n')

                # Average across folds
                avg_keys = [
                    'val_loss',
                    'fvc_acc', 'fvc_sens', 'fvc_spec', 'fvc_auc',
                    'fev_acc', 'fev_sens', 'fev_spec', 'fev_auc',
                    'macro_acc', 'macro_sens', 'macro_spec', 'macro_auc',
                    # tuned
                    'fvc_acc_tuned', 'fvc_sens_tuned', 'fvc_spec_tuned',
                    'fev_acc_tuned', 'fev_sens_tuned', 'fev_spec_tuned',
                    'macro_acc_tuned', 'macro_sens_tuned', 'macro_spec_tuned',
                ]
                avg = {k: float(np.nanmean([e.get(k, np.nan) for e in fold_results])) for k in avg_keys}

                method_summary = {
                    'backbone': backbone,
                    'fusion': fusion,
                    'tabular_type': tabular_type,
                    'fold_avg': avg,
                    'augmentations': [t.__class__.__name__ for t in train_transform.transforms],
                    'num_folds': NUM_FOLDS
                }
                with open(METHOD_SUMMARY_FILE, 'a') as f:
                    f.write(json.dumps(method_summary) + '\n')
                    
                save_combined_roc(
                    fold_results,
                    save_dir=RESULTS_DIR,
                    backbone=backbone,
                    fusion=fusion,
                    tabular_type=tabular_type
                )

    print("Training complete.")
