import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt

# pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Project modules
from dataset import AISTabularXRDataset, custom_collate
from model import MultiModalMTLModel

# ==========================================
# Configuration
# ==========================================
class Config:
    DATA_DIR = 'data'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images')
    TABULAR_PATH = os.path.join(DATA_DIR, 'ais_tabular_sampling.csv')
    LABEL_PATH = os.path.join(DATA_DIR, 'ais_labels.csv')
    REFERENCE_DCM_PATH = os.path.join(IMAGE_DIR, '<reference_study>.dcm')
    
    # Multimodal checkpoint directory (e.g. efficientnet_b0_film_mlp)
    CKPT_ROOT = 'checkpoints/efficientnet_b0_film_mlp'
    BACKBONE = 'efficientnet_b0'
    
    # Output root
    SAVE_ROOT = 'results/gradcam'
    
    BATCH_SIZE = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NUM_SAMPLES = 60 

# ==========================================
# Utils
# ==========================================
class GrayTo3CH(object):
    def __call__(self, x):
        if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x

def build_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        GrayTo3CH(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

# ==========================================
# Multimodal wrapper for Grad-CAM
# ==========================================
class MultimodalGradCAMWrapper(nn.Module):
    """
    The Grad-CAM library calls forward(x) with the image alone, so the tabular
    features are injected beforehand with set_tabs() and combined inside forward.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Expose the backbone so Grad-CAM can attach its hooks
        self.backbone = model.backbone 
        self.current_tabs = None

    def set_tabs(self, tabs):
        self.current_tabs = tabs

    def forward(self, x):
        # The library passes only the image; call the real model with the stored tabs
        return self.model(x, self.current_tabs)

# ==========================================
# Main Visualization Logic
# ==========================================
def run_gradcam(model, loader, device, save_dir, prefix="val"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Wrap the model
    wrapped_model = MultimodalGradCAMWrapper(model)
    
    # 2. Target layer: the last conv of the EfficientNet backbone
    target_layers = [wrapped_model.backbone.conv_head]
    
    # 3. Build the Grad-CAM object on the wrapped model
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    print(f"   Generating {prefix} samples...")
    
    count = 0
    for batch in tqdm(loader, leave=False):
        if count >= Config.NUM_SAMPLES:
            break
            
        # The tabular features are needed as well
        imgs, tabs, labels, pids = batch
        if imgs is None: continue
        
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        
        # Inject this batch's tabular features into the wrapper
        wrapped_model.set_tabs(tabs)
        
        # 1. FVC Heatmap
        grayscale_cam_fvc = cam(input_tensor=imgs, targets=[ClassifierOutputTarget(0)])
        
        # 2. FEV1 Heatmap
        grayscale_cam_fev1 = cam(input_tensor=imgs, targets=[ClassifierOutputTarget(1)])

        # Save the visualizations
        for i in range(imgs.size(0)):
            pid = pids[i]
            y_fvc = int(labels['FVC_class'][i])
            y_fev = int(labels['FEV1_class'][i])
            
            rgb_img = denormalize(imgs[i])
            
            vis_fvc = show_cam_on_image(rgb_img, grayscale_cam_fvc[i, :], use_rgb=True)
            vis_fev = show_cam_on_image(rgb_img, grayscale_cam_fev1[i, :], use_rgb=True)
            
            # Annotate and compose
            original_bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            fvc_bgr = cv2.cvtColor(vis_fvc, cv2.COLOR_RGB2BGR)
            fev_bgr = cv2.cvtColor(vis_fev, cv2.COLOR_RGB2BGR)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_bgr, f"ID: {pid}", (10, 30), font, 0.8, (255, 255, 255), 2)
            cv2.putText(fvc_bgr, f"FVC (GT: {y_fvc})", (10, 30), font, 0.8, (255, 255, 255), 2)
            cv2.putText(fev_bgr, f"FEV1 (GT: {y_fev})", (10, 30), font, 0.8, (255, 255, 255), 2)
            
            combined = np.hstack([original_bgr, fvc_bgr, fev_bgr])
            
            save_path = os.path.join(save_dir, f"{prefix}_{pid}_gradcam.png")
            cv2.imwrite(save_path, combined)
            count += 1
            
    print(f"   -> Saved {count} images to {save_dir}")

if __name__ == "__main__":
    print("Initializing Dataset...")
    transform = build_transforms()
    full_dataset = AISTabularXRDataset(
        image_dir=Config.IMAGE_DIR,
        tabular_path=Config.TABULAR_PATH,
        label_path=Config.LABEL_PATH,
        transform=transform,
        reference_dcm_path=Config.REFERENCE_DCM_PATH,
        verbose=False
    )
    full_dataset.tabular_path = Config.TABULAR_PATH
    full_dataset.label_path = Config.LABEL_PATH
    full_dataset.reference_dcm_path = Config.REFERENCE_DCM_PATH
    
    y_multi = full_dataset.df[['FVC_class','FEV1_class']].values
    skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(full_dataset.df, y_multi))

    # Number of tabular features, needed to build the model
    num_tab_features = len(full_dataset.feature_cols)

    for fold in range(1, 6):
        print(f"\n========================================")
        print(f" Processing Fold {fold}")
        print(f"========================================")
        
        ckpt_path = os.path.join(Config.CKPT_ROOT, f'fold{fold}.pt')
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint not found: {ckpt_path}. Skipping...")
            continue
            
        print(f"Loading MultiModal model from {ckpt_path}...")
        
        # Build the multimodal model (not the image-only variant)
        base_model = MultiModalMTLModel(
            num_tabular_features=num_tab_features,
            backbone_name=Config.BACKBONE,
            fusion_type='film',   # must match the checkpoint
            tabular_type='mlp',   # must match the checkpoint
            pretrained=False,
            num_tasks=2,
            dropout=0.3
        )
        
        state_dict = torch.load(ckpt_path, map_location=Config.DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        base_model.load_state_dict(new_state_dict)
        base_model.to(Config.DEVICE)
        base_model.eval()
        
        # Data Split
        train_idx, val_idx = splits[fold - 1]
        
        train_dataset = AISTabularXRDataset(
            image_dir=Config.IMAGE_DIR, tabular_path=Config.TABULAR_PATH, label_path=Config.LABEL_PATH,
            transform=transform, reference_dcm_path=Config.REFERENCE_DCM_PATH, verbose=False
        )
        train_dataset.df = full_dataset.df.iloc[train_idx].reset_index(drop=True)
        train_dataset.feature_cols = full_dataset.feature_cols
        
        val_dataset = AISTabularXRDataset(
            image_dir=Config.IMAGE_DIR, tabular_path=Config.TABULAR_PATH, label_path=Config.LABEL_PATH,
            transform=transform, reference_dcm_path=Config.REFERENCE_DCM_PATH, verbose=False
        )
        val_dataset.df = full_dataset.df.iloc[val_idx].reset_index(drop=True)
        val_dataset.feature_cols = full_dataset.feature_cols
        
        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
        
        fold_save_dir = os.path.join(Config.SAVE_ROOT, f"fold{fold}")
        
        # Run Grad-CAM
        run_gradcam(base_model, val_loader, Config.DEVICE, fold_save_dir, prefix="val")
        run_gradcam(base_model, train_loader, Config.DEVICE, fold_save_dir, prefix="train")

    print("\nAll folds processed successfully.")