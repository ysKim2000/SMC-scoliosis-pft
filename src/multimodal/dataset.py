import os
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt


#  Image Preprocessing Utilities
def apply_windowing(pixel_array, center, width, invert=False):
    """Window/level to 8-bit."""
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(pixel_array, lower, upper)
    windowed = (windowed - lower) / (upper - lower) * 255.0
    img = windowed.astype(np.uint8)
    if invert:
        img = 255 - img
    return img

def apply_rescale(arr, ds):
    """Apply DICOM RescaleSlope/Intercept if present."""
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter

def _to_float(x):
    if isinstance(x, pydicom.multival.MultiValue):
        return float(x[0])
    return float(x)

def make_top_roi_mask(h, w, top_band_ratio=0.22, corner_ratio=0.22):
    """
    Restrict the ROI to the upper region where burned-in text usually appears:
    - the top band
    - the top-left and top-right corner boxes
    """
    roi = np.zeros((h, w), dtype=np.uint8)
    top_h = int(h * top_band_ratio)
    c = int(min(h, w) * corner_ratio)

    roi[:top_h, :] = 255        # top band
    roi[:c, :c] = 255           # top-left
    roi[:c, w-c:] = 255         # top-right
    return roi

def build_text_mask_roi_strong(img_u8, roi_u8,
                               k1=31, k2=51, pct=98.5,
                               close_k=3, dilate_k=7, dilate_iter=2,
                               min_area=10, max_area=12000):
    """
    Within the ROI only:
    - enhance bright characters with a multi-scale top-hat
    - select candidate text pixels with a percentile threshold
    - close and dilate to cover the anti-aliased fringe
    - drop overly large connected components (anatomy, not text)
    """
    h, w = img_u8.shape
    img_roi = img_u8.copy()
    img_roi[roi_u8 == 0] = 0

    blur = cv2.GaussianBlur(img_roi, (3, 3), 0)

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    th1 = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, se1)
    th2 = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, se2)
    tophat = cv2.max(th1, th2)

    vals = tophat[roi_u8 > 0]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return np.zeros_like(img_u8, dtype=np.uint8)

    thr = np.percentile(vals, pct)
    bw = (tophat >= thr).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    bw = cv2.dilate(bw, k_d, iterations=dilate_iter)

    bw[roi_u8 == 0] = 0

    # component filter
    mask = np.zeros((h, w), dtype=np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if ww > 0.6*w or hh > 0.35*h:  # discard components too large to be text
            continue
        mask[lab == i] = 255

    return mask

def inpaint_text(img_u8, mask_u8, radius=6, method="telea"):
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img_u8, mask_u8, inpaintRadius=radius, flags=flag)

def remove_burned_in_text_inpaint(
    img_u8,
    top_band_ratio=0.22,
    corner_ratio=0.22,
    pct=98.5,
    tophat_k1=31,
    tophat_k2=51,
    close_k=3,
    dilate_k=7,
    dilate_iter=2,
    inpaint_radius=6,
    method="telea",
    two_pass=True,
):
    """
    Remove burned-in text from an 8-bit radiograph by inpainting, searching the
    upper ROI only.
    """
    h, w = img_u8.shape
    roi = make_top_roi_mask(h, w, top_band_ratio, corner_ratio)

    mask1 = build_text_mask_roi_strong(
        img_u8, roi, k1=tophat_k1, k2=tophat_k2, pct=pct,
        close_k=close_k, dilate_k=dilate_k, dilate_iter=dilate_iter
    )
    out1 = inpaint_text(img_u8, mask1, radius=inpaint_radius, method=method)

    if not two_pass:
        return out1

    # Second pass, to catch the residual grey fringe left by the first
    mask2 = build_text_mask_roi_strong(
        out1, roi, k1=tophat_k1, k2=tophat_k2, pct=min(pct, 98.8),
        close_k=close_k, dilate_k=dilate_k, dilate_iter=max(1, dilate_iter)
    )
    mask_all = cv2.bitwise_or(mask1, mask2)
    out2 = inpaint_text(img_u8, mask_all, radius=inpaint_radius, method=method)
    return out2



def apply_clahe(img):
    """CLAHE for local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def histogram_match(source, reference):
    """Histogram match grayscale images."""
    return match_histograms(source, reference, channel_axis=None)


#  Dataset
class AISTabularXRDataset(Dataset):
    """
    DICOM + Tabular + Dual Labels(FVC_class, FEV1_class).
    """
    def __init__(
        self,
        image_dir,
        tabular_path,
        label_path,
        transform=None,
        reference_dcm_path=None,
        image_size=(512, 512),
        verbose=True,
    ):
        self.image_dir = image_dir
        self.tabular_df = pd.read_csv(tabular_path)
        self.label_df   = pd.read_csv(label_path)
        self.transform  = transform or transforms.ToTensor()
        self.image_size = image_size
        self.failed_patient_ids = {}

        # Merge: Tabular + Labels
        df = pd.merge(self.tabular_df, self.label_df, on="Patient_ID", how="left")

        # Map DICOM files
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(".dcm")]
        # Key on the first 8 characters (zero-padded Patient_ID)
        self.image_paths = {f[:8]: os.path.join(image_dir, f) for f in files}

        # Standardize Patient_ID -> 8-digit zero-padded string
        def _pid_str(x):
            # Accept float, int, or str
            try:
                return str(int(float(x))).zfill(8)
            except Exception:
                return str(x).strip().zfill(8)

        df["Patient_ID_str"] = df["Patient_ID"].apply(_pid_str)

        # Filter: Only patients with matching DICOM
        valid_mask = df["Patient_ID_str"].isin(self.image_paths)
        self.failed_patient_ids.update(
            {pid: "DICOM file not found" for pid in df.loc[~valid_mask, "Patient_ID_str"]}
        )
        df = df.loc[valid_mask].reset_index(drop=True)

        # Check that labels exist
        missing_labels = df["FVC_class"].isna() | df["FEV1_class"].isna()
        if missing_labels.any():
            for pid in df.loc[missing_labels, "Patient_ID_str"]:
                self.failed_patient_ids[pid] = "Missing FVC_class / FEV1_class"
            df = df.loc[~missing_labels].reset_index(drop=True)

        # Build feature list (auto)
        exclude_cols = {"Patient_ID", "Patient_ID_str", "FVC_class", "FEV1_class"}
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Coerce feature dtypes to numeric
        df_features = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        df_features = df_features.fillna(0.0).astype(np.float32)
        df[self.feature_cols] = df_features

        self.df = df

        # Load reference DICOM for histogram matching (optional)
        self.reference_image = None
        if reference_dcm_path and os.path.exists(reference_dcm_path):
            try:
                ref_ds = pydicom.dcmread(reference_dcm_path)
                ref_arr = ref_ds.pixel_array.astype(np.float32)
                ref_arr = apply_rescale(ref_arr, ref_ds)
                
                if hasattr(ref_ds, "WindowCenter") and hasattr(ref_ds, "WindowWidth"):
                    wc = ref_ds.WindowCenter
                    ww = ref_ds.WindowWidth
                    center = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
                    width  = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
                else:
                    center, width = ref_arr.mean(), float(np.ptp(ref_arr))
                invert = (getattr(ref_ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1")
                ref_img = apply_windowing(ref_arr, center, width, invert)
                ref_img = remove_burned_in_text_inpaint(ref_img)
                self.reference_image = ref_img
            except Exception as e:
                print(f"[Warning] reference DICOM load failed: {e}")

        if verbose:
            print(f"[AISTabularXRDataset] Loaded {len(self.df)} patients.")
            print(f"[AISTabularXRDataset] Num tabular features: {len(self.feature_cols)}")
            print(f"[AISTabularXRDataset] Feature columns: {self.feature_cols}")

    def __len__(self):
        return len(self.df)

    def _load_process_dicom(self, dcm_path):
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)

        # Window/level (fallback to full dynamic range if missing)
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            center = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
            width  = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
        else:
            center, width = arr.mean(), float(np.ptp(arr))
        invert = (getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1")
        
        img = apply_windowing(arr, center, width, invert)
        
        img = remove_burned_in_text_inpaint(
            img,
            top_band_ratio=0.22,
            corner_ratio=0.22,
            pct=98.5,
            tophat_k1=31,
            tophat_k2=51,
            close_k=3,
            dilate_k=7,
            dilate_iter=2,
            inpaint_radius=6,
            method="telea",
            two_pass=True,
        )

        # Histogram matching (optional)
        if self.reference_image is not None:
            img = histogram_match(img, self.reference_image).astype(np.uint8)

        # CLAHE
        img = apply_clahe(img)
        
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient_ID_str"]
        dcm_path = self.image_paths[pid]

        try:
            # Image
            img = self._load_process_dicom(dcm_path)

            # Resize & ToTensor
            img = Image.fromarray(img).resize(self.image_size)
            img = self.transform(img)  # [1,H,W] or [C,H,W] depending on transform

            # Tabular
            tab_vals = row[self.feature_cols].values.astype(np.float32)
            tabular = torch.from_numpy(tab_vals)

            # Labels
            labels = {
                "FVC_class":  torch.tensor(int(row["FVC_class"]),  dtype=torch.long),
                "FEV1_class": torch.tensor(int(row["FEV1_class"]), dtype=torch.long),
            }

            return img, tabular, labels, pid

        except Exception as e:
            self.failed_patient_ids[pid] = str(e)
            return None


#  Collate
def custom_collate(batch):
    """Filter out failed samples (None) and stack."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None

    imgs, tabs, labs, pids = zip(*batch)

    imgs_tensor = torch.stack(imgs)
    tabs_tensor = torch.stack(tabs)

    labels_dict = {
        "FVC_class":  torch.stack([l["FVC_class"]  for l in labs]),
        "FEV1_class": torch.stack([l["FEV1_class"] for l in labs]),
    }
    return imgs_tensor, tabs_tensor, labels_dict, pids


#  Visualization Utility: Grid of Sample PNGs
def save_grid_plot(sample_dir, out_path, rows=3, cols=4):
    imgs = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".png")])[: rows * cols]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for ax, fname in zip(axes.flatten(), imgs):
        img = Image.open(os.path.join(sample_dir, fname))
        ax.imshow(img, cmap="gray")
        ax.set_title(fname, fontsize=8)
        ax.axis("off")

    for ax in axes.flatten()[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved grid plot to {out_path}")


# Main (example usage)
if __name__ == "__main__":
    # Paths
    image_dir    = "data/images"
    tabular_path = "data/ais_tabular_sampling.csv"  
    label_path   = "data/ais_labels.csv"
    reference_dcm_path = "data/images/<reference_study>.dcm"

    save_dir = "samples/new_preprocessing"
    os.makedirs(save_dir, exist_ok=True)

    # Load the dataset
    dataset = AISTabularXRDataset(
        image_dir=image_dir,
        tabular_path=tabular_path,
        label_path=label_path,
        transform=transforms.ToTensor(),  # yields a [1, H, W] grayscale tensor
        reference_dcm_path=reference_dcm_path,
        verbose=True,
    )

    # Data loader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,  # keep at 0 when debugging DICOM I/O errors
    )

    # Save sample images
    for imgs, tabs, labels, pids in dataloader:
        if imgs is None:
            continue
        for img, pid in zip(imgs, pids):
            out = transforms.ToPILImage()(img)
            out.save(os.path.join(save_dir, f"{pid}.png"))
            print(f"Saved: {pid}.png")

    # Report patients that failed to load
    if dataset.failed_patient_ids:
        print("\nFailed Patient_IDs with reasons:")
        for pid, reason in dataset.failed_patient_ids.items():
            print(f"  {pid}: {reason}")
    else:
        print("\nAll patients processed successfully.")

    # Sample grid for the figure
    sample_dir = "samples/figure"
    os.makedirs(sample_dir, exist_ok=True)
    out_path   = os.path.join(sample_dir, "histogram_clahe_plot.png")
    save_grid_plot(save_dir, out_path, rows=3, cols=4)

    # Dataset summary
    total = len(dataset)
    fvc_dist  = dataset.df["FVC_class"].value_counts().to_dict()
    fev1_dist = dataset.df["FEV1_class"].value_counts().to_dict()
    print(f"\nDataset Summary:")
    print(f"  Total patients: {total}")
    print(f"  FVC_class distribution: {fvc_dist}")
    print(f"  FEV1_class distribution: {fev1_dist}")
    print(f"  Num tabular features: {len(dataset.feature_cols)}")
