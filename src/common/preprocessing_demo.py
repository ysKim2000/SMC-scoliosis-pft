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


def apply_windowing(pixel_array, center, width, invert=False):
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(pixel_array, lower, upper)
    windowed = (windowed - lower) / (upper - lower + 1e-8) * 255.0
    img = windowed.astype(np.uint8)
    if invert:
        img = 255 - img
    return img


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def histogram_match(source, reference):
    return match_histograms(source, reference, channel_axis=None)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_preprocessing_steps(dcm_path, pid, out_root, reference_image=None,
                             size=(512, 512), save_raw_minmax=True):
    """
    Save the output of each preprocessing stage as a PNG.
    Output layout:
      preprocessing/0_raw_minmax/ (optional)
      preprocessing/1_windowing/
      preprocessing/2_histogram_matching/
      preprocessing/3_clahe/
      preprocessing/4_resized_512x512/
    """
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)

        # 0) raw, min-max normalized for display (optional)
        if save_raw_minmax:
            raw_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0).astype(np.uint8)
            d0 = os.path.join(out_root, "1_raw_minmax")
            ensure_dir(d0)
            Image.fromarray(raw_norm).save(os.path.join(d0, f"minmax_{pid}.png"))

        # Window parameters
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            wc = ds.WindowCenter; ww = ds.WindowWidth
            center = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
            width  = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
        else:
            center, width = float(arr.mean()), float(np.ptp(arr))
        invert = (getattr(ds, 'PhotometricInterpretation','').upper() == "MONOCHROME1")

        # 1) windowing
        img_win = apply_windowing(arr, center, width, invert)
        d1 = os.path.join(out_root, "2_windowing")
        ensure_dir(d1)
        Image.fromarray(img_win).save(os.path.join(d1, f"windowing_{pid}.png"))

        # 2) histogram matching (without a reference, save the windowed image as is)
        if reference_image is not None:
            img_hm = histogram_match(img_win, reference_image).astype(np.uint8)
        else:
            img_hm = img_win.copy()
        d2 = os.path.join(out_root, "3_histogram_matching")
        ensure_dir(d2)
        Image.fromarray(img_hm).save(os.path.join(d2, f"histogram_match_{pid}.png"))

        # 3) CLAHE
        img_clahe = apply_clahe(img_hm)
        d3 = os.path.join(out_root, "4_clahe")
        ensure_dir(d3)
        Image.fromarray(img_clahe).save(os.path.join(d3, f"clahe_{pid}.png"))

        # 4) resize (512x512)
        img_resized = Image.fromarray(img_clahe).resize(size)
        d4 = os.path.join(out_root, f"5_resized_{size[0]}x{size[1]}")
        ensure_dir(d4)
        img_resized.save(os.path.join(d4, f"resized_{size[0]}x{size[1]}_{pid}.png"))

        return True, None
    except Exception as e:
        return False, str(e)


class AISTabularXRDataset(Dataset):
    def __init__(self, image_dir, tabular_path, label_path, transform=None, reference_dcm_path=None):
        self.image_dir = image_dir
        self.tabular_df = pd.read_csv(tabular_path)
        self.label_df   = pd.read_csv(label_path)
        self.transform  = transform or transforms.ToTensor()

        # merge tabular + label
        df = pd.merge(self.tabular_df, self.label_df, on="Patient_ID")

        # Map DICOM files
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(".dcm")]
        self.image_paths = {f[:8]: os.path.join(image_dir, f) for f in files}

        # Normalize Patient_ID to a string and filter to valid patients
        df['Patient_ID_str'] = df['Patient_ID'].apply(lambda x: str(int(float(x))).zfill(8))
        valid_mask = df['Patient_ID_str'].isin(self.image_paths)
        self.failed_patient_ids = {pid: "DICOM file not found" for pid in df.loc[~valid_mask, 'Patient_ID_str']}
        self.df = df.loc[valid_mask].reset_index(drop=True)

        # Load the reference DICOM used for histogram matching
        self.reference_image = None
        if reference_dcm_path and os.path.exists(reference_dcm_path):
            try:
                ref_ds = pydicom.dcmread(reference_dcm_path)
                ref_arr = ref_ds.pixel_array.astype(np.float32)
                if hasattr(ref_ds, 'WindowCenter') and hasattr(ref_ds, 'WindowWidth'):
                    wc = ref_ds.WindowCenter; ww = ref_ds.WindowWidth
                    center = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
                    width  = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
                else:
                    center, width = float(ref_arr.mean()), float(np.ptp(ref_arr))
                invert = (getattr(ref_ds, 'PhotometricInterpretation','').upper() == "MONOCHROME1")
                ref_img = apply_windowing(ref_arr, center, width, invert)
                self.reference_image = ref_img
            except Exception as e:
                print(f"[Warning] reference DICOM load failed: {e}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['Patient_ID_str']
        dcm_path = self.image_paths[pid]

        try:
            ds = pydicom.dcmread(dcm_path)
            arr = ds.pixel_array.astype(np.float32)

            # windowing
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc = ds.WindowCenter; ww = ds.WindowWidth
                center = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
                width  = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
            else:
                center, width = float(arr.mean()), float(np.ptp(arr))
            invert = (getattr(ds, 'PhotometricInterpretation','').upper() == "MONOCHROME1")
            img = apply_windowing(arr, center, width, invert)

            # histogram matching
            if self.reference_image is not None:
                img = histogram_match(img, self.reference_image).astype(np.uint8)

            # CLAHE
            img = apply_clahe(img)

            # resize & to tensor
            img = Image.fromarray(img).resize((512, 512))
            img = self.transform(img)

            # tabular features
            exclude = ["Patient_ID","FVC_class","FEV1_class","Patient_ID_str"]
            tab_vals = row.drop(labels=exclude).astype(np.float32).values
            tabular = torch.tensor(tab_vals)

            labels = {
                "FVC_class":  torch.tensor(int(row["FVC_class"]),  dtype=torch.long),
                "FEV1_class": torch.tensor(int(row["FEV1_class"]), dtype=torch.long),
            }

            return img, tabular, labels, pid

        except Exception as e:
            self.failed_patient_ids[pid] = str(e)
            return None

def custom_collate(batch):
    # Drop failed samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None

    # Unpack imgs, tabs, labs, pids
    imgs, tabs, labs, pids = zip(*batch)

    imgs_tensor = torch.stack(imgs)
    tabs_tensor = torch.stack(tabs)
    labels_dict = {
        "FVC_class":  torch.stack([l["FVC_class"]  for l in labs]),
        "FEV1_class": torch.stack([l["FEV1_class"] for l in labs]),
    }
    return imgs_tensor, tabs_tensor, labels_dict, pids


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
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved grid plot to {out_path}")


if __name__ == "__main__":
    image_dir    = "data/images"
    tabular_path = "data/ais_tabular.csv"
    label_path   = "data/ais_labels.csv"
    reference_dcm_path  = "data/images/<reference_study>.dcm"

    # --- Dataset ---
    dataset = AISTabularXRDataset(
        image_dir, tabular_path, label_path,
        transform=transforms.ToTensor(),
        reference_dcm_path=reference_dcm_path
    )

    # --- Save each preprocessing stage ---
    out_root = "preprocessing_"
    ensure_dir(out_root)
    failures = {}

    for _, row in dataset.df.iterrows():
        pid = row["Patient_ID_str"]
        dcm_path = dataset.image_paths[pid]
        ok, reason = save_preprocessing_steps(
            dcm_path=dcm_path,
            pid=pid,
            out_root=out_root,
            reference_image=dataset.reference_image,
            size=(512, 512),
            save_raw_minmax=True
        )
        if not ok:
            failures[pid] = reason
            print(f"[Fail] {pid}: {reason}")

    if failures:
        print("\nFailed Patient_IDs with reasons:")
        for pid, reason in failures.items():
            print(f"  {pid}: {reason}")
    else:
        print("\nAll patients processed successfully (preprocessing steps saved).")

    # --- Optional: build a sample grid for a given stage ---
    # Example: a grid of histogram-matched samples
    sample_dir = os.path.join(out_root, "2_histogram_matching")
    ensure_dir("samples/figure")
    out_path   = os.path.join("samples/figure", "histogram_clahe_plot.png")
    save_grid_plot(sample_dir, out_path, rows=3, cols=4)

    # --- Dataset summary ---
    total = len(dataset)
    fvc_dist  = dataset.df["FVC_class"].value_counts().to_dict()
    fev1_dist = dataset.df["FEV1_class"].value_counts().to_dict()
    print(f"\nDataset Summary:")
    print(f"  Total patients: {total}")
    print(f"  FVC_class distribution: {fvc_dist}")
    print(f"  FEV1_class distribution: {fev1_dist}")
