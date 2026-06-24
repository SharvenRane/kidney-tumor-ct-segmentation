"""Datalist construction and MONAI transforms for kidney + tumor CT segmentation.

Labels: 0 background, 1 kidney, 2 tumor. Inputs are the aligned NIfTI pairs
produced by convert_to_nifti.py. Training is patch based so a full abdominal CT
never has to fit in VRAM at once.
"""
import json
import os

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

# Abdominal soft-tissue window covering kidney and renal-tumor HU on contrast CT.
HU_MIN, HU_MAX = -200.0, 300.0
PIXDIM = (1.5, 1.5, 1.5)


def build_datalist(dataset_json, val_frac=0.15, test_frac=0.15, seed=42):
    """Deterministic patient-level train/val/test split."""
    import random

    items = json.load(open(dataset_json))
    items = [{"image": it["image"], "label": it["label"], "patient_id": it["patient_id"]}
             for it in items]
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return {"train": train, "val": val, "test": test}


def train_transforms(roi=(96, 96, 96), num_samples=4):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=roi,
            pos=2, neg=1, num_samples=num_samples, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])


def val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
