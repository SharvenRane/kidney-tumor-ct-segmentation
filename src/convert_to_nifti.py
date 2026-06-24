"""Convert downloaded C4KC-KiTS DICOM + DICOM-SEG pairs into aligned NIfTI.

Real DICOM ingestion happens here and is validated case by case. We read the CT
series and the DICOM-SEG through the *same* MONAI PydicomReader so both land in
one RAS coordinate convention, combine the two binary SEG segments into a single
labelmap (0 background, 1 kidney, 2 tumor, tumor taking priority on any overlap),
put both on an identical grid with Orientation + ResampleToMatch, and write
image.nii.gz / label.nii.gz. Training then loads the cached NIfTI, which is fast
and sidesteps the DICOM slice-order and LPS/RAS pitfalls at every epoch.

Usage:
    python src/convert_to_nifti.py --in C:/Users/sharv/data/c4kc-kits \
        --out C:/Users/sharv/data/c4kc-kits-nifti
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from monai.data import MetaTensor, PydicomReader
from monai.transforms import EnsureChannelFirst, Orientation, ResampleToMatch

KIDNEY, TUMOR = 1, 2


def load_ct(ct_dir):
    # Pass the directory (not a file list) so the DICOM series is read as one
    # 3D volume; a list would be stacked channel-wise and fail on per-slice affines.
    reader = PydicomReader()
    arr, meta = reader.get_data(reader.read(ct_dir))
    img = EnsureChannelFirst(channel_dim="no_channel")(
        MetaTensor(torch.as_tensor(np.ascontiguousarray(arr)).float(), meta=meta)
    )
    return img


def combine_segments(arr):
    """Combine a (H, W, D, n_segments) binary SEG into a labelmap.

    0 background, 1 kidney, 2 tumor, with tumor taking priority on any overlap.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:  # single segment edge case
        arr = arr[..., None]
    lab = np.zeros(arr.shape[:3], dtype=np.uint8)
    lab[arr[..., 0] > 0] = KIDNEY
    if arr.shape[-1] > 1:
        lab[arr[..., 1] > 0] = TUMOR  # tumor overrides kidney on overlap
    return lab


def load_label(seg_file):
    reader = PydicomReader(label_dict={KIDNEY: "kidney", TUMOR: "tumor"})
    arr, meta = reader.get_data(reader.read(seg_file))  # (H, W, D, n_segments)
    lab = combine_segments(arr)
    label = EnsureChannelFirst(channel_dim="no_channel")(
        MetaTensor(torch.as_tensor(lab).float(), meta=meta)
    )
    return label


def convert_pair(ct_dir, seg_file):
    img = Orientation(axcodes="RAS")(load_ct(ct_dir))
    label = Orientation(axcodes="RAS")(load_label(seg_file))
    label = ResampleToMatch(mode="nearest")(label, img)
    return img, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    args = ap.parse_args()
    import nibabel as nib

    man = json.load(open(os.path.join(args.indir, "pairs_manifest.json")))
    os.makedirs(args.outdir, exist_ok=True)
    out_manifest = []
    for i, m in enumerate(man):
        pid = m["patient_id"]
        ct_dir = os.path.join(args.indir, m["ct_uid"])
        seg_file = glob.glob(os.path.join(args.indir, m["seg_uid"], "*.dcm"))[0]
        try:
            img, label = convert_pair(ct_dir, seg_file)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(man)}] {pid}: FAILED {e}")
            continue
        assert img.shape == label.shape, f"{pid} shape mismatch {img.shape} {label.shape}"
        lvals = torch.unique(label).tolist()
        n_kid = int((label == KIDNEY).sum())
        n_tum = int((label == TUMOR).sum())
        case_dir = os.path.join(args.outdir, pid)
        os.makedirs(case_dir, exist_ok=True)
        aff = img.affine.numpy()
        nib.save(nib.Nifti1Image(img[0].numpy().astype(np.float32), aff),
                 os.path.join(case_dir, "image.nii.gz"))
        nib.save(nib.Nifti1Image(label[0].numpy().astype(np.uint8), aff),
                 os.path.join(case_dir, "label.nii.gz"))
        out_manifest.append({**m, "image": os.path.join(case_dir, "image.nii.gz"),
                             "label": os.path.join(case_dir, "label.nii.gz"),
                             "shape": list(img.shape[1:]), "n_kidney": n_kid, "n_tumor": n_tum})
        print(f"[{i+1}/{len(man)}] {pid}: shape {tuple(img.shape[1:])} labels {lvals} "
              f"kidney={n_kid} tumor={n_tum}")
    with open(os.path.join(args.outdir, "dataset.json"), "w") as f:
        json.dump(out_manifest, f, indent=1)
    print(f"Converted {len(out_manifest)}/{len(man)} -> {args.outdir}/dataset.json")


if __name__ == "__main__":
    main()
