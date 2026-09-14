"""Save the model's test set predictions as NIfTI label maps on the evaluated 1.5 mm grid.

evaluate.py scores predictions in memory and keeps only the numbers. Surface and mesh
analysis (src/vtk_surfaces.py) needs the masks themselves, so this script reruns the same
sliding window inference on the same 32 held out cases and writes, per case:

    <out>/<patient_id>/label_1p5mm.nii.gz   clinician reference after Spacingd (nearest)
    <out>/<patient_id>/pred_1p5mm.nii.gz    model prediction (argmax)

It recomputes per case Dice and checks it against outputs/metrics.json, so the saved
masks are provably the ones behind the validation report.

Usage:
    python src/save_predictions.py --out C:/Users/sharv/data/c4kc-kits-pred --device cpu
"""
import argparse
import json
import os
import time

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

from data import build_datalist, val_transforms

ROI = (96, 96, 96)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C:/Users/sharv/data/c4kc-kits-nifti/dataset.json")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--metrics", default="outputs/metrics.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    test = build_datalist(args.dataset)["test"]
    validated = {c["patient_id"]: c for c in json.load(open(args.metrics))["per_case"]}

    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=3, init_filters=16,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = DataLoader(CacheDataset(test, val_transforms(), cache_rate=0.0, num_workers=0),
                        batch_size=1, shuffle=False, num_workers=0)
    rows = []
    for batch, item in zip(loader, test):
        pid = item["patient_id"]
        t0 = time.time()
        with torch.no_grad():
            logits = sliding_window_inference(batch["image"].to(device), ROI, 2, model,
                                              overlap=0.5, mode="gaussian")
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        gt = batch["label"][0, 0].numpy().astype(np.uint8)
        affine = batch["image"].affine[0].numpy() if batch["image"].affine.ndim == 3 else batch["image"].affine.numpy()
        case_dir = os.path.join(args.out, pid)
        os.makedirs(case_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(gt, affine), os.path.join(case_dir, "label_1p5mm.nii.gz"))
        nib.save(nib.Nifti1Image(pred, affine), os.path.join(case_dir, "pred_1p5mm.nii.gz"))
        row = {"patient_id": pid, "seconds": round(time.time() - t0, 1)}
        for ci, cname in ((1, "kidney"), (2, "tumor")):
            row[cname] = round(dice(pred == ci, gt == ci), 4)
            row[f"{cname}_validated"] = validated[pid][cname]["dice"]
        rows.append(row)
        print(json.dumps(row), flush=True)

    worst = max(abs(r[c] - r[f"{c}_validated"]) for r in rows for c in ("kidney", "tumor"))
    summary = {"n": len(rows), "device": args.device, "max_abs_dice_diff_vs_validated": round(worst, 4),
               "cases": rows}
    with open(os.path.join(args.out, "predictions_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {len(rows)} cases, max |Dice - validated| = {worst:.4f}")


if __name__ == "__main__":
    main()
