"""Standalone (analytical) performance evaluation on the held-out test set.

This produces the analytical / standalone performance evidence an FDA-style
submission expects for an imaging algorithm: per-class Dice and 95th-percentile
Hausdorff distance, voxel-level sensitivity and specificity, all with
nonparametric 95% bootstrap confidence intervals, plus a subgroup breakdown by
sex, age band and scanner manufacturer drawn from the DICOM metadata. It writes
outputs/metrics.json and fills docs/VALIDATION_REPORT.md with the real numbers.

It is analytical performance only (algorithm vs the clinician DICOM-SEG reference
standard). Clinical validation (a reader study of the human-AI team) is out of
scope for a portfolio project and is described, not claimed, in the docs.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SegResNet
from monai.transforms import AsDiscrete, Compose

from data import PIXDIM, build_datalist, val_transforms

ROI = (96, 96, 96)
CLASSES = ["kidney", "tumor"]


def bootstrap_ci(values, n_boot=2000, seed=0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (round(lo, 4), round(hi, 4))


def sens_spec(pred_bin, gt_bin):
    tp = float((pred_bin & gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    tn = float((~pred_bin & ~gt_bin).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def age_band(age_str):
    try:
        a = int(str(age_str).rstrip("Yy").lstrip("0") or "0")
    except Exception:  # noqa: BLE001
        return "unknown"
    if a < 50:
        return "<50"
    if a < 65:
        return "50-64"
    return ">=65"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C:/Users/sharv/data/c4kc-kits-nifti/dataset.json")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--report", default="docs/VALIDATION_REPORT.md")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_by_pid = {it["patient_id"]: it for it in json.load(open(args.dataset))}
    split = build_datalist(args.dataset)
    test = split["test"]
    print(f"Evaluating {len(test)} held-out cases on {device}")

    ds = CacheDataset(test, val_transforms(), cache_rate=0.0, num_workers=0)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=3, init_filters=16,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
    post_label = Compose([AsDiscrete(to_onehot=3)])
    dice_m = DiceMetric(include_background=False, reduction="none")
    hd_m = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    per_case = []
    for batch, item in zip(loader, test):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = sliding_window_inference(x, ROI, 2, model, overlap=0.5, mode="gaussian")
        pred = [post_pred(p) for p in decollate_batch(logits)]
        lab = [post_label(l) for l in decollate_batch(y)]
        d = dice_m(y_pred=pred, y=lab)[0].tolist()
        # spacing makes HD95 millimetres; without it MONAI returns voxel units of the 1.5 mm grid
        h = hd_m(y_pred=pred, y=lab, spacing=PIXDIM)[0].tolist()
        pred_np = torch.argmax(logits, dim=1)[0].cpu().numpy()
        gt_np = y[0, 0].cpu().numpy()
        rec = {"patient_id": item["patient_id"]}
        for ci, cname in enumerate(CLASSES, start=1):
            s, sp = sens_spec(pred_np == ci, gt_np == ci)
            rec[cname] = {"dice": round(float(d[ci - 1]), 4),
                          "hd95": round(float(h[ci - 1]), 2),
                          "sensitivity": round(s, 4), "specificity": round(sp, 4)}
        m = meta_by_pid.get(item["patient_id"], {})
        rec["sex"] = m.get("sex"); rec["age_band"] = age_band(m.get("age"))
        rec["manufacturer"] = m.get("manufacturer")
        per_case.append(rec)
        print(rec["patient_id"], {c: rec[c]["dice"] for c in CLASSES})

    def agg(cname, metric, subset=None):
        vals = [c[cname][metric] for c in per_case
                if (subset is None or subset(c)) and not np.isnan(c[cname][metric])]
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 4), "ci95": bootstrap_ci(vals), "n": len(vals)}

    overall = {c: {m: agg(c, m) for m in ["dice", "hd95", "sensitivity", "specificity"]}
               for c in CLASSES}
    subgroups = {}
    for field, levels in [("sex", sorted({c["sex"] for c in per_case if c["sex"]})),
                          ("age_band", ["<50", "50-64", ">=65"]),
                          ("manufacturer", sorted({c["manufacturer"] for c in per_case
                                                   if c["manufacturer"]}))]:
        subgroups[field] = {
            lv: {c: agg(c, "dice", subset=lambda r, f=field, v=lv: r[f] == v) for c in CLASSES}
            for lv in levels
        }

    metrics = {"n_test": len(per_case), "classes": CLASSES, "overall": overall,
               "subgroups": subgroups, "per_case": per_case}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    write_report(args.report, metrics)
    print(f"Wrote {args.outdir}/metrics.json and {args.report}")


def write_report(path, m):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = []
    L.append("# Validation Report: Kidney + Tumor CT Segmentation\n")
    L.append("> Illustrative analytical (standalone) performance evaluation for a portfolio "
             "project. Not a regulatory submission. Reference standard: clinician DICOM-SEG "
             "annotations from the C4KC-KiTS collection.\n")
    L.append(f"\n**Held-out test cases:** {m['n_test']} (patient-level split, no patient in "
             "more than one partition).\n")
    L.append("\n## Standalone performance (algorithm vs reference standard)\n")
    L.append("| Class | Dice (95% CI) | HD95 mm (95% CI) | Sensitivity (95% CI) | "
             "Specificity (95% CI) |")
    L.append("|---|---|---|---|---|")
    for c in m["classes"]:
        o = m["overall"][c]
        def cell(x):
            return f"{x['mean']} ({x['ci95'][0]}-{x['ci95'][1]})" if x else "n/a"
        L.append(f"| {c} | {cell(o['dice'])} | {cell(o['hd95'])} | {cell(o['sensitivity'])} "
                 f"| {cell(o['specificity'])} |")
    L.append("\n## Subgroup performance (Dice)\n")
    L.append("Per CLAIM items 33/34/36 and FUTURE-AI fairness: performance stratified by sex, "
             "age band and scanner manufacturer to surface best- and worst-performing "
             "subpopulations.\n")
    for field, levels in m["subgroups"].items():
        L.append(f"\n**By {field}**\n")
        L.append("| " + field + " | " + " | ".join(m["classes"]) + " |")
        L.append("|" + "---|" * (len(m["classes"]) + 1))
        for lv, byc in levels.items():
            cells = []
            for c in m["classes"]:
                x = byc[c]
                cells.append(f"{x['mean']} (n={x['n']})" if x else "n/a")
            L.append(f"| {lv} | " + " | ".join(cells) + " |")
    L.append("\n## Notes and limitations\n")
    L.append("- This is **analytical/standalone** performance only. Clinical validation of the "
             "human-AI team (an MRMC reader study, the primary evaluation for imaging aids per "
             "FDA's Jan 2025 draft guidance) is described in the model card, not performed here.\n")
    L.append("- The reference standard is semiautomatic clinician segmentation; inter-reader "
             "variability is a known source of label noise.\n")
    L.append("- The test set is single-collection; multi-site, multi-scanner external "
             "validation would be required before any clinical claim.\n")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
