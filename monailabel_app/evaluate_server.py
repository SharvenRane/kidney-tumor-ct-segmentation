"""Drive a running MONAI Label server the way an annotation client does, and measure it.

For each held out test CT this script:
  1. asks the server for a pre segmentation over the REST API (POST /infer/kidney_tumor)
  2. scores the returned label, in the ORIGINAL CT grid, against the clinician reference
  3. saves a label back (PUT /datastore/label), standing in for a radiologist's corrected mask
and records request latency. It then asks the active learning endpoint for the next study and
checks the datastore counts the saved labels.

    python monailabel_app/evaluate_server.py --server http://127.0.0.1:8000 \
        --dataset <c4kc-kits-nifti/dataset.json> --out outputs/monailabel_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import build_datalist  # noqa: E402

CLASSES = {"kidney": 1, "tumor": 2}


def dice(pred: np.ndarray, ref: np.ndarray, cls: int) -> float:
    p, r = pred == cls, ref == cls
    denom = p.sum() + r.sum()
    return float(2.0 * (p & r).sum() / denom) if denom else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--path-map", nargs=2, metavar=("FROM", "TO"),
                    help="rewrite a path prefix in dataset.json, e.g. a Windows drive to a WSL mount")
    ap.add_argument("--save-labels", type=int, default=3, help="how many corrected labels to submit")
    ap.add_argument("--out", default="outputs/monailabel_eval.json")
    args = ap.parse_args()
    s = requests.Session()

    info = s.get(f"{args.server}/info/").json()
    test = build_datalist(args.dataset)["test"]
    fix = (lambda p: p.replace("\\", "/").replace(args.path_map[0], args.path_map[1])) if args.path_map else (lambda p: p)
    everything = s.get(f"{args.server}/datastore/", params={"output": "all"}).json()
    listed = set(everything.get("objects", everything).keys())

    rows = []
    for i, item in enumerate(test):
        pid = item["patient_id"]
        if pid not in listed:
            raise RuntimeError(f"{pid} is not in the server datastore")
        t0 = time.perf_counter()
        r = s.post(f"{args.server}/infer/kidney_tumor", params={"image": pid, "output": "image"},
                   data={"params": json.dumps({})})
        seconds = time.perf_counter() - t0
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            f.write(r.content)
            pred_path = f.name
        pred_img = nib.load(pred_path)
        ref_img = nib.load(fix(item["label"]))
        pred = np.asarray(pred_img.dataobj).astype(np.int16).squeeze()
        ref = np.asarray(ref_img.dataobj).astype(np.int16)
        same_grid = pred.shape == ref.shape and np.allclose(pred_img.affine, ref_img.affine, atol=1e-3)
        row = {"patient_id": pid, "seconds": seconds, "shape": list(ref.shape), "same_grid_as_reference": bool(same_grid),
               **{f"dice_{c}": dice(pred, ref, k) if pred.shape == ref.shape else None for c, k in CLASSES.items()}}
        rows.append(row)
        print(f"[{i + 1}/{len(test)}] {pid} {seconds:.1f}s kidney {row['dice_kidney']:.4f} tumor {row['dice_tumor']:.4f} grid ok {same_grid}", flush=True)

        if i < args.save_labels:
            with open(fix(item["label"]), "rb") as lf:
                put = s.put(f"{args.server}/datastore/label", params={"image": pid, "tag": "final"},
                            files={"label": (f"{pid}.nii.gz", lf, "application/octet-stream")},
                            data={"params": json.dumps({})})
            put.raise_for_status()
        os.remove(pred_path)

    stats = s.get(f"{args.server}/datastore/", params={"output": "stats"}).json()
    nxt = s.post(f"{args.server}/activelearning/random", json={}).json()
    nxt.pop("path", None)  # server side filesystem path, not useful outside this machine

    def agg(key):
        v = np.array([r[key] for r in rows if r[key] is not None and not np.isnan(r[key])])
        return {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(len(v))}

    secs = np.array([r["seconds"] for r in rows])
    out = {
        "server_models": list(info.get("models", {}).keys()),
        "n_cases": len(rows),
        "all_labels_on_reference_grid": all(r["same_grid_as_reference"] for r in rows),
        "dice_kidney_original_space": agg("dice_kidney"),
        "dice_tumor_original_space": agg("dice_tumor"),
        "seconds_per_request": {"median": float(np.median(secs)), "p95": float(np.percentile(secs, 95)),
                                "first": float(secs[0])},
        "labels_saved": args.save_labels,
        "datastore_stats_after": stats,
        "next_sample_from_random_strategy": nxt,
        "per_case": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_case"}, indent=1))


if __name__ == "__main__":
    main()
