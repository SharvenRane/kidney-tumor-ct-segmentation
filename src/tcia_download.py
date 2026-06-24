"""Download C4KC-KiTS (kidney + tumor CT segmentation) from TCIA.

Each patient has one DICOM-SEG (kidney + tumor mask) and one or more multi-phase
CT series. The SEG references exactly one CT series through its
ReferencedSeriesSequence, so we download every SEG, read which CT it was drawn on,
and pull only that CT. This keeps image and mask perfectly registered and avoids
downloading the unused phases.

Usage:
    python src/tcia_download.py --out C:/Users/sharv/data/c4kc-kits --limit 3
    python src/tcia_download.py --out C:/Users/sharv/data/c4kc-kits          # all 210
"""
import argparse
import json
import os

import pydicom
from tcia_utils import nbia

COLLECTION = "C4KC-KiTS"


def referenced_ct_uid(seg_dir):
    """Read the SEG file in seg_dir and return the CT SeriesInstanceUID it references."""
    files = [f for f in os.listdir(seg_dir) if f.lower().endswith(".dcm")]
    ds = pydicom.dcmread(os.path.join(seg_dir, files[0]), stop_before_pixels=True)
    ref = ds.ReferencedSeriesSequence[0]
    return ref.SeriesInstanceUID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="number of patients (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    series = nbia.getSeries(collection=COLLECTION)
    seg_series = [s for s in series if s.get("Modality") == "SEG"]
    ct_by_uid = {s["SeriesInstanceUID"]: s for s in series if s.get("Modality") == "CT"}
    seg_series.sort(key=lambda s: s["PatientID"])
    if args.limit:
        seg_series = seg_series[: args.limit]

    print(f"Downloading {len(seg_series)} SEG series first...")
    nbia.downloadSeries(
        [s["SeriesInstanceUID"] for s in seg_series], input_type="list", path=args.out
    )

    manifest = []
    ct_to_get = []
    for s in seg_series:
        seg_dir = os.path.join(args.out, s["SeriesInstanceUID"])
        try:
            ct_uid = referenced_ct_uid(seg_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN {s['PatientID']}: cannot read SEG ref ({e}); skipping")
            continue
        ct_meta = ct_by_uid.get(ct_uid)
        manifest.append(
            {
                "patient_id": s["PatientID"],
                "sex": s.get("PatientSex"),
                "age": s.get("PatientAge"),
                "site": s.get("Site"),
                "manufacturer": ct_meta.get("Manufacturer") if ct_meta else None,
                "model": ct_meta.get("ManufacturerModelName") if ct_meta else None,
                "ct_uid": ct_uid,
                "seg_uid": s["SeriesInstanceUID"],
                "ct_resolved": ct_meta is not None,
            }
        )
        ct_to_get.append(ct_uid)

    print(f"Downloading {len(ct_to_get)} referenced CT series...")
    nbia.downloadSeries(ct_to_get, input_type="list", path=args.out)

    man_path = os.path.join(args.out, "pairs_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
    n_ok = sum(m["ct_resolved"] for m in manifest)
    print(f"Done. {len(manifest)} pairs ({n_ok} CT resolved). Manifest: {man_path}")


if __name__ == "__main__":
    main()
