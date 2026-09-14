"""Resampling to the 1.5 mm training grid with ITK, measured against MONAI's Spacingd.

Training and evaluate.py resample every CT to 1.5 mm isotropic with MONAI Spacingd (bilinear
image, nearest label). This script redoes that step with ITK's ResampleImageFilter and answers
three questions on the 32 held out cases:

1. Do ITK and MONAI produce the same 1.5 mm grid and the same voxels? (grid, HU, label agreement)
2. How much does resampling change the clinician reference itself? Label volumes in ml before and
   after, and the Dice of the reference against itself after a round trip to 1.5 mm and back, for
   nearest neighbour and for ITK's LabelImageGaussianInterpolateImageFunction.
3. How much of the gap between Dice on the 1.5 mm grid (evaluate.py) and Dice on the original CT
   grid (the MONAI Label evaluation) comes from the grid? The saved predictions are restored to
   the original grid and scored there.

Usage:
    python src/itk_resample.py --pred-dir C:/Users/sharv/data/c4kc-kits-pred --out outputs/itk_resample.json
"""
import argparse
import json
import os
import sys
import time

import itk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KIDNEY, TUMOR = 1, 2
CLASSES = ((KIDNEY, "kidney"), (TUMOR, "tumor"))
OUT_SPACING = 1.5
GAUSS_SIGMA_MM = 0.75  # half the output voxel
GAUSS_ALPHA = 3.0  # kernel cut off at three sigma


def interpolator(image, kind):
    if kind == "nearest":
        return itk.NearestNeighborInterpolateImageFunction.New(image)
    if kind == "linear":
        return itk.LinearInterpolateImageFunction.New(image)
    if kind == "label_gaussian":
        f = itk.LabelImageGaussianInterpolateImageFunction.New(image)
        f.SetSigma([GAUSS_SIGMA_MM] * 3)
        f.SetAlpha(GAUSS_ALPHA)
        return f
    raise ValueError(kind)


def isotropic_grid(image, spacing=OUT_SPACING):
    """The grid MONAI Spacingd produces: same origin and direction, and a size that spans the input
    voxel centres, round((n - 1) * spacing_in / spacing_out) + 1 (monai.data.utils.compute_shape_offset)."""
    size = np.round((np.array(itk.size(image)) - 1) * np.array(itk.spacing(image)) / spacing + 1.0).astype(int)
    return {"spacing": [spacing] * 3, "origin": itk.origin(image), "direction": image.GetDirection(),
            "size": [int(s) for s in size]}


def grid_of(image):
    return {"spacing": list(itk.spacing(image)), "origin": itk.origin(image), "direction": image.GetDirection(),
            "size": [int(s) for s in itk.size(image)]}


def pad_border(image, width=1):
    """Replicate edge voxels outward, which is MONAI Spacingd's default padding_mode="border".

    The 1.5 mm grid spans voxel centres, so its last row can fall up to 1.5 mm past the last input
    voxel centre. ITK treats that as outside the image and returns 0; MONAI clamps to the edge.
    Padding keeps physical coordinates, so resampling from the padded image matches MONAI there.
    """
    f = itk.ZeroFluxNeumannPadImageFilter.New(image)
    f.SetPadLowerBound([width] * 3)
    f.SetPadUpperBound([width] * 3)
    f.Update()
    return f.GetOutput()


def resample(image, grid, kind, border=None):
    # Border padding exists to reproduce MONAI, so it applies to nearest and linear only. With the
    # padded input the label Gaussian interpolator gave wrong labels (tumor round trip Dice on
    # KiTS-00037 fell from 0.976 to 0.61), so it runs on the unpadded image.
    if border is None:
        border = kind != "label_gaussian"
    if border:
        image = pad_border(image)
    f = itk.ResampleImageFilter.New(image)
    f.SetInterpolator(interpolator(image, kind))
    f.SetOutputSpacing(grid["spacing"])
    f.SetOutputOrigin(grid["origin"])
    f.SetOutputDirection(grid["direction"])
    f.SetSize(grid["size"])
    f.SetDefaultPixelValue(0)
    f.Update()
    return f.GetOutput()


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def volume_ml(arr, image, c):
    return float((arr == c).sum() * np.prod(itk.spacing(image)) / 1000.0)


def monai_resample(case_dir):
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Spacingd

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(OUT_SPACING,) * 3, mode=("bilinear", "nearest")),
    ])({"image": os.path.join(case_dir, "image.nii.gz"), "label": os.path.join(case_dir, "label.nii.gz")})


def monai_to_itk_order(arr):
    """MONAI arrays are (x, y, z) in RAS; ITK arrays of the same NIfTI are (z, y, x)."""
    return np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))


def as_image(arr, like_grid):
    img = itk.image_from_array(np.ascontiguousarray(arr))
    img.SetSpacing(like_grid["spacing"])
    img.SetOrigin(like_grid["origin"])
    img.SetDirection(like_grid["direction"])
    return img


def analyse_case(case_dir, pred_dir=None):
    img = itk.imread(os.path.join(case_dir, "image.nii.gz"), itk.F)
    lab = itk.imread(os.path.join(case_dir, "label.nii.gz"), itk.UC)
    # The cache from convert_to_nifti.py is RAS, so MONAI's Orientationd is a no op and the two
    # pipelines resample the same grid. Anything else would make the comparison meaningless.
    if not np.allclose(itk.array_from_matrix(lab.GetDirection()), np.diag([-1.0, -1.0, 1.0]), atol=1e-4):
        raise ValueError(f"{case_dir}: expected a RAS oriented NIfTI as written by convert_to_nifti.py")
    lab_arr = itk.array_from_image(lab)
    orig_grid = grid_of(lab)
    iso = isotropic_grid(lab)
    row = {"original_spacing_mm": [round(s, 4) for s in itk.spacing(lab)], "iso_size": iso["size"]}

    t0 = time.time()
    img_itk = resample(img, iso, "linear")
    lab_nn = resample(lab, iso, "nearest")
    row["seconds_itk_linear_plus_nearest"] = round(time.time() - t0, 2)
    t0 = time.time()
    lab_gauss = resample(lab, iso, "label_gaussian")
    row["seconds_itk_label_gaussian"] = round(time.time() - t0, 2)
    t0 = time.time()
    m = monai_resample(case_dir)
    row["seconds_monai_load_orientation_spacingd"] = round(time.time() - t0, 2)

    m_img = monai_to_itk_order(m["image"][0].numpy())
    m_lab = monai_to_itk_order(m["label"][0].numpy()).astype(np.uint8)
    m_aff = m["image"].affine.numpy()
    # MONAI's RAS affine origin, flipped to LPS, must equal the ITK grid origin.
    monai_origin_lps = np.array([-m_aff[0, 3], -m_aff[1, 3], m_aff[2, 3]])
    row["grid_origin_diff_mm"] = round(float(np.linalg.norm(monai_origin_lps - np.array(iso["origin"]))), 5)
    row["grid_shape_equal"] = list(m_img.shape) == list(itk.array_view_from_image(img_itk).shape)

    nn_arr, gauss_arr = itk.array_from_image(lab_nn), itk.array_from_image(lab_gauss)
    if row["grid_shape_equal"]:
        diff = np.abs(itk.array_view_from_image(img_itk) - m_img)
        row["hu_abs_diff_itk_vs_monai"] = {"max": round(float(diff.max()), 4), "mean": round(float(diff.mean()), 6)}
        disagree = nn_arr != m_lab
        row["label_voxels_disagree_itk_nearest_vs_monai"] = int(np.count_nonzero(disagree))
        # Nearest neighbour ties: output voxels whose continuous input index is exactly k + 0.5 on
        # some axis. ITK rounds those up, torch's nearest mode rounds half to even.
        cont = [np.arange(n) * OUT_SPACING / s for n, s in zip(iso["size"], itk.spacing(lab))]  # x, y, z
        tie = [np.isclose(c % 1.0, 0.5, atol=1e-6) for c in cont]
        tie_mask = tie[2][:, None, None] | tie[1][None, :, None] | tie[0][None, None, :]
        row["label_voxels_disagree_not_at_half_voxel_ties"] = int(np.count_nonzero(disagree & ~tie_mask))

    methods = {"monai_nearest": m_lab, "itk_nearest": nn_arr, "itk_label_gaussian": gauss_arr}
    for c, name in CLASSES:
        v0 = volume_ml(lab_arr, lab, c)
        entry = {"original_ml": round(v0, 3)}
        for mname, arr in methods.items():
            v = float((arr == c).sum() * OUT_SPACING ** 3 / 1000.0)
            entry[f"{mname}_ml"] = round(v, 3)
            entry[f"{mname}_volume_error_pct"] = round(100.0 * (v - v0) / v0, 3) if v0 else None
        row[name] = entry

    # Round trip of the reference itself: 1.5 mm and back to the original grid.
    for mname, arr, kind in (("monai_nearest", m_lab, "nearest"), ("itk_nearest", nn_arr, "nearest"),
                             ("itk_label_gaussian", gauss_arr, "label_gaussian")):
        back = itk.array_from_image(resample(as_image(arr, iso), orig_grid, kind))
        for c, name in CLASSES:
            row[name][f"{mname}_roundtrip_dice"] = round(dice(back == c, lab_arr == c), 4)

    if pred_dir and os.path.isdir(pred_dir):
        pred = itk.imread(os.path.join(pred_dir, "pred_1p5mm.nii.gz"), itk.UC)
        ref15 = itk.imread(os.path.join(pred_dir, "label_1p5mm.nii.gz"), itk.UC)
        pred_arr, ref15_arr = itk.array_from_image(pred), itk.array_from_image(ref15)
        row["saved_reference_matches_monai_nearest"] = bool(np.array_equal(ref15_arr, m_lab))
        for kind in ("nearest", "label_gaussian"):
            restored = itk.array_from_image(resample(pred, orig_grid, kind))
            for c, name in CLASSES:
                row[name]["dice_1p5mm_grid"] = round(dice(pred_arr == c, ref15_arr == c), 4)
                row[name][f"dice_original_grid_pred_restored_{kind}"] = round(dice(restored == c, lab_arr == c), 4)
    return row


def summarize(rows):
    out = {"n_cases": len(rows)}
    shape_ok = [r for r in rows if r["grid_shape_equal"]]
    out["grid_identical_cases"] = sum(r["grid_shape_equal"] and r["grid_origin_diff_mm"] < 1e-3 for r in rows)
    if shape_ok:
        out["hu_abs_diff_itk_vs_monai_max"] = max(r["hu_abs_diff_itk_vs_monai"]["max"] for r in shape_ok)
        out["hu_abs_diff_itk_vs_monai_mean_of_means"] = round(float(np.mean([r["hu_abs_diff_itk_vs_monai"]["mean"] for r in shape_ok])), 4)
        out["label_voxels_disagree_itk_nearest_vs_monai_total"] = sum(r["label_voxels_disagree_itk_nearest_vs_monai"] for r in shape_ok)
        out["label_voxels_disagree_not_at_half_voxel_ties_total"] = sum(r["label_voxels_disagree_not_at_half_voxel_ties"] for r in shape_ok)
        out["cases_with_label_disagreement"] = sum(r["label_voxels_disagree_itk_nearest_vs_monai"] > 0 for r in shape_ok)
    for _, name in CLASSES:
        rs = [r[name] for r in rows if r[name]["original_ml"] > 0]
        cls = {"cases_with_structure": len(rs)}
        for key in rs[0]:
            if key == "original_ml":
                continue
            vals = [x[key] for x in rs if x.get(key) is not None]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            if key.endswith("volume_error_pct"):
                cls[key] = {"mean_abs": round(float(np.mean(np.abs(arr))), 3), "max_abs": round(float(np.max(np.abs(arr))), 3),
                            "mean_signed": round(float(np.mean(arr)), 3)}
            elif "dice" in key:
                cls[key] = {"mean": round(float(np.mean(arr)), 4), "min": round(float(np.min(arr)), 4)}
        out[name] = cls
    for k in ("seconds_itk_linear_plus_nearest", "seconds_itk_label_gaussian", "seconds_monai_load_orientation_spacingd"):
        out[f"median_{k}"] = float(np.median([r[k] for r in rows]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", default="C:/Users/sharv/data/c4kc-kits-nifti")
    ap.add_argument("--pred-dir", default="C:/Users/sharv/data/c4kc-kits-pred")
    ap.add_argument("--cases", default="test")
    ap.add_argument("--out", default="outputs/itk_resample.json")
    args = ap.parse_args()

    if args.cases == "test":
        from data import build_datalist

        pids = [it["patient_id"] for it in build_datalist(os.path.join(args.nifti, "dataset.json"))["test"]]
    else:
        pids = args.cases.split(",")
    rows = []
    for pid in pids:
        row = {"patient_id": pid, **analyse_case(os.path.join(args.nifti, pid), os.path.join(args.pred_dir, pid))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"output_spacing_mm": OUT_SPACING, "label_gaussian": {"sigma_mm": GAUSS_SIGMA_MM, "alpha": GAUSS_ALPHA},
               **summarize(rows), "cases": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
