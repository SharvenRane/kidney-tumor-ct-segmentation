"""Independent ITK ingestion of the C4KC-KiTS DICOM, cross checked against the MONAI path.

convert_to_nifti.py reads the CT series and the DICOM-SEG with MONAI's PydicomReader. This
module rebuilds the same pairs through a second, unrelated stack and compares the two voxel by
voxel, which is the check you want before trusting a training cache:

* CT series: ITK GDCMSeriesFileNames sorts the slices by Image Position (Patient) along the
  slice normal, ImageSeriesReader + GDCMImageIO stacks them and applies Rescale Slope and
  Intercept. Geometry (spacing, origin, direction) stays in ITK's LPS convention.
* DICOM-SEG: highdicom reads the binary segments and places every frame on the CT slice it
  references by SOP Instance UID, so the label lands on the CT grid by identity rather than by
  frame order. Each frame's own Image Position (Patient) is then checked against the physical
  position of that CT slice.
* NIfTI: itk.imwrite converts LPS to the RAS header convention (`write_nifti`).

The comparison never resamples. Two grids over the same voxels differ only by axis flips and
permutations, so `reorient_array` maps one array onto the other exactly from the direction
matrices and reports how far apart the origins land.

Negative controls apply realistic ingestion bugs to the ITK result in memory (a left/right label
flip, reversed slice order, a half voxel origin shift) and confirm the comparison flags each one.

Usage:
    python src/itk_dicom.py --in C:/Users/sharv/data/c4kc-kits \
        --monai-nifti C:/Users/sharv/data/c4kc-kits-nifti --out outputs/itk_dicom.json
"""
import argparse
import glob
import json
import os
import sys
import time

import highdicom as hd
import itk
import numpy as np
import pydicom

KIDNEY, TUMOR = 1, 2
MUTATIONS = ("label_left_right_flip", "reverse_slice_order", "half_voxel_origin_shift")


def read_ct_series(ct_dir, pixel_type=itk.SS):
    """Read one CT series with ITK. Returns (image, SOP Instance UIDs in ITK slice order)."""
    names = itk.GDCMSeriesFileNames.New()
    names.SetUseSeriesDetails(True)
    names.SetDirectory(ct_dir)
    uids = names.GetSeriesUIDs()
    if len(uids) != 1:
        raise ValueError(f"expected one series in {ct_dir}, found {len(uids)}")
    files = list(names.GetFileNames(uids[0]))
    reader = itk.ImageSeriesReader[itk.Image[pixel_type, 3]].New()
    reader.SetImageIO(itk.GDCMImageIO.New())
    reader.SetFileNames(files)
    reader.Update()
    sop_uids = [pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["SOPInstanceUID"]).SOPInstanceUID
                for f in files]
    return reader.GetOutput(), sop_uids


def slice_positions(image):
    """Physical LPS position (mm) of voxel (0, 0, k) for every slice k."""
    origin = np.array(itk.origin(image), dtype=float)
    spacing = np.array(itk.spacing(image), dtype=float)
    direction = itk.array_from_matrix(image.GetDirection())
    n = int(itk.size(image)[2])
    return origin + np.outer(np.arange(n) * spacing[2], direction[:, 2])


def read_seg_labelmap(seg_file, sop_uids, reference):
    """Build a 0/1/2 labelmap on the CT grid from a DICOM-SEG. Returns (label image, report)."""
    seg = hd.seg.segread(seg_file)
    numbers = {s.SegmentLabel.lower(): s.SegmentNumber for s in seg.SegmentSequence}
    kidney_n, tumor_n = numbers["kidney"], numbers.get("mass", numbers.get("tumor"))

    # These SEGs omit SpatialLocationsPreserved, so highdicom refuses source frame indexing
    # unless told otherwise. That is safe here because every frame position is verified below.
    per_seg = seg.get_pixels_by_source_instance(
        source_sop_instance_uids=sop_uids, segment_numbers=[kidney_n, tumor_n],
        ignore_spatial_locations=True, assert_missing_frames_are_empty=True, dtype=np.uint8)
    lab = np.zeros(per_seg.shape[:3], dtype=np.uint8)
    lab[per_seg[..., 0] > 0] = KIDNEY
    lab[per_seg[..., 1] > 0] = TUMOR  # mass takes priority on overlap, as in convert_to_nifti.py

    positions = slice_positions(reference)
    uid_to_k = {u: k for k, u in enumerate(sop_uids)}
    errors = []
    for fg in seg.PerFrameFunctionalGroupsSequence:
        ref = fg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        ipp = np.array(fg.PlanePositionSequence[0].ImagePositionPatient, dtype=float)
        errors.append(float(np.linalg.norm(ipp - positions[uid_to_k[ref]])))

    label = itk.image_from_array(lab)  # numpy (z, y, x) is ITK index (x, y, z) reversed
    label.CopyInformation(reference)
    return label, {"frames": int(seg.NumberOfFrames), "max_frame_position_error_mm": round(max(errors), 4)}


def reorient_array(src_image, ref_image):
    """Map src voxels onto ref's index order when both grids cover the same voxels.

    Uses the direction matrices only (axis permutations and flips), never interpolation.
    Returns (array in ref's numpy order, distance in mm between ref's origin and where src's
    voxel that maps to ref index 0 actually sits).
    """
    src = itk.array_from_image(src_image)
    d_src = itk.array_from_matrix(src_image.GetDirection())
    d_ref = itk.array_from_matrix(ref_image.GetDirection())
    spacing = np.array(itk.spacing(src_image), dtype=float)
    size = np.array(itk.size(src_image), dtype=int)
    origin = np.array(itk.origin(src_image), dtype=float)

    m = d_src.T @ d_ref  # m[i, j] near +-1 when src axis i runs along ref axis j
    perm = np.argmax(np.abs(m), axis=0)
    if sorted(perm.tolist()) != [0, 1, 2] or not np.allclose(np.abs(m[perm, np.arange(3)]), 1.0, atol=1e-4):
        raise ValueError("grids are not related by axis permutations and flips")
    signs = np.sign(m[perm, np.arange(3)])

    # ITK index axes (x, y, z) are numpy axes (2, 1, 0).
    arr = np.transpose(src, [2 - perm[2], 2 - perm[1], 2 - perm[0]])
    corner = origin.copy()
    for j in range(3):
        if signs[j] < 0:
            arr = np.flip(arr, axis=2 - j)
            i = perm[j]
            corner = corner + d_src[:, i] * spacing[i] * (size[i] - 1)
    origin_err = float(np.linalg.norm(corner - np.array(itk.origin(ref_image), dtype=float)))
    return np.ascontiguousarray(arr), origin_err


def _mutate(image, axis=None, shift=False):
    arr = itk.array_from_image(image)
    if axis is not None:
        arr = np.ascontiguousarray(np.flip(arr, axis=axis))
    out = itk.image_from_array(arr)
    out.CopyInformation(image)
    if shift:
        o = np.array(itk.origin(image), dtype=float) + 0.5 * np.array(itk.spacing(image), dtype=float)
        out.SetOrigin(o.tolist())
    return out


def apply_mutation(ct, label, mutation):
    if mutation is None:
        return ct, label
    if mutation == "label_left_right_flip":
        return ct, _mutate(label, axis=2)  # numpy axis 2 is ITK x, patient left/right for these scans
    if mutation == "reverse_slice_order":
        return _mutate(ct, axis=0), _mutate(label, axis=0)
    if mutation == "half_voxel_origin_shift":
        return _mutate(ct, shift=True), _mutate(label, shift=True)
    raise ValueError(mutation)


def compare_to_reference(ct, label, ref_img, ref_lab, origin_tol_mm=1e-3):
    img_arr, origin_err = reorient_array(ct, ref_img)
    lab_arr, _ = reorient_array(label, ref_lab)
    ref_img_arr = itk.array_view_from_image(ref_img)
    ref_lab_arr = itk.array_view_from_image(ref_lab)
    out = {
        "spacing_max_abs_diff_mm": float(np.max(np.abs(np.array(itk.spacing(ct)) - np.array(itk.spacing(ref_img))))),
        "origin_mismatch_mm": round(origin_err, 4),
    }
    if img_arr.shape != ref_img_arr.shape:
        out.update(shape_mismatch=[list(img_arr.shape), list(ref_img_arr.shape)], identical=False)
        return out
    out["image_max_abs_diff_hu"] = float(np.max(np.abs(img_arr.astype(np.float32) - ref_img_arr)))
    out["label_voxels_disagree"] = int(np.count_nonzero(lab_arr != ref_lab_arr))
    for ci, name in ((KIDNEY, "kidney"), (TUMOR, "tumor")):
        out[f"{name}_voxels"] = int((lab_arr == ci).sum())
        out[f"{name}_voxels_reference"] = int((ref_lab_arr == ci).sum())
    out["identical"] = (out["image_max_abs_diff_hu"] == 0.0 and out["label_voxels_disagree"] == 0
                        and origin_err < origin_tol_mm and out["spacing_max_abs_diff_mm"] < 1e-4)
    return out


def write_nifti(ct_dir, seg_file, out_dir):
    ct, sop_uids = read_ct_series(ct_dir)
    label, _ = read_seg_labelmap(seg_file, sop_uids, ct)
    os.makedirs(out_dir, exist_ok=True)
    itk.imwrite(ct, os.path.join(out_dir, "image.nii.gz"), compression=True)
    itk.imwrite(label, os.path.join(out_dir, "label.nii.gz"), compression=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="C:/Users/sharv/data/c4kc-kits")
    ap.add_argument("--monai-nifti", default="C:/Users/sharv/data/c4kc-kits-nifti")
    ap.add_argument("--cases", default="test", help="'test' (32 held out), 'all', or a comma list")
    ap.add_argument("--controls", type=int, default=3, help="number of cases also run through each mutation")
    ap.add_argument("--nifti-roundtrip", type=int, default=3, help="cases written with itk.imwrite and reread")
    ap.add_argument("--out", default="outputs/itk_dicom.json")
    args = ap.parse_args()

    manifest = {m["patient_id"]: m for m in json.load(open(os.path.join(args.indir, "pairs_manifest.json")))}
    if args.cases == "test":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data import build_datalist

        pids = [it["patient_id"] for it in build_datalist(os.path.join(args.monai_nifti, "dataset.json"))["test"]]
    elif args.cases == "all":
        pids = [p for p in manifest if os.path.isdir(os.path.join(args.monai_nifti, p))]
    else:
        pids = args.cases.split(",")

    rows = []
    for i, pid in enumerate(pids):
        m = manifest[pid]
        ct_dir = os.path.join(args.indir, m["ct_uid"])
        seg_file = glob.glob(os.path.join(args.indir, m["seg_uid"], "*.dcm"))[0]
        mdir = os.path.join(args.monai_nifti, pid)

        t0 = time.time()
        ct, sop_uids = read_ct_series(ct_dir)
        label, seg_report = read_seg_labelmap(seg_file, sop_uids, ct)
        seconds = time.time() - t0
        ref_img = itk.imread(os.path.join(mdir, "image.nii.gz"), itk.F)
        ref_lab = itk.imread(os.path.join(mdir, "label.nii.gz"), itk.UC)

        row = {"patient_id": pid, "slices": len(sop_uids), "seconds_itk_read": round(seconds, 2),
               "direction_is_identity": bool(np.allclose(itk.array_from_matrix(ct.GetDirection()), np.eye(3))),
               **seg_report, **compare_to_reference(ct, label, ref_img, ref_lab)}
        if i < args.controls:
            row["controls"] = {}
            for mutation in MUTATIONS:
                c = compare_to_reference(*apply_mutation(ct, label, mutation), ref_img, ref_lab)
                row["controls"][mutation] = {k: c.get(k) for k in ("identical", "origin_mismatch_mm", "image_max_abs_diff_hu", "label_voxels_disagree")}
        if i < args.nifti_roundtrip:
            tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_itk_nifti_tmp", pid)
            os.makedirs(tmp, exist_ok=True)
            itk.imwrite(ct, os.path.join(tmp, "image.nii.gz"), compression=True)
            itk.imwrite(label, os.path.join(tmp, "label.nii.gz"), compression=True)
            rt = compare_to_reference(itk.imread(os.path.join(tmp, "image.nii.gz"), itk.SS),
                                      itk.imread(os.path.join(tmp, "label.nii.gz"), itk.UC), ref_img, ref_lab)
            row["itk_written_nifti_vs_reference"] = {k: rt.get(k) for k in ("identical", "origin_mismatch_mm", "image_max_abs_diff_hu", "label_voxels_disagree")}
            for f in ("image.nii.gz", "label.nii.gz"):
                os.remove(os.path.join(tmp, f))
            os.rmdir(tmp)
        rows.append(row)
        print(json.dumps(row), flush=True)

    tmp_root = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_itk_nifti_tmp")
    if os.path.isdir(tmp_root) and not os.listdir(tmp_root):
        os.rmdir(tmp_root)
    controls = [c for r in rows for c in r.get("controls", {}).values()]
    summary = {
        "n_cases": len(rows),
        "cases_identical_to_monai_path": sum(r["identical"] for r in rows),
        "cases_with_identity_direction": sum(r["direction_is_identity"] for r in rows),
        "max_origin_mismatch_mm": max(r["origin_mismatch_mm"] for r in rows),
        "max_spacing_diff_mm": max(r["spacing_max_abs_diff_mm"] for r in rows),
        "max_seg_frame_position_error_mm": max(r["max_frame_position_error_mm"] for r in rows),
        "total_seg_frames": sum(r["frames"] for r in rows),
        "total_slices": sum(r["slices"] for r in rows),
        "median_seconds_itk_read": float(np.median([r["seconds_itk_read"] for r in rows])),
        "controls_run": len(controls),
        "controls_flagged": sum(not c["identical"] for c in controls),
        "itk_nifti_roundtrips_identical": sum(r["itk_written_nifti_vs_reference"]["identical"] for r in rows if "itk_written_nifti_vs_reference" in r),
        "cases": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
