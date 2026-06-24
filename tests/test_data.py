"""Tests for the data split, label combination, and training transforms."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data as D  # noqa: E402
from convert_to_nifti import combine_segments  # noqa: E402


def test_combine_segments_tumor_priority():
    arr = np.zeros((4, 4, 4, 2), dtype=np.uint8)
    arr[..., 0] = 1  # kidney everywhere
    arr[1, 1, 1, 1] = 1  # one tumor voxel inside kidney
    lab = combine_segments(arr)
    assert set(np.unique(lab)) == {1, 2}
    assert lab[1, 1, 1] == 2  # tumor overrides kidney
    assert lab[0, 0, 0] == 1


def test_combine_segments_single_segment():
    arr = np.zeros((3, 3, 3, 1), dtype=np.uint8)
    arr[0, 0, 0, 0] = 1
    lab = combine_segments(arr)
    assert lab[0, 0, 0] == 1 and lab.max() == 1


def test_build_datalist_is_disjoint(tmp_path):
    import json

    items = [{"image": f"i{i}.nii.gz", "label": f"l{i}.nii.gz", "patient_id": f"P{i}",
              "n_tumor": 1} for i in range(20)]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(items))
    split = D.build_datalist(str(p))
    ids = lambda part: {x["patient_id"] for x in split[part]}
    train, val, test = ids("train"), ids("val"), ids("test")
    assert len(train | val | test) == 20  # all accounted for
    assert not (train & val) and not (train & test) and not (val & test)


def test_build_datalist_deterministic(tmp_path):
    import json

    items = [{"image": f"i{i}", "label": f"l{i}", "patient_id": f"P{i}"} for i in range(20)]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(items))
    a = D.build_datalist(str(p))
    b = D.build_datalist(str(p))
    assert [x["patient_id"] for x in a["test"]] == [x["patient_id"] for x in b["test"]]


def test_train_transforms_produce_patches(tmp_path):
    import nibabel as nib

    # synthetic 128^3 CT with a kidney+tumor blob so RandCropByPosNeg has foreground
    img = np.random.RandomState(0).randint(-200, 300, size=(128, 128, 128)).astype(np.float32)
    lab = np.zeros((128, 128, 128), dtype=np.uint8)
    lab[50:80, 50:80, 50:80] = 1
    lab[60:66, 60:66, 60:66] = 2
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    nib.save(nib.Nifti1Image(img, aff), str(tmp_path / "image.nii.gz"))
    nib.save(nib.Nifti1Image(lab, aff), str(tmp_path / "label.nii.gz"))
    item = {"image": str(tmp_path / "image.nii.gz"), "label": str(tmp_path / "label.nii.gz")}
    out = D.train_transforms(roi=(96, 96, 96), num_samples=2)(item)
    assert isinstance(out, list) and len(out) == 2
    for sample in out:
        assert tuple(sample["image"].shape) == (1, 96, 96, 96)
        assert tuple(sample["label"].shape) == (1, 96, 96, 96)
