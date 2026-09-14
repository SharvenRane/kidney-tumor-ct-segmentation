"""Tests for the ITK ingestion, ITK resampling and VTK surface code on synthetic data.

No downloads and no GPU: a small CT series is written with pydicom (filenames deliberately in the
wrong order), a real DICOM-SEG is built from it with highdicom, and label volumes are synthetic
spheres and boxes with known geometry.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

itk = pytest.importorskip("itk")


# ---------------------------------------------------------------- synthetic DICOM

def _write_ct_series(folder, n_slices=6, rows=16, cols=12, spacing=(0.8, 0.7), thickness=2.5, seed=0):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    rng = np.random.RandomState(seed)
    study, series, frame_of_ref = generate_uid(), generate_uid(), generate_uid()
    volume = rng.randint(-1000, 1500, size=(n_slices, rows, cols)).astype(np.int16)
    datasets = []
    order = rng.permutation(n_slices)  # file name order differs from anatomical order
    for k in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame_of_ref
        ds.Modality = "CT"
        ds.PatientID, ds.PatientName, ds.PatientSex, ds.PatientBirthDate = "SYN001", "Synthetic^Case", "O", "19700101"
        ds.StudyDate, ds.StudyTime, ds.AccessionNumber, ds.ReferringPhysicianName, ds.StudyID = "20260101", "120000", "1", "", "1"
        ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
        ds.Manufacturer = "synthetic"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-20.0, 35.5, -100.0 + k * thickness]
        ds.PixelSpacing = list(spacing)
        ds.SliceThickness = thickness
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
        ds.PixelData = (volume[k] + 1024).astype(np.int16).tobytes()
        path = os.path.join(folder, f"img_{order[k]:03d}.dcm")
        ds.save_as(path, enforce_file_format=True)
        datasets.append(pydicom.dcmread(path))
    return volume, datasets


def _write_seg(path, datasets, kidney, tumor):
    import highdicom as hd
    from pydicom.sr.codedict import codes

    def desc(number, label):
        return hd.seg.SegmentDescription(
            segment_number=number, segment_label=label,
            segmented_property_category=codes.SCT.Organ, segmented_property_type=codes.SCT.Kidney,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL)

    mask = np.stack([kidney, tumor], axis=-1).astype(np.uint8)  # (slices, rows, cols, segments)
    seg = hd.seg.Segmentation(
        source_images=datasets, pixel_array=mask, segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[desc(1, "Kidney"), desc(2, "Mass")],
        series_instance_uid=hd.UID(), series_number=2, sop_instance_uid=hd.UID(), instance_number=1,
        manufacturer="synthetic", manufacturer_model_name="test", software_versions="1", device_serial_number="1")
    seg.save_as(path)


@pytest.fixture
def ct_and_seg(tmp_path):
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    volume, datasets = _write_ct_series(str(ct_dir))
    datasets = sorted(datasets, key=lambda d: float(d.ImagePositionPatient[2]))
    kidney = np.zeros(volume.shape, dtype=np.uint8)
    tumor = np.zeros(volume.shape, dtype=np.uint8)
    kidney[1:5, 3:12, 2:9] = 1
    tumor[2:4, 5:8, 4:7] = 1
    kidney[0, 0, 0] = 1  # marks the most inferior slice
    seg_path = str(tmp_path / "seg.dcm")
    _write_seg(seg_path, datasets, kidney, tumor)
    return str(ct_dir), seg_path, volume, kidney, tumor


def test_ct_series_sorted_by_position_and_rescaled(ct_and_seg):
    import itk_dicom as I

    ct_dir, _, volume, _, _ = ct_and_seg
    image, uids = I.read_ct_series(ct_dir)
    arr = itk.array_from_image(image)
    assert arr.shape == volume.shape
    assert np.array_equal(arr, volume)  # slice order from positions, HU after rescale
    assert np.allclose(itk.spacing(image), (0.7, 0.8, 2.5))  # ITK spacing is (column, row, slice)
    assert np.allclose(itk.origin(image), (-20.0, 35.5, -100.0))
    assert len(set(uids)) == volume.shape[0]


def test_seg_frames_land_on_referenced_slices(ct_and_seg):
    import itk_dicom as I

    ct_dir, seg_path, _, kidney, tumor = ct_and_seg
    image, uids = I.read_ct_series(ct_dir)
    label, report = I.read_seg_labelmap(seg_path, uids, image)
    lab = itk.array_from_image(label)
    expected = np.where(tumor > 0, 2, np.where(kidney > 0, 1, 0))
    assert np.array_equal(lab, expected)
    assert report["max_frame_position_error_mm"] == 0.0
    assert lab[0, 0, 0] == 1


def _ras_copy(image):
    """What a RAS writer produces for the same voxels: x and y reversed, origin at the far corner."""
    arr = np.ascontiguousarray(itk.array_from_image(image)[:, ::-1, ::-1])
    out = itk.image_from_array(arr)
    out.SetSpacing(itk.spacing(image))
    size, sp = np.array(itk.size(image)), np.array(itk.spacing(image))
    d = itk.array_from_matrix(image.GetDirection())
    origin = np.array(itk.origin(image)) + d[:, 0] * sp[0] * (size[0] - 1) + d[:, 1] * sp[1] * (size[1] - 1)
    out.SetOrigin(origin.tolist())
    flip = np.diag([-1.0, -1.0, 1.0])
    out.SetDirection(itk.matrix_from_array(d @ flip))
    return out


def test_reorient_array_matches_ras_copy_and_controls_are_flagged(ct_and_seg):
    import itk_dicom as I

    ct_dir, seg_path, _, _, _ = ct_and_seg
    ct, uids = I.read_ct_series(ct_dir)
    label, _ = I.read_seg_labelmap(seg_path, uids, ct)
    ref_img = _ras_copy(ct)
    ref_img = itk.cast_image_filter(ref_img, ttype=(type(ref_img), itk.Image[itk.F, 3]))
    ref_lab = _ras_copy(label)
    result = I.compare_to_reference(ct, label, ref_img, ref_lab)
    assert result["identical"], result
    for mutation in I.MUTATIONS:
        flagged = I.compare_to_reference(*I.apply_mutation(ct, label, mutation), ref_img, ref_lab)
        assert not flagged["identical"], mutation


def test_reorient_array_handles_axis_permutation():
    import itk_dicom as I

    arr = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)  # (z, y, x)
    a = itk.image_from_array(arr)
    a.SetSpacing([1.0, 2.0, 3.0])
    # b stores the same voxels with its index x running along a's z axis and index z along a's x.
    b = itk.image_from_array(np.ascontiguousarray(np.transpose(arr, (2, 1, 0))))
    b.SetSpacing([3.0, 2.0, 1.0])
    b.SetDirection(itk.matrix_from_array(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])))
    mapped, origin_err = I.reorient_array(a, b)
    assert np.array_equal(mapped, itk.array_from_image(b))
    assert origin_err == 0.0


# ---------------------------------------------------------------- resampling

def _nifti_case(tmp_path, spacing=(0.8, 0.8, 5.0), shape=(40, 36, 9)):
    import nibabel as nib

    rng = np.random.RandomState(1)
    img = rng.uniform(-200, 300, size=shape).astype(np.float32)
    lab = np.zeros(shape, dtype=np.uint8)
    lab[8:30, 6:28, 2:7] = 1
    lab[14:20, 12:18, 3:6] = 2
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])  # RAS, as convert_to_nifti.py writes
    affine[:3, 3] = [15.0, 20.0, -40.0]
    nib.save(nib.Nifti1Image(img, affine), str(tmp_path / "image.nii.gz"))
    nib.save(nib.Nifti1Image(lab, affine), str(tmp_path / "label.nii.gz"))
    return str(tmp_path)


def test_itk_resampling_reproduces_monai_spacingd(tmp_path):
    pytest.importorskip("monai")
    import itk_resample as R

    case = _nifti_case(tmp_path)
    row = R.analyse_case(case, pred_dir=None)
    assert row["grid_shape_equal"] and row["grid_origin_diff_mm"] < 1e-6
    assert row["hu_abs_diff_itk_vs_monai"]["max"] < 1e-3
    # Nearest neighbour may only differ where the input index is exactly k + 0.5.
    assert row["label_voxels_disagree_not_at_half_voxel_ties"] == 0


def test_without_border_padding_the_edge_differs(tmp_path):
    pytest.importorskip("monai")
    import itk_resample as R

    case = _nifti_case(tmp_path)
    img = itk.imread(os.path.join(case, "image.nii.gz"), itk.F)
    grid = R.isotropic_grid(img)
    m = R.monai_to_itk_order(R.monai_resample(case)["image"][0].numpy())
    padded = itk.array_from_image(R.resample(img, grid, "linear", border=True))
    unpadded = itk.array_from_image(R.resample(img, grid, "linear", border=False))
    assert np.abs(padded - m).max() < 1e-3
    assert np.abs(unpadded - m).max() > 1.0


def test_label_gaussian_round_trip_beats_nearest_on_a_sphere():
    import itk_resample as R

    n, sp = 64, (0.7, 0.7, 3.0)
    zz, yy, xx = np.meshgrid(np.arange(20) * sp[2], np.arange(n) * sp[1], np.arange(n) * sp[0], indexing="ij")
    sphere = (((xx - 22) ** 2 + (yy - 22) ** 2 + (zz - 30) ** 2) < 12.0 ** 2).astype(np.uint8)
    lab = itk.image_from_array(sphere)
    lab.SetSpacing(list(sp))
    grid = R.isotropic_grid(lab)
    back = {}
    for kind in ("nearest", "label_gaussian"):
        iso = R.resample(lab, grid, kind)
        back[kind] = itk.array_from_image(R.resample(iso, R.grid_of(lab), kind))
    d_nn = R.dice(back["nearest"] == 1, sphere == 1)
    d_g = R.dice(back["label_gaussian"] == 1, sphere == 1)
    assert d_nn > 0.9 and d_g >= d_nn


# ---------------------------------------------------------------- VTK surfaces

vtk = pytest.importorskip("vtk")


def _sphere_label(radius_mm=10.0, spacing=1.0, center=(20.0, 22.0, 18.0), shape=(40, 44, 48), origin=(5.0, -3.0, 7.0),
                  direction=None):
    zz, yy, xx = np.meshgrid(*(np.arange(s) * spacing for s in shape), indexing="ij")
    arr = (((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2) <= radius_mm ** 2).astype(np.uint8)
    img = itk.image_from_array(arr)
    img.SetSpacing([spacing] * 3)
    img.SetOrigin(list(origin))
    if direction is not None:
        img.SetDirection(itk.matrix_from_array(np.array(direction, dtype=float)))
    return img, arr


def test_sphere_mesh_is_closed_and_volume_matches():
    import vtk_surfaces as V

    img, arr = _sphere_label()
    vimg, tf = V.itk_label_to_vtk(img)
    mesh = V.extract_surface(vimg, tf, 1)
    props = V.mesh_properties(mesh)
    analytic_ml = 4.0 / 3.0 * np.pi * 10.0 ** 3 / 1000.0
    assert props["open_or_non_manifold_edges"] == 0
    assert props["connected_components"] == 1
    assert abs(props["volume_ml"] - arr.sum() / 1000.0) / (arr.sum() / 1000.0) < 0.02
    assert abs(props["volume_ml"] - analytic_ml) / analytic_ml < 0.03
    assert abs(props["area_cm2"] - 4 * np.pi * 10.0 ** 2 / 100.0) / (4 * np.pi) < 0.05


def test_mesh_sits_at_the_physical_location_of_the_voxels():
    import vtk_surfaces as V

    flip = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    img, arr = _sphere_label(direction=flip)
    vimg, tf = V.itk_label_to_vtk(img)
    mesh = V.extract_surface(vimg, tf, 1)
    com = vtk.vtkCenterOfMass()
    com.SetInputData(mesh)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    idx = np.argwhere(arr > 0).mean(axis=0)[::-1]  # (x, y, z) index
    d = itk.array_from_matrix(img.GetDirection())
    physical = np.array(itk.origin(img)) + d @ (idx * np.array(itk.spacing(img)))
    assert np.linalg.norm(np.array(com.GetCenter()) - physical) < 0.2


def test_surface_distance_between_concentric_spheres():
    import vtk_surfaces as V

    a, _ = _sphere_label(radius_mm=10.0)
    b, _ = _sphere_label(radius_mm=13.0)
    ma = V.extract_surface(*V.itk_label_to_vtk(a), 1)
    mb = V.extract_surface(*V.itk_label_to_vtk(b), 1)
    d_ab, d_ba, _ = V.surface_distances(ma, mb)
    assert abs(np.abs(d_ab).mean() - 3.0) < 0.5
    assert abs(np.abs(d_ba).mean() - 3.0) < 0.5
    same_ab, same_ba, _ = V.surface_distances(ma, ma)
    assert np.abs(same_ab).max() < 1e-6 and np.abs(same_ba).max() < 1e-6
