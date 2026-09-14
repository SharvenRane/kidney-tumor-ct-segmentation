"""3D surfaces of the kidney and tumor labels with VTK: meshes, mesh based distances, renders.

For each held out case this builds closed triangle meshes from the clinician reference and from
the model prediction (both on the evaluated 1.5 mm grid, see save_predictions.py):

* vtkDiscreteFlyingEdges3D extracts one surface per label value, vtkWindowedSincPolyDataFilter
  smooths it without shrinking, and the mesh is placed in patient LPS millimetres using the
  image origin, spacing and direction.
* vtkMassProperties gives enclosed volume and surface area. The volume is compared with the
  voxel count, which shows how much a surface representation changes a volume measurement.
* vtkFeatureEdges counts boundary and non manifold edges, so "closed surface" is checked, not assumed.
* vtkDistancePolyDataFilter measures point to surface distance from each mesh to the other. HD95
  computed the way MONAI defines it (the larger directed 95th percentile) is compared with MONAI's
  voxel based HD95 from evaluate.py. Distances are sampled at mesh vertices, not weighted by area.
* Meshes are written as .vtp, and selected cases are rendered offscreen to PNG: the reference on
  the left, the prediction coloured by its signed distance to the reference on the right.

Usage:
    python src/vtk_surfaces.py --pred-dir C:/Users/sharv/data/c4kc-kits-pred --out outputs/vtk_surfaces.json
    python src/vtk_surfaces.py --render KiTS-00085,KiTS-00037 --figures docs/figures
"""
import argparse
import json
import os
import time

import itk
import numpy as np
import vtk
from vtk.util import numpy_support

KIDNEY, TUMOR = 1, 2
CLASSES = ((KIDNEY, "kidney"), (TUMOR, "tumor"))
SMOOTH_ITERATIONS = 20
SMOOTH_PASSBAND = 0.01


def itk_label_to_vtk(label):
    """ITK label image (LPS) to vtkImageData in index space scaled by spacing, plus a transform
    that carries it to patient LPS millimetres (origin and direction)."""
    arr = itk.array_from_image(label)  # (z, y, x)
    img = vtk.vtkImageData()
    img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
    img.SetSpacing(*[float(s) for s in itk.spacing(label)])
    scalars = numpy_support.numpy_to_vtk(np.ascontiguousarray(arr).ravel(order="C"), deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    img.GetPointData().SetScalars(scalars)  # VTK point order is x fastest, same as C order of (z, y, x)

    direction = itk.array_from_matrix(label.GetDirection())
    origin = np.array(itk.origin(label), dtype=float)
    m = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            m.SetElement(r, c, float(direction[r, c]))
        m.SetElement(r, 3, float(origin[r]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(m)
    return img, transform


def extract_surface(vtk_image, transform, value, smooth=True):
    fe = vtk.vtkDiscreteFlyingEdges3D()
    fe.SetInputData(vtk_image)
    fe.SetValue(0, value)
    fe.ComputeNormalsOff()
    fe.ComputeGradientsOff()
    fe.Update()
    poly = fe.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return None
    if smooth:
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(poly)
        sm.SetNumberOfIterations(SMOOTH_ITERATIONS)
        sm.SetPassBand(SMOOTH_PASSBAND)
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        poly = sm.GetOutput()
    tf = vtk.vtkTransformFilter()
    tf.SetInputData(poly)
    tf.SetTransform(transform)
    tf.Update()
    clean = vtk.vtkTriangleFilter()
    clean.SetInputData(tf.GetOutput())
    clean.Update()
    return clean.GetOutput()


def mesh_properties(poly):
    mp = vtk.vtkMassProperties()
    mp.SetInputData(poly)
    mp.Update()
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return {
        "volume_ml": mp.GetVolume() / 1000.0,
        "area_cm2": mp.GetSurfaceArea() / 100.0,
        "triangles": poly.GetNumberOfPolys(),
        "open_or_non_manifold_edges": fe.GetOutput().GetNumberOfLines(),
        "connected_components": conn.GetNumberOfExtractedRegions(),
    }


def surface_distances(a, b):
    """Signed point to surface distances (mm) from a's points to b and from b's points to a."""
    f = vtk.vtkDistancePolyDataFilter()
    f.SetInputData(0, a)
    f.SetInputData(1, b)
    f.SignedDistanceOn()
    f.ComputeSecondDistanceOn()
    f.Update()
    d_ab = numpy_support.vtk_to_numpy(f.GetOutput().GetPointData().GetArray("Distance"))
    d_ba = numpy_support.vtk_to_numpy(f.GetSecondDistanceOutput().GetPointData().GetArray("Distance"))
    return np.asarray(d_ab), np.asarray(d_ba), f.GetOutput()


def analyse_case(pred_dir, validated=None, mesh_dir=None):
    ref = itk.imread(os.path.join(pred_dir, "label_1p5mm.nii.gz"), itk.UC)
    pred = itk.imread(os.path.join(pred_dir, "pred_1p5mm.nii.gz"), itk.UC)
    voxel_ml = float(np.prod(itk.spacing(ref))) / 1000.0
    ref_arr, pred_arr = itk.array_view_from_image(ref), itk.array_view_from_image(pred)
    ref_img, ref_tf = itk_label_to_vtk(ref)
    pred_img, pred_tf = itk_label_to_vtk(pred)
    row = {}
    for value, name in CLASSES:
        entry = {"reference_voxel_ml": round(float((ref_arr == value).sum()) * voxel_ml, 3),
                 "prediction_voxel_ml": round(float((pred_arr == value).sum()) * voxel_ml, 3)}
        t0 = time.time()
        meshes = {}
        for tag, img, tf in (("reference", ref_img, ref_tf), ("prediction", pred_img, pred_tf)):
            raw = extract_surface(img, tf, value, smooth=False)
            smooth = extract_surface(img, tf, value, smooth=True)
            if smooth is None:
                entry[tag] = None
                continue
            pr, ps = mesh_properties(raw), mesh_properties(smooth)
            entry[tag] = {
                "raw_volume_ml": round(pr["volume_ml"], 3),
                "smoothed_volume_ml": round(ps["volume_ml"], 3),
                "raw_area_cm2": round(pr["area_cm2"], 2),
                "smoothed_area_cm2": round(ps["area_cm2"], 2),
                "triangles": ps["triangles"],
                "open_or_non_manifold_edges": ps["open_or_non_manifold_edges"],
                "connected_components": ps["connected_components"],
            }
            meshes[tag] = smooth
            if mesh_dir:
                os.makedirs(mesh_dir, exist_ok=True)
                w = vtk.vtkXMLPolyDataWriter()
                w.SetFileName(os.path.join(mesh_dir, f"{name}_{tag}.vtp"))
                w.SetInputData(smooth)
                w.Write()
        entry["seconds_meshing"] = round(time.time() - t0, 2)
        if len(meshes) == 2:
            t0 = time.time()
            d_rp, d_pr, _ = surface_distances(meshes["reference"], meshes["prediction"])
            both = np.concatenate([np.abs(d_rp), np.abs(d_pr)])
            entry["mesh_symmetric_mean_surface_distance_mm"] = round(float(both.mean()), 3)
            entry["mesh_symmetric_hd95_mm"] = round(float(np.percentile(both, 95)), 3)
            # MONAI's HausdorffDistanceMetric takes the larger of the two directed 95th percentiles.
            entry["mesh_hd95_max_of_directed_mm"] = round(float(max(np.percentile(np.abs(d_rp), 95), np.percentile(np.abs(d_pr), 95))), 3)
            entry["mesh_hausdorff_mm"] = round(float(both.max()), 3)
            entry["seconds_distance"] = round(time.time() - t0, 2)
        if validated is not None:
            entry["monai_voxel_hd95_mm"] = validated[name]["hd95"]
        row[name] = entry
    return row


# ---------------------------------------------------------------- rendering

def _actor(poly, color, opacity=1.0, scalars=None, lut=None):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    if scalars is not None:
        poly.GetPointData().SetActiveScalars(scalars)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(lut.GetRange())
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetSpecular(0.2)
    return actor


def _distance_lut(limit_mm):
    lut = vtk.vtkLookupTable()
    lut.SetRange(-limit_mm, limit_mm)
    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(-limit_mm, 0.23, 0.30, 0.75)
    ctf.AddRGBPoint(0.0, 0.87, 0.87, 0.87)
    ctf.AddRGBPoint(limit_mm, 0.71, 0.02, 0.15)
    lut.SetNumberOfTableValues(256)
    for i in range(256):
        c = ctf.GetColor(-limit_mm + 2 * limit_mm * i / 255.0)
        lut.SetTableValue(i, c[0], c[1], c[2], 1.0)
    lut.Build()
    return lut


def _text(renderer, text, size=18):
    t = vtk.vtkTextActor()
    t.SetInput(text)
    t.GetTextProperty().SetFontSize(size)
    t.GetTextProperty().SetColor(0.1, 0.1, 0.1)
    t.SetPosition(12, 12)
    renderer.AddViewProp(t)


def render_case(pred_dir, png_path, title, limit_mm=10.0, size=(1400, 700)):
    """Offscreen render: reference kidney and tumor left, prediction coloured by signed distance
    to the reference surface right (positive means outside the reference)."""
    ref = itk.imread(os.path.join(pred_dir, "label_1p5mm.nii.gz"), itk.UC)
    pred = itk.imread(os.path.join(pred_dir, "pred_1p5mm.nii.gz"), itk.UC)
    ref_img, ref_tf = itk_label_to_vtk(ref)
    pred_img, pred_tf = itk_label_to_vtk(pred)
    lut = _distance_lut(limit_mm)

    left, right = vtk.vtkRenderer(), vtk.vtkRenderer()
    left.SetViewport(0.0, 0.0, 0.5, 1.0)
    right.SetViewport(0.5, 0.0, 1.0, 1.0)
    for r in (left, right):
        r.SetBackground(1.0, 1.0, 1.0)

    ref_k = extract_surface(ref_img, ref_tf, KIDNEY)
    ref_t = extract_surface(ref_img, ref_tf, TUMOR)
    if ref_k is not None:
        left.AddActor(_actor(ref_k, (0.85, 0.72, 0.55), opacity=0.45))
    if ref_t is not None:
        left.AddActor(_actor(ref_t, (0.80, 0.15, 0.15)))
    for value, opacity in ((KIDNEY, 0.55), (TUMOR, 1.0)):
        p = extract_surface(pred_img, pred_tf, value)
        r = extract_surface(ref_img, ref_tf, value)
        if p is None:
            continue
        if r is not None:
            _, _, with_dist = surface_distances(p, r)
            right.AddActor(_actor(with_dist, (1, 1, 1), opacity=opacity, scalars="Distance", lut=lut))
        else:
            right.AddActor(_actor(p, (0.5, 0.5, 0.5), opacity=opacity))

    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle("signed distance to reference (mm)")
    bar.SetNumberOfLabels(5)
    bar.SetOrientationToHorizontal()
    bar.SetPosition(0.2, 0.86)
    bar.SetWidth(0.6)
    bar.SetHeight(0.1)
    for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
        prop.SetColor(0.1, 0.1, 0.1)
        prop.ItalicOff()
        prop.ShadowOff()
    right.AddViewProp(bar)
    _text(left, f"{title}  clinician reference (kidney beige, tumor red)")
    _text(right, "model prediction, coloured by distance to the reference")

    # Anterior view in LPS: the camera sits at -y (anterior) looking toward +y, superior (+z) up.
    left.ResetCamera()
    cam = left.GetActiveCamera()
    fp = np.array(cam.GetFocalPoint())
    dist = cam.GetDistance()
    cam.SetPosition(*(fp + np.array([0.0, -dist, 0.0])))
    cam.SetViewUp(0.0, 0.0, 1.0)
    left.ResetCamera()
    left.GetActiveCamera().Zoom(0.95)
    right.SetActiveCamera(left.GetActiveCamera())

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(*size)
    win.AddRenderer(left)
    win.AddRenderer(right)
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.ReadFrontBufferOff()
    grab.Update()
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(png_path)
    writer.SetInputConnection(grab.GetOutputPort())
    writer.Write()
    win.Finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="C:/Users/sharv/data/c4kc-kits-pred")
    ap.add_argument("--metrics", default="outputs/metrics.json")
    ap.add_argument("--out", default="outputs/vtk_surfaces.json")
    ap.add_argument("--mesh-dir", default="outputs/surfaces")
    ap.add_argument("--render", default="", help="comma list of cases to render instead of analysing")
    ap.add_argument("--figures", default="docs/figures")
    args = ap.parse_args()

    if args.render:
        for pid in args.render.split(","):
            png = os.path.join(args.figures, f"surface_{pid}.png")
            render_case(os.path.join(args.pred_dir, pid), png, pid)
            print(f"wrote {png}")
        return

    validated = {c["patient_id"]: c for c in json.load(open(args.metrics))["per_case"]}
    rows = []
    for pid in [c["patient_id"] for c in json.load(open(args.metrics))["per_case"]]:
        case_dir = os.path.join(args.pred_dir, pid)
        row = {"patient_id": pid, **analyse_case(case_dir, validated[pid], os.path.join(args.mesh_dir, pid))}
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = {"n_cases": len(rows), "smoothing": {"filter": "vtkWindowedSincPolyDataFilter",
               "iterations": SMOOTH_ITERATIONS, "passband": SMOOTH_PASSBAND}}
    for _, name in CLASSES:
        rs = [r[name] for r in rows]
        meshes = [x for r in rs for x in (r["reference"], r["prediction"]) if x]
        ref_ok = [r for r in rs if r["reference"]]
        err_raw = [100 * (r["reference"]["raw_volume_ml"] - r["reference_voxel_ml"]) / r["reference_voxel_ml"] for r in ref_ok]
        err_smooth = [100 * (r["reference"]["smoothed_volume_ml"] - r["reference_voxel_ml"]) / r["reference_voxel_ml"] for r in ref_ok]
        paired = [r for r in rs if "mesh_symmetric_hd95_mm" in r and np.isfinite(r["monai_voxel_hd95_mm"])]
        mesh_hd = np.array([r["mesh_hd95_max_of_directed_mm"] for r in paired])
        vox_hd = np.array([r["monai_voxel_hd95_mm"] for r in paired])
        summary[name] = {
            "meshes": len(meshes),
            "closed_manifold_meshes": sum(m["open_or_non_manifold_edges"] == 0 for m in meshes),
            "reference_raw_mesh_volume_error_pct_vs_voxels": {"mean": round(float(np.mean(err_raw)), 3), "min": round(float(np.min(err_raw)), 3), "max": round(float(np.max(err_raw)), 3)},
            "reference_smoothed_mesh_volume_error_pct_vs_voxels": {"mean": round(float(np.mean(err_smooth)), 3), "min": round(float(np.min(err_smooth)), 3), "max": round(float(np.max(err_smooth)), 3)},
            "reference_area_change_pct_raw_to_smoothed": round(float(np.mean([100 * (r["reference"]["smoothed_area_cm2"] - r["reference"]["raw_area_cm2"]) / r["reference"]["raw_area_cm2"] for r in ref_ok])), 2),
            "cases_with_both_meshes": len(paired),
            "mesh_hd95_mean_mm": round(float(mesh_hd.mean()), 2),
            "monai_voxel_hd95_mean_mm": round(float(vox_hd.mean()), 2),
            "hd95_mesh_minus_voxel_median_mm": round(float(np.median(mesh_hd - vox_hd)), 2),
            "hd95_pearson_r": round(float(np.corrcoef(mesh_hd, vox_hd)[0, 1]), 4),
            "mean_surface_distance_mean_mm": round(float(np.mean([r["mesh_symmetric_mean_surface_distance_mm"] for r in paired])), 2),
            "median_seconds_meshing": float(np.median([r["seconds_meshing"] for r in rs])),
        }
    summary["cases"] = rows
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
