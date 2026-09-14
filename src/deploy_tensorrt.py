"""Deploy the trained SegResNet with TensorRT and measure what each precision costs.

The question a deployment has to answer is not "how fast is FP16" but "how fast is FP16 at
the Dice we validated". So every backend runs the same sliding window inference over the
same 32 held out CT volumes, and each one is scored against the reference segmentation and
against the PyTorch baseline voxel by voxel.

Backends
  torch_fp32   PyTorch eager, TF32 disabled, so it is genuinely FP32
  torch_amp    PyTorch eager under autocast, the configuration evaluate.py validated
  trt_fp32     TensorRT engine, TF32 disabled
  trt_fp16     TensorRT engine with FP16 kernels allowed

The engines are built with the TensorRT Python API directly (builder, ONNX parser,
optimization profile, serialized plan) and executed with execute_async_v3 on PyTorch's
CUDA stream, reading and writing PyTorch tensors in place. No ONNX Runtime in the loop.

    python src/deploy_tensorrt.py --checkpoint outputs/best.pt
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SegResNet
from monai.networks.utils import one_hot

from data import PIXDIM, build_datalist, val_transforms

ROI = (96, 96, 96)
CLASSES = ["kidney", "tumor"]


def make_model():
    return SegResNet(spatial_dims=3, in_channels=1, out_channels=3, init_filters=16,
                     blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))


def _import_tensorrt():
    # On Windows the pip wheel keeps its DLLs in tensorrt_libs, which is on no search path.
    if os.name == "nt":
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            os.add_dll_directory(list(spec.submodule_search_locations)[0])
    import tensorrt as trt
    return trt


def export_onnx(model, path, sw_batch):
    model = model.cpu().eval()
    x = torch.randn(sw_batch, 1, *ROI)
    torch.onnx.export(model, x, str(path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=18, dynamo=False)
    return path


def build_engine(trt, onnx_path, plan_path, precision, sw_batch, logger):
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + "; ".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("image", (1, 1, *ROI), (sw_batch, 1, *ROI), (sw_batch, 1, *ROI))
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if plan is None:
        raise RuntimeError(f"engine build failed for {precision}")
    Path(plan_path).write_bytes(bytes(plan))
    return build_s


class TrtPredictor:
    """Callable that sliding_window_inference can use in place of an nn.Module."""

    def __init__(self, trt, plan_path, logger):
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if n != self.inp)
        assert self.engine.get_tensor_dtype(self.out) == trt.DataType.FLOAT

    def __call__(self, x):
        x = x.contiguous().float()
        self.ctx.set_input_shape(self.inp, tuple(x.shape))
        y = torch.empty(tuple(self.ctx.get_tensor_shape(self.out)), device=x.device,
                        dtype=torch.float32)
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.out, y.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        return y


def run_backend(name, predictor, x, sw_batch, stream):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # A dedicated stream: on the default stream TensorRT inserts extra synchronisation calls
    # inside enqueueV3, which would charge TensorRT for overhead it does not have in deployment.
    with torch.no_grad(), torch.cuda.stream(stream):
        if name == "torch_amp":
            with torch.amp.autocast("cuda"):
                logits = sliding_window_inference(x, ROI, sw_batch, predictor, overlap=0.5,
                                                  mode="gaussian")
        else:
            logits = sliding_window_inference(x, ROI, sw_batch, predictor, overlap=0.5,
                                              mode="gaussian")
    torch.cuda.synchronize()
    return torch.argmax(logits.float(), dim=1, keepdim=True), time.perf_counter() - t0


def mean_ci(vals, n_boot=2000, seed=0):
    vals = [v for v in vals if np.isfinite(v)]
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    boots = np.sort([rng.choice(arr, len(arr)).mean() for _ in range(n_boot)])
    return {"mean": round(float(arr.mean()), 4),
            "ci95": [round(float(boots[int(0.025 * n_boot)]), 4),
                     round(float(boots[int(0.975 * n_boot)]), 4)], "n": len(arr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C:/Users/sharv/data/c4kc-kits-nifti/dataset.json")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--workdir", default="outputs/tensorrt")
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N cases")
    ap.add_argument("--out", default="outputs/tensorrt_metrics.json")
    args = ap.parse_args()

    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    model = make_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    onnx_path = export_onnx(model, work / "segresnet.onnx", args.sw_batch)

    engines = {}
    for prec in ["fp32", "fp16"]:
        plan = work / f"segresnet_{prec}.plan"
        build_s = build_engine(trt, onnx_path, plan, prec, args.sw_batch, logger)
        engines[prec] = {"build_seconds": round(build_s, 1),
                         "plan_mb": round(plan.stat().st_size / 1e6, 1), "path": plan}
        print(f"built {prec}: {build_s:.1f} s, {engines[prec]['plan_mb']} MB")

    model = model.to(dev).eval()
    stream = torch.cuda.Stream()
    backends = {
        "torch_fp32": model,
        "torch_amp": model,
        "trt_fp32": TrtPredictor(trt, engines["fp32"]["path"], logger),
        "trt_fp16": TrtPredictor(trt, engines["fp16"]["path"], logger),
    }

    test = build_datalist(args.dataset)["test"]
    if args.limit:
        test = test[:args.limit]
    loader = DataLoader(Dataset(test, val_transforms()), batch_size=1, num_workers=0)

    # warm every backend on a real volume so first case timings are not build or autotune time
    warm = next(iter(loader))["image"].to(dev)
    for name, pred in backends.items():
        run_backend(name, pred, warm, args.sw_batch, stream)

    dice_m = DiceMetric(include_background=False, reduction="none")
    hd_m = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    rows = {b: {"dice": {c: [] for c in CLASSES}, "hd95": {c: [] for c in CLASSES},
                "seconds": [], "agree_vs_torch_amp": [], "voxels_changed": []} for b in backends}

    for i, (batch, item) in enumerate(zip(loader, test)):
        x = batch["image"].to(dev)
        y = batch["label"].to(dev)
        y1h = one_hot(y, 3)
        preds = {}
        for name, pred in backends.items():
            p, secs = run_backend(name, pred, x, args.sw_batch, stream)
            preds[name] = p
            d = dice_m(y_pred=one_hot(p, 3), y=y1h)[0].tolist()
            h = hd_m(y_pred=one_hot(p, 3), y=y1h, spacing=PIXDIM)[0].tolist()  # millimetres, see evaluate.py
            for ci, c in enumerate(CLASSES):
                rows[name]["dice"][c].append(float(d[ci]))
                rows[name]["hd95"][c].append(float(h[ci]))
            rows[name]["seconds"].append(secs)
        base = preds["torch_amp"]
        for name in backends:
            same = (preds[name] == base)
            rows[name]["agree_vs_torch_amp"].append(float(same.float().mean().item()))
            rows[name]["voxels_changed"].append(int((~same).sum().item()))
        shape = list(x.shape[2:])
        print(f"[{i + 1}/{len(test)}] {item['patient_id']} {shape} " + "  ".join(
            f"{n}: kid {rows[n]['dice']['kidney'][-1]:.3f} tum {rows[n]['dice']['tumor'][-1]:.3f} "
            f"{rows[n]['seconds'][-1]:.2f}s" for n in backends))

    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "tensorrt": trt.__version__, "torch": torch.__version__,
        "n_cases": len(test), "roi": ROI, "sw_batch_size": args.sw_batch, "overlap": 0.5,
        "engines": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in engines.items()},
        "backends": {},
    }
    base_s = np.median(rows["torch_fp32"]["seconds"])
    for name, r in rows.items():
        summary["backends"][name] = {
            "dice": {c: mean_ci(r["dice"][c]) for c in CLASSES},
            "hd95_mm": {c: mean_ci(r["hd95"][c]) for c in CLASSES},
            "median_seconds_per_volume": round(float(np.median(r["seconds"])), 3),
            "total_seconds": round(float(np.sum(r["seconds"])), 1),
            "speedup_vs_torch_fp32": round(float(base_s / np.median(r["seconds"])), 2),
            "min_voxel_agreement_vs_torch_amp": round(float(np.min(r["agree_vs_torch_amp"])), 6),
            "total_voxels_changed_vs_torch_amp": int(np.sum(r["voxels_changed"])),
            "max_abs_dice_delta_vs_torch_amp": {
                c: round(float(np.nanmax(np.abs(np.array(r["dice"][c]) -
                                             np.array(rows["torch_amp"]["dice"][c])))), 4)
                for c in CLASSES},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["backends"], indent=1))


if __name__ == "__main__":
    main()
