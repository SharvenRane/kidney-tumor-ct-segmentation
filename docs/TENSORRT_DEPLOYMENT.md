# TensorRT Deployment: Speed at the Validated Dice

A segmentation model is only deployable at the accuracy it was validated at. So the question
here is not how fast TensorRT FP16 is, but how fast it is while keeping the Dice reported in
`VALIDATION_REPORT.md`. Every backend below ran the same sliding window inference over the same
32 held out CT volumes and was scored against the clinician reference and, voxel by voxel,
against the validated PyTorch configuration.

Produced by `src/deploy_tensorrt.py`; raw numbers in `outputs/tensorrt_metrics.json`.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Software | TensorRT 10.16.1.11, PyTorch 2.11.0+cu128, MONAI 1.5.2 |
| Model | 3D SegResNet, 3 classes, trained weights from `outputs/best.pt` |
| Inference | sliding window, 96 cubed patches, overlap 0.5, Gaussian blending, 4 patches per call |
| Test set | 32 held out patients, 1.5 mm isotropic, patient level split |
| Timing | wall clock per volume around the full sliding window call, synchronised, after a warm up volume; data loading excluded |

Engines are built with the TensorRT Python API directly: ONNX parser, builder config, an
optimization profile over patch batch 1 to 4, and a serialized plan. Inference calls
`execute_async_v3` on a dedicated CUDA stream and reads and writes PyTorch tensors in place.
The FP32 engine and the FP32 PyTorch baseline both have TF32 disabled, so FP32 means FP32.

## Results

| backend | seconds per volume (median) | speedup vs PyTorch FP32 | kidney Dice (95% CI) | tumor Dice (95% CI) | voxels changed vs validated config |
|---|---|---|---|---|---|
| PyTorch FP32 | 3.142 | 1.00x | 0.9201 (0.8901 to 0.9446) | 0.6687 (0.5736 to 0.7534) | 459 |
| PyTorch AMP (validated) | 1.344 | 2.34x | 0.9201 (0.8901 to 0.9446) | 0.6687 (0.5736 to 0.7534) | 0 |
| TensorRT FP32 | 1.299 | 2.42x | 0.9201 (0.8901 to 0.9446) | 0.6687 (0.5736 to 0.7534) | 459 |
| **TensorRT FP16** | **0.360** | **8.72x** | 0.9201 (0.8901 to 0.9446) | 0.6688 (0.5738 to 0.7534) | 758 |

Mean HD95 is 14.2 mm for kidney and 37.4 mm for tumor on every backend. Across all 32 volumes the
largest per case Dice change for TensorRT FP16 is 0.0001 on kidney and 0.0006 on tumor, and its
lowest per volume voxel agreement with the validated configuration is 0.999992.

| engine | build time | plan size |
|---|---|---|
| FP32 | 23.6 s | 20.6 MB |
| FP16 | 19.4 s | 10.2 MB |

## What the numbers say

**FP16 is 3.7x faster than the configuration that was actually validated, at the same Dice.** The
honest comparison is not against FP32 eager, which nobody deploys, but against mixed precision
PyTorch. TensorRT FP16 takes a whole abdominal CT from 1.34 s to 0.36 s and the tumor class,
the minority and clinically important one, moves by one ten thousandth of a Dice point on average.

**TensorRT FP32 reproduces PyTorch FP32 voxel for voxel.** Both differ from the AMP baseline in
the same 459 voxels. That is a useful check that the ONNX export and engine are faithful, and a
reminder that the validated AMP numbers were themselves not bit identical to FP32.

**TensorRT at FP32 buys almost nothing over AMP.** 1.30 s against 1.34 s. Without the precision
change, graph fusion alone did not move this model much; the speed comes from FP16 kernels.

**Per class numbers are the acceptance test, not the mean.** On a chest X-ray classifier in a
companion benchmark, INT8 kept aggregate accuracy identical while tuberculosis recall fell. Here
the per class Dice and the voxel change count are checked for every backend, and FP16 passes
on both classes.

## One thing that cost time

Calling `execute_async_v3` on PyTorch's default CUDA stream works, but TensorRT then inserts extra
stream synchronisations and logs a warning about it. The first run used the default stream; the
numbers above come from a dedicated stream, which is what a real serving loop would use.

## Reproduce

```bash
python src/deploy_tensorrt.py --checkpoint outputs/best.pt
pytest tests/test_tensorrt.py   # engine parity; skipped without a CUDA GPU and TensorRT
```

On Windows the TensorRT wheel keeps its DLLs in `tensorrt_libs`, which is not on the DLL search
path; the script adds it before importing `tensorrt`.

## Limits

One GPU, one model, one sliding window configuration. 32 volumes from a single collection. INT8
was not attempted: calibrating a 3D segmentation model properly needs a representative
calibration set drawn from training cases and a per class acceptance check like the one above,
and FP16 already meets the target without it.
