"""TensorRT engine parity for the deployment path. Skipped without a CUDA GPU and TensorRT."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import deploy_tensorrt as D  # noqa: E402

try:
    trt = D._import_tensorrt()
except ImportError:
    pytest.skip("needs TensorRT", allow_module_level=True)


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    work = tmp_path_factory.mktemp("trt")
    torch.manual_seed(0)
    model = D.make_model().eval()
    logger = trt.Logger(trt.Logger.ERROR)
    onnx_path = D.export_onnx(model, work / "m.onnx", sw_batch=2)
    plans = {}
    for prec in ["fp32", "fp16"]:
        plans[prec] = work / f"m_{prec}.plan"
        D.build_engine(trt, onnx_path, plans[prec], prec, 2, logger)
    return model.cuda(), plans, logger


def _input(n):
    g = torch.Generator(device="cuda").manual_seed(1)
    return torch.randn(n, 1, *D.ROI, device="cuda", generator=g)


def _decided(ref, margin=1e-2):
    # An untrained network produces near ties; only voxels with a clear winner can be compared.
    top2 = ref.topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]) > margin


def _reference(model, x):
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        return model(x)


def test_fp32_engine_matches_pytorch(engines):
    model, plans, logger = engines
    x = _input(2)
    ref = _reference(model, x)
    out = D.TrtPredictor(trt, plans["fp32"], logger)(x)
    torch.cuda.synchronize()
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 1e-3
    m = _decided(ref)
    assert torch.equal(out.argmax(1)[m], ref.argmax(1)[m])


def test_fp16_engine_keeps_float32_io_and_close_logits(engines):
    model, plans, logger = engines
    x = _input(1)  # batch below the optimum still runs
    ref = _reference(model, x)
    out = D.TrtPredictor(trt, plans["fp16"], logger)(x)
    torch.cuda.synchronize()
    assert out.dtype == torch.float32
    m = _decided(ref)
    agree = (out.argmax(1)[m] == ref.argmax(1)[m]).float().mean().item()
    assert agree > 0.999
