"""MONAI Label inference task for the trained kidney and tumor SegResNet.

Preprocessing and sliding window settings are the ones src/evaluate.py validated: RAS orientation,
1.5 mm isotropic resampling, abdominal HU window -200 to 300, 96 cubed Gaussian weighted windows
with 0.5 overlap. The difference is the output: MONAI Label's Restored transform maps the label back
to the original CT grid, which is what a viewer overlays on the scan.
"""
from __future__ import annotations

from typing import Callable, Sequence

from monai.inferers import Inferer, SlidingWindowInferer
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored

LABELS = {"kidney": 1, "tumor": 2}


def kidney_network() -> SegResNet:
    return SegResNet(spatial_dims=3, in_channels=1, out_channels=3, init_filters=16,
                     blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))


class KidneySegmentation(BasicInferTask):
    def __init__(self, path: str, **kwargs):
        super().__init__(
            path=path,
            network=kidney_network(),
            type=InferType.SEGMENTATION,
            labels=LABELS,
            dimension=3,
            description="Kidney and renal tumor segmentation on contrast enhanced abdominal CT (SegResNet)",
            **kwargs,
        )

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            ScaleIntensityRanged(keys="image", a_min=-200.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5, mode="gaussian")

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            Restored(keys="pred", ref_image="image"),
        ]
