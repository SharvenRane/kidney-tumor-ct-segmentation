"""MONAI Label app serving the trained kidney model to annotation clients.

    monailabel start_server --app monailabel_app --studies <folder of .nii.gz CTs> \
        --conf model_path outputs/best.pt

Any MONAI Label client (3D Slicer, OHIF) can then list the studies, request a pre segmentation,
correct it, and save the corrected label back, which lands in the datastore under labels/final.
Two sample selection strategies decide which unlabelled study to annotate next.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random

from lib.infer_kidney import KidneySegmentation

logger = logging.getLogger(__name__)


class KidneyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        default = os.path.join(app_dir, "..", "outputs", "best.pt")
        self.model_path = os.path.realpath(conf.get("model_path", default))
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"trained weights not found at {self.model_path}; pass --conf model_path")
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Kidney and renal tumor CT segmentation",
            description="Pre segmentation of kidney and tumor for human correction",
            version=monailabel.__version__,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {"kidney_tumor": KidneySegmentation(path=self.model_path)}

    def init_strategies(self) -> Dict[str, Strategy]:
        return {"first": First(), "random": Random()}
