"""Explainability overlays for the kidney + tumor segmentation model.

Two complementary explanations:

  * Prediction vs reference overlay (blend_images) on the axial slice with the
    most tumor, so a reader can see where the model agrees and disagrees with the
    clinician mask.
  * Gradient saliency for the tumor class on a window centered on the tumor. It
    backpropagates the summed tumor logit to the input voxels, answering "which
    input regions does the tumor prediction actually depend on". This suits a dense
    segmentation network, where occlusion sensitivity (a classifier method) does
    not map cleanly onto the per voxel output.

Outputs PNGs to outputs/explain/. These support GMLP Principle 9 (give users clear
information) and the transparency guiding principles.
"""
import argparse
import os

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import SegResNet
from monai.visualize import blend_images

from data import build_datalist, val_transforms

ROI = (96, 96, 96)
TUMOR = 2


def tumor_slice(label_np):
    counts = (label_np == TUMOR).sum(axis=(0, 1))
    return int(np.argmax(counts)) if counts.max() > 0 else label_np.shape[2] // 2


def save_overlay(image2d, label2d, pred2d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = image2d[None]  # (1,H,W)
    gt = blend_images(img, label2d[None], alpha=0.5, cmap="hsv", rescale_arrays=True)
    pr = blend_images(img, pred2d[None], alpha=0.5, cmap="hsv", rescale_arrays=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image2d, cmap="gray"); ax[0].set_title("CT")
    ax[1].imshow(np.moveaxis(np.asarray(gt), 0, -1)); ax[1].set_title("Reference (clinician)")
    ax[2].imshow(np.moveaxis(np.asarray(pr), 0, -1)); ax[2].set_title("Model prediction")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def save_saliency(sal_map2d, image2d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image2d, cmap="gray"); ax[0].set_title("CT")
    ax[1].imshow(image2d, cmap="gray")
    im = ax[1].imshow(sal_map2d, cmap="jet", alpha=0.5)
    ax[1].set_title("Tumor gradient saliency")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C:/Users/sharv/data/c4kc-kits-nifti/dataset.json")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--outdir", default="outputs/explain")
    ap.add_argument("--n-cases", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = build_datalist(args.dataset)
    cases = split["test"][: args.n_cases]
    loader = DataLoader(CacheDataset(cases, val_transforms(), cache_rate=0.0, num_workers=0),
                        batch_size=1, num_workers=0)

    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=3, init_filters=16,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    from monai.inferers import sliding_window_inference

    def tumor_window(vol, cy, cx, cz):
        """A 96^3 window centered on the tumor, clamped inside the volume."""
        H, W, D = vol.shape[-3:]
        def span(c, n):
            lo = min(max(0, c - ROI[0] // 2), max(0, n - ROI[0]))
            return slice(lo, lo + min(ROI[0], n))
        return span(cy, H), span(cx, W), span(cz, D)

    for case, batch in zip(cases, loader):
        pid = case["patient_id"]
        x = batch["image"].to(device)
        label_np = batch["label"][0, 0].cpu().numpy()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = sliding_window_inference(x, ROI, 2, model, overlap=0.5, mode="gaussian")
        pred_np = torch.argmax(logits, dim=1)[0].cpu().numpy()
        z = tumor_slice(label_np)
        img2d = x[0, 0, :, :, z].cpu().numpy()
        save_overlay(img2d, label_np[:, :, z], pred_np[:, :, z],
                     os.path.join(args.outdir, f"{pid}_overlay.png"))

        # Gradient saliency for the tumor class on a 96^3 window around the tumor.
        ys, xs = np.where(label_np[:, :, z] == TUMOR)
        if len(xs):
            cy, cx = int(ys.mean()), int(xs.mean())
            sy, sx, sz = tumor_window(x, cy, cx, z)
            patch = x[:, :, sy, sx, sz].clone().requires_grad_(True)
            try:
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                logits = model(patch)
                logits[0, TUMOR].sum().backward()
                sal = patch.grad[0, 0].abs()
                mz = (z - sz.start) if sz.start <= z < sz.stop else sal.shape[-1] // 2
                sal2d = sal[:, :, mz].cpu().numpy()
                if sal2d.max() > 0:
                    sal2d = sal2d / sal2d.max()
                save_saliency(sal2d, patch[0, 0, :, :, mz].detach().cpu().numpy(),
                              os.path.join(args.outdir, f"{pid}_saliency.png"))
            except Exception as e:  # noqa: BLE001
                print(f"  {pid}: saliency skipped ({e})")
        print(f"{pid}: overlay + saliency saved (tumor slice z={z})")

    print(f"Explainability outputs in {args.outdir}/")


if __name__ == "__main__":
    main()
