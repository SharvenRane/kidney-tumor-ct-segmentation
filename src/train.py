"""Train a 3D SegResNet for kidney + tumor CT segmentation on C4KC-KiTS.

Patch-based training with AMP fits comfortably on a 16GB GPU. Validation uses
sliding-window inference over full volumes and reports per-class Dice (kidney,
tumor). The best checkpoint by mean foreground Dice is saved. Metrics are logged
to MLflow and to outputs/metrics.json.
"""
import argparse
import json
import os

import torch
from monai.data import (
    CacheDataset,
    DataLoader,
    PersistentDataset,
    decollate_batch,
    pad_list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

from data import build_datalist, train_transforms, val_transforms

ROI = (96, 96, 96)
CLASSES = ["background", "kidney", "tumor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C:/Users/sharv/data/c4kc-kits-nifti/dataset.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--cache-rate", type=float, default=1.0)
    ap.add_argument("--cache-dir", default=None,
                    help="if set, use disk-backed PersistentDataset (low RAM)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    args = ap.parse_args()

    set_determinism(seed=42)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = build_datalist(args.dataset)
    if args.smoke:
        split["train"] = split["train"][:2] or split["test"][:2]
        split["val"] = split["val"][:1]
        args.epochs, args.val_interval, args.cache_rate = 2, 1, 0.0
        args.num_samples = 2
    print(f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} device={device}")

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        train_ds = PersistentDataset(split["train"], train_transforms(ROI, args.num_samples),
                                     cache_dir=os.path.join(args.cache_dir, "train"))
        val_ds = PersistentDataset(split["val"], val_transforms(),
                                   cache_dir=os.path.join(args.cache_dir, "val"))
    else:
        train_ds = CacheDataset(split["train"], train_transforms(ROI, args.num_samples),
                                cache_rate=args.cache_rate, num_workers=0)
        val_ds = CacheDataset(split["val"], val_transforms(),
                              cache_rate=args.cache_rate, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=pad_list_data_collate)

    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=3,
                      init_filters=16, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                      dropout_prob=0.2).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
    post_label = Compose([AsDiscrete(to_onehot=3)])
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    try:
        import mlflow
        mlflow.set_experiment("kidney-tumor-ct-segmentation")
        mlflow.start_run()
        mlflow.log_params({"model": "SegResNet", "roi": str(ROI), "lr": args.lr,
                           "epochs": args.epochs, "spacing": "1.5iso"})
        use_mlflow = True
    except Exception:  # noqa: BLE001
        use_mlflow = False

    best_dice, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            steps += 1
        scheduler.step()
        epoch_loss /= max(steps, 1)

        line = {"epoch": epoch + 1, "train_loss": round(epoch_loss, 4)}
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    with torch.amp.autocast("cuda"):
                        logits = sliding_window_inference(x, ROI, 2, model, overlap=0.5,
                                                          mode="gaussian")
                    preds = [post_pred(p) for p in decollate_batch(logits)]
                    labels = [post_label(l) for l in decollate_batch(y)]
                    dice_metric(y_pred=preds, y=labels)
            per_class = dice_metric.aggregate().tolist()  # [kidney, tumor]
            mean_dice = sum(per_class) / len(per_class)
            line.update({"dice_kidney": round(per_class[0], 4),
                         "dice_tumor": round(per_class[1], 4),
                         "dice_mean": round(mean_dice, 4)})
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))
                line["saved_best"] = True
        history.append(line)
        print(line)
        if use_mlflow:
            import mlflow
            mlflow.log_metrics({k: v for k, v in line.items()
                                if isinstance(v, (int, float))}, step=epoch + 1)

    summary = {"best_mean_dice": round(best_dice, 4), "epochs": args.epochs,
               "n_train": len(split["train"]), "n_val": len(split["val"]),
               "history": history}
    with open(os.path.join(args.outdir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if use_mlflow:
        import mlflow
        mlflow.end_run()
    print(f"Done. Best mean foreground Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
