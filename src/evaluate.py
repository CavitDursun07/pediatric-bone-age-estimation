"""Evaluate one or more checkpoints (ensemble = mean of predictions), optionally with TTA."""
import argparse
import json

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

import config
from dataset import BoneAgeDataset, get_transforms, load_splits
from model import BoneAgeModel
from train import predict

AGE_BINS = [0, 60, 120, 180, 1000]
AGE_LABELS = ["0-5y", "5-10y", "10-15y", "15+y"]


def tta_transforms(img_size):
    """Original + small deterministic rotations."""
    return [get_transforms(False, img_size)] + [
        get_transforms(False, img_size, extra=[
            A.Affine(rotate=(angle, angle), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0)])
        for angle in (-7, 7)
    ]


def predict_checkpoint(path, part, use_tta, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BoneAgeModel(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    stats = ckpt["stats"]
    transforms = tta_transforms(ckpt["img_size"]) if use_tta else [get_transforms(False, ckpt["img_size"])]
    preds = []
    for t in transforms:
        loader = DataLoader(BoneAgeDataset(part, t, *stats), batch_size=32, num_workers=config.NUM_WORKERS)
        preds.append(predict(model, loader, stats, device))
    return np.mean(preds, axis=0)


def compute_metrics(res):
    metrics = {
        "n": len(res),
        "mae": mean_absolute_error(res["boneage"], res["pred"]),
        "rmse": float(np.sqrt(mean_squared_error(res["boneage"], res["pred"]))),
        "r2": r2_score(res["boneage"], res["pred"]),
    }
    by_age = res.groupby("age_group", observed=True)["abs_error"].agg(["mean", "size"])
    by_sex = res.groupby(res["male"].map({True: "male", False: "female"}))["abs_error"].agg(["mean", "size"])
    return metrics, by_age, by_sex


def plot_results(res, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(res["boneage"], res["pred"], s=8, alpha=0.5)
    lim = [0, max(res["boneage"].max(), res["pred"].max()) + 10]
    ax.plot(lim, lim, "r--", label="y = x")
    ax.set(xlim=lim, ylim=lim, xlabel="True bone age (months)", ylabel="Predicted (months)",
           title=f"Predicted vs true (MAE {res['abs_error'].mean():.2f} mo)")
    ax.legend()
    fig.savefig(out_dir / "scatter.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res["pred"] - res["boneage"], bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0, color="r", linestyle="--")
    ax.set(xlabel="Error = predicted - true (months)", ylabel="Count", title="Error distribution")
    fig.savefig(out_dir / "error_hist.png", dpi=120, bbox_inches="tight")

    worst = res.nlargest(10, "abs_error")
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    for ax, (_, row) in zip(axes.ravel(), worst.iterrows()):
        img = cv2.imread(str(config.CACHE_DIR / f"{row['id']}.png"), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"id {row['id']} {'M' if row['male'] else 'F'}\ntrue {row['boneage']:.0f} / pred {row['pred']:.0f}")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "worst10.png", dpi=100, bbox_inches="tight")
    plt.close("all")


def main(args):
    df, _ = load_splits()
    part = df[df["split"] == args.split].reset_index(drop=True)
    preds = [predict_checkpoint(config.OUTPUT_DIR / run / "best_model.pth", part, args.tta) for run in args.runs]

    res = part.copy()
    res["pred"] = np.mean(preds, axis=0)
    res["abs_error"] = (res["pred"] - res["boneage"]).abs()
    res["age_group"] = pd.cut(res["boneage"], AGE_BINS, labels=AGE_LABELS, right=False)

    name = args.name or ("+".join(args.runs) + ("_tta" if args.tta else ""))
    out_dir = config.OUTPUT_DIR / f"eval_{args.split}" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "predictions.csv", index=False)

    metrics, by_age, by_sex = compute_metrics(res)
    metrics.update({"runs": args.runs, "tta": args.tta, "split": args.split})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_results(res, out_dir)

    print(f"[{args.split}] {name}: MAE {metrics['mae']:.2f} | RMSE {metrics['rmse']:.2f} | R2 {metrics['r2']:.4f}")
    print("MAE by age group:\n", by_age.round(2).to_string())
    print("MAE by sex:\n", by_sex.round(2).to_string())
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["baseline_b0_384"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--name", default=None)
    main(parser.parse_args())
