"""Grad-CAM heatmaps on the last backbone feature map."""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from dataset import BoneAgeDataset, get_transforms, load_splits
from model import BoneAgeModel


def grad_cam(model, img, sex):
    """Weight each feature channel by its mean gradient, sum, ReLU -> heatmap in [0, 1]."""
    feature_map = model.backbone.forward_features(img)
    feature_map.retain_grad()
    pred = model.forward_from_features(feature_map, sex)
    model.zero_grad()
    pred.sum().backward()

    weights = feature_map.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=1)).squeeze(0).detach().cpu().numpy()
    cam = cv2.resize(cam, (img.shape[3], img.shape[2]))
    return cam / (cam.max() + 1e-8), pred.item()


def main(args):
    device = "cuda"
    ckpt = torch.load(config.OUTPUT_DIR / args.run / "best_model.pth", map_location=device, weights_only=False)
    model = BoneAgeModel(ckpt["backbone"], pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    stats = ckpt["stats"]

    df, _ = load_splits()
    part = df[df["split"] == "val"].sort_values("boneage").reset_index(drop=True)
    picks = part.iloc[np.linspace(0, len(part) - 1, args.n).astype(int)].reset_index(drop=True)
    ds = BoneAgeDataset(picks, get_transforms(False, ckpt["img_size"]), *stats)

    fig, axes = plt.subplots(2, args.n, figsize=(2.6 * args.n, 6))
    for i in range(args.n):
        img, sex, _ = ds[i]
        cam, pred = grad_cam(model, img[None].to(device), sex[None].to(device))
        raw = img.permute(1, 2, 0).numpy()
        raw = (raw - raw.min()) / (raw.max() - raw.min())
        axes[0, i].imshow(raw)
        axes[0, i].set_title(f"true {picks.loc[i, 'boneage']:.0f} / pred {pred * stats[1] + stats[0]:.0f}")
        axes[1, i].imshow(raw)
        axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
        for ax in axes[:, i]:
            ax.axis("off")
    plt.tight_layout()
    out = config.OUTPUT_DIR / "gradcam.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="baseline_b0_384")
    parser.add_argument("--n", type=int, default=8)
    main(parser.parse_args())
