"""Dataset, augmentations and image caching."""
from concurrent.futures import ThreadPoolExecutor

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_one(img_id):
    out_path = config.CACHE_DIR / f"{img_id}.png"
    if out_path.exists():
        return
    img = cv2.imread(str(config.TRAIN_IMG_DIR / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
    scale = config.CACHE_SIZE / max(img.shape)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), img)


def cache_images(ids):
    """Resize raw images once (they are ~2000px) so training is not bottlenecked by decoding."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(tqdm(pool.map(_resize_one, ids), total=len(ids), desc="Caching images"))


def get_transforms(train, img_size=config.IMG_SIZE, extra=()):
    resize = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
    ]
    augment = [
        A.Affine(rotate=(-15, 15), translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
    ] if train else []
    normalize = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(resize + augment + list(extra) + normalize)


class BoneAgeDataset(Dataset):
    """Returns (image [3,H,W], sex [1], normalized target [1])."""

    def __init__(self, df, transform, target_mean, target_std):
        self.ids = df["id"].values
        self.sex = df["male"].astype(np.float32).values
        self.target_mean, self.target_std = target_mean, target_std
        self.target = ((df["boneage"].values - target_mean) / target_std).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = cv2.imread(str(config.CACHE_DIR / f"{self.ids[idx]}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 3 channels for the pretrained backbone
        img = self.transform(image=img)["image"]
        sex = torch.tensor([self.sex[idx]])
        target = torch.tensor([self.target[idx]])
        return img, sex, target


def load_splits():
    df = pd.read_csv(config.SPLITS_CSV)
    train = df[df["split"] == "train"]
    stats = (train["boneage"].mean(), train["boneage"].std())
    return df, stats


def make_loader(df, split, stats, img_size=config.IMG_SIZE, train=None, batch_size=config.BATCH_SIZE):
    train = (split == "train") if train is None else train
    part = df[df["split"] == split].reset_index(drop=True)
    ds = BoneAgeDataset(part, get_transforms(train, img_size), *stats)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=config.NUM_WORKERS,
                      pin_memory=True, drop_last=train, persistent_workers=True)


def show_batch(loader, n=8, path=None):
    """Plot a few (augmented) images from a loader with their ages."""
    imgs, sexes, targets = next(iter(loader))
    ds = loader.dataset
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i, ax in enumerate(axes):
        img = imgs[i].permute(1, 2, 0).numpy() * std + mean
        ax.imshow(np.clip(img, 0, 1))
        age = targets[i].item() * ds.target_std + ds.target_mean
        ax.set_title(f"{age:.0f} mo, {'M' if sexes[i].item() else 'F'}")
        ax.axis("off")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=100)
    return fig


if __name__ == "__main__":
    df, stats = load_splits()
    cache_images(df["id"].tolist())
    loader = make_loader(df, "train", stats)
    imgs, sexes, targets = next(iter(loader))
    print("image:", imgs.shape, imgs.dtype, "sex:", sexes.shape, "target:", targets.shape)
    print(f"target mean/std (train): {stats[0]:.2f} / {stats[1]:.2f}")
    show_batch(loader, path=config.OUTPUT_DIR / "augmented_batch.png")
    print(f"Saved {config.OUTPUT_DIR / 'augmented_batch.png'}")
