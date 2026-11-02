"""Training loop with mixed precision, cosine LR schedule and early stopping."""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import load_splits, make_loader
from model import BoneAgeModel


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device):
    model.train()
    total = 0.0
    for img, sex, target in tqdm(loader, desc="train", leave=False):
        img, sex, target = img.to(device), sex.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = loss_fn(model(img, sex), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(img)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, stats, device):
    """Return predictions in months (de-normalized)."""
    model.eval()
    preds = []
    for img, sex, _ in tqdm(loader, desc="predict", leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(img.to(device), sex.to(device))
        preds.append(out.float().cpu())
    return torch.cat(preds).squeeze(1).numpy() * stats[1] + stats[0]


def main(args):
    set_seed()
    device = "cuda"
    df, stats = load_splits()
    train_loader = make_loader(df, "train", stats, args.img_size, batch_size=args.batch_size)
    val_loader = make_loader(df, "val", stats, args.img_size, batch_size=args.batch_size)
    val_true = df[df["split"] == "val"]["boneage"].values

    model = BoneAgeModel(args.backbone).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
        {"params": list(model.sex_fc.parameters()) + list(model.head.parameters()), "lr": config.LR_HEAD},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.L1Loss()

    run_dir = config.OUTPUT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_mae, bad_epochs, log = float("inf"), 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
        val_mae = float(np.mean(np.abs(predict(model, val_loader, stats, device) - val_true)))
        scheduler.step()

        log.append({"epoch": epoch, "train_loss": train_loss, "train_mae_months": train_loss * stats[1],
                    "val_mae_months": val_mae, "seconds": time.time() - t0})
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)

        improved = val_mae < best_mae
        if improved:
            best_mae, bad_epochs = val_mae, 0
            torch.save({"model": model.state_dict(), "stats": stats, "backbone": args.backbone,
                        "img_size": args.img_size, "epoch": epoch, "val_mae": val_mae},
                       run_dir / "best_model.pth")
        else:
            bad_epochs += 1
        print(f"epoch {epoch:02d} | train MAE {train_loss * stats[1]:.2f} mo | val MAE {val_mae:.2f} mo"
              f" | {time.time() - t0:.0f}s {'*' if improved else ''}")
        if bad_epochs >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best val MAE: {best_mae:.2f} months -> {run_dir / 'best_model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="baseline_b0_384")
    parser.add_argument("--backbone", default=config.BACKBONE)
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    main(parser.parse_args())
