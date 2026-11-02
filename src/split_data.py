"""Stratified train/val/test split + simple baselines."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def make_splits(df):
    """Split by (boneage bin, sex) so every split has a similar distribution."""
    age_bin = pd.qcut(df["boneage"], q=config.N_AGE_BINS, labels=False, duplicates="drop")
    strat = age_bin.astype(str) + "_" + df["male"].astype(str)

    train_df, rest_df = train_test_split(
        df, test_size=config.VAL_FRAC + config.TEST_FRAC,
        stratify=strat, random_state=config.SEED,
    )
    rest_strat = strat.loc[rest_df.index]
    val_df, test_df = train_test_split(
        rest_df, test_size=config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
        stratify=rest_strat, random_state=config.SEED,
    )

    df = df.copy()
    df["split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"
    return df


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def baselines(df):
    """Mean and per-sex mean baselines (fit on train, scored on val and test)."""
    train = df[df["split"] == "train"]
    global_mean = train["boneage"].mean()
    sex_means = train.groupby("male")["boneage"].mean()

    rows = []
    for split in ["val", "test"]:
        part = df[df["split"] == split]
        rows.append({
            "split": split,
            "mean_baseline_mae": mae(part["boneage"], np.full(len(part), global_mean)),
            "sex_mean_baseline_mae": mae(part["boneage"], part["male"].map(sex_means)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_CSV)
    df = make_splits(df)
    df.to_csv(config.SPLITS_CSV, index=False)
    print(f"Saved {config.SPLITS_CSV}")
    print(df.groupby("split").agg(n=("id", "size"), age_mean=("boneage", "mean"),
                                  age_std=("boneage", "std"), male_ratio=("male", "mean")))

    result = baselines(df)
    print("\nBaselines (MAE, months):")
    print(result.to_string(index=False))
    result.to_csv(config.OUTPUT_DIR / "baselines.csv", index=False)
