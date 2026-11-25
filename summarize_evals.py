"""
One table from several eval_monuseg.py output folders: mean over images, then mean and spread over
checkpoints (seeds), sorted by boundary F1.

Usage:
    python summarize_evals.py runs/eval/tune/*
"""
import argparse
import os

import pandas as pd

METRICS = ["dice", "iou", "hd95", "assd", "bf1", "aji", "var_auroc", "var_auroc_band"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dirs", nargs="+")
    parser.add_argument("--sort", default="bf1")
    args = parser.parse_args()

    frames = []
    for d in args.eval_dirs:
        f = os.path.join(d, "metrics.csv")
        if not os.path.exists(f):
            print(f"skipping {d}: no metrics.csv (still running or failed)")
            continue
        df = pd.read_csv(f)
        df["eval"] = os.path.basename(os.path.normpath(d))
        frames.append(df)
    df = pd.concat(frames)
    cols = [c for c in METRICS if c in df.columns]

    per_ckpt = df.groupby(["eval", "ckpt"])[cols].mean()
    table = per_ckpt.groupby("eval")[cols].mean()
    table.insert(0, "n_ckpt", per_ckpt.groupby("eval").size())
    if per_ckpt.groupby("eval").size().max() > 1:  # spread over seeds where there is more than one
        table[f"{args.sort}_seed_sd"] = per_ckpt.groupby("eval")[args.sort].std()
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 4):
        print(table.sort_values(args.sort, ascending=args.sort in ("hd95", "assd")))
