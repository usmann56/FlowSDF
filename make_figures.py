"""
Figures for the slides/report, from eval_thresholds.py CSVs and the maps saved by eval_monuseg.py.

  fig_threshold.png  Dice and boundary F1 against the foreground threshold, validation and test
  fig_auroc.png      variance AUROC (whole image and boundary band) per condition, validation and test
  fig_example.png    one test image: contours, variance map and error map for two conditions

Usage:
    python make_figures.py --val-csv runs/eval/final_val_thresholds.csv --test-csv runs/eval/test_thresholds.csv \
        --test-eval-root runs/eval/test --raw-test-dir /data/.../raw/monuseg_test --out figures
"""
import argparse
import glob
import os
import re

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess_data.preprocess_monuseg import find_files, load_sample

CONDS = ["e1cont", "e2", "e3", "e3b"]
LABELS = {"e1": "E1", "e1cont": "E1-cont (baseline)", "e2": "E2 uniform", "e3": "E3 variance",
          "e3b": "E3b entropy", "full37": "37 img bf16", "full37_fp32": "37 img fp32"}
COLORS = {"e1cont": "#6b7280", "e2": "#2563eb", "e3": "#dc2626", "e3b": "#16a34a"}


def condition(name):
    return re.sub(r"^final_|_val$", "", name)


def seed_means(csv):
    """Mean over images per (condition, seed, threshold)."""
    df = pd.read_csv(csv)
    df["cond"] = df["eval"].map(condition)
    return df.groupby(["cond", "ckpt", "thresh"]).mean(numeric_only=True).reset_index()


def fig_threshold(splits, out):
    fig, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 6.5), squeeze=False)
    for j, (title, sm) in enumerate(splits):
        for i, m in enumerate(["dice", "bf1"]):
            ax = axes[i, j]
            for c in CONDS:
                g = sm[sm.cond == c].groupby("thresh")[m]
                if g.ngroups == 0:
                    continue
                x, mean, lo, hi = g.mean().index, g.mean().values, g.min().values, g.max().values
                ax.plot(x, mean, "o-", color=COLORS[c], label=LABELS[c], ms=4)
                ax.fill_between(x, lo, hi, color=COLORS[c], alpha=0.15)  # range over seeds
            ax.axvline(0.0, color="k", lw=0.8, ls=":")
            ax.set_xlabel("foreground threshold (SDF <= t)")
            ax.set_ylabel({"dice": "Dice", "bf1": "boundary F1"}[m])
            ax.set_title(f"{title}: {'Dice' if m == 'dice' else 'boundary F1'}")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_auroc(splits, thresh, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, m, name in zip(axes, ["var_auroc", "var_auroc_band"], ["whole image", "5 px boundary band"]):
        width = 0.8 / len(splits)
        for k, (title, sm) in enumerate(splits):
            s = sm[np.isclose(sm.thresh, thresh)]
            for i, c in enumerate(CONDS):
                vals = s[s.cond == c][m].values
                if len(vals) == 0:
                    continue
                x = i + (k - (len(splits) - 1) / 2) * width
                ax.bar(x, vals.mean(), width, color=COLORS[c], alpha=0.55 if k == 0 else 0.95,
                       edgecolor="k", lw=0.5, label=title if i == 0 else None)
                ax.scatter(np.full(len(vals), x), vals, color="k", s=10, zorder=3)  # seeds
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xticks(range(len(CONDS)), [LABELS[c] for c in CONDS], rotation=15, fontsize=8)
        ax.set_ylim(0.4, 0.85)
        ax.set_ylabel("AUROC (variance vs error)")
        ax.set_title(f"Uncertainty quality, {name}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def find_map(eval_root, cond, image, seed):
    hits = sorted(glob.glob(os.path.join(eval_root, cond, "maps", f"*_seed{seed}__*", f"{image}.npz")))
    if not hits:
        raise FileNotFoundError(f"no map for {cond} seed {seed} image {image} under {eval_root}")
    return np.load(hits[0])


def fig_example(test_csv, eval_root, raw_dir, conds, image, seed, thresh, crop, out):
    df = pd.read_csv(test_csv)
    df["cond"] = df["eval"].map(condition)
    if image is None:  # image where the variance AUROC gains most over the first condition
        s = df[np.isclose(df.thresh, thresh)].groupby(["cond", "image"]).var_auroc.mean()
        image = (s.loc[conds[1]] - s.loc[conds[0]]).idxmax()
    imgs, annots = find_files(raw_dir, (".tif", ".tiff", ".png")), find_files(raw_dir, (".xml",))
    img, _, _, inst = load_sample(imgs[image], annots[image], 500, thresh=15.0)
    gt = inst > 0
    y0, x0, sz = crop if crop else (0, 0, img.shape[0])
    sl = (slice(y0, y0 + sz), slice(x0, x0 + sz))

    fig, axes = plt.subplots(len(conds), 3, figsize=(10, 3.4 * len(conds)), squeeze=False)
    for r, c in enumerate(conds):
        maps = find_map(eval_root, c, image, seed)
        mmse, var = maps["mmse"].astype(np.float32), maps["var"].astype(np.float32)
        pred = mmse <= thresh
        over = img.copy()
        for mask, col in ((gt, (0, 200, 0)), (pred, (40, 40, 230))):
            cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(over, cnt, -1, col, 1)
        err = pred != gt
        row = df[(df.cond == c) & (df.image == image) & np.isclose(df.thresh, thresh) & df.ckpt.str.contains(f"_seed{seed}__")]
        auc = row.var_auroc.mean()
        axes[r, 0].imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB)[sl])
        axes[r, 0].set_title(f"{LABELS.get(c, c)}: GT green, prediction red", fontsize=9)
        axes[r, 1].imshow(var[sl], cmap="inferno", vmax=np.percentile(var, 99.5))
        axes[r, 1].set_title(f"sampling variance (AUROC {auc:.2f})", fontsize=9)
        axes[r, 2].imshow(err[sl], cmap="gray")
        axes[r, 2].set_title(f"error (pred != GT), t = {thresh:g}", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(f"{image}, seed {seed}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", help="eval_thresholds.py output on validation (full threshold range)")
    parser.add_argument("--test-csv", required=True, help="eval_thresholds.py output on test")
    parser.add_argument("--test-eval-root", default="runs/eval/test", help="folder holding <cond>/maps/")
    parser.add_argument("--raw-test-dir", required=True)
    parser.add_argument("--auroc-thresh", type=float, default=0.05, help="threshold whose errors define the AUROC")
    parser.add_argument("--example-conds", nargs=2, default=["e1cont", "e2"])
    parser.add_argument("--example-image", default=None, help="default: largest AUROC gain on test")
    parser.add_argument("--example-seed", type=int, default=0)
    parser.add_argument("--example-thresh", type=float, default=0.05)
    parser.add_argument("--crop", nargs=3, type=int, metavar=("Y", "X", "SIZE"), default=None)
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    splits = ([("validation", seed_means(args.val_csv))] if args.val_csv else []) + [("test", seed_means(args.test_csv))]
    fig_threshold(splits, os.path.join(args.out, "fig_threshold.png"))
    fig_auroc(splits, args.auroc_thresh, os.path.join(args.out, "fig_auroc.png"))
    img = fig_example(args.test_csv, args.test_eval_root, args.raw_test_dir, args.example_conds, args.example_image,
                      args.example_seed, args.example_thresh, args.crop, os.path.join(args.out, "fig_example.png"))
    print(f"wrote {args.out}/fig_threshold.png, fig_auroc.png, fig_example.png (example image {img})")
