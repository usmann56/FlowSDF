"""
Extra figures for the slides/report, from the saved test maps and the training logs.

  fig_gallery.png  several test images: input + ground truth, then control vs boundary loss at the SDF's
                   zero level (contours), and their variance maps
  fig_curves.png   training curves of the 10k-epoch branches: flow-matching loss (all conditions) and the
                   boundary loss (E2, E3, E3b), mean of the seeds, smoothed

Usage:
    python make_extra_figures.py --test-csv runs/eval/test_thresholds.csv --test-eval-root runs/eval/test \
        --raw-test-dir /data/.../raw/monuseg_test --runs-root runs/monuseg --out figures
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

from make_figures import COLORS, LABELS, condition, find_map
from preprocess_data.preprocess_monuseg import find_files, load_sample

LOG_RE = re.compile(r"Epoch (\d+) Loss: ([0-9.eE+-]+), Time: [0-9.]+s(?: \| cfm ([0-9.eE+-]+) boundary ([0-9.eE+-]+))?")
RUNS = {"e1cont": "monuseg_e1cont", "e2": "final_e2", "e3": "final_e3", "e3b": "final_e3b"}


def overlay(img, gt, pred):
    out = img.copy()
    for mask, col in ((gt, (0, 200, 0)), (pred, (40, 40, 230))):  # BGR: ground truth green, prediction red
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnt, -1, col, 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def fig_gallery(args):
    df = pd.read_csv(args.test_csv)
    df["cond"] = df["eval"].map(condition)
    a, b = args.conds
    if args.images:
        names = args.images
    else:  # images where the boundary loss gains most Dice at the zero level, then spread over the ranking
        s = df[np.isclose(df.thresh, args.thresh)].groupby(["cond", "image"]).dice.mean()
        gain = (s.loc[b] - s.loc[a]).sort_values(ascending=False)
        names = [gain.index[i] for i in np.linspace(0, len(gain) // 2, args.n).astype(int)]
    imgs, annots = find_files(args.raw_test_dir, (".tif", ".tiff", ".png")), find_files(args.raw_test_dir, (".xml",))
    cols = ["Image + ground truth", f"{LABELS[a]}, t = {args.thresh:g}", f"{LABELS[b]}, t = {args.thresh:g}",
            f"{LABELS[a]}: variance", f"{LABELS[b]}: variance"]
    fig, axes = plt.subplots(len(names), 5, figsize=(17, 3.6 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        img, _, _, inst = load_sample(imgs[name], annots[name], 500, thresh=15.0)
        gt = inst > 0
        y0 = x0 = (img.shape[0] - args.crop) // 2
        sl = (slice(y0, y0 + args.crop), slice(x0, x0 + args.crop))
        maps = {c: find_map(args.test_eval_root, c, name, args.seed) for c in (a, b)}
        var_max = max(np.percentile(maps[c]["var"].astype(np.float32), 99.5) for c in (a, b))
        panels = [overlay(img, gt, np.zeros_like(gt))]
        panels += [overlay(img, gt, maps[c]["mmse"].astype(np.float32) <= args.thresh) for c in (a, b)]
        for j, p in enumerate(panels):
            axes[r, j].imshow(p[sl])
        for j, c in enumerate((a, b)):
            axes[r, 3 + j].imshow(maps[c]["var"].astype(np.float32)[sl], cmap="inferno", vmin=0, vmax=var_max)
        dice = {c: df[(df.cond == c) & (df.image == name) & np.isclose(df.thresh, args.thresh)].dice.mean() for c in (a, b)}
        for j, c in ((1, a), (2, b)):
            axes[r, j].text(0.03, 0.97, f"Dice {dice[c]:.3f}", transform=axes[r, j].transAxes, ha="left", va="top",
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        axes[r, 0].set_ylabel(name.replace("-01Z-00-DX1", "").replace("-01A-01-TS1", ""), fontsize=11)
        for j in range(5):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if r == 0:
                axes[r, j].set_title(cols[j], fontsize=13)
    fig.suptitle(f"Test images, seed {args.seed}, {args.crop} px centre crops. Green: ground truth, red: prediction. "
                 "Variance maps share one colour scale per row.", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.out, "fig_gallery.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out, names


def read_log(path):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            m = LOG_RE.search(line.decode("utf-8", "ignore"))
            if m:
                ep, loss, cfm, bnd = m.groups()
                rows.append((int(ep), float(cfm) if cfm else float(loss), float(bnd) if bnd else np.nan))
    return pd.DataFrame(rows, columns=["epoch", "cfm", "boundary"]).drop_duplicates("epoch").set_index("epoch")


def fig_curves(args):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for c, pattern in RUNS.items():
        logs = sorted(glob.glob(os.path.join(args.runs_root, f"*_{pattern}_seed*", "simulate.log")))
        if not logs:
            print(f"no logs for {c} ({pattern})")
            continue
        d = pd.concat([read_log(p) for p in logs], axis=1, keys=range(len(logs)))
        for k, m in enumerate(["cfm", "boundary"]):
            series = d.xs(m, axis=1, level=1).mean(axis=1)
            if series.isna().all():
                continue
            axes[k].plot(series.index, series.rolling(args.smooth, min_periods=1).mean(), color=COLORS[c],
                         lw=2, label=f"{LABELS[c]} ({len(logs)} seeds)")
    axes[0].set_title("Flow-matching loss L(CFM)", fontsize=13)
    axes[1].set_title("Boundary sign loss (BCE in the band)", fontsize=13)
    for ax in axes:
        ax.set_xlabel("epoch after the 20k-epoch checkpoint", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    fig.tight_layout()
    out = os.path.join(args.out, "fig_curves.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="runs/eval/test_thresholds.csv")
    parser.add_argument("--test-eval-root", default="runs/eval/test")
    parser.add_argument("--raw-test-dir", required=True)
    parser.add_argument("--runs-root", default="runs/monuseg")
    parser.add_argument("--conds", nargs=2, default=["e1cont", "e2"])
    parser.add_argument("--images", nargs="+", default=None, help="default: picked by Dice gain at --thresh")
    parser.add_argument("--n", type=int, default=3, help="number of gallery images")
    parser.add_argument("--thresh", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop", type=int, default=200)
    parser.add_argument("--smooth", type=int, default=200, help="rolling window in epochs for the curves")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    g, names = fig_gallery(args)
    print(f"wrote {g} (images: {', '.join(names)})")
    print(f"wrote {fig_curves(args)}")
