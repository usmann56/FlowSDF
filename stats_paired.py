"""
Paired per-image comparison of conditions, from the per-image CSV written by eval_thresholds.py.

Each image is scored under every condition, so conditions are compared image by image: for every image the
metric is first averaged over seeds, then the per-image differences (condition - baseline) are tested with
an exact paired sign-flip permutation test on the mean difference (no scipy needed). Reported per row:
mean difference, 95% bootstrap CI over images, number of images where the condition is better, p-value,
and a Holm-adjusted p-value over all rows of the run (for the multiple comparisons).

Usage:
    python stats_paired.py runs/eval/test_thresholds.csv --thresholds 0.0 0.05
    python stats_paired.py runs/eval/final_val_thresholds.csv --thresholds 0.0 0.05
"""
import argparse
import itertools
import re

import numpy as np
import pandas as pd

METRICS = ["dice", "hd95", "assd", "bf1", "aji", "var_auroc", "var_auroc_band", "dist_auroc", "dist_auroc_band",
           "var_auroc_strat"]
LOWER_IS_BETTER = {"hd95", "assd"}


def condition(name):
    # final_e2_val -> e2, e1cont_val -> e1cont, e2 -> e2
    return re.sub(r"^final_|_val$", "", name)


def sign_flip_p(d, n_mc=200_000, seed=0):
    """Two-sided p-value of the mean paired difference under random sign flips (exact for n <= 20)."""
    d = np.asarray(d, float)
    n, obs = len(d), abs(d.mean())
    if n <= 20:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_mc, n))
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))


def bootstrap_ci(d, n_boot=10_000, seed=0):
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    return np.percentile(d[idx].mean(1), [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="per-image CSV from eval_thresholds.py (--out)")
    parser.add_argument("--baseline", default="e1cont")
    parser.add_argument("--compare", nargs="+", default=["e2:e1cont", "e3:e1cont", "e3b:e1cont", "e3:e2", "e3b:e2"],
                        help="condition:reference pairs")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None, help="default: all in the CSV")
    parser.add_argument("--metrics", nargs="+", default=None, help="default: all known metrics in the CSV")
    parser.add_argument("--out", default=None, help="optional CSV for the table")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["cond"] = df["eval"].map(condition)
    args.metrics = args.metrics or [m for m in METRICS if m in df.columns]
    if args.thresholds is not None:
        df = df[np.isclose(df["thresh"].values[:, None], np.array(args.thresholds)[None]).any(1)]
    # per image: mean over seeds (checkpoints) of each condition
    per_img = df.groupby(["cond", "thresh", "image"])[args.metrics].mean()

    rows = []
    for pair in args.compare:
        a, b = pair.split(":")
        for thr in sorted(df["thresh"].unique()):
            try:
                A, B = per_img.loc[(a, thr)], per_img.loc[(b, thr)]
            except KeyError:
                continue
            for m in args.metrics:
                both = pd.concat([A[m], B[m]], axis=1, keys=["a", "b"]).dropna()
                d = (both["a"] - both["b"]).values
                if len(d) == 0:
                    continue
                better = (d < 0) if m in LOWER_IS_BETTER else (d > 0)
                lo, hi = bootstrap_ci(d)
                rows.append({"comparison": f"{a} - {b}", "thresh": thr, "metric": m,
                             "ref_mean": both["b"].mean(), "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "better": f"{better.sum()}/{len(d)}", "p": sign_flip_p(d)})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p"].values)
    res["sig"] = np.where(res["p_holm"] < 0.05, "*", np.where(res["p"] < 0.05, "(*)", ""))
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.precision", 4):
        for comp, g in res.groupby("comparison", sort=False):
            print(f"\n=== {comp} ===")
            print(g.drop(columns="comparison").to_string(index=False))
    print("\n* = significant after Holm correction over all rows; (*) = p < 0.05 uncorrected only."
          "\nbetter = images where the first condition wins (lower is better for hd95/assd).")
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
