"""
Re-score saved evaluation maps at several foreground thresholds, without running the model again.

Reads the per-image MMSE SDF and variance maps that eval_monuseg.py saved under <eval-dir>/maps/,
rebuilds the ground truth from the raw .xml files, and computes all metrics for each threshold
(foreground = SDF <= threshold). Use it on the validation split to choose the threshold.

Usage:
    python eval_thresholds.py --eval-dirs runs/eval/e1_val_official runs/eval/e1_val_standard \
        --raw-train-dir /data/.../raw/monuseg_train
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from eval_monuseg import image_metrics
from preprocess_data.preprocess_monuseg import find_files, load_sample

METRICS = ["dice", "iou", "hd95", "assd", "bf1", "aji", "var_auroc", "var_auroc_band",
           "dist_auroc", "dist_auroc_band", "var_auroc_strat"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dirs", nargs="+", required=True, help="output folders of eval_monuseg.py")
    parser.add_argument("--raw-train-dir", help="raw MoNuSeg training folder (for val results)")
    parser.add_argument("--raw-test-dir", help="raw MoNuSeg test folder (for test results)")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08])
    parser.add_argument("--band", type=float, default=5, help="band width in px for var_auroc_band")
    parser.add_argument("--out", default=None, help="CSV path (default: thresholds.csv in the first eval dir)")
    args = parser.parse_args()

    gt_cache, rows = {}, []
    for d in args.eval_dirs:
        ev = json.load(open(os.path.join(d, "eval_args.json")))
        raw = args.raw_train_dir if ev["split"] == "val" else args.raw_test_dir
        if raw is None:
            raise ValueError(f"{d} is a {ev['split']} evaluation; pass --raw-{'train' if ev['split'] == 'val' else 'test'}-dir")
        images, annots = find_files(raw, (".tif", ".tiff", ".png")), find_files(raw, (".xml",))
        for ck_dir in sorted(glob.glob(os.path.join(d, "maps", "*"))):
            for f in sorted(glob.glob(os.path.join(ck_dir, "*.npz"))):
                name = os.path.splitext(os.path.basename(f))[0]
                if name not in gt_cache:
                    gt_cache[name] = load_sample(images[name], annots[name], ev["resize"], thresh=15.0)[3]
                maps = np.load(f)
                mmse, var = maps["mmse"].astype(np.float32), maps["var"].astype(np.float32)
                for thr in args.thresholds:
                    m = image_metrics(mmse <= thr, gt_cache[name], var, ev["tol"], args.band, mmse=mmse, thresh=thr)
                    rows.append({"eval": os.path.basename(os.path.normpath(d)), "ckpt": os.path.basename(ck_dir),
                                 "image": name, "thresh": thr, **m})
        print(f"scored {d}", flush=True)

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(args.eval_dirs[0], "thresholds.csv")
    df.to_csv(out, index=False)
    # mean over images, then over checkpoints (seeds)
    per_ckpt = df.groupby(["eval", "ckpt", "thresh"])[METRICS].mean()
    summary = per_ckpt.groupby(["eval", "thresh"]).mean()
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.precision", 4):
        print("\nmean over images and checkpoints:\n", summary)
    print(f"\nwrote {out}")
