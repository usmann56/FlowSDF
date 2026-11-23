"""
Hold out validation images from the preprocessed MoNuSeg training set.

Reads <data>/train.csv (full training images written by preprocess_monuseg.py), picks --n-val
images (seeded random, or the names given with --val-names), and writes:
    <data>/train<N>.csv                remaining N training images (full images, cropped while training)
    <data>/val.csv                     overlapping sz x sz patches of the validation images
    <data>/val/{images,masks,sdf}/     the validation patches
    splits/val.txt                     the chosen validation image names (commit this file)

Usage:
    python preprocess_data/make_split.py --data /path/monuseg_rgb/ --n-val 7 --seed 0
"""
import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd

from preprocess_monuseg import patch_starts, save


def name_of(path):
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder with train.csv from preprocess_monuseg.py")
    parser.add_argument("--n-val", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0, help="seed for picking the validation images")
    parser.add_argument("--val-names", nargs="*", help="pick these images instead of a random choice")
    parser.add_argument("--sz", type=int, default=128, help="patch size, must match general.sz")
    parser.add_argument("--stride", type=int, default=64, help="same overlap as the test patches")
    parser.add_argument("--split-file", default="splits/val.txt")
    args = parser.parse_args()

    train = pd.read_csv(os.path.join(args.data, "train.csv"))
    names = sorted(name_of(f) for f in train["filename"])
    if args.val_names:
        missing = set(args.val_names) - set(names)
        if missing:
            raise ValueError(f"Not in train.csv: {sorted(missing)}")
        val_names = sorted(args.val_names)
    else:
        val_names = sorted(random.Random(args.seed).sample(names, args.n_val))

    is_val = train["filename"].map(name_of).isin(val_names)
    train_csv = f"train{(~is_val).sum()}.csv"
    train[~is_val].to_csv(os.path.join(args.data, train_csv), index=False)

    for d in ("images", "masks", "sdf"):
        os.makedirs(os.path.join(args.data, "val", d), exist_ok=True)
    rows = []
    for _, r in train[is_val].iterrows():
        img = cv2.imread(os.path.join(args.data, r["filename"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(args.data, r["maskname"]), cv2.IMREAD_GRAYSCALE)
        sdf = np.load(os.path.join(args.data, r["maskdtname"]))
        h, w = img.shape[:2]
        for y in patch_starts(h, args.sz, args.stride):
            for x in patch_starts(w, args.sz, args.stride):
                sl = (slice(y, y + args.sz), slice(x, x + args.sz))
                rows.append(save(args.data, "val", f"{name_of(r['filename'])}_{y:04d}_{x:04d}",
                                 img[sl], mask[sl], sdf[sl]))
    pd.DataFrame(rows).to_csv(os.path.join(args.data, "val.csv"), index=False)

    os.makedirs(os.path.dirname(args.split_file), exist_ok=True)
    with open(args.split_file, "w") as f:
        f.write("\n".join(val_names) + "\n")

    print(f"validation images ({len(val_names)}): {', '.join(val_names)}")
    print(f"{train_csv}: {(~is_val).sum()} training images | val.csv: {len(rows)} patches")
    print(f"wrote {args.split_file}")
