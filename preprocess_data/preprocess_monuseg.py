"""
Build the FlowSDF MoNuSeg dataset from the raw MoNuSeg release (.tif images + .xml annotations),
following the paper: images resized 1000x1000 -> 500x500, random 128x128 training crops (taken by
the dataloader from the full training images), overlapping 128x128 test patches.

The SDF follows generative-segmentation-sdf/preprocess_data/precompute_sdf.ipynb: negative inside,
positive outside, truncated at +-thresh pixels and scaled to [-1, 1]. It is computed on the full
image before patching, so patch borders do not create fake boundaries. Stain normalisation is not
done here; apply it to the .tif images beforehand if needed.

Output layout (point data_path in the config to <out>/ with a trailing slash):
    <out>/train.csv, <out>/test.csv            columns: filename, maskname, maskdtname
    <out>/{train,test}/images/<name>.png       RGB image (train: full image, test: patch)
    <out>/{train,test}/masks/<name>.png        binary mask (0/255)
    <out>/{train,test}/sdf/<name>.npy          float32 SDF in [-1, 1]

Usage:
    python preprocess_data/preprocess_monuseg.py \
        --train-dir /path/MoNuSeg/train --test-dir /path/MoNuSeg/test \
        --out /path/monuseg_rgb
"""
import argparse
import os
import xml.etree.ElementTree as ET
from glob import glob

import cv2
import numpy as np
import pandas as pd


def find_files(root, exts):
    files = []
    for ext in exts:
        files += glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    # skip macOS metadata files (__MACOSX/._name.tif) that come with some MoNuSeg zips
    files = [f for f in files if "__MACOSX" not in f and not os.path.basename(f).startswith("._")]
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}


def read_polygons(xml_path, scale):
    polys = []
    for region in ET.parse(xml_path).getroot().iter("Region"):
        pts = [(float(v.get("X")), float(v.get("Y"))) for v in region.iter("Vertex")]
        if len(pts) >= 3:
            polys.append(np.round(np.asarray(pts) * scale).astype(np.int32))
    return polys


def instance_map(polys, h, w):
    inst = np.zeros((h, w), dtype=np.int32)
    for i, p in enumerate(polys, start=1):
        cv2.fillPoly(inst, [p], i)
    return inst


def compute_sdf(inst, thresh):
    # boundary = foreground pixels with a 4-neighbour of a different label, so touching
    # nuclei keep the boundary between them instead of merging into one blob
    pad = np.pad(inst, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    diff = (c != pad[:-2, 1:-1]) | (c != pad[2:, 1:-1]) | (c != pad[1:-1, :-2]) | (c != pad[1:-1, 2:])
    boundary = (inst > 0) & diff

    distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = np.where(inst > 0, -distance, distance)
    sdf = np.clip(sdf, -thresh, thresh) / thresh
    return sdf.astype(np.float32)


def patch_starts(n, sz, stride):
    starts = list(range(0, n - sz + 1, stride))
    if starts[-1] != n - sz:
        starts.append(n - sz)  # make sure the image border is covered
    return starts


def load_sample(img_path, xml_path, resize, thresh):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    scale = 1.0
    if resize:
        scale = resize / img.shape[1]
        img = cv2.resize(img, (resize, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    # rasterise the rescaled polygons instead of resizing a mask, to keep boundaries sharp
    inst = instance_map(read_polygons(xml_path, scale), h, w)
    mask = ((inst > 0) * 255).astype(np.uint8)
    return img, mask, compute_sdf(inst, thresh), inst.max()


def save(out, split, name, img, mask, sdf):
    rel = {
        "filename": f"{split}/images/{name}.png",
        "maskname": f"{split}/masks/{name}.png",
        "maskdtname": f"{split}/sdf/{name}.npy",
    }
    cv2.imwrite(os.path.join(out, rel["filename"]), img)
    cv2.imwrite(os.path.join(out, rel["maskname"]), mask)
    np.save(os.path.join(out, rel["maskdtname"]), sdf)
    return rel


def process_split(src_dir, split, out, args):
    images = find_files(src_dir, (".tif", ".tiff", ".png"))
    annots = find_files(src_dir, (".xml",))
    names = sorted(set(images) & set(annots))
    if not names:
        raise RuntimeError(f"No matching image/.xml pairs found under {src_dir}")
    for d in ("images", "masks", "sdf"):
        os.makedirs(os.path.join(out, split, d), exist_ok=True)

    rows = []
    for name in names:
        img, mask, sdf, n_nuclei = load_sample(images[name], annots[name], args.resize, args.thresh)
        h, w = img.shape[:2]
        if split == "train":
            rows.append(save(out, split, name, img, mask, sdf))
        else:
            for y in patch_starts(h, args.sz, args.test_stride):
                for x in patch_starts(w, args.sz, args.test_stride):
                    sl = (slice(y, y + args.sz), slice(x, x + args.sz))
                    rows.append(save(out, split, f"{name}_{y:04d}_{x:04d}", img[sl], mask[sl], sdf[sl]))
        print(f"[{split}] {name}: {h}x{w}, {n_nuclei} nuclei")

    pd.DataFrame(rows).to_csv(os.path.join(out, f"{split}.csv"), index=False)
    print(f"[{split}] {len(names)} images -> {len(rows)} entries in {split}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, help="raw MoNuSeg training folder (.tif + .xml)")
    parser.add_argument("--test-dir", required=True, help="raw MoNuSeg test folder (.tif + .xml)")
    parser.add_argument("--out", required=True, help="output folder; its name must contain 'rgb'")
    parser.add_argument("--resize", type=int, default=500, help="target width (paper: 500); 0 keeps original size")
    parser.add_argument("--sz", type=int, default=128, help="test patch size, must match general.sz")
    parser.add_argument("--test-stride", type=int, default=64, help="64 -> each pixel covered by 2-4 patches")
    parser.add_argument("--thresh", type=float, default=15.0, help="SDF truncation distance in pixels")
    args = parser.parse_args()

    if "rgb" not in args.out:
        raise ValueError("Output path must contain 'rgb', otherwise the dataloader reads images as grayscale")

    process_split(args.train_dir, "train", args.out, args)
    process_split(args.test_dir, "test", args.out, args)
