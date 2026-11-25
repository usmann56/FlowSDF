"""
Evaluate FlowSDF checkpoints on full MoNuSeg images.

For every image of the chosen split, the image is resized as in preprocessing, cut into
overlapping sz x sz tiles, and K samples are drawn per tile. Overlapping tiles are averaged back
into full-image SDFs, the K samples are averaged (MMSE) and thresholded, and metrics are computed
per image against ground truth built directly from the raw .xml annotations:

    dice, iou          region overlap of the binary mask
    hd95, assd         boundary distances in pixels (at the resized resolution)
    bf1                boundary F1 with a --tol pixel tolerance
    aji                Aggregated Jaccard Index (instances = connected components of the prediction)
    var_auroc          how well the per-pixel variance over the K samples ranks misclassified pixels

Usage:
    python eval_monuseg.py --config cfg/monuseg_e1_train30.yaml --split val \
        --raw-train-dir /data/.../raw/monuseg_train --raw-test-dir /data/.../raw/monuseg_test \
        --ckpt runs/monuseg/<run>/models/<file>.pt [more .pt files] --out runs/eval/<name>
"""
import argparse
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from models import unet_segdiff
from preprocess_data.preprocess_monuseg import find_files, load_sample, patch_starts

CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


# --------------------- metrics ---------------------

def boundary(mask):
    # pixels outside the image count as foreground, so the image border is not a boundary
    return mask & ~cv2.erode(mask.astype(np.uint8), CROSS, borderType=cv2.BORDER_REPLICATE).astype(bool)


def surface_distances(pred, gt):
    """Distances from each pred boundary pixel to the gt boundary, and vice versa."""
    pb, gb = boundary(pred), boundary(gt)
    if not pb.any() or not gb.any():
        return None, None
    dt_g = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dt_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dt_g[pb], dt_p[gb]


def aji(gt_inst, pred_inst):
    """Aggregated Jaccard Index (Kumar et al. 2017)."""
    n_g, n_p = gt_inst.max() + 1, pred_inst.max() + 1
    inter = np.bincount(gt_inst.ravel() * n_p + pred_inst.ravel(), minlength=n_g * n_p).reshape(n_g, n_p)
    area_g, area_p = inter.sum(1), inter.sum(0)
    inter, area_g, area_p = inter[1:, 1:], area_g[1:], area_p[1:]
    if len(area_g) == 0:
        return np.nan
    if len(area_p) == 0:
        return 0.0
    union = area_g[:, None] + area_p[None, :] - inter
    iou = np.where(inter > 0, inter / np.maximum(union, 1), 0)
    C = U = 0
    used = np.zeros(len(area_p), bool)
    for g in range(len(area_g)):
        if iou[g].max() > 0:
            p = iou[g].argmax()
            C += inter[g, p]
            U += union[g, p]
            used[p] = True
        else:
            U += area_g[g]
    U += area_p[~used].sum()
    return C / U if U > 0 else np.nan


def auroc(score, positive):
    """Probability that a random positive pixel scores higher than a random negative one."""
    n_pos = positive.sum()
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.empty(score.size)
    ranks[np.argsort(score, kind="stable")] = np.arange(1, score.size + 1)
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def image_metrics(pred, gt_inst, var, tol, band=5):
    gt = gt_inst > 0
    inter, total = (pred & gt).sum(), pred.sum() + gt.sum()
    union = total - inter
    out = {
        "dice": 2 * inter / total if total else 1.0,
        "iou": inter / union if union else 1.0,
    }
    d_pg, d_gp = surface_distances(pred, gt)
    if d_pg is None:
        out.update(hd95=np.nan, assd=np.nan, bf1=0.0)
    else:
        prec, rec = (d_pg <= tol).mean(), (d_gp <= tol).mean()
        out.update(
            hd95=max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)),
            assd=(d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)),
            bf1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        )
    _, pred_inst = cv2.connectedComponents(pred.astype(np.uint8), connectivity=4)
    out["aji"] = aji(gt_inst, pred_inst)
    err = pred != gt
    out["var_auroc"] = auroc(var.ravel(), err.ravel())
    # same, restricted to pixels within `band` px of a true nucleus edge (where E2/E3 act)
    near = cv2.distanceTransform((~boundary(gt)).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= band
    out["var_auroc_band"] = auroc(var[near], err[near])
    return out


# --------------------- sampling ---------------------

def load_cfg(path):
    with open(path) as f:
        return json.loads(json.dumps(yaml.safe_load(f)), object_hook=lambda d: SimpleNamespace(**d))


def build_model(cfg, device):
    return unet_segdiff.UNetModel(
        in_channels=cfg.model.n_cin, model_channels=cfg.model.n_fm, out_channels=cfg.model.n_cin,
        num_res_blocks=3, attention_resolutions=(16, 8), dropout=0, channel_mult=tuple(cfg.model.mults),
        conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, num_heads=1,
        num_heads_upsample=-1, use_scale_shift_norm=False, rrdb_blocks=12,
    ).to(device)


@torch.no_grad()
def sample_tiles(net, cond, n_points, sigma_min, sampler, gen, amp):
    """Euler integration from noise to an SDF for a batch of tiles; returns (B, 1, sz, sz)."""
    m0 = torch.randn(cond.shape[0], 1, *cond.shape[2:], device=cond.device, generator=gen)
    m = m0.clone()
    ts = torch.linspace(0, 1, n_points, device=cond.device)
    for i in range(n_points - 1):
        t = ts[i]
        if sampler == "official":  # as in sampler.py / trainer.py
            m_ipt = (1 - (1 - sigma_min) * t) * m0 + t * m
        else:  # standard flow matching: the state is the sample at time t
            m_ipt = m
        with torch.autocast(cond.device.type, dtype=torch.bfloat16, enabled=amp):
            v = net(m_ipt, torch.full((m.shape[0], 1), float(t), device=m.device), img_cond=cond)
        m = m + (ts[i + 1] - t) * v.float()
    return m


def predict_image(net, img, args, cfg, gen, device):
    """K full-image SDF samples, each the average of the overlapping tiles."""
    sz = cfg.general.sz
    h, w = img.shape[:2]
    pos = [(y, x) for y in patch_starts(h, sz, args.stride) for x in patch_starts(w, sz, args.stride)]
    x = torch.from_numpy((img.astype(np.float32) / 255. - 0.5) * 2).permute(2, 0, 1)  # as in training
    tiles = torch.stack([x[:, y:y + sz, xx:xx + sz] for y, xx in pos]).to(device)

    count = np.zeros((h, w), np.float32)
    for y, xx in pos:
        count[y:y + sz, xx:xx + sz] += 1
    sums = np.zeros((args.k, h, w), np.float32)
    jobs = [(k, i) for k in range(args.k) for i in range(len(pos))]
    for s in range(0, len(jobs), args.batch):
        batch = jobs[s:s + args.batch]
        out = sample_tiles(net, tiles[[i for _, i in batch]], cfg.inference.ode_steps, cfg.fm.sigma_min,
                           args.sampler, gen, args.amp).cpu().numpy()[:, 0]
        for (k, i), o in zip(batch, out):
            y, xx = pos[i]
            sums[k, y:y + sz, xx:xx + sz] += o
    return sums / count


def save_maps(out_dir, name, img, gt, pred, mmse, var):
    os.makedirs(out_dir, exist_ok=True)
    overlay = img.copy()
    for mask, color in ((gt, (0, 200, 0)), (pred, (40, 40, 230))):  # BGR: ground truth green, prediction red
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, color, 1)
    cv2.imwrite(os.path.join(out_dir, f"{name}_overlay.png"), overlay)
    v = np.clip(var / max(np.percentile(var, 99.5), 1e-8), 0, 1)
    cv2.imwrite(os.path.join(out_dir, f"{name}_var.png"), cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), mmse=mmse.astype(np.float16), var=var.astype(np.float16))


def ckpt_label(path):
    run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    return f"{run}/{os.path.splitext(os.path.basename(path))[0]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="training config (model, fm and inference settings)")
    parser.add_argument("--ckpt", nargs="+", required=True, help="one or more .pt checkpoints")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--raw-train-dir", help="raw MoNuSeg training folder (needed for --split val)")
    parser.add_argument("--raw-test-dir", help="raw MoNuSeg test folder (needed for --split test)")
    parser.add_argument("--val-file", default="splits/val.txt")
    parser.add_argument("--out", required=True, help="output folder for metrics and maps")
    parser.add_argument("--k", type=int, default=25, help="samples per image (paper: 25)")
    parser.add_argument("--sampler", choices=["official", "standard"], default="official")
    parser.add_argument("--stride", type=int, default=64, help="tile stride; 64 -> each pixel in 2-4 tiles")
    parser.add_argument("--fg-thresh", type=float, default=None, help="foreground if SDF <= this (default 3 x inference.thresh, as in sampler.py)")
    parser.add_argument("--tol", type=float, default=2.0, help="boundary F1 tolerance in pixels")
    parser.add_argument("--resize", type=int, default=500, help="must match preprocessing")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="bf16 inference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-maps", action="store_true", help="skip saving overlays and uncertainty maps")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fg_thresh = args.fg_thresh if args.fg_thresh is not None else 3 * cfg.inference.thresh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_dir = args.raw_train_dir if args.split == "val" else args.raw_test_dir
    if raw_dir is None:
        raise ValueError(f"--split {args.split} needs --raw-{'train' if args.split == 'val' else 'test'}-dir")
    images, annots = find_files(raw_dir, (".tif", ".tiff", ".png")), find_files(raw_dir, (".xml",))
    if args.split == "val":
        names = [n.strip() for n in open(args.val_file) if n.strip()]
    else:
        names = sorted(set(images) & set(annots))
    missing = [n for n in names if n not in images or n not in annots]
    if missing:
        raise ValueError(f"Missing image or .xml under {raw_dir}: {missing}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ground truth once, shared by all checkpoints
    data = {}
    for n in names:
        img, _, _, inst = load_sample(images[n], annots[n], args.resize, thresh=15.0)
        data[n] = (img, inst)

    net = build_model(cfg, device).eval()
    rows = []
    for ck in args.ckpt:
        net.load_state_dict(torch.load(ck, map_location=device)["state_dict"])
        label = ckpt_label(ck)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        for n in names:
            img, inst = data[n]
            samples = predict_image(net, img, args, cfg, gen, device)
            mmse, var = samples.mean(0), samples.var(0)
            pred = mmse <= fg_thresh
            row = {"ckpt": label, "split": args.split, "image": n, **image_metrics(pred, inst, var, args.tol)}
            rows.append(row)
            print(f"{label} {n}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, (float, np.floating))), flush=True)
            if not args.no_maps:
                save_maps(os.path.join(args.out, "maps", label.replace("/", "__")), n, img, inst > 0, pred, mmse, var)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    metric_cols = ["dice", "iou", "hd95", "assd", "bf1", "aji", "var_auroc", "var_auroc_band"]
    summary = df.groupby("ckpt")[metric_cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 4):
        print("\nmean over images:\n", df.groupby("ckpt")[metric_cols].mean())
    print(f"\nwrote {args.out}/metrics.csv and summary.csv")
