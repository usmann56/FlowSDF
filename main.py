import random
import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import yaml

from SimulationHelper.simulation import Simulation
from datasets.config_dl import config_dl
from models import unet_segdiff
import trainer

parser = argparse.ArgumentParser("")
parser.add_argument(
    "--config", default="cfg/monuseg.yaml", type=str, help="path to .yaml config"
)
parser.add_argument("--seed", default=0, type=int, help="seed for reproducibility")
parser.add_argument("--init-ckpt", default=None, type=str, help="start from these weights (strict load), e.g. an E1 models_sanity checkpoint")
args = parser.parse_args()

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


if __name__ == "__main__":
    with open(args.config) as file:
        yaml_cfg = yaml.safe_load(file)
        cfg = json.loads(
            json.dumps(yaml_cfg), object_hook=lambda d: SimpleNamespace(**d)
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        device_name = "Apple Metal (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"Using device: {device} ({device_name})")
    train_dl, test_dl = config_dl(cfg)

    model = unet_segdiff.UNetModel(
        in_channels=cfg.model.n_cin,
        model_channels=cfg.model.n_fm,
        out_channels=cfg.model.n_cin,
        num_res_blocks=3,
        attention_resolutions=(16, 8),
        dropout=0,
        channel_mult=tuple(cfg.model.mults),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=getattr(cfg.model, "use_checkpoint", False),  # gradient checkpointing to save GPU memory
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        rrdb_blocks=12,
    ).to(device)

    optim = Adam(model.parameters(), cfg.learning.lr)

    if cfg.general.resume_training:
        load_path = os.path.join(
            os.getcwd(),
            "runs", 
            f"{cfg.general.modality}",
            cfg.general.load_path,
            cfg.general.resume_model_dir,
        )
        fnames = sorted([f for f in os.listdir(load_path) if f.endswith(".pt")])
        cp = fnames[-1] if cfg.general.resume_cp == "None" else cfg.general.resume_cp

        model.load_state_dict(
            torch.load(os.path.join(load_path, cp), map_location=device)["state_dict"],
            strict=False,
        )
        print(f"Loaded model from: {os.path.join(load_path, cp)}")

    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device)["state_dict"])
        print(f"Initialised from: {args.init_ckpt}")

    print(f"\nNetwork has {count_parameters(model)} parameters")

    sim_name = f"{cfg.general.modality}"

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    with Simulation(sim_name=sim_name, output_root=os.path.join(os.getcwd(), "runs"), suffix=f"{cfg_name}_seed{args.seed}") as simulation:
        writer = SummaryWriter(os.path.join(simulation.outdir, "tensorboard"))
        cfg.inference.load_exp = simulation.outdir.split("/")[-1]

        with open(os.path.join(simulation.outdir, "cfg.yaml"), "w") as f:
            yaml.dump({k: v.__dict__ for k, v in cfg.__dict__.items()}, f)

        if cfg.general.corr_mode in ("sdf", "binary"):
            trainer.TrainFlow(
                fm_type=cfg.fm.type,
                sigma_min=cfg.fm.sigma_min,
                ode_steps=cfg.inference.ode_steps,
                n_val=cfg.learning.n_val,
                val_dl=train_dl,
                ema_decay=cfg.learning.ema_decay,
                thresh=cfg.inference.thresh,
                amp=getattr(cfg.learning, "amp", False),
            ).do(
                model,
                train_dl,
                cfg.learning.epochs,
                cfg.learning.clip,
                optim=optim,
                device=device,
                simulation=simulation,
                writer=writer,
                img_cond=cfg.general.img_cond,
            )