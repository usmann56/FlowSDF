"""
Boundary-band supervision for FlowSDF (conditions E2, E3, E3b).

The one-step estimate of the clean SDF, m_hat = m_t + (1 - t) * v, is turned into an inside
probability p = sigmoid(-k * m_hat) and compared with the true mask by BCE. Only pixels within
`delta` px of a true nucleus edge count (the band B), and only samples with t >= t_min, where
m_hat is no longer a blurry average. Each band pixel is weighted by w = 1 + gamma * U, with U
detached and normalised to mean 1 over the band:

    uniform    U = 0                                           (E2)
    variance   U = variance of K one-step estimates with fresh (t, noise) draws   (E3)
    entropy    U = binary entropy of p                         (E3b)

    L_boundary = sum(B * w * bce) / (sum(B * w) + eps)        (proposal, slide 21)

Config section (all keys required when the section is present):
    boundary:
      weighting: uniform | variance | entropy
      lam: 0.1        # weight of L_boundary in the total loss
      delta: 5        # band half-width in px
      t_min: 0.5      # only samples with t >= t_min contribute
      k: 6.0          # calibration scale in p = sigmoid(-k * m_hat)
      gamma: 1.0      # strength of the uncertainty weighting (unused for uniform)
      n_var: 4        # K, extra one-step estimates for variance
      u_clip: 10.0    # cap on the normalised U
      sdf_trunc: 15.0 # SDF truncation distance in px used in preprocessing
"""
import torch
import torch.nn.functional as F


def one_step(mt, t, v):
    """Estimate of the clean SDF from the noisy input and the predicted velocity."""
    return mt + (1 - t)[:, None, None, None] * v


@torch.no_grad()
def estimate_variance(net, m, x, cfg, sigma_min, amp):
    """Per-pixel variance of K one-step estimates, each from its own (t, noise) draw."""
    estimates = []
    for _ in range(cfg.n_var):
        t = cfg.t_min + (1 - cfg.t_min) * torch.rand(m.shape[0], device=m.device)
        eta = torch.randn_like(m)
        mt = t[:, None, None, None] * m + (1 - (1 - sigma_min) * t)[:, None, None, None] * eta
        with torch.autocast(m.device.type, dtype=torch.bfloat16, enabled=amp):
            v = net(mt, t.reshape(-1, 1), img_cond=x)
        estimates.append(one_step(mt, t, v.float()))
    return torch.stack(estimates).var(0, unbiased=False)


def boundary_loss(cfg, m, mt, t, v, unc_var=None):
    """Returns (loss, stats). `unc_var` is the output of estimate_variance for weighting=variance."""
    band = (m.abs() <= cfg.delta / cfg.sdf_trunc) & (t >= cfg.t_min)[:, None, None, None]
    if not band.any():
        return v.sum() * 0.0, {"band_frac": 0.0, "w_mean": 1.0}

    logits = -cfg.k * one_step(mt, t, v.float())
    bce = F.binary_cross_entropy_with_logits(logits, (m <= 0).float(), reduction="none")

    with torch.no_grad():
        if cfg.weighting == "uniform":
            w = torch.ones_like(m)
        else:
            if cfg.weighting == "variance":
                unc = unc_var
            elif cfg.weighting == "entropy":
                p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
                unc = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
            else:
                raise ValueError(f"Unknown boundary weighting {cfg.weighting!r}")
            unc = (unc / (unc[band].mean() + 1e-8)).clamp(max=cfg.u_clip)
            w = 1 + cfg.gamma * unc
        wb = w * band

    loss = (wb * bce).sum() / (wb.sum() + 1e-8)
    return loss, {"band_frac": band.float().mean().item(), "w_mean": w[band].mean().item()}
