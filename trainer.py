import random
import numpy as np
import time
import copy
import os
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchdiffeq import odeint

from SimulationHelper.simulation import Simulation
from datasets.config_dl import config_dl
from datasets.transform_factory import inv_normalize, transform_factory
from boundary_loss import boundary_loss, estimate_variance

def update_ema_variables(model, ema_model, ema_decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.copy_(ema_param.data * ema_decay + (1 - ema_decay) * param.data)


class TrainFlow:
    def __init__(self, fm_type, sigma_min, ode_steps, n_val, val_dl, ema_decay, thresh, amp=False, boundary=None):
        self.fm_type = fm_type
        self.sigma_min = sigma_min
        self.ode_steps = ode_steps
        self.ema_decay = ema_decay
        self.thresh = thresh
        self.amp = amp  # bf16 mixed precision for the training forward pass
        self.boundary = boundary  # boundary-band loss settings (E2/E3), None for the original loss

        batch_val = next(iter(val_dl))
        self.x_val = batch_val['image'][:n_val]

    def get_grad_norm(self, model):
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        norms = [p.grad.detach().abs().max().item() for p in parameters]
        return np.asarray(norms).max()

    def do(self, net, dl, n_epochs, clip, optim, device, simulation, writer, img_cond):
        best_loss = float("inf")
        ema_model = copy.deepcopy(net).to(device)

        for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            epoch_cfm, epoch_bnd, epoch_w = [], [], []
            grad_norms_epoch = []
            start_time = time.time()

            for step, batch in enumerate(tqdm(dl, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                m = batch['mask'].to(device)
                if dl.dataset.corr_mode == 'binary':
                    m = torch.where(m <= 1e-2, torch.ones_like(m), torch.zeros_like(m))
                x = batch['image'].to(device) if img_cond else None
                n = len(m)

                t = torch.rand(n).float().to(device)
                eta = torch.randn_like(m).to(device)

                sigma_t = 1 - (1 - self.sigma_min) * t
                mu_t = t[:, None, None, None] * m
                mt = mu_t + sigma_t[:, None, None, None] * eta

                u = (m - (1 - self.sigma_min) * mt) / (1 - (1 - self.sigma_min) * t[:, None, None, None])
                bnd = self.boundary
                # computed before the training forward pass, so it adds no activations to the graph
                unc_var = estimate_variance(net, m, x, bnd, self.sigma_min, self.amp) \
                    if bnd is not None and bnd.weighting == 'variance' else None
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=self.amp):
                    v = net(mt, t.reshape(n, -1), img_cond=x)

                loss_cfm = ((v.float() - u) ** 2).mean()
                loss = loss_cfm
                if bnd is not None:
                    loss_bnd, stats = boundary_loss(bnd, m, mt, t, v, unc_var)
                    loss = loss_cfm + bnd.lam * loss_bnd
                    epoch_bnd.append(loss_bnd.item())
                    epoch_w.append(stats['w_mean'])
                epoch_cfm.append(loss_cfm.item())

                optim.zero_grad()
                loss.backward()

                if isinstance(clip, float):
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip, norm_type='inf')

                grad_norms_epoch.append(self.get_grad_norm(net))
                optim.step()
                epoch_loss += loss.item() * len(x) / len(dl.dataset)

                update_ema_variables(net, ema_model, self.ema_decay)

            end_time = time.time()
            elapsed_time = end_time - start_time

            extra = f" | cfm {np.mean(epoch_cfm):.6f} boundary {np.mean(epoch_bnd):.6f} w {np.mean(epoch_w):.3f}" if epoch_bnd else ""
            print(f"Epoch {epoch + 1} Loss: {epoch_loss:.8f}, Time: {elapsed_time:.2f}s{extra}")
            writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            if epoch_bnd:
                writer.add_scalar('train/epoch_loss_cfm', np.mean(epoch_cfm), epoch)
                writer.add_scalar('train/epoch_loss_boundary', np.mean(epoch_bnd), epoch)
                writer.add_scalar('train/epoch_weight_mean', np.mean(epoch_w), epoch)
            writer.add_scalar('train/epoch_max_grad', np.asarray(grad_norms_epoch).max(), epoch)
            writer.add_scalar('train/epoch_mean_grad', np.asarray(grad_norms_epoch).mean(), epoch)

            if epoch > 0 and epoch % 100 == 0:
                ema_model.eval()
                m0 = torch.randn_like(m)[:self.x_val.shape[0]]

                def func_conditional(t, m_):
                    t_curr = torch.ones(m_.shape[0], device=m_.device) * t
                    m_ipt = t * m_ + (1 - (1 - self.sigma_min) * t) * m0
                    return ema_model(m_ipt, t_curr.reshape(m_.shape[0], -1), img_cond=self.x_val.to(m0.device))

                with torch.no_grad():
                    # fixed-step euler as in sampler.py; the default adaptive solver (dopri5) can take
                    # thousands of steps near t=1 and stall training
                    traj = odeint(func_conditional, m0, torch.linspace(0, 1, self.ode_steps, device=device),
                                  method='euler')

                sample = traj[-1]
                if len(torch.unique(m)) == 2:
                    sample_thresh = torch.where(sample > 0.5, torch.ones_like(sample), torch.zeros_like(sample))
                else:
                    sample_thresh = torch.where(sample > 3 * self.thresh, torch.zeros_like(sample), torch.ones_like(sample))

                if sample.shape[1] > 1:
                    for c in range(sample.shape[1]):
                        writer.add_image(f'train/samples_{c}', make_grid(sample[:, c][:, None], normalize=True), epoch)
                        writer.add_image(f'train/samples_thresh_{c}', make_grid(sample_thresh[:, c][:, None], normalize=True), epoch)
                else:
                    writer.add_image('train/samples', make_grid(sample, normalize=True), epoch)
                    writer.add_image('train/samples_thresh', make_grid(sample_thresh, normalize=True), epoch)

            if epoch % 10000 == 0:
                checkpoint = {'state_dict': ema_model.state_dict()}
                simulation.save_pytorch(checkpoint, overwrite=False, subdir='models_sanity',
                                        epoch='_' + '{0:07}'.format(epoch))

            # only check for a new best every 100 epochs: each save writes ~2 GB (model + optimizer)
            if (epoch % 100 == 0 and best_loss > epoch_loss) or epoch == n_epochs - 1:
                best_loss = epoch_loss
                cp_dir = os.path.join(simulation._outdir, 'models')
                if epoch > 0 and os.path.exists(cp_dir):
                    pt_files = [f for f in os.listdir(cp_dir) if f.endswith('.pt')]
                    if len(pt_files) == 3:
                        oldest = sorted(pt_files)[0]
                        os.remove(os.path.join(cp_dir, oldest))

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': ema_model.state_dict(),
                    'optimizer': optim.state_dict()
                }
                simulation.save_pytorch(checkpoint, overwrite=False, epoch='_' + '{0:07}'.format(epoch))
                print(" --> Best model ever (stored)")
