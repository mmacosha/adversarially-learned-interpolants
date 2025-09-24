import matplotlib.pyplot as plt
import torch

import os
import wandb
import warnings
from tqdm.auto import trange, tqdm
from hydra import compose, initialize
import numpy as np
import random

from omegaconf import OmegaConf

from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from torchcfm.conditional_flow_matching import OTPlanSampler

import ali_cfm.training.training_utils as utils
from ali_cfm.loggin_and_metrics import compute_emd
from ali_cfm.data_utils import get_dataset, denormalize, denormalize_gradfield
from utils import Plotter
from scipy import interpolate


from ali_cfm.nets import MLP


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_ot_interpolant_given_coupling(coupled_x, unobserved, t, bs):
    device = coupled_x.device
    xhat_t = torch.zeros((bs, 2), device=device)
    dx_t = torch.zeros((bs, 2), device=device)

    observed_t = (1 - unobserved)
    idx_i = t.squeeze(-1) > observed_t
    idx_j = t.squeeze(-1) < observed_t

    # Linear interpolant between t = observed_t and t = 1
    denom_i = (1 - observed_t)
    a_i = (1. - t[idx_i]) / denom_i
    b_i = (t[idx_i] - observed_t) / denom_i
    xhat_t[idx_i] = a_i * coupled_x[idx_i, 1] + b_i * coupled_x[idx_i, 2]
    dx_t[idx_i] = coupled_x[idx_i, 2] / denom_i - coupled_x[idx_i, 1] / denom_i

    # Linear interpolant between t = 0 and t = observed_t
    denom_j = (observed_t - 0)
    a_j = (observed_t - t[idx_j]) / denom_j
    b_j = (t[idx_j] - 0) / denom_j
    xhat_t[idx_j] = a_j * coupled_x[idx_j, 0] + b_j * coupled_x[idx_j, 1]
    dx_t[idx_j] = coupled_x[idx_j, 1] / denom_j - coupled_x[idx_j, 0] / denom_j

    return xhat_t, dx_t


def train_ot_cfm(
        ot_cfm_model, ot_cfm_optimizer, interpolant,
         ot_sampler, train_data, batch_size, min_max,
         n_ot_cfm_epochs,
         metric_prefix="", ot='border', device='cpu', times=(0, -1)
    ):

    if interpolant == "cubic":
        try:
            assert ot == 'mmot'
        except AssertionError:
            print("Setting ot to 'mmot'")
            ot = 'mmot'
    else:
        try:
            assert ot == 'full'
        except AssertionError:
            print("Setting ot to 'full'")
            ot = 'full'

    for step in trange(n_ot_cfm_epochs,
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()


        x0, x1, xt, t_j = utils.sample_gan_batch(train_data, batch_size, ot_sampler, 3, ot=ot)
        t0 = torch.zeros_like(t_j)
        t1 = torch.ones_like(t_j)
        t = torch.rand(x0.shape[0], 1, device=device)

        if interpolant == 'linear':
            coupled_x = torch.stack([x0, xt, x1], dim=1)
            xt, ut = get_ot_interpolant_given_coupling(coupled_x, times[1] / 3, t, batch_size)  # (K, n, d)
        else:
            X = torch.stack([x0, xt, x1], dim=1)

            y = X.detach().cpu().numpy()
            x = np.array(times) / 3  # (K,)

            # vectorized spline fit over (n, 2)
            splines = interpolate.CubicSpline(x, y, axis=1)  # no loops

            t_np = t.squeeze(-1).detach().cpu().numpy()  # (T,)

            xt = torch.tensor(splines(t_np), device=device, dtype=torch.float32).squeeze(1)
            xt = torch.einsum("nni->ni", xt)
            ut = torch.tensor(splines(t_np, 1), device=device, dtype=torch.float32).squeeze(1)
            ut = torch.einsum("nni->ni", ut)

        vt = ot_cfm_model(torch.cat([xt, t], dim=-1))

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        ot_cfm_optimizer.step()

        wandb.log({
            f"{metric_prefix}/cfm_loss": loss.item(),
            f"{metric_prefix}_cfm_step": step
        })


# @hydra.main(config_path="./configs", config_name="ali")
def main(cfg, wandb_run=None):
    os.environ["HYDRA_FULL_ERROR"] = '1'
    seed_list = cfg.seed_list
    warnings.filterwarnings("ignore")
    ot_sampler = OTPlanSampler('exact', reg=0.1)


    data, min_max = get_dataset("ST", cfg.n_data_dims, normalize=cfg.normalize_dataset)
    timesteps_list = [t for t in range(len(data))]

    #pl = Plotter("../data/ST_images/ref_U5_warped_images",
    #             [t / 3 for t in range(len(data))], coordinate_scaling=1)
    #plot_fn = pl.plot_fn
    interpolant = cfg.interpolant

    int_results, cfm_results = {}, {}

    run = wandb.init(
        name=f"{cfg.gan_loss}-{cfg.wandb_name}-{cfg.dataset}-{cfg.n_data_dims}D",
        mode=cfg.wandb_mode,
        project=f"ot-cfm-{cfg.dataset}-{cfg.n_data_dims}D",
        config=OmegaConf.to_object(cfg)
    )

    os.makedirs(f"{run.dir}/checkpoints", exist_ok=True)

    for seed in tqdm(seed_list, desc="Seeds"):
        fix_seed(seed)
        int_results[f"seed={seed}"] = []
        cfm_results[f"seed={seed}"] = []

        for removed_t in tqdm(timesteps_list[1: -1], desc="Timesteps", leave=False):
            curr_timesteps = [x for x in timesteps_list if x != removed_t]
            metric_prefix = f"t={removed_t}_{seed=}"

            ot_cfm_model = MLP(dim=cfg.dim, time_varying=True, w=cfg.net_hidden).to(cfg.device)
            ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), cfg.lr_CFM)

            if cfg.train_cfm:
                # Train OT-CFM using GAN interpolant
                cfm_metric_prefix = f"{metric_prefix}_cfm"
                wandb.define_metric(f"{cfm_metric_prefix}/*",
                                    step_metric=f"{cfm_metric_prefix}_step")
                train_ot_cfm(
                    ot_cfm_model, ot_cfm_optimizer, interpolant, ot_sampler,
                    data, cfg.batch_size, min_max, cfg.n_ot_cfm_epochs,
                    metric_prefix=cfm_metric_prefix, ot=cfg.cfm_ot,
                    device=cfg.device, times=curr_timesteps
                )
                # Save artifacts for given seed and t
                checkpoint = {
                    "ot_cfm_model": ot_cfm_model.state_dict()
                }
                save_path = os.path.join(
                    run.dir, "checkpoints", f"{metric_prefix}_ali_cfm.pth"
                )
                torch.save(checkpoint, save_path)
            else:
                PATH = (f"/Users/oskarkviman/Documents/phd/mixture_FM_loss/ST/wandb/{wandb_run}/files/checkpoints")
                load_checkpoint = torch.load(PATH + f"/{metric_prefix}_ali_cfm.pth", weights_only=True)
                ot_cfm_model.load_state_dict(load_checkpoint['ot_cfm_model'])


            # Compute metrics for OT-CFM
            node = NeuralODE(torch_wrapper(ot_cfm_model),
                             solver="dopri5", sensitivity="adjoint")

            t_s = torch.linspace((removed_t - 1) / max(timesteps_list), removed_t / max(timesteps_list), 101)
            with torch.no_grad():
                cfm_traj = node.trajectory(data[removed_t - 1],
                                           t_s
                                           )

            cfm_emd = compute_emd(
                denormalize(data[removed_t], min_max).to(cfg.device),
                denormalize(cfm_traj[-1], min_max).to(cfg.device),
            )
            cfm_results[f"seed={seed}"].append(cfm_emd.item())

    print("t = 1")
    print(np.mean([cfm_results[f"seed={seed}"][0] for seed in cfg.seed_list]),
          np.std([cfm_results[f"seed={seed}"][0] for seed in cfg.seed_list]))
    print("t = 2")
    print(np.mean([cfm_results[f"seed={seed}"][1] for seed in cfg.seed_list]),
          np.std([cfm_results[f"seed={seed}"][1] for seed in cfg.seed_list]))

    wandb.finish()


if __name__ == "__main__":
    with initialize(config_path="./configs"):
        cfg = compose(config_name="ot_cfm.yaml")
        cubic_run = "run-20250922_045634-czsst8q4"
        cfg.interpolant = 'cubic'
        linear_run = "run-20250922_002843-hnpqaahl"
        # cfg.interpolant = 'linear'
        main(cfg, wandb_run=cubic_run)