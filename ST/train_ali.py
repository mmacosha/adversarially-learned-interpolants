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
from ali_cfm.training.training_funcs import pretain_interpolant, train_interpolant_with_gan
from ali_cfm.loggin_and_metrics import compute_emd
from ali_cfm.data_utils import get_dataset, denormalize, denormalize_gradfield
from utils import Plotter


from ali_cfm.nets import Discriminator, MLP, TrainableInterpolant


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_ot_cfm(
        ot_cfm_model, ot_cfm_optimizer, interpolant,
         ot_sampler, train_data, batch_size, min_max,
         n_ot_cfm_epochs,
         metric_prefix="", ot='border', device='cpu', times=(0, -1)
    ):
    for step in trange(n_ot_cfm_epochs,
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        x0, x1 = utils.sample_x0_x1(train_data, batch_size, device=device)
        if ot == 'border' or ot == 'full':
            x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0], 1, device=device)

        xt = interpolant(x0, x1, t, training=False).detach()
        ut = interpolant.dI_dt(x0, x1, t).detach()

        vt = ot_cfm_model(torch.cat([xt, t], dim=-1))

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        ot_cfm_optimizer.step()

        wandb.log({
            f"{metric_prefix}/cfm_loss": loss.item(),
            f"{metric_prefix}_cfm_step": step
        })


# @hydra.main(config_path="./configs", config_name="ali")
def train_ali(cfg):
    os.environ["HYDRA_FULL_ERROR"] = '1'
    seed_list = cfg.seed_list
    warnings.filterwarnings("ignore")
    ot_sampler = OTPlanSampler('exact', reg=0.1)


    data, min_max = get_dataset("ST", cfg.n_data_dims, normalize=cfg.normalize_dataset)
    timesteps_list = [t for t in range(len(data))]

    pl = Plotter("../data/ST_images/ref_U5_warped_images",
                 [t / 3 for t in range(len(data))], coordinate_scaling=1)
    plot_fn = pl.plot_fn

    int_results, cfm_results = {}, {}

    run = wandb.init(
        name=f"{cfg.gan_loss}-{cfg.wandb_name}-{cfg.dataset}-{cfg.n_data_dims}D",
        mode=cfg.wandb_mode,
        project=f"ali-cfm-{cfg.dataset}-{cfg.n_data_dims}D",
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


            # Configure neural networks
            interpolant = TrainableInterpolant(cfg.dim, cfg.net_hidden, cfg.t_smooth, True,
                                                    regulariser=cfg.regulariser
                                                    ).to(cfg.device)
            pretrain_optimizer_G = torch.optim.Adam(interpolant.parameters(), lr=1e-3)

            discriminator = Discriminator(
                cfg.dim + 1, cfg.net_hidden, apply_sigmoid=False
            ).to(cfg.device)
            gan_optimizer_G = torch.optim.Adam(interpolant.parameters(), lr=cfg.lr_G)
            gan_optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_D)

            ot_cfm_model = MLP(dim=cfg.dim, time_varying=True, w=cfg.net_hidden).to(cfg.device)
            ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), cfg.lr_CFM)

            if cfg.train_interpolants:
                # Pretrain interpolant
                if cfg.n_pretrain_epochs > 0:
                    pretain_metric_prefix = f"{metric_prefix}_pretrain"
                    wandb.define_metric(f"{pretain_metric_prefix}/*",
                                        step_metric=f"{pretain_metric_prefix}_step")
                    pretain_interpolant(
                        interpolant, pretrain_optimizer_G, ot_sampler, data,
                        cfg.n_pretrain_epochs, cfg.batch_size, curr_timesteps,
                        ot=cfg.pretain_ot, metric_prefix=pretain_metric_prefix,
                        device=cfg.device
                )

                # Train interpolant with GAN
                interpolant_metric_prefix = f"{metric_prefix}_interpolant"
                wandb.define_metric(f"{interpolant_metric_prefix}/*",
                                    step_metric=f"{interpolant_metric_prefix}_step")

                train_interpolant_with_gan(
                    interpolant, discriminator, ot_sampler,
                    data,
                    gan_optimizer_G, gan_optimizer_D,
                    cfg.n_epochs, cfg.batch_size, cfg.correct_coeff,
                    curr_timesteps, seed, min_max, ot=cfg.interpolant_ot,
                    metric_prefix=interpolant_metric_prefix,
                    device=cfg.device, gan_loss=cfg.gan_loss, plot_fn=plot_fn, compute_emd_flag=True
                )

                checkpoint = {
                    "interpolant": interpolant.state_dict(),
                    "discriminator": discriminator.state_dict()
                }
                save_path = os.path.join(
                    run.dir, "checkpoints", f"{metric_prefix}_ali_cfm.pth"
                )
                torch.save(checkpoint, save_path)
            else:
                PATH = ("/home/oskar/phd/interpolnet/Mixture-FMLs/rotating_MNIST/wandb/"
                        "run-20250915_145507-2k36hqa2/files/checkpoints")
                load_checkpoint = torch.load(PATH + f"/{metric_prefix}_ali_cfm.pth", weights_only=True)
                interpolant.load_state_dict(load_checkpoint['interpolant'])


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
                    "ot_cfm_model": ot_cfm_model.state_dict(),
                    "interpolant": interpolant.state_dict(),
                    "discriminator": discriminator.state_dict()
                }
                save_path = os.path.join(
                    run.dir, "checkpoints", f"{metric_prefix}_ali_cfm.pth"
                )
                torch.save(checkpoint, save_path)
            else:
                PATH = ("/home/oskar/phd/interpolnet/Mixture-FMLs/rotating_MNIST/wandb"
                        "/run-20250915_155452-15yae40c/files/checkpoints")
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



    wandb.finish()


if __name__ == "__main__":
    with initialize(config_path="./configs"):
        cfg = compose(config_name="ali.yaml")
        train_ali(cfg)