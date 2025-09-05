import torch
from torch import nn

import os
import wandb
import warnings
from matplotlib import pyplot as plt
from tqdm.auto import trange, tqdm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hydra import compose, initialize
import numpy as np

from omegaconf import OmegaConf

from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from torchcfm.conditional_flow_matching import OTPlanSampler

from data import (
    get_dataset,
    denormalize,
    denormalize_gradfield
)

from utils import (
    sample_gan_batch,
    sample_x0_x1,
    compute_emd,
    fix_seed,
    integrate_interpolant,
    remove,
    finish_results_table,
    CellOverlayViewer
)
from interpolants import TrainableInterpolant, Discriminator, MLP


def pretain_interpolant(interpolant, pretrain_optimizer_G, ot_sampler,
                        train_data, n_pretrain_epochs, gan_batch_size,
                        train_timesteps, ot='none', metric_prefix="", device='cpu'):
    if n_pretrain_epochs == 0:
        return

    for epoch in trange(n_pretrain_epochs, desc="Pretraining Interpolant", leave=False):
        pretrain_optimizer_G.zero_grad()
        batch = sample_gan_batch(train_data, gan_batch_size, divisor=max(train_timesteps), ot_sampler=ot_sampler, ot=ot,
                                 times=train_timesteps)
        x0, x1, xt, t = (x.to(device) for x in batch)
        xt_fake = interpolant(x0, x1, t)

        loss = (xt_fake - xt).pow(2).mean()
        loss.backward()

        pretrain_optimizer_G.step()
        wandb.log({
            f"{metric_prefix}/pretrain_loss": loss.item(),
            f"{metric_prefix}_step": epoch}
        )


def train_interpolant_with_gan(
        interpolant, discriminator, ot_sampler, data,
        gan_optimizer_G, gan_optimizer_D, n_epochs,
        gan_batch_size, correct_coeff, train_timesteps, seed, min_max,
        ot='none', plot_freaquency=5_000, metric_prefix="", device='cpu', gan_loss='vanilla'
):
    cov = CellOverlayViewer('/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01/')
    t_max = max(train_timesteps)
    curr_epoch = 0
    denormalized_data = [denormalize(x, min_max) for x in data]
    for epoch in trange(curr_epoch, n_epochs,
                        desc="Training GAN Interpolant", leave=False):
        # if epoch > 40_000:
        #     gan_optimizer_G.param_groups[0]['lr'] = 1e-5
        #     gan_optimizer_D.param_groups[0]['lr'] = 5e-5
        # elif epoch > 100_000:
        #     gan_optimizer_G.param_groups[0]['lr'] = 1e-6
        #     gan_optimizer_D.param_groups[0]['lr'] = 5e-6

        curr_epoch += 1

        batch = sample_gan_batch(data, gan_batch_size, divisor=t_max, ot_sampler=ot_sampler, ot=ot,
                                 times=train_timesteps)
        x0, x1, xt, t = (x.to(device) for x in batch)
        xt_fake = interpolant(x0, x1, t)

        real_inputs = torch.cat([xt, t], dim=-1)
        fake_inputs = torch.cat([xt_fake.detach(), t], dim=-1)

        real_proba = discriminator(real_inputs)
        fake_proba = discriminator(fake_inputs)

        # Train discriminator
        gan_optimizer_D.zero_grad()
        if (gan_loss == "RpGAN") or (gan_loss == "R3GAN"):
            # Using the relativistic pairing GAN loss
            r1 = 0
            r2 = 0
            if gan_loss == "R3GAN":
                xt_fake_ = torch.cat([xt_fake, t], dim=-1).detach().requires_grad_(True)
                xt_ = torch.cat([xt, t], dim=-1).detach().requires_grad_(True)
                disc_score_fake = discriminator(xt_fake_)
                disc_score_real = discriminator(xt_)
                grad_D = torch.autograd.grad(outputs=disc_score_real.sum(), inputs=xt_, create_graph=True)[0]
                grad_G = torch.autograd.grad(outputs=disc_score_fake.sum(), inputs=xt_fake_, create_graph=True)[0]
                gamma = 1.
                r1 = gamma / 2 * ((grad_D.view(gan_batch_size, -1).norm(2, dim=1) - 0) ** 2)
                r2 = gamma / 2 * ((grad_G.view(gan_batch_size, -1).norm(2, dim=1) - 0) ** 2)
            d_loss = torch.nn.functional.softplus((fake_proba - real_proba) + r1 + r2).mean()

            d_loss.backward()
            gan_optimizer_D.step()

            # Train generator
            gan_optimizer_G.zero_grad()
            real_inputs = torch.cat([xt, t], dim=-1)
            fake_inputs = torch.cat([xt_fake, t], dim=-1)

            real_proba = discriminator(real_inputs)
            fake_proba = discriminator(fake_inputs)
            g_loss_ = torch.nn.functional.softplus((real_proba - fake_proba)).mean()
        else:
            d_real_loss = nn.functional.softplus(-real_proba).mean()
            d_fake_loss = nn.functional.softplus(fake_proba).mean()
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            gan_optimizer_D.step()

            # Train generator
            gan_optimizer_G.zero_grad()
            fake_inputs = torch.cat([xt_fake, t], dim=-1)
            fake_proba = discriminator(fake_inputs)

            g_loss_ = nn.functional.softplus(-fake_proba).mean()
        reg_weight_loss = interpolant.get_reg_term(x0, x1, t, xt_fake)
        g_loss = g_loss_ + correct_coeff * reg_weight_loss

        g_loss.backward()
        gan_optimizer_G.step()

        wandb.log({
            f"{metric_prefix}/d_loss": d_loss.item(),
            f"{metric_prefix}/g_loss": g_loss.item(),
            f"{metric_prefix}/reg_weight_loss": reg_weight_loss.item(),
            f"{metric_prefix}/fake_proba": fake_proba.mean().item(),
            f"{metric_prefix}/real_proba": real_proba.mean().item(),
            f"{metric_prefix}_step": epoch
        })

        if epoch % plot_freaquency == 0:
            with torch.no_grad():
                batch = sample_gan_batch(data, gan_batch_size, divisor=t_max, ot_sampler=ot_sampler, ot=ot,
                                         times=train_timesteps)
                x0, x1, _, _ = (x.to(device) for x in batch)
                # n_steps = 200
                # t_linspace = torch.linspace(0, t_max, n_steps).to(device) / t_max
                inferred_mask = torch.zeros((len(train_timesteps), x0.shape[0], 2), device=device)
                for i, t_ in enumerate(train_timesteps):
                    t = t_ / t_max * torch.ones((x0.shape[0], 1), device=device)
                    xt = interpolant(x0, x1, t, training=False)
                    inferred_mask[i] = denormalize(xt, min_max)

            cov.animate_masks(inferred_mask.cpu().numpy(), denormalized_data,
                              save_path=f"/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking"
                                        f"/inferred_gifs/interpolants/ali_{epoch}.gif")

            n_steps, K, _ = inferred_mask.shape

            # Points: (n_steps*K, 2)
            pts = inferred_mask.reshape(-1, 2).cpu().numpy()

            # Color per point = its t; repeat each t for all K samples
            tvals = torch.repeat_interleave(torch.tensor(train_timesteps, device=device) / t_max, K).cpu().numpy()

            fig, ax = plt.subplots(figsize=(6, 5))

            norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)  # ensures 0 bottom, 1 top
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=tvals, cmap="viridis", norm=norm,
                            s=6, alpha=0.5)

            # tight colorbar on the side
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label(r"$t$")
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # optional

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_ylim(200, 500)
            ax.set_xlim(200, 500)
            plt.tight_layout()
            wandb.log({f"{metric_prefix}/scatter": wandb.Image(fig), f"{metric_prefix}_step": epoch})
            plt.close(fig)

            T = len(data)

            # Concatenate all points
            points = np.concatenate(data, axis=0)  # (sum_i n_i, 2)

            # Build time labels: repeat time index for each sample
            labels = np.concatenate([np.full(len(arr), i) for i, arr in enumerate(data)])

            # Normalize to [0,1] for colormap
            tvals = labels / (T - 1)

            # Scatter
            fig, ax = plt.subplots(figsize=(6, 5))
            norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            sc = ax.scatter(points[:, 0], points[:, 1], c=tvals, cmap="viridis", norm=norm, s=6, alpha=0.3)

            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label("time")
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels(["start", "¼", "½", "¾", "end"])

            plt.tight_layout()
            wandb.log({f"{metric_prefix}/scatter_real": wandb.Image(fig), f"{metric_prefix}_step": epoch})
            plt.close(fig)


def train_ot_cfm(ot_cfm_model, ot_cfm_optimizer, interpolant,
                 ot_sampler, train_data, batch_size, min_max,
                 n_ot_cfm_epochs, metric_prefix="", ot='border', device='cpu', times=(0, -1)):
    for step in trange(n_ot_cfm_epochs,
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        if ot == 'mmot':
            t_max = max(times)
            batch = sample_gan_batch(train_data, batch_size, divisor=t_max, ot_sampler=ot_sampler, ot=ot, times=times)
            x0, x1, xt, _ = (x.to(device) for x in batch)
        else:
            x0, x1 = sample_x0_x1(train_data, batch_size, device=device)
            if ot == 'border' or ot == 'full':
                x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0], 1, device=device)

        xt = interpolant(x0, x1, t, training=False).detach()
        ut = interpolant.dI_dt(x0, x1, t).detach()

        ut = denormalize_gradfield(ut, min_max).float()
        xt = denormalize(xt, min_max).float()

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
    n_samples = 10

    data, min_max = get_dataset(cfg.dataset, cfg.n_data_dims,
                                cfg.normalize_dataset, cfg.whiten)
    timesteps_list = [t for t in range(len(data))]
    # This code assumes that timesteps are in [0, ..., T_max]
    num_int_steps = cfg.num_int_steps_per_timestep

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

        curr_timesteps = timesteps_list
        metric_prefix = f"t={seed=}"

        N_min = min([x.shape[0] for x in data])
        idx = np.random.choice(np.arange(0, N_min), n_samples, replace=False)
        # subset_data = [x[np.random.choice(np.arange(0, x.shape[0]), n_samples, replace=False)] for x in data]
        subset_data = [x[idx] for x in data]

        # Configure neural networks
        interpolant = TrainableInterpolant(cfg.dim, cfg.net_hidden, cfg.t_smooth, True).to(cfg.device)
        pretrain_optimizer_G = torch.optim.Adam(interpolant.parameters(), lr=1e-3)

        discriminator = Discriminator(
            cfg.dim + 1, cfg.net_hidden, apply_sigmoid=False
        ).to(cfg.device)
        gan_optimizer_G = torch.optim.Adam(interpolant.parameters(), lr=cfg.lr_G)
        gan_optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_D)

        ot_cfm_model = MLP(dim=cfg.dim, time_varying=True, w=cfg.net_hidden).to(cfg.device)
        ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), 1e-4)

        if cfg.train_interpolants:
            # Pretrain interpolant
            pretain_metric_prefix = f"{metric_prefix}_pretrain"
            wandb.define_metric(f"{pretain_metric_prefix}/*",
                                step_metric=f"{pretain_metric_prefix}_step")
            pretain_interpolant(
                interpolant, pretrain_optimizer_G, ot_sampler, subset_data,
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
                subset_data,
                gan_optimizer_G, gan_optimizer_D,
                cfg.n_epochs, cfg.batch_size, cfg.correct_coeff,
                curr_timesteps, seed, min_max, ot=cfg.interpolant_ot,
                metric_prefix=interpolant_metric_prefix,
                device=cfg.device, gan_loss=cfg.gan_loss
            )
        else:
            PATH = ("/home/oskar/phd/interpolnet/Mixture-FMLs/Mixture-FMLs-kirill_single_cell_experiments/"
                    "ali_cfm/wandb/run-20250822_091600-1kw46hn4/files/checkpoints")
            load_checkpoint = torch.load(PATH + f"/{metric_prefix}_ali_cfm.pth", weights_only=True)
            interpolant.load_state_dict(load_checkpoint['interpolant'])

        # Compute metrics for GAN interpolant
        # x0 = subset_data[0].to(cfg.device)
        # x1 = subset_data[-1].to(cfg.device)
        # x0, x1 = ot_sampler.sample_plan(x0, x1)
        #
        # int_traj = integrate_interpolant(
        #     x0, x1, num_int_steps, interpolant,
        # )

        # TODO: plot interpolant trajectories

        # t = removed_t / (len(timesteps_list) - 1)
        #
        # int_emd = compute_emd(
        #     denormalize(data[removed_t], min_max),
        #     denormalize(int_traj[int(100 * t)], min_max)
        # )
        # int_results[f"seed={seed}"].append(int_emd.item())

        if cfg.train_cfm:
            # Train OT-CFM using GAN interpolant
            cfm_metric_prefix = f"{metric_prefix}_cfm"
            wandb.define_metric(f"{cfm_metric_prefix}/*",
                                step_metric=f"{cfm_metric_prefix}_step")
            train_ot_cfm(
                ot_cfm_model, ot_cfm_optimizer, interpolant, ot_sampler,
                subset_data, cfg.batch_size, min_max, cfg.n_ot_cfm_epochs,
                metric_prefix=cfm_metric_prefix, ot=cfg.cfm_ot,
                device=cfg.device, times=curr_timesteps
            )
        else:
            PATH = ("/home/oskar/phd/interpolnet/Mixture-FMLs/Mixture-FMLs-kirill_single_cell_experiments/"
                    "ali_cfm/wandb/run-20250822_091600-1kw46hn4/files/checkpoints")
            load_checkpoint = torch.load(PATH + f"/{metric_prefix}_ali_cfm.pth", weights_only=True)
            ot_cfm_model.load_state_dict(load_checkpoint['ot_cfm_model'])

        # Compute metrics for OT-CFM
        node = NeuralODE(torch_wrapper(ot_cfm_model),
                         solver="dopri5", sensitivity="adjoint")

        mean_emd = 0
        for t in timesteps_list[1:]:
            t_prev = (t - 1) / (len(timesteps_list) - 1)
            t_curr = t / (len(timesteps_list) - 1)
            with torch.no_grad():
                ot_cfm_traj = node.trajectory(
                    denormalize(data[t - 1], min_max).to(cfg.device),
                    t_span=torch.linspace(t_prev, t_curr, num_int_steps + 1),
                )
                cfm_emd = compute_emd(
                    denormalize(data[t], min_max).to(cfg.device),
                    ot_cfm_traj[-1].float().to(cfg.device),
                )
                mean_emd += cfm_emd / (len(timesteps_list) - 1)
        cfm_results[f"seed={seed}"].append(cfm_emd.item())

        # TODO: plot CFM trajectories and save on wandb

        wandb.log({
            f"{metric_prefix}_cfm/cfm_result": cfm_emd.item()
        })

        # Save artifacts for given seed and t
        # checkpoint = {
        #    "interpolant": interpolant.state_dict(),
        #    "discriminator": discriminator.state_dict(),
        #    "ot_cfm_model": ot_cfm_model.state_dict(),
        # }
        # save_path = os.path.join(
        #    run.dir, "checkpoints", f"{metric_prefix}_ali_cfm.pth"
        # )
        # torch.save(checkpoint, save_path)

    # int_results = wandb.Table(
    #     dataframe=finish_results_table(int_results, timesteps_list[1: -1])
    # )
    # wandb.log({"interpolant_results": int_results})
    #
    # cfm_results = wandb.Table(
    #     dataframe=finish_results_table(cfm_results, timesteps_list[1: -1])
    # )
    # wandb.log({"cfm_results": cfm_results})

    wandb.finish()


if __name__ == "__main__":
    with initialize(config_path="./configs"):
        cfg = compose(config_name="ali.yaml")
        train_ali(cfg)