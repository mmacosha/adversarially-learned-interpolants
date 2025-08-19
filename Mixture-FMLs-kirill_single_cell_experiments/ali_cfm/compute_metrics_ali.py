import torch
from torch import nn

import click
import os
import wandb
import warnings
from matplotlib import pyplot as plt
from tqdm.auto import trange, tqdm

from omegaconf import OmegaConf
from sklearn.decomposition import PCA

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
    get_run_dir,
    finish_results_table,
)
from interpolants import TrainableInterpolant, Discriminator, MLP


def pretain_interpolant(interpolant, pretrain_optimizer_G, ot_sampler, 
                        train_data, n_pretrain_epochs, gan_batch_size, 
                        train_timesteps, ot='none', metric_prefix="", device='cpu'):
    if n_pretrain_epochs == 0:
        return
    
    for epoch in trange(n_pretrain_epochs, desc="Pretraining Interpolant", leave=False):
        pretrain_optimizer_G.zero_grad()
        batch = sample_gan_batch(train_data, gan_batch_size, 
                                         ot_sampler=ot_sampler,
                                         ot=ot, times=train_timesteps)
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
    ot='none', plot_freaquency=5000, metric_prefix="", device='cpu'
):
    t_max = max(train_timesteps)
    curr_epoch = 0
    for epoch in trange(curr_epoch, n_epochs, 
                        desc="Training GAN Interpolant", leave=False):
        if epoch > 40_000:
            gan_optimizer_G.param_groups[0]['lr'] = 1e-5
            gan_optimizer_D.param_groups[0]['lr'] = 5e-5
        elif epoch > 100_000:
            gan_optimizer_G.param_groups[0]['lr'] = 1e-6
            gan_optimizer_D.param_groups[0]['lr'] = 5e-6
        
        curr_epoch += 1
        if epoch % 5_000 == 0:
            with torch.no_grad():
                for time in range(1, 4):
                    test_batch = sample_gan_batch(
                        data, 2300, ot_sampler=ot_sampler, 
                        time=time, ot='border')
                    x0_test, x1_test, xt_test, t_test = (
                        x.to(device) for x in test_batch
                    )
                    
                    xt_fake_test = interpolant(x0_test, x1_test, t_test)
                    emd_t = compute_emd(
                            denormalize(xt_test, min_max), 
                            denormalize(xt_fake_test, min_max)
                        )
                    wandb.log({
                        f"{metric_prefix}/emd_t={time}": emd_t.item(), 
                        f"{metric_prefix}_step": epoch
                    })
                    

        batch = sample_gan_batch(data, gan_batch_size, 
                                 ot_sampler=ot_sampler, ot=ot, 
                                 times=train_timesteps)
        x0, x1, xt, t = (x.to(device) for x in batch)
        xt_fake = interpolant(x0, x1, t)

        real_inputs = torch.cat([xt, t], dim=-1)
        fake_inputs = torch.cat([xt_fake.detach(), t], dim=-1)
        
        real_proba = discriminator(real_inputs)
        fake_proba = discriminator(fake_inputs)
        
        # Train discriminator
        gan_optimizer_D.zero_grad()
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
                batch = sample_gan_batch(data, 256, ot_sampler=ot_sampler, 
                                         ot='full', times=train_timesteps)
                x0, x1, xt, t = (x.to(device) for x in batch)
                xt_fake = interpolant(x0, x1, t)

                pca = PCA(n_components=2, random_state=seed)
                
                xt_pca = pca.fit_transform(xt.cpu())
                xt_fake_pca = pca.transform(xt_fake.cpu())
                fig = plt.figure()
                
                plt.scatter(*xt_fake_pca.T, c='red', alpha=0.5, label="Fake")
                plt.scatter(*xt_pca.T, c='blue', alpha=0.5, label="Real")
                plt.legend()
                plt.title(f"PCA of `True` and `Fake` samples for t={int(t[0] * t_max)}")

                wandb.log({
                    f"{metric_prefix}/scatter_image": wandb.Image(fig), 
                    f"{metric_prefix}_step": epoch
                })
                plt.close(fig) 


def train_ot_cfm(ot_cfm_model, ot_cfm_optimizer, interpolant, 
                 ot_sampler, train_data, batch_size, min_max,
                 n_ot_cfm_epochs, metric_prefix="", ot='border', device='cpu'):
    for step in trange(n_ot_cfm_epochs, 
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        x0, x1 = sample_x0_x1(train_data, batch_size, device=device)
        
        if ot == 'border' or ot == 'full':
            x0, x1 = ot_sampler.sample_plan(x0, x1)
        
        t = torch.rand(x0.shape[0], 1, device=device)
        
        xt = interpolant(x0, x1, t).detach()
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

@click.command()
@click.option("--run_id", type=str)
@click.option("--wandb_mode", type=str, default="online")
def compute_metrics_ali(run_id: str, wandb_mode: str):
    def _load_config(path):
        cfg = OmegaConf.load(path)
        del cfg._wandb
        for key in cfg.keys():
            cfg[key] = cfg[key]['value']

        cfg.num_int_steps_per_timestep = 100

        if 'net_hidden' not in cfg:
            cfg.net_hidden = 64
        
        return cfg

    run_dir = get_run_dir(run_id)
    cfg = _load_config(run_dir / "config.yaml")
    cfg.wandb_mode = wandb_mode

    seed_list = cfg.seed_list
    warnings.filterwarnings("ignore")
    ot_sampler = OTPlanSampler('exact', reg=0.1)

    data, min_max = get_dataset(cfg.dataset, cfg.n_data_dims, 
                                cfg.normalize_dataset, cfg.whiten)
    timesteps_list = [t for t in range(len(data))]
    # This code assumes that timesteps are in [0, ..., T_max]
    num_int_steps = cfg.num_int_steps_per_timestep * max(timesteps_list)
    
    print(num_int_steps)
    

    assert 0
    int_results, cfm_results = {}, {}

    run = wandb.init(
        name=f"{cfg.wandb_name}-{cfg.dataset}-{cfg.n_data_dims}D-metric_compute",
        mode=cfg.wandb_mode,
        project="ali-cfm",
        config=OmegaConf.to_object(cfg)
    )

    os.makedirs(f"{run.dir}/checkpoints", exist_ok=True)
    
    for seed in tqdm(seed_list, desc="Seeds"):
        fix_seed(seed)
        int_results[f"seed={seed}"] = []
        cfm_results[f"seed={seed}"] = []

        for removed_t in tqdm(timesteps_list[1: -1], desc="Timesteps", leave=False):
            metric_prefix = f"t={removed_t}_{seed=}"
            checkpoint = torch.load(
                run_dir / "checkpoints" / f"{metric_prefix}_ali_cfm.pth", 
                map_location=cfg.device
            )

            interpolant = TrainableInterpolant(cfg.dim, cfg.net_hidden, 0.01, True)
            interpolant.load_state_dict(checkpoint["interpolant"])
            interpolant.to(cfg.device)

            ot_cfm_model = MLP(dim=cfg.dim, time_varying=True, w=cfg.net_hidden)
            ot_cfm_model.load_state_dict(checkpoint["ot_cfm_model"])
            ot_cfm_model.to(cfg.device)

            
            # Compute metrics for GAN interpolant
            x0 = data[0].to(cfg.device)
            x1 = data[-1].to(cfg.device)
            x0, x1 = ot_sampler.sample_plan(x0, x1)

            int_traj = integrate_interpolant(
                x0, x1, num_int_steps, interpolant,
            )

            int_emd = compute_emd(
                denormalize(data[removed_t], min_max), 
                denormalize(int_traj[100 * removed_t], min_max)
            )
            int_results[f"seed={seed}"].append(int_emd.item())
            
            # Compute metrics for OT-CFM
            node = NeuralODE(torch_wrapper(ot_cfm_model),
                            solver="dopri5", sensitivity="adjoint")

            with torch.no_grad():
                ot_cfm_traj = node.trajectory(
                    denormalize(data[0], min_max).to(cfg.device),
                    t_span=torch.linspace(0, 1, num_int_steps + 1),
                )

            cfm_emd = compute_emd(
                denormalize(data[removed_t], min_max).to(cfg.device), 
                ot_cfm_traj[100 * removed_t].float().to(cfg.device),
            )
            cfm_results[f"seed={seed}"].append(cfm_emd.item())

    int_results = wandb.Table(
        dataframe=finish_results_table(int_results, timesteps_list[1: -1])
    )
    wandb.log({"interpolant_results": int_results})

    cfm_results = wandb.Table(
        dataframe=finish_results_table(cfm_results, timesteps_list[1: -1])
    )
    wandb.log({"cfm_results": cfm_results})
    
    wandb.finish()


if __name__ == "__main__":
    compute_metrics_ali()
