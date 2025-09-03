import random
import numpy as np
import torch
from torch import nn

import os
import wandb

import click
import hydra
from hydra import compose, initialize

import warnings
from tqdm.auto import trange, tqdm

from omegaconf import OmegaConf, DictConfig

from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from torchcfm.conditional_flow_matching import OTPlanSampler

from ali_cfm.data_utils import get_dataset, denormalize
from ali_cfm.loggin_and_metrics import finish_results_table, compute_emd
from ali_cfm.nets import TrainableInterpolant, Discriminator, MLP
from ali_cfm.training import (
    pretain_interpolant, train_ot_cfm, 
    train_interpolant_with_gan, integrate_interpolant,
    init_cfm_from_checkpoint, init_interpolant_from_checkpoint,
)


def fix_seed(seed: int = 42):
    random.seed(seed)               
    np.random.seed(seed)            
    torch.manual_seed(seed)         
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_ali(cfg):
    seed_list = cfg.seed_list
    ot_sampler = OTPlanSampler('exact', reg=0.1)

    data, min_max = get_dataset(
        name=cfg.dataset, 
        n_data_dims=cfg.dim, 
        normalize=cfg.normalise, 
        whiten=cfg.whiten
    )
    # This code assumes that timesteps are in [0, ..., T_max]
    timesteps_list = [t for t in range(len(data))]
    
    int_results, cfm_results = {}, {}

    run = wandb.init(
        name=f"{cfg.gan_loss}-{cfg.wandb_name}-{cfg.dataset}-{cfg.dim}D",
        mode=cfg.wandb_mode,
        project=f"ali-cfm-{cfg.dataset}-{cfg.dim}D",
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
            interpolant = TrainableInterpolant(
                dim=cfg.dim,
                h_dim=cfg.interpolant_dim,
                t_smooth=cfg.t_smooth,
            ).to(cfg.device)
            pretrain_optimizer_G = torch.optim.Adam(
                interpolant.parameters(), 
                lr=cfg.pretrain_lr,
            )

            discriminator = Discriminator(
                in_dim=cfg.dim + 1, 
                w=cfg.disc_dim, 
                apply_sigmoid=False,
            ).to(cfg.device)
            gan_optimizer_G = torch.optim.Adam(
                interpolant.parameters(), 
                lr=cfg.interpolant_gen_lr
            )
            gan_optimizer_D = torch.optim.Adam(
                discriminator.parameters(), 
                lr=cfg.interpolant_disc_lr
            )

            ot_cfm_model = MLP(
                dim=cfg.dim, 
                time_varying=True, 
                w=cfg.cfm_dim,
            ).to(cfg.device)
            ot_cfm_optimizer = torch.optim.Adam(
                ot_cfm_model.parameters(), 
                lr=cfg.cfm_lr,
            )

            # Train interpolant model
            if cfg.train_interpolants:
                # Pretrain interpolant
                pretain_metric_prefix = f"{metric_prefix}_pretrain"
                wandb.define_metric(f"{pretain_metric_prefix}/*",
                                    step_metric=f"{pretain_metric_prefix}_step")
                pretain_interpolant(
                    interpolant, pretrain_optimizer_G, ot_sampler, data,
                    cfg.num_ali_pretrain_steps, cfg.batch_size, curr_timesteps,
                    ot=cfg.pretain_ot, 
                    metric_prefix=pretain_metric_prefix,
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
                    cfg.num_ali_train_steps, cfg.batch_size, cfg.correct_coeff,
                    curr_timesteps, seed, min_max, 
                    ot=cfg.interpolant_ot,
                    metric_prefix=interpolant_metric_prefix,
                    device=cfg.device, 
                    gan_loss=cfg.gan_loss
                )
            else:
                init_interpolant_from_checkpoint()
            
            # Compute metrics for GAN interpolant
            x0 = data[0].to(cfg.device)
            x1 = data[-1].to(cfg.device)
            x0, x1 = ot_sampler.sample_plan(x0, x1)

            int_traj = integrate_interpolant(
                x0, x1, 
                cfg.num_int_steps_per_timestep * timesteps_list[-1], 
                interpolant,
            )

            int_emd = compute_emd(
                denormalize(data[removed_t], min_max), 
                denormalize(int_traj[100 * removed_t], min_max)
            )
            int_results[f"seed={seed}"].append(int_emd.item())

            # Train CFM model using GAN interpolant
            if cfg.train_cfm:
                cfm_metric_prefix = f"{metric_prefix}_cfm"
                wandb.define_metric(f"{cfm_metric_prefix}/*",
                                    step_metric=f"{cfm_metric_prefix}_step")
                train_ot_cfm(
                    ot_cfm_model, ot_cfm_optimizer, 
                    interpolant, ot_sampler,
                    data, 
                    cfg.batch_size, min_max, cfg.num_cfm_train_steps,
                    metric_prefix=cfm_metric_prefix, 
                    ot=cfg.cfm_ot,
                    device=cfg.device, 
                    times=curr_timesteps
                )
            else:
                init_cfm_from_checkpoint()

            # Compute metrics for OT-CFM
            node = NeuralODE(
                vector_field=torch_wrapper(ot_cfm_model),
                solver="dopri5", 
                sensitivity="adjoint"
            )

            with torch.no_grad():
                ot_cfm_traj = []
                _batch_size = cfg.trajectory_simulation_batch_size
                for i in trange(
                        (len(data[removed_t - 1]) + _batch_size - 1) // _batch_size,
                        leave=False,
                        desc='Collecting CFM trajectories.'
                    ):
                    data_ = data[removed_t - 1][i * _batch_size: (i + 1) * _batch_size]
                    batched_traj = node.trajectory(
                        data_.to(cfg.device),
                        t_span=torch.linspace(
                            start=(removed_t - 1) / max(timesteps_list), 
                            end=removed_t / max(timesteps_list), 
                            steps=cfg.num_int_steps_per_timestep
                        ),
                    )
                    ot_cfm_traj.append(batched_traj)
                ot_cfm_traj = torch.cat(ot_cfm_traj, dim=1).float()

            cfm_emd = compute_emd(
                denormalize(data[removed_t], min_max).to(cfg.device), 
                denormalize(ot_cfm_traj[-1], min_max).to(cfg.device),
            )
            cfm_results[f"seed={seed}"].append(cfm_emd.item())

            wandb.log({
                f"{metric_prefix}_cfm/cfm_result": cfm_emd.item()
            })

            # Save artifacts for given seed and t
            checkpoint = {
               "interpolant": interpolant.state_dict(),
               "discriminator": discriminator.state_dict(),
               "ot_cfm_model": ot_cfm_model.state_dict(),
            }
            save_path = os.path.join(
               run.dir, "checkpoints", f"{metric_prefix}_ali_cfm.pth"
            )
            torch.save(checkpoint, save_path)

    int_results = finish_results_table(int_results, timesteps_list[1: -1])
    cfm_results = finish_results_table(cfm_results, timesteps_list[1: -1])
    int_results.to_csv(exp_path / 'int_results.csv', index=False)  
    cfm_results.to_csv(exp_path / 'cfm_results.csv', index=False)  

    wandb.log({"interpolant_results": wandb.Table(int_results)})
    wandb.log({"cfm_results": wandb.Table(cfm_results)})
    wandb.finish()


@click.command()
@click.option("--config", type=click.Path(exists=True), help="Path to the config file.")
@click.argument("overrides", nargs=-1)
def main(config, overrides):
    base_cfg = OmegaConf.load(config)
    overrides_cfg = OmegaConf.from_dotlist(list(overrides))
    cfg = OmegaConf.merge(base_cfg, overrides_cfg)

    os.environ["HYDRA_FULL_ERROR"] = '1'
    warnings.filterwarnings("ignore")
    train_ali(cfg)


if __name__ == "__main__":
    main()
