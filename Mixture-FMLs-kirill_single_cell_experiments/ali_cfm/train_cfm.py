import torch

import hydra
from tqdm.auto import  tqdm, trange

from interpolants import MLP
from  torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ConditionalFlowMatcher,
)
from cubic_fm import MultiMarginalFlowMatcher
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

from omegaconf import OmegaConf

import wandb
import utils

from data import get_dataset, denormalize


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="cfm",
)
def train_ot_cfm(cfg):
    if cfg.interpolant == "linear-ot":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
        get_batch_fn = utils.get_batch
    elif cfg.interpolant == "linear-i":
        FM = ConditionalFlowMatcher(sigma=0.1)
        get_batch_fn = utils.get_batch
    elif cfg.interpolant in {"lagrange", "cubic"}:
        FM = MultiMarginalFlowMatcher(sigma=0.1, interpolation=cfg.interpolant)
        get_batch_fn = utils.get_batch_for_cubic
    else:
        raise ValueError(
            f"Unknown interpolant: {cfg.interpolant}. "
            "Available options: linear-ot, linear-i, lagrange, cubic."
        )
    
    data, min_max = get_dataset(
        cfg.dataset, cfg.n_data_dims,
        whiten=cfg.whiten,
        normalize=cfg.normalize_dataset
    )
    
    timesteps_list = [t for t in range(len(data))]
    t_max = max(timesteps_list)
    # This code assumes that timesteps are in [0, ..., T_max]
    num_int_steps = cfg.num_int_steps_per_timestep * t_max

    cfm_results = {}

    wandb.init(
        name=f"{cfg.interpolant}-cfm-{cfg.dataset}-{cfg.n_data_dims}D",
        project="ali-cfm",
        config=OmegaConf.to_object(cfg),
        mode=cfg.wandb_mode,
    )

    for seed in tqdm(cfg.seed_list, desc="Seed", leave=True):
        utils.fix_seed(seed)
        cfm_results[f"seed={seed}"] = []
        
        for removed_t in tqdm(timesteps_list[1: -1], desc="timesteps", leave=False):
            curr_timesteps = [t for t in timesteps_list if t != removed_t]
            wandb.define_metric(
                f"t={removed_t}_{seed=}_cfm_loss", 
                step_metric=f"t={removed_t}_{seed=}_cfm_loss"
            )

            ot_cfm_model = MLP(
                dim=cfg.dim, 
                time_varying=cfg.time_varying, 
                w=cfg.net_hidden
            ).to(cfg.device)

            ot_cfm_optimizer = torch.optim.AdamW(
                ot_cfm_model.parameters(), 
                lr=cfg.lr, 
                weight_decay=cfg.weight_decay
            )
            
            for step in trange(cfg.n_iter, desc="Training Flow Matching", leave=False):
                ot_cfm_optimizer.zero_grad()
                t, xt, ut = get_batch_fn(
                    FM, data, cfg.batch_size, 
                    timesteps=curr_timesteps,
                    device=cfg.device
                )

                vt = ot_cfm_model(torch.cat([xt, t[:, None]], dim=-1))
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()

                wandb.log(
                    {
                        f"t={removed_t}_{seed=}/cfm_loss": loss.item(),
                        f"t={removed_t}_{seed=}_step": step
                    }
                )
                ot_cfm_optimizer.step()

            node = NeuralODE(torch_wrapper(ot_cfm_model), 
                            solver="dopri5", sensitivity="adjoint")
            
            with torch.no_grad():
                ot_cfm_traj = node.trajectory(
                    data[0].float().to(cfg.device),
                    t_span=torch.linspace(0, t_max, num_int_steps + 1),
                ).cpu()

            cfm_emd = utils.compute_emd(
                denormalize(data[removed_t].float(), min_max),
                denormalize(ot_cfm_traj[100 * removed_t], min_max),
            )
            cfm_results[f"seed={seed}"].append(cfm_emd.item())
    
    cfm_results = wandb.Table(
        dataframe=utils.finish_results_table(
            cfm_results, 
            timesteps=timesteps_list[1: -1]
        )
    )
    wandb.log({"cfm_results": cfm_results})
    wandb.finish()


if __name__ == "__main__":
    train_ot_cfm()
