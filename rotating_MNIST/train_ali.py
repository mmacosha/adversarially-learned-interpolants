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
from mnist_utils import plot_fn


from ali_cfm.nets import Discriminator, TrainableInterpolantMNIST, CFMNet, DiscriminatorMNIST, RotationCFM


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sobel_xy(img):  # img: (B,1,H,W)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=img.device, dtype=img.dtype).view(1,1,3,3)/8
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=img.device, dtype=img.dtype).view(1,1,3,3)/8
    gx = torch.nn.functional.conv2d(img, kx, padding=1); gy = torch.nn.functional.conv2d(img, ky, padding=1)
    return gx, gy


def train_ot_cfm(
        ot_cfm_model, ot_cfm_optimizer, interpolant,
         ot_sampler, train_data, batch_size, min_max,
         n_ot_cfm_epochs,
         metric_prefix="", ot='border', device='cpu', times=(0, -1), viz_every=1000
    ):
    for step in trange(n_ot_cfm_epochs,
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        x0, x1 = utils.sample_x0_x1(train_data, batch_size, device=device)
        if ot == 'border' or ot == 'full':
            x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0], 1, device=device)
        t = torch.clamp(t, 0.001, 1.0 - 0.01)

        xt = interpolant(x0, x1, t, training=False).detach()
        ut = interpolant.dI_dt(x0, x1, t).detach()

        vt = ot_cfm_model(torch.cat([xt, t], dim=-1))

        Ix, Iy = sobel_xy(xt.view(-1, 1, 16, 16))
        w = (Ix.pow(2) + Iy.pow(2)).sqrt().clamp_min(1e-3).view_as(vt).detach()
        loss = (w * (vt - ut)).pow(2).mean() / (w.mean() + 1e-8)

        # loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        ot_cfm_optimizer.step()

        wandb.log({
            f"{metric_prefix}/cfm_loss": loss.item(),
            f"{metric_prefix}_cfm_step": step
        })
        # --- Visual probe every viz_every steps ---
        if step % viz_every == 0:
            with torch.no_grad():
                B = min(4, xt.shape[0])
                imgs = xt[:B].cpu().view(B, 16, 16)  # input
                tgt = ut[:B].cpu().view(B, 16, 16)  # target
                pred = vt[:B].cpu().view(B, 16, 16)  # predicted
                err = (pred - tgt).cpu().view(B, 16, 16)

                fig, axes = plt.subplots(B, 4, figsize=(8, 2 * B))
                for i in range(B):
                    for j, arr in enumerate([imgs[i], tgt[i], pred[i], err[i]]):
                        ax = axes[i, j]
                        im = ax.imshow(arr.numpy(), cmap="gray")
                        ax.axis("off")
                        if i == 0:
                            ax.set_title(["x_t", "target dI/dt", "pred vθ", "error"][j])
                plt.tight_layout()
                wandb.log({f"{metric_prefix}/cfm_probe": wandb.Image(fig)})
                plt.close(fig)


# @hydra.main(config_path="./configs", config_name="ali")
def train_ali(cfg):
    os.environ["HYDRA_FULL_ERROR"] = '1'
    seed_list = cfg.seed_list
    warnings.filterwarnings("ignore")
    ot_sampler = OTPlanSampler('exact', reg=0.1)


    data, min_max = get_dataset("RotatingMNIST_train", cfg.n_data_dims, normalize=cfg.normalize_dataset)
    timesteps_list = [t for t in range(len(data))]

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

        cfg.dim = data[0].shape[1]

        # Configure neural networks
        interpolant = TrainableInterpolantMNIST(cfg.dim, cfg.net_hidden, cfg.t_smooth, True,
                                                regulariser=cfg.regulariser
                                                ).to(cfg.device)
        pretrain_optimizer_G = torch.optim.Adam(interpolant.parameters(), lr=1e-3)

        discriminator = DiscriminatorMNIST(
            cfg.dim + 1, cfg.net_hidden, apply_sigmoid=False
        ).to(cfg.device)
        gan_optimizer_G = torch.optim.Adam(interpolant.parameters(), lr=cfg.lr_G)
        gan_optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_D)

        ot_cfm_model = RotationCFM(cfg.dim).to(cfg.device)
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
                    device=cfg.device, mnist=True
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
                device=cfg.device, gan_loss=cfg.gan_loss, plot_fn=plot_fn, compute_emd_flag=False
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
                    "run-20250919_112624-gcvh356b/files/checkpoints")
            load_checkpoint = torch.load(PATH + f"/{metric_prefix}_ali_cfm.pth", weights_only=True)
            interpolant.load_state_dict(load_checkpoint['interpolant'])

        # plot_fn(interpolant, 0, seed, len(timesteps_list) - 1, data, ot_sampler, cfg.device, metric_prefix,
        #         timesteps_list, None, min_max)

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
                    "/run-20250919_134235-agse3txs/files/checkpoints")
            load_checkpoint = torch.load(PATH + f"/{metric_prefix}_ali_cfm.pth", weights_only=True)
            ot_cfm_model.load_state_dict(load_checkpoint['ot_cfm_model'])


        # Compute metrics for OT-CFM
        node = NeuralODE(torch_wrapper(ot_cfm_model),
                         solver="dopri5", sensitivity="adjoint")

        test_data, min_max = get_dataset("RotatingMNIST_test", cfg.n_data_dims, normalize=cfg.normalize_dataset)
        # test_data, min_max = get_dataset("RotatingMNIST_train", cfg.n_data_dims, normalize=cfg.normalize_dataset)
        timesteps_list_test = [t for t in range(len(test_data))]

        X0 = torch.tensor(test_data[0], dtype=torch.float32).to(cfg.device)
        # t_s = torch.linspace(0, 1, 101)
        t_s = torch.tensor(timesteps_list_test, device=cfg.device, dtype=torch.float32) / max(timesteps_list_test)
        with torch.no_grad():
            cfm_traj = node.trajectory(X0,
                                       t_s
                                       )

        img_dim = int(np.sqrt(cfg.dim))
        fig, ax = plt.subplots(1 + 4, len(timesteps_list_test), figsize=(20, 10))
        for i, t in enumerate(timesteps_list_test):
            cfm_t = t  # torch.argmin(torch.abs(t_s - t / max(timesteps_list_test)))
            cfm_emd = compute_emd(
                denormalize(test_data[t], min_max).to(cfg.device),
                cfm_traj[cfm_t].float().to(cfg.device),
            )

            t_in = torch.tensor(t / max(timesteps_list_test), device=cfg.device).view(1, 1).expand(X0.shape[0], 1)
            xt = interpolant(X0, X0, t_in, training=False).detach()

            ax[0, i].imshow(cfm_traj[cfm_t][0].reshape(img_dim, img_dim).cpu(), cmap='gray')
            ax[0, i].set_title(f"t={360 * (t / max(timesteps_list_test)):.0f}°")
            ax[0, i].axis('off')
            cfm_results[f"seed={seed}"].append(cfm_emd.item())

            for j in range(1, ax.shape[0]):
                ax[j, i].imshow(xt[j].reshape(img_dim, img_dim).cpu(), cmap='gray')
                ax[j, i].axis('off')

            wandb.log({
                f"{metric_prefix}_cfm/cfm_result_t={t}": cfm_emd.item()
            })
        plt.tight_layout()
        plt.show()


        # mean_emd = 0
        # for t in timesteps_list[1:]:
        #     t_prev = (t - 1) / (len(timesteps_list) - 1)
        #     t_curr = t / (len(timesteps_list) - 1)
        #     with torch.no_grad():
        #         ot_cfm_traj = node.trajectory(
        #             denormalize(data[t - 1], min_max).to(cfg.device),
        #             t_span=torch.linspace(t_prev, t_curr, num_int_steps + 1),
        #         )
        #         cfm_emd = compute_emd(
        #             denormalize(data[t], min_max).to(cfg.device),
        #             ot_cfm_traj[-1].float().to(cfg.device),
        #         )
        #         mean_emd += cfm_emd / (len(timesteps_list) - 1)
        # cfm_results[f"seed={seed}"].append(cfm_emd.item())
        #
        # # TODO: plot CFM trajectories and save on wandb
        #
        # wandb.log({
        #     f"{metric_prefix}_cfm/cfm_result": cfm_emd.item()
        # })



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