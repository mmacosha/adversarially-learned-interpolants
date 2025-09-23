import random
import wandb
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from ali_cfm.loggin_and_metrics import compute_emd
from ali_cfm.data_utils import denormalize

from . import training_utils as utils


def pretain_interpolant(
    interpolant, pretrain_optimizer_G, ot_sampler, 
    train_data, train_timesteps, metric_prefix, cfg,
):
    if cfg.num_ali_pretrain_steps == 0:
        return
    
    for epoch in trange(cfg.num_ali_pretrain_steps, 
                        desc="Pretraining Interpolant", leave=False):
        pretrain_optimizer_G.zero_grad()
        batch = utils.sample_gan_batch(
            train_data, cfg.batch_size, 
            divisor=max(train_timesteps), 
            ot_sampler=ot_sampler, 
            ot=cfg.pretain_ot, 
            times=train_timesteps
        )
        x0, x1, xt, t = (x.to(cfg.device) for x in batch)
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
    gan_optimizer_G, gan_optimizer_D, metric_prefix,
    train_timesteps, seed, min_max, cfg,
    plot_freaquency=5000,
):
    t_max = max(train_timesteps)
    curr_epoch = 0
    for epoch in trange(curr_epoch, cfg.num_ali_train_steps, 
                        desc="Training GAN Interpolant", leave=False):
        # if epoch > 40_000:
        #     gan_optimizer_G.param_groups[0]['lr'] = 1e-5
        #     gan_optimizer_D.param_groups[0]['lr'] = 5e-5
        # elif epoch > 100_000:
        #     gan_optimizer_G.param_groups[0]['lr'] = 1e-6
        #     gan_optimizer_D.param_groups[0]['lr'] = 5e-6
        
        curr_epoch += 1
        if epoch % 5_000 == 0:
            with torch.no_grad():
                for time in range(1, train_timesteps[-1]):
                    test_batch = utils.sample_gan_batch(
                        data, 2300, 
                        divisor=t_max,
                        ot_sampler=ot_sampler, 
                        time=time, 
                        ot='border'
                    )
                    x0_test, x1_test, xt_test, t_test = (
                        x.to(cfg.device) for x in test_batch
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
                    
        # batch = utils.sample_gan_batch(
        #     data, gan_batch_size, 
        #     divisor=t_max, 
        #     ot_sampler=ot_sampler, 
        #     ot=ot, 
        #     times=train_timesteps
        # )
        # x0, x1, xt, t = (x.to(device) for x in batch)
        
        xts, t = utils.sample_full_batch(
            data, cfg.batch_size,
            ot_sampler=ot_sampler,
            divisor=t_max,
            ot=cfg.interpolant_ot,
            times=train_timesteps,
        )
        xts = {t: x.to(cfg.device) for t, x in xts.items()}
        x0, x1, xt = xts[0], xts[1], xts[t]
        t = torch.ones(cfg.batch_size, 1, device=cfg.device) * t

        xt_fake = interpolant(x0, x1, t)

        real_inputs = torch.cat([xt, t], dim=-1)
        fake_inputs = torch.cat([xt_fake.detach(), t], dim=-1)
        
        real_proba = discriminator(real_inputs)
        fake_proba = discriminator(fake_inputs)

        # Train discriminator
        gan_optimizer_D.zero_grad()
        if (cfg.gan_loss == "RpGAN") or (cfg.gan_loss == "R3GAN"):
            # Using the relativistic pairing GAN loss
            r1 = 0
            r2 = 0
            if cfg.gan_loss == "R3GAN":
                xt_fake_ = torch.cat([xt_fake, t], dim=-1).detach().requires_grad_(True)
                xt_ = torch.cat([xt, t], dim=-1).detach().requires_grad_(True)
                disc_score_fake = discriminator(xt_fake_)
                disc_score_real = discriminator(xt_)
                grad_D = torch.autograd.grad(
                    outputs=disc_score_real.sum(), inputs=xt_, create_graph=True
                )[0]
                grad_G = torch.autograd.grad(
                    outputs=disc_score_fake.sum(), inputs=xt_fake_, create_graph=True
                )[0]
                r1 = 0.5 * ((grad_D.view(cfg.batch_size, -1).norm(2, dim=1) - 0) ** 2)
                r2 = 0.5 * ((grad_G.view(cfg.batch_size, -1).norm(2, dim=1) - 0) ** 2)
            d_loss = F.softplus((fake_proba - real_proba) + r1 + r2).mean()

            d_loss.backward()
            gan_optimizer_D.step()

            # Train generator
            gan_optimizer_G.zero_grad()
            real_inputs = torch.cat([xt, t], dim=-1)
            fake_inputs = torch.cat([xt_fake, t], dim=-1)

            real_proba = discriminator(real_inputs)
            fake_proba = discriminator(fake_inputs)
            g_loss_ = F.softplus((real_proba - fake_proba)).mean()
        else:
            d_real_loss = F.softplus(-real_proba).mean()
            d_fake_loss = F.softplus(fake_proba).mean()
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            gan_optimizer_D.step()

            # Train generator
            gan_optimizer_G.zero_grad()
            fake_inputs = torch.cat([xt_fake, t], dim=-1)
            fake_proba = discriminator(fake_inputs)

            g_loss_ = F.softplus(-fake_proba).mean()
        
        match cfg.reg_term_type:
            case "picewise_oskar_version":
                t = torch.rand(x0.shape[0], 1, device=cfg.device)
                xt_fake = interpolant(x0, x1, t, training=False)

                coupled_x = torch.stack([x0, xt, x1], dim=1) # (B, 3, D)
                bs, K, d = coupled_x.shape
                times = torch.tensor(train_timesteps, device=cfg.device) / t_max

                idx = torch.bucketize(t, times) - 1
                idx = idx.clamp(0, K - 2)  # keep in valid range

                t0 = times[idx]  # (bs,)
                t1 = times[idx + 1]  # (bs,)
                denom = (t1 - t0)  # (bs,)

                a = (t1 - t) / denom
                b = (t - t0) / denom

                # Select endpoints for each batch element
                x0 = coupled_x[torch.arange(bs), idx]  # (bs, d)
                x1 = coupled_x[torch.arange(bs), idx + 1]  # (bs, d)

                xhat_t = a.unsqueeze(-1) * x0 + b.unsqueeze(-1) * x1
                diff = xt_fake - xhat_t

                if cfg.reg_metric == 'land_norm':
                    reg_weight_loss = (diff ** 2).mean()
                elif cfg.reg_metric == 'l1':
                    xs, ts = get_marginals(xts)
                    G = compute_time_dependent_metric(
                        xhat_t, t, xs, ts, 
                        gamma=land_gamma, 
                        t_gamma=land_t_gamma,
                        normalize_t=True,
                    )
                    reg_weight_loss = (diff**2 * G).mean()
                else:
                    raise ValueError(f"Unknown metric {cfg.reg_metric}")

            case 'linear':
                reg_weight_loss = \
                    interpolant.compute_linear_reg_term(
                        xt_fake, xts, t, 
                        cfg.reg_metric, cfg.land_gamma, cfg.land_t_gamma
                    )

            case 'piecewise':
                t1_idx = random.randint(0, len(train_timesteps) - 2)
                t1, t2 = train_timesteps[t1_idx], train_timesteps[t1_idx + 1]
                t1, t2 = t1 / t_max, t2 / t_max

                reg_weight_loss = \
                    interpolant.compute_piecewise_reg_term(
                        xts, t1, t2, 
                        cfg.reg_metric, cfg.land_gamma, cfg.land_t_gamma
                    )

            case '2nd_derivative':
                reg_weight_loss = \
                    interpolant.compute_length_reg_term(
                        xts, 
                        cfg.reg_metric, cfg.land_gamma, cfg.land_t_gamma, 
                        cfg.num_t_steps
                    )

            case _:
                raise ValueError(f"Unknown regularization term {cfg.reg_term_type}")
        
        g_loss = g_loss_ + cfg.correct_coeff * reg_weight_loss
        
        g_loss.backward()
        gan_optimizer_G.step()

        wandb.log({
            f"{metric_prefix}/d_loss": d_loss.item(),
            f"{metric_prefix}/g_loss": g_loss.item(),
            f"{metric_prefix}/g_loss_": g_loss_.item(),
            f"{metric_prefix}/reg_weight_loss": reg_weight_loss.item(),
            f"{metric_prefix}/fake_proba": F.sigmoid(fake_proba).mean().item(),
            f"{metric_prefix}/real_proba": F.sigmoid(real_proba).mean().item(),
            f"{metric_prefix}_step": epoch
        })

        if epoch % plot_freaquency == 0:
            with torch.no_grad():
                batch = utils.sample_gan_batch(
                    data, 256, 
                    divisor=t_max, 
                    ot_sampler=ot_sampler, 
                    ot='full', 
                    times=train_timesteps
                )
                x0, x1, xt, t = (x.to(cfg.device) for x in batch)
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


def train_ot_cfm(
    ot_cfm_model, ot_cfm_optimizer, interpolant, ot_sampler, 
    train_data, metric_prefix, cfg, train_timesteps, min_max,
):
    for step in trange(cfg.num_cfm_train_steps, 
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        if cfg.cfm_ot == 'mmot':
            t_max = max(train_timesteps)
            batch = utils.sample_gan_batch(
                train_data, cfg.batch_size, 
                divisor=t_max, 
                ot_sampler=ot_sampler, 
                ot=cfg.cfm_ot, 
                times=train_timesteps
            )
            x0, x1, xt, _ = (x.to(cfg.device) for x in batch)
        else:
            x0, x1 = utils.sample_x0_x1(
                train_data, cfg.batch_size, device=cfg.device
            )
            if cfg.cfm_ot in {'border', 'full'}:
                x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0], 1, device=cfg.device)
        
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
