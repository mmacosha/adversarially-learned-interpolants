import wandb
import torch
from tqdm.auto import trange

from ali_cfm.loggin_and_metrics import compute_emd
from ali_cfm.data_utils import denormalize

from . import training_utils as utils


def pretain_interpolant(
        interpolant, pretrain_optimizer_G, ot_sampler, 
        train_data, n_pretrain_epochs, gan_batch_size, 
        train_timesteps, 
        ot='none', metric_prefix="", device='cpu'
    ):
    if n_pretrain_epochs == 0:
        return
    
    for epoch in trange(n_pretrain_epochs, 
                        desc="Pretraining Interpolant", leave=False):
        pretrain_optimizer_G.zero_grad()
        batch = utils.sample_gan_batch(
            train_data, gan_batch_size, 
            divisor=max(train_timesteps), 
            ot_sampler=ot_sampler, 
            ot=ot, 
            times=train_timesteps
        )
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
    ot='none', plot_freaquency=5000, metric_prefix="", compute_emd_flag=True,
    device='cpu', gan_loss='vanilla', plot_fn=utils.sc_plot_fn,
):
    t_max = max(train_timesteps)
    curr_epoch = 0
    for epoch in trange(curr_epoch, n_epochs, 
                        desc="Training GAN Interpolant", leave=False):
        # if epoch > 40_000:
        #     gan_optimizer_G.param_groups[0]['lr'] = 1e-5
        #     gan_optimizer_D.param_groups[0]['lr'] = 5e-5
        # elif epoch > 100_000:
        #     gan_optimizer_G.param_groups[0]['lr'] = 1e-6
        #     gan_optimizer_D.param_groups[0]['lr'] = 5e-6
        
        curr_epoch += 1
        if (epoch % 5_000 == 0) and compute_emd_flag:
            with torch.no_grad():
                for time in range(1, 4):
                    test_batch = utils.sample_gan_batch(
                        data, 2300, 
                        divisor=t_max,
                        ot_sampler=ot_sampler, 
                        time=time, 
                        ot='border'
                    )
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
                    
        batch = utils.sample_gan_batch(
            data, gan_batch_size, 
            divisor=t_max, 
            ot_sampler=ot_sampler, 
            ot=ot, 
            times=train_timesteps
        )

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
                grad_D = torch.autograd.grad(
                    outputs=disc_score_real.sum(), inputs=xt_, create_graph=True
                )[0]
                grad_G = torch.autograd.grad(
                    outputs=disc_score_fake.sum(), inputs=xt_fake_, create_graph=True
                )[0]
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
            d_real_loss = torch.nn.functional.softplus(-real_proba).mean()
            d_fake_loss = torch.nn.functional.softplus(fake_proba).mean()
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            gan_optimizer_D.step()

            # Train generator
            gan_optimizer_G.zero_grad()
            fake_inputs = torch.cat([xt_fake, t], dim=-1)
            fake_proba = discriminator(fake_inputs)

            g_loss_ = torch.nn.functional.softplus(-fake_proba).mean()
        
        reg_weight_loss = interpolant.get_reg_term(x0, x1, t, xt_fake, xt)
        g_loss = g_loss_ + correct_coeff * reg_weight_loss
        
        g_loss.backward()
        gan_optimizer_G.step()

        wandb.log({
            f"{metric_prefix}/d_loss": d_loss.item(),
            f"{metric_prefix}/g_loss": g_loss_      .item(),
            f"{metric_prefix}/reg_weight_loss": reg_weight_loss.item(),
            f"{metric_prefix}/fake_proba": fake_proba.mean().item(),
            f"{metric_prefix}/real_proba": real_proba.mean().item(),
            f"{metric_prefix}_step": epoch
        })

        if epoch % plot_freaquency == 0:
            plot_fn(interpolant, epoch, seed, t_max, data, ot_sampler, device, metric_prefix, train_timesteps, wandb, min_max)


def train_ot_cfm(
        ot_cfm_model, ot_cfm_optimizer, interpolant, 
        ot_sampler, train_data, batch_size, min_max,
        n_ot_cfm_epochs, 
        metric_prefix="", ot='border', device='cpu', times=(0, -1)
    ):
    for step in trange(n_ot_cfm_epochs, 
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        if ot == 'mmot':
            t_max = max(times)
            batch = utils.sample_gan_batch(
                train_data, batch_size, 
                divisor=t_max, 
                ot_sampler=ot_sampler, 
                ot=ot, 
                times=times
            )
            x0, x1, xt, _ = (x.to(device) for x in batch)
        else:
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
