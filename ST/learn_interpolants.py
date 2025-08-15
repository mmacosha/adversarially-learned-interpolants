import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from time import time

from utils import sample_interm_4_slides, Plotter, load_data, Dataset
from learnable_interpolants import CorrectionInterpolant
from toy_experiment.plot_cubic_splines import couple_marginals
import copy


def train_interpolants(args, plot=False, verbose=True):
    T.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    X0, Xt1, Xt2, X1 = args['X']
    validation_set = args['validation_set']
    unobserved_data = args['unobserved_data']
    device = args['device']
    unobserved = args['unobserved']
    observed = args['observed']
    ds = args['ds']

    # INIT PLOT OBJECT
    scale_factor = args['scale_factor']
    pl = Plotter("../data/ST_images/ref_U5_warped_images",
                 [0., 0.25, 0.75, 1.], coordinate_scaling=scale_factor, ds=ds)

    start_time = time()

    otplan = OTPlanSampler('exact')
    pi1 = otplan.get_map(X0, validation_set)
    pi2 = otplan.get_map(validation_set, X1)
    pi = [pi1, pi2]
    # pi = otplan.get_map(X0, X1)
    N = 10000
    eval_idx_x0, eval_idx_x1 = otplan.sample_map(otplan.get_map(X0, X1), N, replace=True)


    # INITIALIZE MODELS
    g_hidden = args['g_hidden']
    interpolant = CorrectionInterpolant(2, g_hidden)
    interpolant = interpolant.to(device)

    d_hidden = g_hidden // 2
    discriminator = T.nn.Sequential(
        T.nn.Linear(3, d_hidden), T.nn.ELU(),
        T.nn.Linear(d_hidden, d_hidden), T.nn.ELU(),
        T.nn.Linear(d_hidden, 1)
    ).to(device)

    opt_interp = T.optim.Adam(interpolant.parameters(), lr=args['lr_generator'])
    opt_disc = T.optim.Adam(discriminator.parameters(), lr=args['lr_discriminator'])

    time_stamps = [observed]
    bs = args['bs']
    reg_weight = args['reg_weight']
    losses = []
    target_emds = {"t1": [], "t2": []}

    best_emd = 10000
    pretraining_steps = 1_001

    for it in range(args['n_iterations']):
        # i, j = otplan.sample_map(pi, bs, replace=True)
        # x0, x1 = X0[i], X1[j]
        aligned = couple_marginals([X0, validation_set, X1], bs, pi)
        x0, xhat_t, x1 = aligned[:, 0], aligned[:, 1], aligned[:, 2]

        opt_interp.zero_grad()
        if it < pretraining_steps:
            t = np.random.uniform(0, 1, size=bs)
            t = T.tensor(t, dtype=T.float32, device=device)
            t = t.unsqueeze(-1)
            xt_fake = T.cat([interpolant(x0, x1, t), t], 1)
            loss_reg = interpolant.regularizing_term(x0, x1, t, xt_fake)
            loss_reg.backward()
            opt_interp.step()
            loss_interp = 0
            loss_disc = 0
            r1 = 0
            r2 = 0
            losses.append((loss_interp, loss_reg.item(), loss_disc, r1, r2))
        else:
            # t_ = np.random.uniform(0, 1, size=bs)
            # t_ = T.tensor(t_, dtype=T.float32, device=device)
            # t_ = t_.unsqueeze(-1)
            # xt_extra = T.cat([interpolant(x0, x1, t_), t_], 1)
            # loss_reg = interpolant.regularizing_term(x0, x1, t_, xt_extra)

            t = np.random.choice(time_stamps, size=bs)
            t = T.tensor(t, dtype=T.float32, device=device)
            t = t.unsqueeze(-1)
            xt_fake = T.cat([interpolant(x0, x1, t), t], 1)
            # disc_score_fake = discriminator(xt_fake)  # (1-discriminator(xt_fake)).log()
            # loss_interp = T.nn.functional.softplus(disc_score_fake).mean()
            # loss_interp = -disc_score_fake.mean()
            loss_reg = interpolant.regularizing_term(x0, x1, t, xt_fake)
            # (loss_interp + reg_weight * loss_reg).backward()
            # opt_interp.step()



            # xhat_t = sample_interm_4_slides(bs, t.squeeze(-1), Xt1, Xt2)
            xt_real = T.cat([xhat_t, t], 1)
            # x_negative = sample_gmm_negatives(bs, device)
            # x_negative = T.cat([x_negative, t], 1)
            # disc_score_real = discriminator(xt_real)
            # disc_score_fake = discriminator(xt_fake)


            # WGAN
            # disc_score_fake = discriminator(xt_fake.detach())
            # # loss_disc_n = (T.nn.functional.softplus(- disc_score_real).mean() +
            # #                T.nn.functional.softplus(disc_score_fake)).mean()
            # loss_disc = - disc_score_real.mean() + disc_score_fake.mean()
            #
            # # gradient penalty
            # epsilon = T.rand(bs, 1, 1, 1).to(device)
            # x_hat = epsilon * xt_real + (1 - epsilon) * xt_fake.detach()
            # x_hat.requires_grad_(True)
            # d_hat = discriminator(x_hat)
            # grad = T.autograd.grad(outputs=d_hat, inputs=x_hat,
            #                        grad_outputs=T.ones_like(d_hat),
            #                        create_graph=True, retain_graph=True)[0]
            # L = 1  # Lipschitz constant
            # gp = ((grad.view(bs, -1).norm(2, dim=1) - L) ** 2).mean()
            # lambda_gp = 10 / (L ** 2)
            # loss_disc += lambda_gp * gp

            # RpGAN
            # disc_score_negative = discriminator(x_negative)
            disc_score_real = discriminator(xt_real)
            disc_score_fake = discriminator(xt_fake)
            loss_raw = T.nn.functional.softplus((disc_score_real - disc_score_fake)).mean()
            (loss_raw + reg_weight * loss_reg).backward()
            opt_interp.step()

            opt_disc.zero_grad()
            xt_fake += 0.01 * T.randn_like(xt_fake)
            xt_fake = xt_fake.detach().requires_grad_(True)
            disc_score_fake = discriminator(xt_fake)
            xt_real += 0.01 * T.randn_like(xt_real)
            xt_real = xt_real.detach().requires_grad_(True)
            disc_score_real = discriminator(xt_real)

            grad_D = T.autograd.grad(outputs=disc_score_real.sum(), inputs=xt_real, create_graph=True)[0]
            grad_G = T.autograd.grad(outputs=disc_score_fake.sum(), inputs=xt_fake, create_graph=True)[0]
            gamma = args['gamma']
            r1 = gamma / 2 * ((grad_D.view(bs, -1).norm(2, dim=1) - 0) ** 2)
            r2 = gamma / 2 * ((grad_G.view(bs, -1).norm(2, dim=1) - 0) ** 2)

            loss =  (T.nn.functional.softplus(-(disc_score_real - disc_score_fake)) + r1 + r2).mean()

            loss.backward()
            opt_disc.step()
            # losses.append((loss_interp.item(), loss_reg.item(), loss_disc.item()))
            losses.append((loss_raw.item(), loss_reg.item(), loss.item(), r1.mean().item(), r2.mean().item()))


        if it % 1000 == 0:
            print("iteration: ", it)
            if verbose:
                print(f"it={it}, mean losses (last 100): {np.array(losses)[-100:].mean(0)}")

            x0, x1 = X0[eval_idx_x0], X1[eval_idx_x1]

            for i, (t_i, target) in enumerate(zip([0.25, 0.75], [Xt1, Xt2])):
                t = T.ones(N, dtype=T.float32, device=x0.device) * t_i
                t = t.unsqueeze(-1)
                with T.no_grad():
                    xt_fake = interpolant(x0, x1, t)
                # if t_i == unobserved:
                if t_i == observed:
                    W1 = wasserstein(ds.denormalize(validation_set) * scale_factor, ds.denormalize(xt_fake) * scale_factor, power=1)
                    # xhat_t = sample_interm_4_slides(10000, t.squeeze(-1), Xt1, Xt2)
                    # xt_real = T.cat([xhat_t, t], 1)
                    # disc_score_real = discriminator(xt_real)
                    # plt.scatter(validation_set[:, 1].cpu(), validation_set[:, 0].cpu(), alpha=0.3)
                    # plt.scatter(xt_fake[:, 1].detach().cpu(), xt_fake[:, 0].detach().cpu(), alpha=0.3)
                    # with T.no_grad():
                    #     x = interpolant.linear_interpolant(x0, x1, t)
                    # plt.scatter(x[:, 1].detach().cpu(), x[:, 0].detach().cpu(), alpha=0.3)
                    # plt.axis('equal')
                    # plt.show()
                    # xs, ys = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
                    # xy = np.stack([xs.ravel(), ys.ravel()], axis=1)
                    # t_val = np.full((xy.shape[0], 1), fill_value=0.25)  # e.g., fixed t
                    # xt = T.tensor(np.concatenate([xy, t_val], axis=1), dtype=T.float32).to(device)
                    #
                    # with T.no_grad():
                    #     scores = discriminator(xt).cpu().numpy().reshape(100, 100)
                    #
                    # plt.figure()
                    # plt.imshow(scores, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
                    # plt.colorbar(label='Discriminator score')
                    # plt.title(f"Discriminator output over 2D plane {it}")
                    # plt.contour(xs, ys, scores, levels=20, colors='white', linewidths=0.5)
                    # plt.show()
                    #
                    # pl.plot_interpolants(interpolant, [x0, Xt1, Xt2, x1], [1, 1, 0, 1])
                    if (W1 < best_emd) and (it > pretraining_steps):
                        best_interpolant = copy.deepcopy(interpolant)
                        best_emd = W1
                        # pl.plot_interpolants(best_interpolant, [x0, Xt1, Xt2, x1], [1, 1, 0, 1])

                else:
                    W1 = wasserstein(ds.denormalize(target) * scale_factor, ds.denormalize(xt_fake) * scale_factor, power=1)
                if verbose:
                    print(f"t = {t_i}, EMD = {W1:.4f}")
                target_emds[f"t{i+1}"].append(W1)
            if verbose:
                print(f"Best EMD: {best_emd:.4f}")
                print("")

    if verbose:
        print(f"Total training time: {time() - start_time:.2f} seconds")
        # time reported in the article was computed with plotting disabled

        print("")
        print("Averaging EMDs using 'best' and final interpolant...")
    n_test_runs = 20
    test_emds = []
    test_emds_final = []
    T.manual_seed(0)  # reset manual seed to compare with other methods
    for run in range(n_test_runs):
        N = min(X0.shape[0], Xt1.shape[0], Xt2.shape[0], X1.shape[0])
        test_idx_x0, test_idx_x1 = otplan.sample_map(otplan.get_map(X0, X1), N, replace=False)
        x0, x1 = X0[test_idx_x0], X1[test_idx_x1]

        t = T.ones(N, dtype=T.float32, device=x0.device) * unobserved
        t = t.unsqueeze(-1)
        with T.no_grad():
            xt_fake = best_interpolant(x0, x1, t)
        W1 = wasserstein(ds.denormalize(unobserved_data) * scale_factor, ds.denormalize(xt_fake) * scale_factor, power=1)
        test_emds.append(W1)

        with T.no_grad():
            xt_fake = interpolant(x0, x1, t)
        W1 = wasserstein(ds.denormalize(unobserved_data) * scale_factor, ds.denormalize(xt_fake) * scale_factor, power=1)
        test_emds_final.append(W1)

    pl.plot_interpolants(best_interpolant, [x0, Xt1, Xt2, x1], [1, 1, 0, 1])

    if verbose:
        print(f"EMD using 'best' model, at t = {unobserved}")
        print(f"  mean = {np.mean(test_emds):.4f}, std = {np.std(test_emds):.4f}")

        print(f"EMD using final model, at t = {unobserved}")
        print(np.mean(test_emds_final), np.std(test_emds_final))
        print("")
        print("Time including evaluation:", time() - start_time)


    if plot:
        plt.subplot(1, 3, 1)
        plt.plot(target_emds["t1"], label="EMD t=0.25")
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(target_emds["t2"], label="EMD t=0.75")
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(np.array(losses)[:, 1], label="reg loss")
        plt.legend()
        plt.show()
        plt.subplot(1, 2, 1)
        plt.plot(np.array(losses)[2000:, 0], label="generator loss")
        plt.subplot(1, 2, 2)
        plt.plot(np.array(losses)[2000:, 2], label="discriminator loss")
        plt.legend()
        plt.show()

    output = {"best_interpolant": best_interpolant,
              "interpolant": interpolant,
              "losses": losses,
              "target_emds": target_emds,
              "test_emds": test_emds,
              "test_emds_final": test_emds_final}

    return output


if __name__ == '__main__':
    device = 'cuda' if T.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # LOAD DATA
    scale_factor = 1
    X0_, Xt1_, Xt2_, X1_ = load_data(scale_factor, device)
    ds = Dataset([x.cpu().numpy() for x in [X0_, Xt1_, Xt2_, X1_]], [0, 0.25, 0.75, 1], normalize=True)
    X0, Xt1, Xt2, X1 = (T.tensor(x, device=device, dtype=T.float32) for x in ds.data)


    observed = 0.75
    unobserved = 1 - observed

    if unobserved == 0.75:
        unobserved_data = Xt2.clone()
        observed_intermediate_data = Xt1.clone()
    else:
        unobserved_data = Xt1.clone()
        observed_intermediate_data = Xt2.clone()

    args = {}
    args['bs'] = 512
    args['validation_set'] = observed_intermediate_data
    args['observed'] = observed
    args['unobserved'] = unobserved
    args['device'] = device
    args['X'] = (X0, Xt1, Xt2, X1)
    args['unobserved_data'] = unobserved_data
    args['lr_generator'] = 1e-4
    args['lr_discriminator'] = 1e-4
    args['reg_weight'] = .0001
    args['n_iterations'] = 50_001
    args['g_hidden'] = 128
    args['scale_factor'] = scale_factor
    args['ds'] = ds
    args['gamma'] = 20

    scores = []
    for seed in range(5):
        args['seed'] = seed
        out = train_interpolants(args, verbose=True, plot=True)
        print(np.mean(out["test_emds"]))
        scores.append(np.mean(out["test_emds"]))
    print(np.mean(scores), np.std(scores))
