import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from time import time
from utils import sample_interm_4_slides, Plotter, load_data, pre_compute_OT_minibatches
from learnable_interpolants import CorrectionInterpolant
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

    # INIT PLOT OBJECT
    scale_factor = args['scale_factor']
    pl = Plotter("../data/ST_images/ref_U5_warped_images",
                 [0., 0.25, 0.75, 1.], coordinate_scaling=scale_factor)

    start_time = time()
    # PRE-PROCESS MINIBATCHES
    n_batches = 50000
    # coupled_indices = pre_compute_OT_minibatches(X0, X1, args['bs'], n_batches)
    otplan = OTPlanSampler('exact')
    pi = otplan.get_map(X0, X1)
    idx_x0, idx_x1 = otplan.sample_map(pi, n_batches, replace=True)
    N = min(X0.shape[0], Xt1.shape[0], Xt2.shape[0], X1.shape[0])
    eval_idx_x0, eval_idx_x1 = otplan.sample_map(pi, N, replace=True)


    # INITIALIZE MODELS
    g_hidden = args['g_hidden']
    interpolant = CorrectionInterpolant(2, g_hidden,
                                        'linear',
                                        correction_scale_factor=None,
                                        interpolnet_input='')
    interpolant = interpolant.to(device)

    discriminator = T.nn.Sequential(
        T.nn.Linear(3, g_hidden), T.nn.ELU(),
        T.nn.Linear(g_hidden, g_hidden), T.nn.ELU(),
        T.nn.Linear(g_hidden, 1), T.nn.Sigmoid()
    ).to(device)

    opt_interp = T.optim.Adam(interpolant.parameters(), lr=args['lr_generator'])
    opt_disc = T.optim.Adam(discriminator.parameters(), lr=args['lr_discriminator'])

    time_stamps = [observed]
    bs = args['bs']
    reg_weight = args['reg_weight']  # 0.01
    losses = []
    target_emds = {"t1": [], "t2": []}

    best_emd = 10000

    for it in range(args['n_iterations']):

        t = np.random.choice(time_stamps, size=bs)
        t = T.tensor(t, dtype=T.float32, device=device)

        #  sample a minibatch
        # idx = T.randint(0, n_batches, (1,))
        # coupling = coupled_indices[idx].squeeze()
        # idx_x0, idx_x1 = coupling[:, 0], coupling[:, 1]
        idx = T.randint(0, n_batches, (bs,))
        # idx_x0, idx_x1 = otplan.sample_map(pi, bs, replace=False)
        x0, x1 = X0[idx_x0[idx]], X1[idx_x1[idx]]

        t = t.unsqueeze(-1)

        opt_interp.zero_grad()
        xt_fake = T.cat([interpolant(x0, x1, t), t], 1)
        disc_score_fake = (1-discriminator(xt_fake)).log() #discriminator(xt_fake).log()  #
        loss_interp = disc_score_fake.mean()
        loss_reg = interpolant.regularizing_term(x0, x1, t, xt_fake)
        (loss_interp + reg_weight * loss_reg).backward()
        opt_interp.step()

        opt_disc.zero_grad()
        xhat_t = sample_interm_4_slides(bs, t.squeeze(-1), Xt1, Xt2)
        xt_real = T.cat([xhat_t, t], 1)
        disc_score_real = discriminator(xt_real).log()
        disc_score_fake = (1 - discriminator(xt_fake.detach())).log()
        loss_disc = - (disc_score_real.mean() + disc_score_fake.mean())
        loss_disc.backward()
        opt_disc.step()

        losses.append((loss_interp.item(), loss_reg.item(), loss_disc.item()))

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
                if t_i == unobserved:
                    W1 = wasserstein(validation_set * scale_factor, xt_fake * scale_factor, power=1)
                    if W1 < best_emd:
                        best_interpolant = copy.deepcopy(interpolant)
                        best_emd = W1
                        # pl.plot_interpolants(interpolant, [x0, Xt1, Xt2, x1], [1, 1, 0, 1])
                else:
                    W1 = wasserstein(target * scale_factor, xt_fake * scale_factor, power=1)
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
        test_idx_x0, test_idx_x1 = otplan.sample_map(pi, N, replace=False)
        x0, x1 = X0[test_idx_x0], X1[test_idx_x1]

        t = T.ones(N, dtype=T.float32, device=x0.device) * unobserved
        t = t.unsqueeze(-1)
        with T.no_grad():
            xt_fake = best_interpolant(x0, x1, t)
        W1 = wasserstein(unobserved_data * scale_factor, xt_fake * scale_factor, power=1)
        test_emds.append(W1)

        with T.no_grad():
            xt_fake = interpolant(x0, x1, t)
        W1 = wasserstein(unobserved_data * scale_factor, xt_fake * scale_factor, power=1)
        test_emds_final.append(W1)

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
        plt.plot(np.array(losses)[100:, 0], label="generator loss")
        plt.subplot(1, 2, 2)
        plt.plot(np.array(losses)[100:, 2], label="discriminator loss")
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
    scale_factor = 100
    X0, Xt1, Xt2, X1 = load_data(scale_factor, device)

    observed = 0.25
    unobserved = 1 - observed

    if unobserved == 0.75:
        unobserved_data = Xt2.clone()
    else:
        unobserved_data = Xt1.clone()

    # SET SEEDS
    T.manual_seed(0)
    np.random.seed(0)

    eval_idx_target = T.tensor(
        np.random.choice(np.arange(0, unobserved_data.shape[0]), unobserved_data.shape[0] // 5, replace=False),
    dtype=T.int, device=device)
    validation_set = unobserved_data[eval_idx_target]

    args = {}
    args['bs'] = 512
    args['validation_set'] = validation_set
    args['observed'] = observed
    args['unobserved'] = unobserved
    args['device'] = device
    args['X'] = (X0, Xt1, Xt2, X1)
    args['unobserved_data'] = unobserved_data
    args['lr_generator'] = 1e-3
    args['lr_discriminator'] = 1e-4
    args['reg_weight'] = 0.01
    args['n_iterations'] = 40000
    args['g_hidden'] = 512
    args['scale_factor'] = scale_factor

    scores = []
    for seed in range(5):
        args['seed'] = seed
        out = train_interpolants(args, verbose=True, plot=True)
        print(np.mean(out["test_emds"]))
        scores.append(np.mean(out["test_emds"]))
    print(np.mean(scores), np.std(scores))
