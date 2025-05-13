from torch import nn, Tensor
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from time import time
from learnable_interpolants import CorrectionInterpolant
from utils import Plotter, load_data, pad_a_like_b
from learn_interpolants import train_interpolants
from infer_mmot import get_ot_interpolant, get_ot_interpolant_given_coupling, couple_marginals
from toy_experiment.plot_cubic_splines import get_cubic_spline_interpolation


class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(T.cat((t, x_t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

        return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2,
                                              x_t=x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


if __name__ == '__main__':
    device = 'cuda' if T.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # LOAD DATA
    scale_factor = 100
    X0, Xt1, Xt2, X1 = load_data(scale_factor, device)

    # INIT PLOT OBJECT
    pl = Plotter("../data/ST_images/ref_U5_warped_images",
                 [0., 0.25, 0.75, 1.], coordinate_scaling=scale_factor)

    # SET SEEDS
    T.manual_seed(0)
    np.random.seed(0)

    observed = 0.25
    unobserved = 1 - observed
    if unobserved == 0.75:
        unobserved_data = Xt2.clone()
    else:
        unobserved_data = Xt1.clone()

    eval_idx_target = T.tensor(
        np.random.choice(np.arange(0, unobserved_data.shape[0]), unobserved_data.shape[0] // 5, replace=False),
        dtype=T.int, device=device)
    validation_set = unobserved_data[eval_idx_target]

    # INPUTS TO INTERPOLANT TRAINING/LOADING
    interpolant_class = 'linear' # cs, linear, geodesic
    load_interpolant = True
    bs = 512

    args = {}
    args['seed'] = 0
    if interpolant_class == 'ali':
        args['bs'] = bs
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
        if load_interpolant:
            print("Loading interpolant...")
            interpolant = CorrectionInterpolant(2, args["g_hidden"],
                                                'linear',
                                                correction_scale_factor=None,
                                                interpolnet_input='')
            interpolant = interpolant.to(device)
            interpolant.load_state_dict(T.load("interpolant_models/" + interpolant_class + str(args["seed"]) + '.pth'))
        else:
            print("Training interpolant...")
            out = train_interpolants(args, verbose=False, plot=True)
            interpolant = out['best_interpolant']
            print(np.mean(out["test_emds"]))
            T.save(interpolant.state_dict(), "interpolant_models/" + interpolant_class + str(args["seed"]) + '.pth')
    elif (interpolant_class == 'linear') or (interpolant_class == 'cs'):
        print("Precomputing couplings...")
        start_time = time()
        otplan = OTPlanSampler('exact')
        if observed == 0.25:
            pi1 = otplan.get_map(X0, Xt1)
            pi2 = otplan.get_map(Xt1, X1)
            observed_data = Xt1.clone()
        else:
            pi1 = otplan.get_map(X0, Xt2)
            pi2 = otplan.get_map(Xt2, X1)
            observed_data = Xt2.clone()
        pi = [pi1, pi2]
        # pre-compute splines (not the correct way, according to authors code)
        # if interpolant_class == 'cs':
        #     observed_t = T.zeros((bs, 3))
        #     observed_t[:, 0] = 0.
        #     observed_t[:, 1] = observed
        #     observed_t[:, 2] = 1.
        #     splines = get_cubic_spline_interpolation([X0, observed_data, X1], observed_t, pi)
        print("Finished coupling after: ", time() - start_time, " sec")

    # TRAIN VECTOR FIELD
    seed = args['seed']
    T.manual_seed(seed)
    np.random.seed(seed)

    flow = Flow(dim=2, h=512).to(device)
    optimizer = T.optim.Adam(flow.parameters(), 1e-4)

    if interpolant_class == 'ali':
        start_time = time()
        # PRE-PROCESS MINIBATCHES
        otplan = OTPlanSampler('exact')
        pi = otplan.get_map(X0, X1)
        idx_x0, idx_x1 = otplan.sample_map(pi, 50000, replace=True)

    loss_fn = nn.MSELoss()

    losses = []
    emds = []
    best_mse_loss = 10000
    for it in range(40000):

        t = T.rand(bs, 1, device=device)

        if interpolant_class == 'ali':
            idx = T.randint(0, idx_x0.size, (bs,))
            x0, x1 = X0[idx_x0[idx]], X1[idx_x1[idx]]
            with T.no_grad():
                xt = interpolant(x0, x1, t)
                dxt = interpolant.dI_dt(x0, x1, t)
        elif interpolant_class == 'linear':
            # sample an OT minibatch
            coupling = couple_marginals([X0, observed_data, X1], bs, pi)
            xt, dxt = get_ot_interpolant_given_coupling(coupling, unobserved, t, bs)
        elif interpolant_class == 'cs':
            observed_t = T.zeros((bs, 3))
            observed_t[:, 0] = 0.
            observed_t[:, 1] = observed
            observed_t[:, 2] = 1.
            splines = get_cubic_spline_interpolation([X0, observed_data, X1], observed_t, pi)

            xt_list, dxt_list = [], []
            for k, spline in enumerate(splines):
                t_k = t[k].cpu().numpy()
                xt_list.append(spline(t_k))
                dxt_list.append(spline(t_k, nu=1))

            xt_np = np.concatenate(xt_list, axis=0)
            dxt_np = np.concatenate(dxt_list, axis=0)

            xt = T.tensor(xt_np, device=device, dtype=T.float32)
            dxt = T.tensor(dxt_np, device=device, dtype=T.float32)

        optimizer.zero_grad()
        loss = loss_fn(flow(t, xt), dxt)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(it)
            print(np.array(losses)[-100:].mean(0))


            n_steps = 1000
            time_steps = T.linspace(0, unobserved, n_steps + 1, device=device)

            x = X0.clone()
            with T.no_grad():
                for i in range(n_steps):
                    x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
            W1 = wasserstein(unobserved_data * scale_factor, x * scale_factor, power=1)
            print("EMD: ", W1)
            print()
            emds.append(W1)
            if loss.item() < best_mse_loss:
                reported_W1 = W1
                best_mse_loss = loss.item()

    print("Total runtime: ", time() - start_time)
    print("Reported EMD: ", W1)
    plt.plot(emds)
    plt.title('EMD Scores')
    plt.show()




