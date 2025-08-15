from torch import nn, Tensor
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from time import time
from learnable_interpolants import CorrectionInterpolant
from utils import Plotter, load_data, Dataset
from learn_interpolants import train_interpolants
from infer_mmot import get_ot_interpolant_given_coupling
from toy_experiment.plot_cubic_splines import get_cubic_spline_interpolation, couple_marginals, couple_marginals_markov


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
    scale_factor = 1
    X0, Xt1, Xt2, X1 = load_data(scale_factor, device)

    # INIT PLOT OBJECT
    pl = Plotter("../data/ST_images/ref_U5_warped_images",
                 [0., 0.25, 0.75, 1.], coordinate_scaling=scale_factor)
    ds = Dataset([x.cpu().numpy() for x in [X0, Xt1, Xt2, X1]], [0, 0.25, 0.75, 1], normalize=True)
    X0, Xt1, Xt2, X1 = (T.tensor(x, device=device, dtype=T.float32) for x in ds.data)

    observed = 0.25
    unobserved = 1 - observed
    if unobserved == 0.75:
        unobserved_data = Xt2.clone()
        observed_data = Xt1.clone()
        unobserved_label = "U4"
    else:
        unobserved_data = Xt1.clone()
        observed_data = Xt2.clone()
        unobserved_label = "U3"

    # INPUTS TO INTERPOLANT TRAINING/LOADING
    for interpolant_class in ['ali']:
        # interpolant_class = 'ali' # ali, cs, linear, geodesic
        load_interpolant = False
        bs = 512

        for seed in [0, 1, 2, 3, 4]:
            args = {}
            args['seed'] = seed
            if interpolant_class == 'ali':
                args['bs'] = bs
                args['ds'] = ds
                args['validation_set'] = observed_data
                args['unobserved_data'] = unobserved_data
                args['observed'] = observed
                args['unobserved'] = unobserved
                args['device'] = device
                args['X'] = (X0, Xt1, Xt2, X1)
                args['lr_generator'] = 1e-4
                args['lr_discriminator'] = 1e-4
                args['reg_weight'] = 0.01
                args['gamma'] = 20
                args['n_iterations'] = 30_001
                args['g_hidden'] = 128
                args['scale_factor'] = scale_factor
                if load_interpolant:
                    print("Loading interpolant...")
                    interpolant = CorrectionInterpolant(2, args["g_hidden"])
                    interpolant = interpolant.to(device)
                    interpolant.load_state_dict(T.load("interpolant_models/" + interpolant_class + str(args["seed"]) + unobserved_label + '.pth'))
                else:
                    print("Training interpolant...")
                    out = train_interpolants(args, verbose=False, plot=True)
                    interpolant = out['best_interpolant']
                    print(np.mean(out["test_emds"]))
                    T.save(interpolant.state_dict(), "interpolant_models/" +
                           interpolant_class + str(args["seed"]) + unobserved_label + '.pth')
            elif (interpolant_class == 'linear') or (interpolant_class == 'cs'):
                print("Precomputing couplings...")
                start_time = time()
                otplan = OTPlanSampler('exact')
                if observed == 0.25:
                    pi1 = otplan.get_map(X0, Xt1)
                    pi2 = otplan.get_map(Xt1, X1)
                else:
                    pi1 = otplan.get_map(X0, Xt2)
                    pi2 = otplan.get_map(Xt2, X1)
                    observed_data = Xt2.clone()
                pi = [pi1, pi2]
                print("Finished coupling after: ", time() - start_time, " sec")

            # TRAIN VECTOR FIELD
            seed = args['seed']
            T.manual_seed(seed)
            np.random.seed(seed)

            flow = Flow(dim=2, h=512).to(device)
            optimizer = T.optim.Adam(flow.parameters(), 1e-4)

            if interpolant_class == 'ali':
                start_time = time()
                otplan = OTPlanSampler('exact')
                pi = otplan.get_map(X0, X1)

            loss_fn = nn.MSELoss()

            losses = []
            emds = []
            best_mse_loss = 10000
            for it in range(40000):

                t = T.rand(bs, 1, device=device)

                if interpolant_class == 'ali':
                    i, j = otplan.sample_map(pi, bs, replace=True)
                    x0, x1 = X0[i], X1[j]
                    with T.no_grad():
                        xt = interpolant(x0, x1, t)
                        dxt = interpolant.dI_dt(x0, x1, t)
                elif interpolant_class == 'linear':
                    coupling = couple_marginals_markov([X0, observed_data, X1], bs, pi)
                    xt, dxt = get_ot_interpolant_given_coupling(coupling, unobserved, t, bs)
                elif interpolant_class == 'cs':
                    observed_t = T.zeros((bs, 3))
                    observed_t[:, 0] = 0.
                    observed_t[:, 1] = observed
                    observed_t[:, 2] = 1.
                    splines = get_cubic_spline_interpolation([X0, observed_data, X1], observed_t, pi)

                    t_np = t.squeeze().cpu().numpy()
                    xt_np = np.einsum("nni->ni", splines(t_np))
                    dxt_np = np.einsum("nni->ni", splines(t_np, 1))

                    xt = T.tensor(xt_np, device=device, dtype=T.float32)
                    dxt = T.tensor(dxt_np, device=device, dtype=T.float32)

                optimizer.zero_grad()
                loss = loss_fn(flow(t, xt), dxt)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                if it % 1000 == 0:
                    n_steps = 1000
                    time_steps = T.linspace(0, unobserved, n_steps + 1, device=device)

                    x = X0.clone()
                    with T.no_grad():
                        for i in range(n_steps):
                            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
                    W1 = wasserstein(ds.denormalize(unobserved_data), ds.denormalize(x), power=1)
                    # print("EMD: ", W1)
                    # print()
                    emds.append(W1)
                    if loss.item() < best_mse_loss:
                        reported_W1 = W1
                        best_mse_loss = loss.item()

            print("Total runtime: ", time() - start_time)
            print("Reported EMD: ", W1)
            plt.plot(emds)
            plt.title('EMD Scores')
            plt.show()
            plt.plot(losses, label='$L_2$ loss')
            plt.legend()
            plt.show()
            T.save(flow.state_dict(), "CFM_models/" +
                   interpolant_class +"-CFM_" + str(args["seed"]) + unobserved_label + '.pth')




