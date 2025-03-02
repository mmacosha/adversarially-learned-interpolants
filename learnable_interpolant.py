import torch
from torch import nn, Tensor
import numpy as np
from losses import sq_mahalanobis

import matplotlib.pyplot as plt
from tqdm import tqdm
from linear_interpolants import Flow


class Interpolnet(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            # nn.Linear(h, dim - 1), nn.Tanh())
            nn.Linear(h, dim - 1), )

    def forward(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
        net_output = self.net(torch.cat((t, x_0, x_1), -1))
        interpolant = t * x_1 + (1 - t) * x_0 + t * (1 - t) * net_output
        return interpolant

    def dt(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
        # x_0 and x_1 have shapes (batch size, d), while t has shape (batch size, 1)
        t_ = t.clone().requires_grad_(requires_grad=True)
        interpolant = self(t_, x_0, x_1)
        grad_t = torch.autograd.grad(interpolant, t_, grad_outputs=torch.ones_like(interpolant), create_graph=True)[0]
        return grad_t


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    flow = Flow(dim=1)
    interpolnet = Interpolnet()

    params = list(flow.parameters()) + list(interpolnet.parameters())
    optimizer = torch.optim.Adam(params, 1e-2)
    loss_fn = nn.MSELoss()
    pi = 0.8

    priors = {0.25: (torch.tensor(np.array(2.)), torch.tensor(np.array(1. / 1.))),
              0.75: (torch.tensor(np.array(-2.)), torch.tensor(np.array(1. / 2.)))}
    for _ in tqdm(range(1000)):
        n = 256
        x_1 = torch.randn((256, 1))  # Tensor(make_moons(256, noise=0.05)[0])
        x_0 = torch.randn_like(x_1)

        optimizer.zero_grad()
        # with prob pi, do MSELoss, else: do sqMD by uniformly picking prior distribution (i.e. t_j)
        unif = np.random.uniform(0, 1)
        if unif < pi:
            t = torch.rand(len(x_1), 1)
            x_t = interpolnet(t, x_0, x_1)
            loss_fn(flow(t=t, x_t=x_t), interpolnet.dt(t, x_0, x_1)).backward()
        else:
            t = np.random.choice([0.25, 0.75])
            prior = priors[t]
            t = np.tile(t, (256, 1))
            t = torch.tensor(t).type_as(x_0)
            x_t = interpolnet(t, x_0, x_1)
            sq_mahalanobis(x_t, prior[0], prior[1]).backward()

        optimizer.step()

    torch.manual_seed(1)
    np.random.seed(1)
    n = 300
    x_1 = torch.randn(n, 1)
    x_0 = torch.randn_like(x_1)
    t = torch.linspace(0., 1.0, 1000)
    # plot the interpolation paths for the different pairs

    x_list = np.zeros((n, 1000))
    for i, t_ in enumerate(t):
        t_ = torch.tile(t_, (n, 1))
        x = interpolnet(t_, x_0, x_1)
        x_list[:, i] = x.detach().numpy().squeeze()

    x_list[:, 0] = x_0.detach().numpy().squeeze()
    x_list[:, -1] = x_1.detach().numpy().squeeze()

    for x in x_list:
        plt.plot(t, x)
    plt.show()

    torch.manual_seed(1)
    np.random.seed(1)

    x = torch.randn(300, 1)
    n_steps = 100
    # fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1)
    x_interpolants = np.zeros((300, n_steps + 1))
    x_interpolants[:, 0] = x.detach().numpy().squeeze()

    for i in range(n_steps):
        x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
        x_interpolants[:, i + 1] = x.detach().numpy().squeeze()
    for i in range(300):
        plt.plot(time_steps, x_interpolants[i])

    plt.show()


