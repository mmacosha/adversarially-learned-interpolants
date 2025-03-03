
import torch
from torch import nn, Tensor
import numpy as np
from losses import sq_mahalanobis

import matplotlib.pyplot as plt
from tqdm import tqdm
from plots import density_and_trajectories_plot


class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

        return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2,
                                              x_t=x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)

    flow = Flow(dim=1)

    optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
    loss_fn = nn.MSELoss()
    pi = 1.

    priors = {0.25: (torch.tensor(np.array(2.)), torch.tensor(np.array(1. / 2.))),
              0.75: (torch.tensor(np.array(0.)), torch.tensor(np.array(1. / 2.)))}

    for _ in tqdm(range(1000)):
        x_1 = torch.randn((256, 1))  # Tensor(make_moons(256, noise=0.05)[0])
        x_0 = torch.randn_like(x_1)
        dx_t = x_1 - x_0


        optimizer.zero_grad()
        # with prob pi, do MSELoss, else: do sqMD by uniformly picking prior distribution (i.e. t_j)
        unif = np.random.uniform(0, 1)
        if unif < pi:
            t = torch.rand(len(x_1), 1)

            x_t = (1 - t) * x_0 + t * x_1

            loss_fn(flow(t=t, x_t=x_t), dx_t).backward()
        else:
            t = np.random.choice([0.25, 0.75])

            prior = priors[t]

            n_steps = 10
            time_steps = torch.linspace(0, t, n_steps + 1)
            x = x_0
            for i in range(n_steps):
                x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])

            # batch_mean = torch.mean(x, 0)

            sq_mahalanobis(x, prior[0], prior[1]).backward()

        optimizer.step()

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
    density_and_trajectories_plot(time_steps, x_interpolants, 0, 0, 1, 1)

    """
    for i in range(300):
        plt.plot(time_steps, x_interpolants[i])
    plt.show()
    """






