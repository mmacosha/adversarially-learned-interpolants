import torch
from torch import nn
import numpy as np


def sq_mahalanobis(batch_mean, mu, inv_cov):
    # batch_mean.shape = d
    diff = (batch_mean - mu)

    if batch_mean.size()[0] == 1:
        return diff ** 2 * inv_cov
    else:
        return diff.T @ diff * inv_cov
    # return diff.T @ inv_cov @ diff


def l2(u_t, dx_t):
    return nn.MSELoss(u_t, dx_t)


if __name__ == '__main__':
    d = 2
    batch_mean = torch.tensor(np.random.normal(0, 1, d))
    mu = torch.zeros(d)
    inv_cov = torch.tensor(np.array([[1/2, 0],[0, 1/3]]))
    print(sq_mahalanobis(batch_mean, mu, inv_cov))
