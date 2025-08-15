import torch as T
from torch import nn, Tensor
from torch.func import jacrev

import torch.nn as nn
import math



class CorrectionInterpolant(nn.Module):
    # learns the correction from the reference trajectory
    # interpolant = reference_trajectory(x0, x1, t) + correction_scale_factor_t * NN(x0, x1, t)
    # regularizing term = || reference_trajectory(x0, x1, t) - interpolant || ** 2
    #                   = correction_scale_factor_t ** 2 || interpolant || ** 2

    def __init__(self, dim, h_dim, t_smooth=0.01, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.t_smooth = t_smooth
        self.interpolant_net = nn.Sequential(
            nn.Linear(2 * dim + 1, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, dim)
        )

    @staticmethod
    def linear_interpolant(x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t
        return t * x1 + (1 - t) * x0

    def forward(self, x0, x1, t, training=True):
        t = t[..., None] if t.ndim == 1 else t

        xt = self.linear_interpolant(x0, x1, t)

        if training:
            t_input = t + T.randn_like(t) * self.t_smooth if self.t_smooth > 0 else t
        else:
            t_input = t
        input_ = T.cat([x0, x1, t_input], dim=-1)

        correction = self.interpolant_net(input_)
        return xt + t * (1 - t) * correction

    def regularizing_term(self, x0, x1, t, xt_fake):
        loss_reg = xt_fake[:, :-1] - self.linear_interpolant(x0, x1, t)
        if len(loss_reg.shape) > 1:
            loss_reg = (loss_reg ** 2).sum(1).mean()
        else:
            loss_reg = (loss_reg ** 2).mean()
        return loss_reg

    def dI_dt(self, x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t

        def _interpolnet(x0i, x1i, ti):
            input_ = T.cat([x0i, x1i, ti], dim=0).unsqueeze(0)
            out = self.interpolant_net(input_)
            return out, out

        (corr_jac, corr_output) = T.vmap(
            jacrev(_interpolnet, argnums=2, has_aux=True))(x0, x1, t)
        if x0.shape[-1] == 1:
            return (
            (x1 - x0) + (1 - 2 * t) * corr_output.squeeze().unsqueeze(-1) +
            t * (1 - t) * corr_jac.squeeze().unsqueeze(-1)
        )
        return (
            (x1 - x0) + (1 - 2 * t) * corr_output.squeeze() +
            t * (1 - t) * corr_jac.squeeze()
        )


class AffineTransformInterpolant(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64, reference_trajectory='linear',
                 correction_scale_factor=None, xt_input=0.01):
        super().__init__()
        self.xt_input = xt_input

        if reference_trajectory == 'linear':
            self.phi_ref = lambda x0, x1, t: x1 * t + x0 * (1 - t)

        if (correction_scale_factor is None) & (reference_trajectory == 'linear'):
            self.c_t = lambda t: t * (1 - t)

        self.interpolnet = T.nn.Sequential(
            T.nn.Linear(2 * dim + 1, h), T.nn.ELU(),
            # T.nn.Linear(h, h), T.nn.ELU(),
            T.nn.Linear(h, dim)
        )

        self.shiftnet = T.nn.Sequential(
            T.nn.Linear(1, h), T.nn.ELU(),
            # T.nn.Linear(h, h), T.nn.ELU(),
            T.nn.Linear(h, dim)
        )

        self.scalenet = T.nn.Sequential(
            T.nn.Linear(1, h), T.nn.ELU(),
            T.nn.Linear(h, h), T.nn.ELU(),
            T.nn.Linear(h, dim)
        )

        self.f = None

    def scale_fn(self, t):
        return 1 - self.c_t(t) * self.scalenet(t)

    def shift_fn(self, t):
        return self.c_t(t) * self.shiftnet(t)

    def forward(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        input = T.cat([x0, x1, t], 1)

        correction = self.interpolnet(input)
        scale_t = self.scale_fn(t)
        shift_t = self.shift_fn(t)

        self.f =  self.c_t(t) * correction

        return shift_t + scale_t * self.phi_ref(x0, x1, t) + self.f

    def regularizing_term(self, x0, x1, t, xt_fake):
        loss_reg = self.f
        # loss_reg = xt_fake[:, :-1] - self.phi_ref(x0, x1, t)
        if len(loss_reg.shape) > 1:
            loss_reg = (loss_reg ** 2).sum(1).mean()
        else:
            loss_reg = (loss_reg ** 2).mean()
        return loss_reg

    def dt(self,  x0: Tensor, x1: Tensor, t: Tensor,) -> Tensor:
        pass

class GaussianProbabilityPath(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64, reference_trajectory='linear',
                 correction_scale_factor=None, sigma_constant=0.01):
        super().__init__()

        if reference_trajectory == 'linear':
            self.phi_ref = lambda x0, x1, t: x1 * t + x0 * (1 - t)

        if (correction_scale_factor is None) & (reference_trajectory == 'linear'):
            self.c_t = lambda t: t * (1 - t)

        elif correction_scale_factor == 'sqrt':
            self.c_t = lambda t: T.sqrt(t * (1 - t))

        self.sigma_constant = sigma_constant

        self.mu_interpolnet = T.nn.Sequential(
            T.nn.Linear(2 * dim + 1, h), T.nn.ELU(),
            T.nn.Linear(h, h), T.nn.ELU(),
            T.nn.Linear(h, dim))

        self.log_sigma_interpolnet = T.nn.Sequential(
            T.nn.Linear(2 * dim + 1, h), T.nn.ELU(),
            T.nn.Linear(h, h), T.nn.ELU(),
            T.nn.Linear(h, dim), nn.Hardtanh(min_val=-6., max_val=2.))

    def forward(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        input = T.cat([x0, x1, t], 1)

        # input shape is (bs, 2 + 2 + 1)

        self.mu_t = self.phi_ref(x0, x1, t) + self.c_t(t) * self.mu_interpolnet(input)
        self.sigma_t = T.sqrt(self.sigma_constant ** 2 + self.c_t(t) * T.exp(2 * self.log_sigma_interpolnet(input)))

        return self.sample_interpolant(self.mu_t, self.sigma_t)

    @staticmethod
    def sample_interpolant(mu_t, sigma_t):
        return mu_t + sigma_t * T.randn(mu_t.shape, device=mu_t.device)

    def regularizing_term(self, x0, x1, t, xt_fake):
        loss_reg = xt_fake[:, :-1] - self.phi_ref(x0, x1, t)
        if len(loss_reg.shape) > 1:
            loss_reg = (loss_reg ** 2).sum(1).mean()
        else:
            loss_reg = (loss_reg ** 2).mean()
        return loss_reg

    """
    def regularizing_term(self, x0, x1, t, xt_fake):
        loss_reg_mu = self.mu_t - self.phi_ref(x0, x1, t)
        loss_reg_sigma = self.sigma_t - self.sigma_constant
        if len(loss_reg_mu.shape) > 1:
            loss_reg = (loss_reg_mu ** 2).sum(1).mean() + (loss_reg_sigma ** 2).sum(1).mean()
        else:
            loss_reg = (loss_reg_mu ** 2).mean() + (loss_reg_sigma ** 2).mean()
        return loss_reg
    """

class SpatialCorrectionInterpolant(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64, reference_trajectory='linear',
                 correction_scale_factor=None, interpolnet_input='reference', coordinate_dims=1):
        super().__init__()
        self.interpolnet_input = interpolnet_input

        if reference_trajectory == 'linear':
            self.phi_ref = lambda x0, x1, t: x1 * t + x0 * (1 - t)

        if (correction_scale_factor is None) & (reference_trajectory == 'linear'):
            self.c_t = lambda t: t * (1 - t)

        elif correction_scale_factor == 'sqrt':
            self.c_t = lambda t: T.sqrt(t * (1 - t))

        if interpolnet_input == 'reference':
            self.interpolnet = T.nn.Sequential(
                T.nn.Linear(dim + 1 + coordinate_dims, h), T.nn.ELU(),
                T.nn.Linear(h, h), T.nn.ELU(),
                T.nn.Linear(h, dim), T.nn.ReLU())
        else:
            self.interpolnet = T.nn.Sequential(
                T.nn.Linear(2 * dim + 1 + coordinate_dims, h), T.nn.ELU(),
                T.nn.Linear(h, h), T.nn.ELU(),
                T.nn.Linear(h, dim), T.nn.ReLU())

    def forward(self, x0: Tensor, x1: Tensor, t: Tensor, c: Tensor) -> Tensor:
        if self.interpolnet_input == 'reference':
            input = T.cat([self.phi_ref(x0, x1, t), t, c], 1)
        else:
            input = T.cat([x0, x1, t, c], 1)

        self.f = T.log(self.interpolnet(input) + 1)
        # shape of f is (bs, 2)

        return self.phi_ref(x0, x1, t) + self.c_t(t) * self.f

    def regularizing_term(self, x0, x1, t, xt_fake):
        # loss_reg = xt_fake[:, :-2] - self.phi_ref(x0, x1, t)
        loss_reg = self.f * self.c_t(t)
        if len(loss_reg.shape) > 1:
            loss_reg = (loss_reg ** 2).sum(1).mean()
        else:
            loss_reg = (loss_reg ** 2).mean()
        return loss_reg

    def dt(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        pass





