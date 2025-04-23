import torch as T
from torch import nn, Tensor
import torch.linalg as LA


class CorrectionInterpolant(nn.Module):
    # learns the correction from the reference trajectory
    # interpolant = reference_trajectory(x0, x1, t) + correction_scale_factor_t * NN(x0, x1, t)
    # regularizing term = || reference_trajectory(x0, x1, t) - interpolant || ** 2
    #                   = correction_scale_factor_t ** 2 || interpolant || ** 2

    def __init__(self, dim: int = 2, h: int = 64, reference_trajectory='linear', correction_scale_factor=None):
        super().__init__()

        if reference_trajectory == 'linear':
            self.phi_ref = lambda x0, x1, t: x1 * t + x0 * (1 - t)

        if (correction_scale_factor is None) & (reference_trajectory == 'linear'):
            self.c_t = lambda t: t * (1 - t)

        elif correction_scale_factor == 'sqrt':
            self.c_t = lambda t: T.sqrt(t * (1 - t))

        self.interpolnet = T.nn.Sequential(
            T.nn.Linear(2 * dim + 1, h), T.nn.ELU(),
            T.nn.Linear(h, h), T.nn.ELU(),
            T.nn.Linear(h, dim)).type(T.float32)

    def forward(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        input = T.cat([x0, x1, t], 1)

        # input shape is (bs, 2 + 2 + 1)
        f = self.interpolnet(input)
        # shape of f is (bs, 2)

        return self.phi_ref(x0, x1, t) + self.c_t(t) * f

    def regularizing_term(self, x0, x1, t, xt_fake):
        loss_reg = xt_fake[:, :-1] - self.phi_ref(x0, x1, t)
        if len(loss_reg.shape) > 1:
            loss_reg = (loss_reg ** 2).sum(1).mean()
        else:
            loss_reg = (loss_reg ** 2).mean()
        return loss_reg

    def dt(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        pass


class AffineTransformInterpolant(nn.Module):
    # TODO: make class inherent CorrectionInterpolant?
    def __init__(self, dim: int = 2, h: int = 64, reference_trajectory='linear', correction_scale_factor=None):
        super().__init__()

        if reference_trajectory == 'linear':
            self.phi_ref = lambda x0, x1, t: x1 * t + x0 * (1 - t)

        if (correction_scale_factor is None) & (reference_trajectory == 'linear'):
            self.c_t = lambda t: t * (1 - t)

        self.interpolnet = T.nn.Sequential(
            T.nn.Linear(2 * dim + 1, h), T.nn.ELU(),
            T.nn.Linear(h, h), T.nn.ELU(),
            T.nn.Linear(h, dim)
            )

        self.shiftnet = T.nn.Sequential(
            T.nn.Linear(1, h), T.nn.ELU(),
            T.nn.Linear(h, dim)
            )

        self.scalenet = T.nn.Sequential(
            T.nn.Linear(1, h), T.nn.ELU(),
            T.nn.Linear(h, dim)
            # T.nn.Linear(h, dim * dim),
            # T.nn.Unflatten(1, (dim, dim))
            )

        self.f = None

    def forward(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        input = T.cat([x0, x1, t], 1)

        # input shape is (bs, 2 + 2 + 1)
        self.f = self.interpolnet(input)  # save for computation of regularization term
        # shape of f is (bs, 2)

        # G = self.scalenet(t)  # (bs, d, d)
        # scale by t(1-t), then exponentiate as a matrix
        # scale_t = LA.matrix_exp((t * (1 - t)).view(t.shape[0], 1, 1) * G)
        # scale_t = T.einsum("bij,bj->bi", scale_t, self.phi_ref(x0, x1, t))

        scale_t = T.exp((t * (1 - t)) * self.scalenet(t))
        shift_t = t * (1 - t) * self.shiftnet(t)

        return shift_t + scale_t * self.phi_ref(x0, x1, t) + self.c_t(t) * self.f

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
            T.nn.Linear(h, dim))

    def forward(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        input = T.cat([x0, x1, t], 1)

        # input shape is (bs, 2 + 2 + 1)

        self.mu_t = self.phi_ref(x0, x1, t) + self.c_t(t) * self.mu_interpolnet(input)
        self.sigma_t = self.sigma_constant + self.c_t(t) * T.exp(self.log_sigma_interpolnet(input))

        return self.sample_interpolant(self.mu_t, self.sigma_t)

    @staticmethod
    def sample_interpolant(mu_t, sigma_t):
        return mu_t + sigma_t * T.randn(mu_t.shape)

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



