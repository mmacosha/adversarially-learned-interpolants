import torch
import torch.nn as nn
from torch.func import jacrev, vmap


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_dim, w=64, apply_sigmoid: bool = True):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.net = torch.nn.Sequential(
            nn.Linear(in_dim, w), nn.ELU(),
            nn.Linear(w, w), nn.ELU(),
            nn.Linear(w, 1), nn.Sigmoid() if apply_sigmoid else nn.Identity()
        )

    def forward(self, x):
        return self.net(x)


class TrainableInterpolant(nn.Module):
    def __init__(self, dim, h_dim, t_smooth=0.01, time_varying=False, regulariser='linear'):
        super().__init__()
        self.time_varying = time_varying
        self.t_smooth = t_smooth
        self.interpolant_net = nn.Sequential(
            nn.Linear(2 * dim + 1, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, dim)
        )
        self.regularizer = regulariser

    def forward(self, x0, x1, t, training=True):
        t = t[..., None] if t.ndim == 1 else t

        xt = self.linear_interpolant(x0, x1, t)
        if training:
            t_input = t + torch.randn_like(t) * self.t_smooth if self.t_smooth > 0 else t
        else:
            t_input = t
        input_ = torch.cat([x0, x1, t_input], dim=-1) 
        
        correction = self.interpolant_net(input_)
        return xt + t * (1 - t) * correction

    @staticmethod
    def linear_interpolant(x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t
        return t * x1 + (1 - t) * x0

    def get_reg_term(self, x0, x1, t, xt_fake, xt):
        if self.regularizer == 'length':
            return self.compute_length_reg_term(x0, x1)
        elif self.regularizer == 'piecewise':
            return self.compute_piecewise_reg_term(x0, x1, xt, t)
        elif self.regularizer == 'linear':
            return self.compute_linear_reg_term(x0, x1, t, xt_fake)
        else:
            raise ValueError(f"Unknown regularizer type: {self.regularizer}")

    def compute_length_reg_term(self, x0, x1, num_t_steps=100, h=0.001):
        def _interpolant(x0, x1, t):
            xt = self.linear_interpolant(x0, x1, t)
            input_ = torch.cat([x0, x1, t], dim=-1) 
            correction = self.interpolant_net(input_)
            return xt + t * (1 - t) * correction

        def estimate_second_derivative(x0, x1, t, h=0.001):
            t_p_h, t_m_h = t + h, t - h
            second_derivative = (
                _interpolant(x0, x1, t_p_h) + _interpolant(x0, x1, t_m_h) - \
                2 * _interpolant(x0, x1, t)
            ) / h**2
            return second_derivative

        integral = 0.0
        for t in torch.linspace(0, 1, num_t_steps):
            t = torch.ones(x0.shape[0], 1, device=x0.device) * t
            second_derivative = estimate_second_derivative(x0, x1, t, h)
            integral += second_derivative.pow(2).sum(-1)
        return integral

    def compute_piecewise_reg_term(self, x0, x1, xt, t_exact):
        t = t_exact + torch.rand_like(t_exact)
        t_left = t[t < t_exact]
        t_right = t[t >= t_exact]

        x0 = x0[torch.randint(0, x0.shape[0], size=(t_left.shape[0],))]
        x1 = x1[torch.randint(0, x1.shape[0], size=(t_right.shape[0],))]

        xt_linear = torch.zeros_like(xt)
        xt_linear[t < t_exact] = ((t_exact - t_left) * x0 + t_left * xt[t < t_left]) / t_exact
        xt_linear[t >= t_exact] = ((1 - t_right) * xt[t >= t_exact] + t_right * x1) / (1 - t_exact)

        t = t[..., None] if t.ndim == 1 else t
        xt = self.linear_interpolant(x0, x1, t)
        input_ = torch.cat([x0, x1, t], dim=-1) 
        correction = self.interpolant_net(input_)
        xt_interpolant = xt + t * (1 - t) * correction

        return (xt_linear - xt_interpolant).pow(2).mean()


    def compute_linear_reg_term(self, x0, x1, t, xt):
        correction = xt - self.linear_interpolant(x0, x1, t)
        batch_size = correction.shape[0]
        if correction.ndim == 1:
            reg_term = correction.pow(2)
        else:
            reg_term = correction.reshape(batch_size, -1).pow(2).sum(-1)
        return reg_term.mean()

    def dI_dt(self, x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t

        def _interpolnet(x0i, x1i, ti):
            input_ = torch.cat([x0i, x1i, ti], dim=0).unsqueeze(0)
            out = self.interpolant_net(input_)
            return out, out

        (corr_jac, corr_output) = vmap(
            jacrev(_interpolnet, argnums=2, has_aux=True))(x0, x1, t)
        return (
            (x1 - x0) + (1 - 2 * t) * corr_output.squeeze() + 
            t * (1 - t) * corr_jac.squeeze()
        )


class CubicSplineInterpolant:
    def __init__(self):
        pass
