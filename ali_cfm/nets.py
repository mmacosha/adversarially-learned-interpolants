import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from .land_metric import compute_time_dependent_metric


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


def get_marginals(xts):
    xs, ts = [], []
    
    for t_, x_ in xts.items():
        ts.append(t_ * torch.ones(x_.size(0), 1, device=x_.device))
        xs.append(x_)
    
    xs = torch.cat(xs, dim=0)
    ts = torch.cat(ts, dim=0)

    # print(xs.shape, ts.shape)
    # assert 0

    return xs, ts


class TrainableInterpolant(nn.Module):
    def __init__(self, dim, h_dim, t_smooth=0.01, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.t_smooth = t_smooth
        self.interpolant_net = nn.Sequential(
            nn.Linear(2 * dim + 1, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, dim)
        )

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

    def compute_length_reg_term(
            self, xts, 
            metric='l2', land_gamma=0.2, land_t_gamma=0.2,
            num_t_steps=10, h=0.001
        ):
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

        x0, x1 = xts.get(0), xts.get(1)
        integral = 0.0
        for t in torch.arange(0, num_t_steps):
            t = torch.ones(x0.shape[0], 1, device=x0.device) * t / num_t_steps
            t += torch.rand_like(t)
            second_derivative = estimate_second_derivative(x0, x1, t, h)
            
            if metric == 'l2':
                integral += second_derivative.pow(2).sum(-1)
            
            elif metric == 'land':
                x_t = _interpolant(x0, x1, t)
                xs, ts = get_marginals(xts)
                G = compute_time_dependent_metric(
                    x_t, t, xs, ts,
                    gamma=land_gamma, 
                    t_gamma=land_t_gamma,
                    normalize_t=False,
                )
                integral += ((second_derivative**2) * G).sum(dim=-1)

            elif metric == 'land_norm':
                x_t = _interpolant(x0, x1, t)
                xs, ts = get_marginals(xts)
                G = compute_time_dependent_metric(
                    x_t, t, xs, ts,
                    gamma=land_gamma, 
                    t_gamma=land_t_gamma,
                    normalize_t=True,
                )
                integral += ((second_derivative**2) * G).sum(dim=-1)
            
            else:
                raise ValueError(f"Unknown metric {metric}")
        
        return integral.mean()

    def compute_piecewise_reg_term(
            self, xts, t1, t2, 
            metric='l2', land_gamma=0.2, land_t_gamma=0.2,
        ):
        x0, x1 = xts.get(0), xts.get(1)
        xt1, xt2 = xts.get(t1), xts.get(t2)
        t = t1 + torch.rand(x0.size(0), 1, device=x0.device) * (t2 - t1)
        
        xt_linear = xt1 * t + xt2 * (1 - t)
        
        t = t[..., None] if t.ndim == 1 else t
        xt = self.linear_interpolant(x0, x1, t)
        input_ = torch.cat([x0, x1, t], dim=-1) 
        correction = self.interpolant_net(input_)
        xt_interpolant = xt + t * (1 - t) * correction
        diff = xt_linear - xt_interpolant

        if metric == 'l2':
            ret_term = (diff).pow(2).mean()
        
        elif metric == 'land':
            xs, ts = get_marginals(xts)
            G = compute_time_dependent_metric(
                xt_linear, t, xs, ts, 
                gamma=land_gamma, 
                t_gamma=land_t_gamma,
                normalize_t=False,
            )
            ret_term = ((diff)**2 * G).mean()
        
        elif metric == 'land_norm':
            xs, ts = get_marginals(xts)
            G = compute_time_dependent_metric(
                xt_linear, t, xs, ts, 
                gamma=land_gamma, 
                t_gamma=land_t_gamma,
                normalize_t=True,
            )
            ret_term = ((diff)**2 * G).mean()
        
        else:
            raise ValueError(f"Unknown metric {metric}")

        return ret_term

    def compute_linear_reg_term(
            self, xt_pred, xts, t, 
            metric='l2', land_gamma=0.2, land_t_gamma=0.2,
        ):
        x0, x1 = xts.get(0), xts.get(1)
        correction = xt_pred - self.linear_interpolant(x0, x1, t)

        if metric == 'l2':
            reg_term = correction.reshape(correction.size(0), -1).pow(2).sum(-1)
        
        elif metric == 'land':
            xs, ts = get_marginals(xts)
            G = compute_time_dependent_metric(
                xt_pred, t, xs, ts, 
                gamma=land_gamma, 
                t_gamma=land_t_gamma,
                normalize_t=False,
            )

        elif metric == 'land_norm':
            xs, ts = get_marginals(xts)

            G = compute_time_dependent_metric(
                xt_pred, t, xs, ts, 
                gamma=land_gamma, 
                t_gamma=land_t_gamma,
                normalize_t=True,
            )

            assert correction.size(1) == G.size(1)
            reg_term = torch.sqrt(((correction**2) * G).sum(dim=-1))

        else:
            raise ValueError(f"Unknown metric {metric}")

        return reg_term

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
