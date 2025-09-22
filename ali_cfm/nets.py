import random

import torch
import torch.nn as nn
from numpy.matlib import randn
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
        elif self.regularizer == 'regression':
            return self.compute_regression_term(xt_fake, xt)
        elif self.regularizer == 'linear':
            return self.compute_linear_reg_term(x0, x1, t, xt_fake)
        else:
            raise ValueError(f"Unknown regularizer type: {self.regularizer}")

    def compute_length_reg_term(self, x0, x1, num_t_steps=10, h=0.001):
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
        return integral.mean()

    def compute_piecewise_reg_term(self, x0, x1, xt, t_exact):
        t = t_exact + torch.randn_like(t_exact)
        t_left = t[t < t_exact]
        t_right = t[t >= t_exact]

        x0_ = x0[torch.randperm(x0.shape[0])[:t_left.shape[0]]]
        x1_ = x1[torch.randperm(x1.shape[0])[:t_right.shape[0]]]
        # x0 = x0[torch.randint(0, x0.shape[0], size=(t_left.shape[0],))]
        # x1 = x1[torch.randint(0, x1.shape[0], size=(t_right.shape[0],))]

        xt_linear = torch.zeros_like(xt)
        xt_linear[(t < t_exact).squeeze()] = ((t_exact[t < t_exact] - t_left).unsqueeze(-1) * x0_ + t_left.unsqueeze(-1) * xt[(t < t_exact).squeeze()]) / (t_exact[t < t_exact].unsqueeze(-1))
        xt_linear[(t >= t_exact).squeeze()] = ((1 - t_right).unsqueeze(-1) * xt[(t >= t_exact).squeeze()] + t_right.unsqueeze(-1) * x1_) / (1 - t_exact[t >= t_exact].unsqueeze(-1))

        t = t[..., None] if t.ndim == 1 else t
        xt = self.linear_interpolant(x0, x1, t)
        input_ = torch.cat([x0, x1, t], dim=-1) 
        correction = self.interpolant_net(input_)
        xt_interpolant = xt + t * (1 - t) * correction

        return (xt_linear - xt_interpolant).pow(2).mean()

    def compute_regression_term(self, x_hat, xt):
        return (x_hat - xt).pow(2).mean()
        # return torch.abs(x_hat - xt).mean()  # no theoretical uniqueness guarantees

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


class TrainableInterpolantST(TrainableInterpolant):
    def __init__(self, dim, h_dim, t_smooth=0.01, time_varying=False, regulariser='linear'):
        super().__init__(dim=dim, h_dim=h_dim, t_smooth=t_smooth, time_varying=time_varying, regulariser=regulariser)
        # self.interpolant_net = RotationGenerator(dim=dim)
        self.interpolant_net = nn.Sequential(
            nn.Linear(2 * dim + 1, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, dim), nn.Sigmoid()
        )


# Horrible MNIST things below this line

class DiscriminatorMNIST(torch.nn.Module):
    def __init__(self, in_dim, w=64, apply_sigmoid: bool = True):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.net = torch.nn.Sequential(
            nn.Linear(in_dim - 1 + 64, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 1), nn.Sigmoid() if apply_sigmoid else nn.Identity()
        )
        self.t_emb = TimeEmbedding(n_frequencies=6, out_dim=64)

    def forward(self, x):
        t = x[..., -1:]
        t_emb = self.t_emb(t) * torch.pi  # (B, 64)
        x = torch.cat([x[..., :-1], t_emb], dim=-1)
        return self.net(x)


class TrainableInterpolantMNIST(TrainableInterpolant):
    def __init__(self, dim, h_dim, t_smooth=0.01, time_varying=False, regulariser='linear'):
        super().__init__(dim=dim, h_dim=h_dim, t_smooth=t_smooth, time_varying=time_varying, regulariser=regulariser)
        # self.interpolant_net = RotationGenerator(dim=dim)
        self.interpolant_net = RotationGenerator(dim=dim, latent_dim=64, hidden=1024)

    def compute_linear_reg_term(self, x0, x1, t, xt):
        correction = xt - x0
        batch_size = correction.shape[0]
        if correction.ndim == 1:
            reg_term = correction.pow(2)
        else:
            reg_term = correction.reshape(batch_size, -1).pow(2).sum(-1)
        return reg_term.mean()

    def compute_length_reg_term(self, x0, x1, num_t_steps=1, h=0.001):
        def _interpolant(x, t):
            xt = self.linear_interpolant(x, x, t)
            input_ = torch.cat([x, t], dim=-1)
            correction = self.interpolant_net(input_)
            return xt + t * (1 - t) * correction

        def estimate_second_derivative(x, t, h=0.001):
            t_p_h, t_m_h = t + h, t - h
            second_derivative = (
                _interpolant(x, t_p_h) + _interpolant(x, t_m_h) - \
                2 * _interpolant(x, t)
            ) / h**2
            return second_derivative

        x = x0
        integral = 0.0
        for t in torch.linspace(0, 1, num_t_steps):
            t = torch.ones(x0.shape[0], 1, device=x0.device) * t
            second_derivative = estimate_second_derivative(x, t, h)
            integral += second_derivative.pow(2).sum(-1)
        return integral.mean()

    def forward(self, x0, x1, t, training=True):
        """
        x0,x1: (B,dim), t: (B,) or (B,1)
        Returns: (B,dim)
        """
        if t.ndim == 1:
            t = t[:, None]  # (B,1)

        x = x0

        xt = self.linear_interpolant(x, x, t)

        # x = torch.randn_like(x)

        if training and self.t_smooth > 0:
            t_input = t + torch.randn_like(t) * self.t_smooth
            t_input = t_input.clamp(0, 1)
        else:
            t_input = t

        input_ = torch.cat([x, t_input], dim=-1)

        correction = self.interpolant_net(input_)

        return xt + (t * (1 - t)) * correction
        # return correction

    def dI_dt(self, x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t

        def _interpolnet(x0i, ti):
            input_ = torch.cat([x0i, ti], dim=0).unsqueeze(0)
            out = self.interpolant_net(input_, didt=True)
            return out, out

        (corr_jac, corr_output) = vmap(
            jacrev(_interpolnet, argnums=1, has_aux=True))(x0, t)
        return (
            (1 - 2 * t) * 1e-5 / ((t * (1 - t) + 1e-5) ** 2) * (corr_output.squeeze() - x0) +
                t * (1 - t) / (t * (1 - t) + 1e-5) * corr_jac.squeeze()
        )


class RotationGenerator(nn.Module):
    def __init__(self, dim, latent_dim=64, hidden=1024):
        super().__init__()

        # Encode input digit (x0, flattened 784)
        self.mlp = nn.Sequential(
            nn.Linear(dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim), nn.Tanh(),
        )
        self.t_embed = TimeEmbedding(n_frequencies=6, out_dim=latent_dim)

    def forward(self, input_, didt=False):
        """
        x0: (B,784) flattened MNIST image
        theta: (B,1) rotation angle (in radians or normalized [0,1])
        """
        x, t = input_[..., :-1], input_[..., -1:]
        theta = self.t_embed(t) * torch.pi

        z = torch.cat([x, theta], dim=-1)
        out = self.mlp(z)  # (B,2 * 784)
        if didt:
            return out
        else:
            return (out - x) / (t * (1 - t) + 1e-5)


class RotationCFM(nn.Module):
    def __init__(self, dim, latent_dim=64, hidden=1024):
        super().__init__()

        # Encode input digit (x0, flattened 784)
        self.mlp = nn.Sequential(
            nn.Linear(dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.t_embed = TimeEmbedding(n_frequencies=6, out_dim=latent_dim)

    def forward(self, input_):
        """
        x0: (B,784) flattened MNIST image
        theta: (B,1) rotation angle (in radians or normalized [0,1])
        """
        x, t = input_[..., :-1], input_[..., -1:]
        theta = self.t_embed(t) * torch.pi

        z = torch.cat([x, theta], dim=-1)
        out = self.mlp(z)  # (B,2 * 784)
        return out



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)

class CorrectionUNet(nn.Module):
    def __init__(self, in_ch=2, base=32, interpolant=False):
        super().__init__()
        self.interpolant = interpolant

        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.down = nn.MaxPool2d(2)

        self.mid  = ConvBlock(base*2, base*4)

        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ConvBlock(base*4 + base*2, base*2)
        self.dec1 = ConvBlock(base*2 + base, base)

        self.out = nn.Conv2d(base, 1, 1)

        self.t_embed = TimeEmbedding(n_frequencies=6, out_dim=256)

    def forward(self, input_):
        if self.interpolant == True:
            dim = (input_.shape[-1] - 1) // 2
            xt, _, t = input_[..., :dim], input_[..., dim:2 * dim], input_[..., -1:]
        else:
            xt, t = input_[..., :-1], input_[..., -1:]
        xt = xt.view(xt.shape[0], 1, int(xt.shape[1] ** 0.5), int(xt.shape[1] ** 0.5))
        B, _, H, W = xt.shape
        # theta = self.theta_net(t) * torch.pi
        theta = self.t_embed(t) * torch.pi  # (B, 256)
        theta_channel = theta.view(B, 1, H, W)# .expand(B, 1, H, W)
        inp = torch.cat([xt, theta_channel], dim=1)  # (B,2,H,W)
        # inp = xt

        e1 = self.enc1(inp)         # (B, base, H, W)
        x  = self.down(e1)
        e2 = self.enc2(x)           # (B, 2*base, H/2, W/2)
        x  = self.down(e2)

        # theta_channel = theta.view(B, 1, 1, 1).expand(B, 1, x.shape[-2], x.shape[-1])
        # x = torch.cat([x, theta_channel], dim=1)

        x  = self.mid(x)            # (B, 4*base, H/4, W/4)

        x  = self.up(x)             # (B, 4*base, H/2, W/2)
        x  = torch.cat([x, e2], dim=1)  # (B, 6*base, H/2, W/2)
        x  = self.dec2(x)           # (B, 2*base, H/2, W/2)

        x  = self.up(x)             # (B, 2*base, H, W)
        x  = torch.cat([x, e1], dim=1)  # (B, 3*base, H, W)
        x  = self.dec1(x)           # (B, base, H, W)

        return self.out(x).view(B, -1)  # (B, H*W)



class TimeEmbedding(nn.Module):
    def __init__(self, n_frequencies=6, out_dim=64):
        super().__init__()
        self.nf = n_frequencies
        self.proj = nn.Sequential(
            nn.Linear(2 * n_frequencies, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, t):  # t: (B, 1)
        # Build [sin(2^k pi t), cos(2^k pi t)]
        device = t.device
        k = torch.arange(self.nf, device=device, dtype=t.dtype)
        phases = (2.0 ** k)[None, :] * torch.pi * t  # (B, nf)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # (B, 2*nf)
        return self.proj(emb)  # (B, out_dim)


class FiLM(nn.Module):
    def __init__(self, emb_dim, n_channels):
        super().__init__()
        self.to_scale = nn.Linear(emb_dim, n_channels)
        self.to_shift = nn.Linear(emb_dim, n_channels)

    def forward(self, x, emb):  # x: (B, C, H, W), emb: (B, E)
        s = self.to_scale(emb).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(emb).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b

class UNetCFM(nn.Module):
    def __init__(self, in_ch=1, base=32, t_emb_dim=64):
        super().__init__()
        self.t_embed = TimeEmbedding(n_frequencies=6, out_dim=t_emb_dim)

        self.enc1 = ConvBlock(in_ch, base)
        self.film1 = FiLM(t_emb_dim, base)
        self.enc2 = ConvBlock(base, base*2)
        self.film2 = FiLM(t_emb_dim, base*2)
        self.down = nn.MaxPool2d(2)

        self.mid  = ConvBlock(base*2, base*4)
        self.filmM = FiLM(t_emb_dim, base*4)

        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ConvBlock(base*4 + base*2, base*2)
        self.film3 = FiLM(t_emb_dim, base*2)
        self.dec1 = ConvBlock(base*2 + base, base)
        self.film4 = FiLM(t_emb_dim, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, input_):

        inp, t = input_[..., :-1], input_[..., -1:]
        inp = inp.view(inp.shape[0], 1, int(inp.shape[1] ** 0.5), int(inp.shape[1] ** 0.5))
        B, _, H, W = inp.shape

        te = self.t_embed(t) * torch.pi  # (B, t_emb_dim)

        e1 = self.enc1(inp); e1 = self.film1(e1, te)
        x  = self.down(e1)
        e2 = self.enc2(x);   e2 = self.film2(e2, te)
        x  = self.down(e2)

        x  = self.mid(x);    x  = self.filmM(x, te)

        x  = self.up(x)
        x  = torch.cat([x, e2], dim=1)
        x  = self.dec2(x);   x  = self.film3(x, te)

        x  = self.up(x)
        x  = torch.cat([x, e1], dim=1)
        x  = self.dec1(x);   x  = self.film4(x, te)

        return self.out(x).view(B, -1)


def sinusoidal_embedding(t, dim=16, max_freq=10.0):
    """
    t: (B,1) in [0,1]
    returns: (B, 2*dim)
    """
    device = t.device
    freqs = torch.linspace(1.0, max_freq, dim, device=device)  # (dim,)
    angles = 2 * torch.pi * freqs[None, :] * t  # (B,dim)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,2*dim)
    return emb


import torch.nn.functional as F
class CFMNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, t_dim=16):
        super().__init__()
        # project time embedding to channel dim
        self.time_proj = nn.Linear(2*t_dim, base_ch*2)

        # Encoder
        self.enc1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)       # 16x16 → 16x16
        self.enc2 = nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1)  # 16x16 → 8x8

        # Bottleneck
        self.bottleneck = nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)  # 8x8 → 16x16
        self.outc = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, input_):
        """
        input_: (B, 257) with flattened image (256) + time (1)
        returns: (B, 256) vector field
        """
        x, t = input_[..., :-1], input_[..., -1:]  # (B,256), (B,1)
        B = x.shape[0]
        x = x.view(B, 1, 16, 16)

        # sinusoidal embedding of t
        temb = sinusoidal_embedding(t, dim=16)  # (B,32)
        temb = self.time_proj(temb)[:, :, None, None]  # (B,base_ch,1,1)

        # Encoder
        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1))

        # Add time conditioning
        h2 = h2 + temb.expand(-1, -1, h2.size(2), h2.size(3))

        # Bottleneck
        h = F.silu(self.bottleneck(h2))

        # Decoder
        h = F.silu(self.dec1(h))
        out = self.outc(h).view(B, -1)

        return out
