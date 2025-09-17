import torch


def t_weighting_function(t, t_sample, t_gamma):
    pairwise_sq_diff = (t[:, None] - t_sample[None, :]) ** 2
    weights = torch.exp(-pairwise_sq_diff / (2 * t_gamma**2))
    return weights.squeeze(-1)


def weighting_function(x, samples, gamma):
    pairwise_sq_diff = (x[:, None, :] - samples[None, :, :]) ** 2
    pairwise_sq_dist = pairwise_sq_diff.sum(-1)
    weights = torch.exp(-pairwise_sq_dist / (2 * gamma**2))
    return weights


def compute_time_dependent_metric(x, t, samples, t_samples, gamma, 
                                  t_gamma, normalize_t=True, rho=1e-6):
    weights = weighting_function(x, samples, gamma)

    if t_gamma > 0: 
        t_weights = t_weighting_function(t, t_samples, t_gamma)
        if normalize_t:
            t_weights = t_weights / (t_weights.sum(-1, keepdim=True) + 1e-8)

        weights = weights * t_weights

    differences = samples[None, :, :] - x[:, None, :]
    squared_differences = differences**2

    M_dd_diag = \
        torch.einsum("bn,bnd->bd", weights, squared_differences) + rho

    M_dd_inv_diag = 1.0 / M_dd_diag  
    return M_dd_inv_diag
