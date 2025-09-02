import numpy as np
import torch
from scipy import interpolate


def pad_a_like_b(a, b):
    """Pad a to have the same number of dimensions as b."""
    if isinstance(a, float | int):
        return a
    return a.reshape(-1, *([1] * (b.dim() - 1)))


class MultiMarginalFlowMatcher:
    """Multi-Marginal Flow Matcher.

    Structure inspired by Alex Tong's CFM implementation:
    https://github.com/atong01/conditional-flow-matching
    """

    def __init__(self, sigma: float | int | str = 0.0, interpolation: str = "cubic"):
        """Initialize the Multi-Marginal Flow Matcher.

        Args:
            sigma (float | int | str): The variance of the probability density path N(mu_t(x), sigma_t(x)).
                If sigma is an int/float, the variance is constant.
                If sigma is a string, the variance is adaptive and depends on the timepoints.
                The string should be in the format "adaptiveX-{M}-{d}" where X is the adaptive method,
                M is the maximum variance and d is the minimum variance.
            interpolation (str): The interpolation method to use for the mean function.
                Can be "linear", "lagrange" or "cubic".
        """
        self.sigma = sigma
        self.interpolation = interpolation

        if self.interpolation not in ["linear", "lagrange", "cubic"]:
            raise ValueError(
                "Interpolation method must be either 'lagrange', 'cubic' or 'linear'."
            )

    def compute_mu_t(self, xs, t):
        """Compute the mean mu_t(x) of the probability density path N(mu_t(x), sigma_t(x)).

        Args:
            xs (torch.Tensor): The data points.
            t (torch.Tensor): The timepoints.
        """
        t = pad_a_like_b(t, xs)
        return self.P(t, 0)

    def compute_sigma_t(self, t, derative=0):
        """Compute the variance sigma_t(x) of the probability density path N(mu_t(x), sigma_t(x)).

        Args:
            t (torch.Tensor): The timepoints.
            derative (int): The derivative order to compute.
        """
        if derative == 0:
            # Description name: "adaptiveX-{M}-{d}"
            if isinstance(self.sigma, float | int):
                return self.sigma
            else:
                if "adaptive1" in self.sigma:
                    # Adaptive1: M * sqrt(t * (1 - t))
                    # Always reaches variance 1 between two timepoints
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint to be between 0 and 1
                    t_np = (t_np - lower_idx) / (upper_idx - lower_idx)
                    std = M * np.sqrt(t_np * (1 - t_np))
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive2" in self.sigma:
                    # Adaptive2:
                    # phi = lambda x, t0, t1: M * np.sqrt((x - t0) ** 2 * (x - t1) ** 2 / ((t1 - t0) ** 2))
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint
                    std = M * np.sqrt(
                        (t_np - lower_idx) ** 2
                        * (t_np - upper_idx) ** 2
                        / ((upper_idx - lower_idx) ** 2)
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive3" in self.sigma:
                    # Adaptive3:
                    # M = 1
                    # phi = lambda x, t0, t1: M * (1 - ((2 * (x - t0) / (t1 - t0) - 1) ** 2)) * ((t1 - t0) ** 2)
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint
                    std = M * np.sqrt(
                        (
                            1
                            - (
                                (2 * (t_np - lower_idx) / (upper_idx - lower_idx) - 1)
                                ** 2
                            )
                        )
                        * ((upper_idx - lower_idx) ** 2)
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive4" in self.sigma:
                    # M = 16
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    std = (
                        M
                        * (t_np - lower_idx) ** 2
                        * (upper_idx - t_np) ** 2
                        / (upper_idx - lower_idx) ** 2
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
        elif derative == 1:
            if isinstance(self.sigma, float | int):
                return torch.tensor(0.0).to(t.device)
            else:
                if "adaptive1" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    t_np = (t_np - lower_idx) / (upper_idx - lower_idx)
                    sigma_prime = M * (1 - 2 * t_np) / (2 * np.sqrt(t_np * (1 - t_np)))
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive2" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * (
                            (2 * (t_np - lower_idx) ** 2 * (t_np - upper_idx))
                            / (upper_idx - lower_idx) ** 2
                            + (2 * (t_np - lower_idx) * (t_np - upper_idx) ** 2)
                            / (upper_idx - lower_idx) ** 2
                        )
                        / (
                            2
                            * np.sqrt(
                                ((t_np - lower_idx) ** 2 * (t_np - upper_idx) ** 2)
                                / (upper_idx - lower_idx) ** 2
                            )
                        )
                    )
                    # At position where t == lower_idx/upper_idx, set sigma_prime to +-M
                    sigma_prime[np.isclose(t_np, lower_idx)] = M
                    sigma_prime[np.isclose(t_np, upper_idx)] = -M
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive3" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * 4
                        * (lower_idx + upper_idx - 2 * t_np)
                        / ((lower_idx - t_np) * (t_np - upper_idx))
                    )
                    # At position where t == lower_idx/upper_idx, set sigma_prime to +-M
                    sigma_prime[np.isclose(t_np, lower_idx)] = M
                    sigma_prime[np.isclose(t_np, upper_idx)] = -M
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive4" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * (
                            2
                            * (t_np - lower_idx)
                            * (upper_idx - t_np)
                            * (lower_idx + upper_idx - 2 * t_np)
                        )
                        / (lower_idx - upper_idx) ** 2
                    )
                    return torch.tensor(sigma_prime).to(t.device)
        else:
            raise ValueError("Only derivatives 0 and 1 are supported")

    def sample_xt(self, xs, t, epsilon):
        """Sample from the conditional distribution N(mu_t(x), sigma_t(x)).

        Args:
            xs (torch.Tensor): The data points.
            t (torch.Tensor): The timepoints.
            epsilon (torch.Tensor): The noise to add to the sample.

        Returns:
            torch.Tensor: The sampled location x_t.
        """
        mu_t = self.compute_mu_t(xs, t)
        sigma_t = self.compute_sigma_t(t, 0)
        sigma_t = pad_a_like_b(sigma_t, xs)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, xs, t, xt):
        """Compute the conditional flow u_t(x | z).

        Args:
            xs (torch.Tensor): The data points.
            t (torch.Tensor): The timepoints.
            xt (torch.Tensor): The sampled location x_t.

        Returns:
            torch.Tensor: The conditional flow u_t(x | z).
        """
        if isinstance(self.sigma, float | int):
            # The derivatives of a constant variance is zero, hence we only return the derivative of the
            # mean function
            t = pad_a_like_b(t, xs)
            return self.P(t, 1)
        else:
            # We need to evaluate the full formula
            # sigma' / sigma * (x - mu) + mu'
            return (self.compute_sigma_t(t, 1) / self.compute_sigma_t(t, 0)).reshape(
                -1, 1, 1
            ) * (xt - self.P(t, 0)) + self.P(t, 1)

    def sample_location_and_conditional_flow(self, xs, timepoints, t=None, return_noise=False):
        """Sample a location x_t from probability density path p_t(x) and conditional flow u_t(x | z) at time t.

        Args:
            xs (torch.Tensor): The data points.
            timepoints (torch.Tensor): The timepoints.
            t (torch.Tensor): The timepoints to sample at. If None, samples are drawn from a uniform distribution.
        """
        if self.interpolation == "basic":
            if xs.shape[1] != 2:
                raise ValueError("Basic interpolation requires exactly 2 data points")
        if t is None:
            t = torch.rand(xs.shape[0]).type_as(xs) * timepoints.max()
            # TODO: Make sure the values in t are not too close to the self.timepoints
            # because derivatives will be unstable
        assert len(t) == xs.shape[0], "t has to have batch size dimension"

        # Convert timepoints to numpy array (check if it's on cuda first)
        if isinstance(timepoints, torch.Tensor):
            timepoints = timepoints.cpu().detach().numpy()
        self.timepoints = timepoints
        eps = torch.randn_like(xs)[:, 0].unsqueeze(1)

        if self.interpolation == "cubic":
            self.P = CubicSplineInterpolation(xs, self.timepoints)
        elif self.interpolation == "lagrange":
            self.P = LagrangeInterpolation(xs, self.timepoints)
        elif self.interpolation == "linear":
            self.P = LinearInterpolation(xs, self.timepoints)
        xt = self.sample_xt(xs, t, eps)
        ut = self.compute_conditional_flow(xs, t, xt)

        return t, xt, ut, xs, self.P


class LagrangePolynomial:
    """Construct Lagrange Polynomial and its derivative.

    This class is in a batched fashion in class LagrangeInterpolation.
    """

    def __init__(self, t_anchor, x):
        """Initialize the Lagrange Polynomial."""
        self.x = x
        self.t_anchor = t_anchor
        self.n = len(t_anchor)
        self.dtype = x.dtype
        self.device = "cpu"

    def _basis_derivative(self, j, t_val, t):
        n = len(t)
        return sum(1 / (t_val - t[i]) for i in range(n) if i != j)

    def _basis_polynomial(self, j, t_val, t, n):
        L_j = 1
        for i in range(n):
            if j != i:
                L_j *= (t_val - t[i]) / (t[j] - t[i])
        return L_j

    def __call__(self, t_query, derivative):
        """Evaluate the x-derivative of the Lagrange Polynomial at t_query."""
        result = torch.zeros((1, self.x.shape[-1]), device=self.device)

        if derivative == 0:
            for i in range(self.n):
                result += self.x[i] * self._basis_polynomial(
                    i, t_query, self.t_anchor, self.n
                )
        elif derivative == 1:
            for i in range(self.n):
                result += (
                    self.x[i]
                    * self._basis_polynomial(i, t_query, self.t_anchor, self.n)
                    * self._basis_derivative(i, t_query, self.t_anchor)
                )
        else:
            raise ValueError("Only derivatives 0 and 1 are supported")
        return result


class LagrangeInterpolation:
    """Construct Lagrange Interpolation for a batch of data."""

    def __init__(self, xs, t_anchor):
        self.device = xs.device
        self.x_dim = xs.dim()
        self.dtype = xs.dtype

        self.P = self.get_lagrange_interpolation(xs, t_anchor)

    def get_lagrange_interpolation(self, xs, t_anchor):
        """Create Lagrange interpolation functions for each batch."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().numpy()

        # Return list of polynomials for each batch
        return [
            LagrangePolynomial(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
            )
            for b in range(xs.shape[0])
        ]

    def __call__(self, t_query, derivative):
        """Evaluate the x-derivative of the Lagrange Interpolation at t_query."""
        return (
            torch.from_numpy(self.eval_lagrange_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def eval_lagrange_interpolation(self, t_query, derivative):
        """Helper function for __call__ taking care of batching & data converting."""
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, int | float):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, len(self.P))
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        results = np.concatenate(
            [lp(t_query[k], derivative) for k, lp in enumerate(self.P)], axis=0
        )
        results = results.reshape(t_query.shape[0], 1, -1)
        return results


class CubicSplineInterpolation:
    """Construct Cubic Spline Interpolation for a batch of data."""

    def __init__(self, xs, t_anchor):
        self.splines = self.get_cubic_spline_interpolation(xs, t_anchor)
        self.device = xs.device
        self.x_dim = xs.dim()

    def __call__(self, t_query, derivative):
        """Evaluate the x-derivative of the Cubic Spline Interpolation at t_query."""
        return (
            torch.from_numpy(self.eval_cubic_spline_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def get_cubic_spline_interpolation(self, xs, t_anchor):
        """Create cubic spline interpolation functions for each batch."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().numpy()
        # Create cubic spline interpolation functions for each batch
        return [
            interpolate.CubicSpline(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
            )
            for b in range(xs.shape[0])
        ]

    def eval_cubic_spline_interpolation(self, t_query, derivative=0):
        """Evaluate the x-derivative of the Cubic Spline Interpolation at t_query."""
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, int | float):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, len(self.splines))
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        return np.concatenate(
            [
                spline(t_query[k], nu=derivative)
                for k, spline in enumerate(self.splines)
            ],
            axis=0,
        )


class LinearInterpolation:
    """Construct Linear Interpolation for a batch of data."""

    def __init__(self, xs, t_anchor):
        self.linear_interpolations = self.get_linear_interpolation(xs, t_anchor)
        self.device = getattr(xs, "device", None)
        self.x_dim = xs.ndim if isinstance(xs, np.ndarray) else xs.dim()

    def __call__(self, t_query, derivative):
        """Evaluate the Linear Interpolation at t_query."""
        return (
            torch.from_numpy(self.eval_linear_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def get_linear_interpolation(self, xs, t_anchor):
        """Create linear interpolation functions for each batch."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().numpy()

        return [
            interpolate.interp1d(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
                axis=0,
                fill_value="extrapolate",
            )
            for b in range(xs.shape[0])
        ]

    def eval_linear_interpolation(self, t_query, derivative=0):
        """Evaluate the Linear Interpolation or its derivative at t_query."""
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, int | float):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, len(self.splines))
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        if derivative == 0:
            results = np.concatenate(
                [
                    interp(t_query[k])
                    for k, interp in enumerate(self.linear_interpolations)
                ],
                axis=0,
            )
        elif derivative == 1:
            results = np.concatenate(
                [
                    self.compute_derivative(interp, t_query[k])
                    for k, interp in enumerate(self.linear_interpolations)
                ],
                axis=0,
            )
        else:
            raise ValueError(
                "Derivative order must be 0 or 1 for linear interpolation."
            )

        return results

    def compute_derivative(self, interp, t):
        """Compute the derivative of the linear interpolation."""
        eps = 1e-6
        return (interp(t + eps) - interp(t - eps)) / (2 * eps)
