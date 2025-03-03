import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def density_and_trajectories_plot(time_steps, x, m1, m2, s1, s2, priors=None):
    x1 = x[:, 0]
    x2 = x[:, -1]

    kde0 = norm(m1, s1)
    kde1 = norm(m2, s2)

    y_min = min(x1.min(), x2.min()) - 1
    y_max = max(x1.max(), x2.max()) + 1
    y_vals = np.linspace(y_min, y_max, 200)

    density0 = kde0.pdf(y_vals)
    density1 = kde1.pdf(y_vals)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Main plot: connecting lines between samples (assuming pairing by index)
    for i in range(x.shape[0]):
        ax.plot(time_steps, x[i])
    ax.set_xlabel('$t$')
    ax.set_xlim(-0., 1.)

    # Create marginal axes for the densities
    divider = make_axes_locatable(ax)
    ax_left = divider.append_axes("left", size="20%", pad=0.3, sharey=ax)
    ax_right = divider.append_axes("right", size="20%", pad=0.3, sharey=ax)

    # Plot the densities in the marginal axes
    ax_left.plot(density0, y_vals, color='blue', lw=2)
    ax_left.set_title('$p_0$')
    ax_left.invert_xaxis()  # Invert so the density appears to the left of t=0
    ax_left.axis('off')     # Optionally turn off axis lines/ticks

    ax_right.plot(density1, y_vals, color='red', lw=2)
    ax_right.set_title('$p_1$')
    ax_right.axis('off')    # Optionally turn off axis lines/ticks

    plt.suptitle('Flows')
    plt.show()
