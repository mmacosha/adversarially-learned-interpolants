import numpy as np


def x(t, std):
    t = 3 * (t / t.max()) - 1.5
    size = t.shape[0]
    assert size % 3 == 0

    t1, t2, t3 = t[:size // 3], t[size // 3:-size // 3], t[-size // 3:]

    x1 = 3 * (t1 + 0.5)
    x2 = np.cos(2 * np.pi * (t2 - 0.75))
    x3 = 3 * (t3 - 0.5)

    x = np.concatenate([x1, x2, x3])
    return x.reshape((-1, 1)) + np.random.randn(x.shape[0], 10) * std
    # return x + np.random.randn(*x.shape) * std


def y(t, std):
    t = 3 * (t / t.max()) - 1.5
    size = t.shape[0]
    assert size % 3 == 0

    t1, t2, t3 = t[:size // 3], t[size // 3:-size // 3], t[-size // 3:]
    y1 = - np.tanh(5 * (t1 + 1)) / 2 + 0.5
    y2 = np.sin(2 * np.pi * (t2 - 0.75)) + 1
    y3 = np.tanh(5 * (t3 - 1)) / 2 + 0.5

    y = np.concatenate([y1, y2, y3])
    return y.reshape((-1, 1)) + np.random.randn(y.shape[0], 10) * std
    # return y + np.random.randn(*y.shape) * std


def loop_distribution(size, std):
    assert size % 3 == 0
    t = np.linspace(0, 3, size)
    xt = np.stack([x(t, std), y(t, std)]).T
    # x0 = np.random.randn(2, size, 10).T * std + np.array([-3, 1])
    # x1 = np.random.randn(2, size, 10).T * std + np.array([3, 1])
    x0 = xt[:, 0]
    x1 = xt[:, -1]
    return x0, xt[:, 1:-1], x1, t / 3