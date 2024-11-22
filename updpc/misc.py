import matplotlib.pyplot as plt
import numpy as np


def plot_xy(x, y, fig=None, ax=None, color=None):
    """Plot (x,y) with trajectory of gray line, and set aspect as equal."""
    if ax is None:
        fig, ax = plt.subplots()
    if color is None:
        ax.plot(x, y, ".-")
    else:
        ax.plot(x, y, color="gray")
        sc = ax.scatter(x, y, c=color, cmap="jet")
        fig.colorbar(sc, ax=ax)
    if (y.max() - y.min()) / (x.max() - x.min()) < 10:
        ymid = (y.max() + y.min()) / 2
        dx = x.max() - x.min()
        plt.ylim([ymid - dx / 10, ymid + dx / 10])
    ax.set_aspect("equal")
    return fig, ax


def plot_xy_slope(x, y, slope, fig=None, ax=None, color=None):
    """Plot (x,y) with trajectory of gray line, draw line with the slope, and set aspect as equal."""
    if ax is None:
        fig, ax = plt.subplots()

    def npmean(a):
        return np.mean(a, axis=0)

    meanx = npmean(x)
    meany = npmean(y)
    fig, ax = plot_xy(x, y, fig, ax, color=color)
    xs = np.array([np.min(x), np.max(x)])
    ys = slope * (xs - meanx) + meany
    if (y.max() - y.min()) * 10 < (x.max() - x.min()):
        ymid = (y.max() + y.min()) / 2
        dx = x.max() - x.min()
        plt.ylim([ymid - dx / 10, ymid + dx / 10])
    ax.plot(xs, ys, color="gray", linewidth=5, alpha=0.5, label="fitting")
    return fig, ax


def best_line_projection(y, x, ax=None, color=None, return_slope=False):
    def npmean(a):
        return np.mean(a, axis=0)

    meanx = npmean(x)
    meany = npmean(y)
    dx = x - meanx
    dy = y - meany
    Sxx = npmean(dx**2)
    Sxy = npmean(dx * dy)
    Syy = npmean(dy**2)
    dS = Syy - Sxx
    if Sxy == 0:
        a = 0
    else:
        a = (dS + np.sqrt(dS**2 + 4 * Sxy**2)) / 2 / Sxy

    if ax is not None:
        fig = ax.get_figure()
        fig, ax = plot_xy_slope(x, y, a, fig, ax, color=color)
    if return_slope:
        return (x + a * y) / np.sqrt(1 + a**2), a
    return (x + a * y) / np.sqrt(1 + a**2)


def best_line_coord(y, x, return_slope=False):
    def npmean(a):
        return np.mean(a, axis=0)

    meanx = npmean(x)
    meany = npmean(y)
    dx = x - meanx
    dy = y - meany
    Sxx = npmean(dx**2)
    Sxy = npmean(dx * dy)
    Syy = npmean(dy**2)
    dS = Syy - Sxx
    if Sxy == 0:
        a = 0
    else:
        a = (dS + np.sqrt(dS**2 + 4 * Sxy**2)) / 2 / Sxy
    denom = np.sqrt(1 + a**2)
    if return_slope:
        return (x + a * y) / denom, (y - a * x) / denom, a
    return (x + a * y) / denom, (y - a * x) / denom
