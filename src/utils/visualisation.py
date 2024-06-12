import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

from utils import COLORMAP, MARKERS, METHOD_NAMES

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15


def plot_accuracy_fairness_tradeoff(ax, df, x, y, style, config_cols, empty_line= False, order=None, x_range=None,
                                    n_std=1.0):
    group_cols = [col for col in config_cols if col != 'seed']
    grouped_df = df.groupby(group_cols, dropna=False)
    for config_vals, point_vals in grouped_df:
        if len(point_vals) == 1:
            continue
        method = config_vals[group_cols.index('method/name')]
        color = COLORMAP[method]
        if (x_range is None
                or (np.mean(point_vals[x].values) >= x_range[0]) and (np.mean(point_vals[x].values) <= x_range[1])):
            plot_confidence_ellipse(point_vals[x].values, point_vals[y].values, ax, facecolor=color,
                                    n_std=n_std, alpha=.2)
    df = grouped_df[[x, y]].mean(numeric_only=True).reset_index()

    if empty_line:
        sns.lineplot(data=df, x=x, y=y, hue='method/name', style=style, palette=COLORMAP, markers=MARKERS,
                     markerfacecolor='white', markeredgecolor="black")
    else:
        sns.lineplot(data=df, x=x, y=y, hue='method/name', style=style, palette=COLORMAP, markers=MARKERS)


def plot_confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)

    # IMPORTANT: we care about the sample mean, not the individual observations. Hence, divide by n
    cov /= x.shape[0]

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_legend(methods):
    x = np.linspace(1, 100, 10)
    y = x
    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot")
    ax = fig.add_subplot(141)
    lines = []
    for method in methods:
        line1 = ax.scatter(x, y, color=COLORMAP[method], marker=MARKERS[method], edgecolor='black', linewidth=0.5)
        lines.append(line1)
    methods_renamed = [METHOD_NAMES[method] for method in methods]
    legendFig.legend(lines, methods_renamed, loc='upper center', ncol=4, edgecolor="k",
                     fancybox=False, columnspacing=0.8, handletextpad=0.4)
    legendFig.tight_layout()
    return legendFig


def save_fig(fig, name):
    if not name.endswith(".pdf"):
        name += ".pdf"
    fig.savefig(os.path.join("manuscripts\\NeurIPS 2024\\Figs", name), format="pdf", bbox_inches='tight')

