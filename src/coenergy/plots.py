"""Style et figures réutilisables de l'analyse exploratoire (matplotlib).

Conventions : un seul axe des ordonnées par panneau, au plus trois couleurs
catégorielles par panneau, rampe bleue séquentielle pour les grandeurs continues.
"""

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
"""Trois premières couleurs catégorielles (bleu, orange, aqua)."""
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues_ref", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style() -> None:
    """Applique le style commun à toutes les figures."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "figure.dpi": 100,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def plot_series_pairs(t: npt.NDArray, u: npt.NDArray, labels: Sequence[str],
                      dt_min: float = 1.0) -> Figure:
    """Petits multiples : T_int (gauche) et U (droite), une ligne par simulation."""
    n = len(labels)
    hours = np.arange(t.shape[1]) * dt_min / 60.0
    fig, axes = plt.subplots(n, 2, figsize=(12, 1.9 * n), sharex=True)
    axes = np.atleast_2d(axes)
    for row in range(n):
        axes[row, 0].plot(hours, t[row], color=SERIES[1])
        axes[row, 0].set_ylabel("T_int (°C)")
        axes[row, 0].set_title(labels[row], loc="left", fontsize=9, color=TEXT_SECONDARY)
        axes[row, 1].step(hours, u[row] / 1000.0, where="post", color=SERIES[0])
        axes[row, 1].set_ylabel("U (kW)")
        axes[row, 1].set_ylim(-0.4, 7.6)
    axes[-1, 0].set_xlabel("temps (h)")
    axes[-1, 1].set_xlabel("temps (h)")
    fig.tight_layout()
    return fig


def plot_band(ax: Axes, x: npt.NDArray, mean: npt.NDArray, std: npt.NDArray,
              color: str, label: str) -> None:
    """Moyenne et bande ±1 écart-type."""
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, label=label)


def plot_curves_by_value(ax: Axes, x: npt.NDArray, curves: npt.NDArray,
                         values: npt.NDArray, label: str) -> None:
    """Courbes colorées par une grandeur continue (rampe séquentielle), avec barre de couleur.

    La barre est placée dans un encart à droite de l'axe, sans réduire sa largeur, afin que
    les axes temporels restent alignés avec les autres panneaux de la figure.
    """
    norm = mpl.colors.Normalize(vmin=np.quantile(values, 0.02), vmax=np.quantile(values, 0.98))
    for k in np.argsort(values):
        ax.plot(x, curves[k], color=BLUES(norm(values[k])), linewidth=0.7, alpha=0.8)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=BLUES)
    cax = ax.inset_axes((1.01, 0.0, 0.015, 1.0))
    ax.figure.colorbar(sm, cax=cax, label=label)
