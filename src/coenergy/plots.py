"""Style et figures réutilisables de l'analyse exploratoire (matplotlib).

Conventions : un seul axe des ordonnées par panneau, au plus trois couleurs
catégorielles par panneau, rampe bleue séquentielle pour les grandeurs continues.
"""

from collections.abc import Sequence

import pandas as pd
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


MUTED_GREY = "#b9b8b3"
TRACK_GREY = "#d9d8d4"


def figure_title(fig: Figure, main: str, subtitle: str) -> None:
    """Titre qui énonce le message de la figure, et sous-titre qui décrit ce qu'elle montre."""
    fig.text(0.01, 0.985, main, fontsize=13, fontweight="bold", color=TEXT_PRIMARY, va="top")
    fig.text(0.01, 0.925, subtitle, fontsize=9.5, color=TEXT_SECONDARY, va="top")


def plot_gain_dumbbell(errors: pd.DataFrame, baseline: str, intermediate: str, retained: str,
                       names: dict[str, str], labels: dict[str, str] | None = None,
                       unit: str = "%") -> Figure:
    """Haltères : pour chaque cible, l'erreur de la référence, d'un modèle intermédiaire et
    du modèle retenu, triées par gain décroissant (le plus fort gain en haut).

    Args:
        errors: une ligne par modèle, une colonne par cible.
        baseline, intermediate, retained: noms des lignes à tracer (gris, orange, bleu).
        names: libellé affiché de chaque modèle.
        labels: libellé affiché de chaque cible (par défaut, le nom de la colonne).
    """
    labels = labels or {c: c for c in errors.columns}
    gain = errors.loc[baseline] - errors.loc[retained]
    order = gain.sort_values().index
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.80, left=0.10, right=0.97)
    for yi, target in enumerate(order):
        a, b, c = (errors.loc[m, target] for m in (baseline, intermediate, retained))
        ax.plot([c, a], [yi, yi], color=TRACK_GREY, linewidth=6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [yi], s=70, color=MUTED_GREY, zorder=3)
        ax.scatter([b], [yi], s=70, color=SERIES[1], zorder=3)
        ax.scatter([c], [yi], s=110, color=SERIES[0], zorder=4, edgecolor="white", linewidth=1.5)
        ax.text(c - 0.35, yi, f"{c:.1f} {unit}", ha="right", va="center", fontsize=9,
                color=TEXT_PRIMARY, fontweight="bold")
        ax.text(a + 0.35, yi, f"−{a - c:.1f} pt", ha="left", va="center", fontsize=9,
                color=TEXT_SECONDARY)
    ax.set_yticks(range(len(order)), [labels[t] for t in order], fontsize=11)
    lo, hi = errors.loc[retained].min(), errors.loc[baseline].max()
    ax.set_xlim(lo - 2.8, hi + 2.3)
    ax.grid(axis="y", visible=False)
    for x, model, color in [(0.62, baseline, MUTED_GREY), (0.79, intermediate, SERIES[1]),
                            (0.88, retained, SERIES[0])]:
        fig.text(x, 0.86, "●", color=color, fontsize=12, va="center")
        fig.text(x + 0.018, 0.86, names[model], fontsize=9, color=TEXT_SECONDARY, va="center")
    return fig


def plot_pred_vs_true(ax: Axes, y_true: npt.NDArray, y_pred: npt.NDArray, title: str,
                      unit: str) -> None:
    """Densité des couples (vrai, prédit) en hexbin, avec la diagonale y = x."""
    lo, hi = np.quantile(y_true, [0.002, 0.998])
    ax.hexbin(y_true, y_pred, gridsize=60, bins="log", cmap=BLUES, mincnt=1,
              extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], color=SERIES[1], linewidth=1.2, label="prédiction parfaite")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"vrai ({unit})")
    ax.set_ylabel(f"prédit ({unit})")
    ax.set_title(title)
    ax.grid(False)


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
