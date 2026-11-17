"""Métriques d'évaluation, en unités physiques (W/K, h, MJ/K).

Les métriques sont calculées après transformation inverse, sur les cibles physiques, et
non dans l'espace standardisé où le modèle est entraîné.

Outre les six cibles de l'énoncé, on évalue les **capacités effectives**
``C = tau x HLC`` (MJ/K), calculées à partir des cibles prédites. L'analyse
d'identifiabilité (:mod:`coenergy.identifiability`) montre qu'un essai de 8 h renseigne
surtout sur la chaleur stockée (capacité) et peu sur la chaleur perdue (HLC) : ces
métriques mesurent ce que le modèle a effectivement appris de la physique de l'essai.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from coenergy.dataset import TARGET_FAMILIES, TARGET_LABELS

UNITS = {"HLC": "W/K", "HLC_mit": "W/K", "HLC_ext": "W/K",
         "tau_b": "h", "tau_n": "h", "tau_inf": "h"}
CAPACITY_LABELS: tuple[str, ...] = ("C_b", "C_n", "C_inf")
"""Capacités effectives associées à ``tau_b``, ``tau_n`` et ``tau_inf``."""


def effective_capacities(y: npt.NDArray) -> npt.NDArray[np.float64]:
    """Capacités effectives ``C = tau x HLC`` en MJ/K, forme ``(n, 3)``.

    Pour un système du premier ordre, ``tau = C / HLC`` ; ``C`` est la capacité thermique
    qui « répond » sur l'échelle de temps de ``tau``.
    """
    y = np.asarray(y, np.float64)
    hlc = y[:, TARGET_LABELS.index("HLC")]
    taus = y[:, [TARGET_LABELS.index(t) for t in ("tau_b", "tau_n", "tau_inf")]]
    return taus * 3600.0 * hlc[:, None] / 1e6


def _metrics(y_true: npt.NDArray, y_pred: npt.NDArray, labels: tuple[str, ...]) -> pd.DataFrame:
    yt, yp = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    err = yp - yt
    abs_err = np.abs(err)
    return pd.DataFrame({
        "RMSE": np.sqrt((err**2).mean(axis=0)),
        "MAE": abs_err.mean(axis=0),
        "P95 |erreur|": np.percentile(abs_err, 95, axis=0),
        "erreur relative (%)": 100 * (abs_err / np.abs(yt)).mean(axis=0),
        "R²": 1 - (err**2).sum(axis=0) / ((yt - yt.mean(axis=0)) ** 2).sum(axis=0),
    }, index=list(labels))


def capacity_metrics(y_true: npt.NDArray, y_pred: npt.NDArray) -> pd.DataFrame:
    """Métriques sur les capacités effectives (MJ/K) déduites des cibles prédites."""
    out = _metrics(effective_capacities(y_true), effective_capacities(y_pred), CAPACITY_LABELS)
    out.insert(0, "unité", "MJ/K")
    return out


def r2_by_group(y_true: npt.NDArray, y_pred: npt.NDArray,
                groups: Mapping[str, npt.NDArray[np.bool_]]) -> pd.DataFrame:
    """R² par cible et par sous-ensemble (par exemple par type d'excitation).

    Le R² de chaque sous-ensemble est calculé par rapport à la variance **globale** des
    cibles, afin que les valeurs soient comparables entre sous-ensembles (un
    sous-ensemble de faible variance n'est pas artificiellement pénalisé).
    """
    yt, yp = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    mean = yt.mean(axis=0)
    rows = {}
    for name, mask in groups.items():
        m = np.asarray(mask, bool)
        if not m.any():
            continue
        r2 = 1 - ((yp[m] - yt[m]) ** 2).sum(axis=0) / ((yt[m] - mean) ** 2).sum(axis=0)
        rows[name] = {**dict(zip(TARGET_LABELS, r2)), "simulations": int(m.sum())}
    return pd.DataFrame(rows).T


def excitation_groups(descriptors: pd.DataFrame) -> dict[str, npt.NDArray[np.bool_]]:
    """Sous-ensembles par type d'excitation, à partir des descripteurs de U de l'EDA.

    Args:
        descriptors: lignes de ``per_sim`` (EDA) alignées sur les prédictions, avec les
            colonnes ``all_off``, ``n_switches`` et ``duty``.
    """
    n_sw, duty, off = descriptors["n_switches"], descriptors["duty"], descriptors["all_off"]
    groups = {
        "U ≡ 0": off,
        "1 commutation": n_sw == 1,
        "2 commutations": n_sw == 2,
        "3 à 10 commutations": n_sw.between(3, 10),
        "plus de 10 commutations": n_sw > 10,
        "taux d'allumage < 0,3 (hors U ≡ 0)": (duty < 0.3) & ~off,
        "taux d'allumage 0,3 à 0,7": duty.between(0.3, 0.7),
        "taux d'allumage > 0,7": duty > 0.7,
    }
    return {k: np.asarray(v, bool) for k, v in groups.items()}


def regression_metrics(y_true: npt.NDArray, y_pred: npt.NDArray) -> pd.DataFrame:
    """Métriques par cible : RMSE, MAE, 95ᵉ percentile de l'erreur absolue, erreur
    relative moyenne (%) et R².

    Args:
        y_true, y_pred: tableaux ``(n, 6)`` dans l'ordre ``TARGET_LABELS``.
    """
    out = _metrics(y_true, y_pred, TARGET_LABELS)
    out.insert(0, "unité", [UNITS[lab] for lab in TARGET_LABELS])
    return out


def family_metrics(per_target: pd.DataFrame) -> pd.DataFrame:
    """Moyenne, par famille de l'énoncé (HLC, Hb, CDT), des métriques sans unité."""
    rows = {family: per_target.loc[list(labels), ["erreur relative (%)", "R²"]].mean()
            for family, labels in TARGET_FAMILIES.items()}
    return pd.DataFrame(rows).T


def physical_checks(y_pred: npt.NDArray) -> pd.Series:
    """Respect des contraintes physiques par les prédictions."""
    y = np.asarray(y_pred, np.float64)
    i = {lab: k for k, lab in enumerate(TARGET_LABELS)}
    residual = y[:, i["HLC"]] - (y[:, i["HLC_mit"]] + y[:, i["HLC_ext"]])
    return pd.Series({
        "|HLC - (HLC_mit + HLC_ext)| max (W/K)": float(np.abs(residual).max()),
        "|résidu| relatif max": float(np.abs(residual / y[:, i["HLC"]]).max()),
        "part avec tau_n < tau_inf < tau_b": float(
            ((y[:, i["tau_n"]] < y[:, i["tau_inf"]]) & (y[:, i["tau_inf"]] < y[:, i["tau_b"]])).mean()),
        "part avec toutes les sorties > 0": float((y > 0).all(axis=1).mean()),
    })
