"""Analyses dérivées des résultats agrégés de la passe complète (:mod:`coenergy.eda`).

Chaque fonction prend le tableau ``per_sim`` (une ligne par simulation, 1 000 000) ou
les métadonnées ``meta``, et renvoie un tableau ou un dictionnaire prêt à afficher.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from coenergy.eda import TARGET_LABELS

N_STEPS = 480
POWER_W = 7000.0


def inventory_table(meta: dict) -> pd.DataFrame:
    """Forme et type de chaque tenseur, avec vérification de leur constance entre batchs."""
    batches = meta["batches"]
    first = batches[0]
    rows = []
    for name, shape in first["shapes"].items():
        rows.append({
            "champ": name,
            "forme (par batch)": tuple(shape),
            "dtype": first["dtypes"][name],
            "identique dans les 20 batchs": all(
                b["shapes"][name] == shape and b["dtypes"][name] == first["dtypes"][name]
                for b in batches),
        })
    return pd.DataFrame(rows)


def solicitation_table(meta: dict) -> pd.DataFrame:
    """Bornes de chaque champ de ``solicitation`` sur l'ensemble des batchs."""
    batches = meta["batches"]
    names = batches[0]["solicitation_field_names"]
    mins = np.array([b["solicitation_min"] for b in batches]).min(axis=0)
    maxs = np.array([b["solicitation_max"] for b in batches]).max(axis=0)
    return pd.DataFrame({"champ": names, "min": mins, "max": maxs,
                         "constant sur 1M": mins == maxs})


def quality_summary(meta: dict, per_sim: pd.DataFrame) -> pd.Series:
    """Contrôles de qualité et de cohérence globale des séries."""
    batches = meta["batches"]
    u_values = sorted({v for b in batches for v in b["u_unique_values"]})
    return pd.Series({
        "valeurs NaN (T et U)": sum(b["n_nan"]["T"] + b["n_nan"]["U"] for b in batches),
        "valeurs infinies (T et U)": sum(b["n_inf"]["T"] + b["n_inf"]["U"] for b in batches),
        "valeurs distinctes de U": u_values,
        "valeurs de U hors {0, max_power}": int(per_sim["n_off_grid"].sum()),
        "T(0) distinctes": np.unique(per_sim["t0"]).tolist(),
        "T min (°C)": float(per_sim["t_min"].min()),
        "T max (°C)": float(per_sim["t_max"].max()),
        "labels Hb identiques": all(b["hb_labels"] == batches[0]["hb_labels"] for b in batches),
        "labels Hb": batches[0]["hb_labels"],
    })


def time_step_table(per_sim: pd.DataFrame,
                    thresholds_h: Sequence[float] = (0.1, 0.25, 0.5, 1.0)) -> pd.DataFrame:
    """Rapport entre la plus courte série intérieure de U (en pas) et ``t_min`` (en minutes).

    Une série intérieure contient au moins un bloc PRBS, donc dure au moins ``t_min``.
    Si un pas vaut une minute, le rapport minimal doit être proche de 1 (à l'arrondi des
    bornes de bloc près) ; pour un pas de ``k`` minutes, il serait proche de ``1/k``.
    """
    has_run = per_sim["min_run"] > 0
    tmin = per_sim.loc[has_run, "prbs_min_duration_hours"].astype(np.float64)
    ratio = per_sim.loc[has_run, "min_run"] / (tmin * 60.0)
    rows = []
    for th in thresholds_h:
        r = ratio[tmin > th]
        rows.append({"t_min >": f"{th} h", "simulations": len(r),
                     "rapport min": r.min(), "quantile 0,1 %": r.quantile(0.001),
                     "médiane": r.median()})
    return pd.DataFrame(rows)


def shared_draws_evidence(per_sim: pd.DataFrame) -> pd.Series:
    """Indices d'une séquence aléatoire commune à toutes les simulations (``prbs_seed`` = 42).

    Si chaque simulation réutilise les mêmes tirages uniformes, l'état du premier bloc
    vaut « allumé » si et seulement si ``p_on`` dépasse le premier tirage : la frontière
    en ``p_on`` doit être nette, et les deux groupes ne pas se chevaucher.
    """
    starts_on = per_sim["first_on"] == 0
    p_on = per_sim["prbs_probability_on"]
    off = per_sim[per_sim["all_off"]]
    counts = per_sim["u_hash"].value_counts()
    n_batches = per_sim.groupby("u_hash")["batch"].nunique()
    in_all = per_sim["u_hash"].map(n_batches) == per_sim["batch"].nunique()
    return pd.Series({
        "p_on min des simulations démarrant allumées": float(p_on[starts_on].min()),
        "p_on max des simulations démarrant éteintes": float(p_on[~starts_on].max()),
        "U ≡ 0 : p_on max": float(off["prbs_probability_on"].max()),
        "U ≡ 0 : t_min min (h)": float(off["prbs_min_duration_hours"].min()),
        "signaux U distincts": int(counts.size),
        "simulations dont le signal U existe dans les 20 batchs": float(in_all.mean()),
        "effectif médian par signal": float(counts.median()),
        "effectif du signal le plus fréquent": int(counts.iloc[0]),
    })


def target_summary(per_sim: pd.DataFrame) -> pd.DataFrame:
    """Statistiques descriptives des 6 cibles, avec asymétrie brute et en logarithme."""
    y = per_sim[list(TARGET_LABELS)].astype(np.float64)
    out = y.describe(percentiles=[0.01, 0.5, 0.99]).T
    out["asymétrie"] = y.skew()
    out["asymétrie de log"] = np.log(y).skew()
    out["unité"] = ["W/K"] * 3 + ["h"] * 3
    return out


def constraint_residuals(per_sim: pd.DataFrame) -> pd.Series:
    """Résidu de la contrainte HLC = HLC_mit + HLC_ext, en float32 (stockage) et float64."""
    y32 = per_sim[["HLC", "HLC_mit", "HLC_ext"]]
    res32 = y32["HLC"] - (y32["HLC_mit"] + y32["HLC_ext"])
    y64 = y32.astype(np.float64)
    res64 = y64["HLC"] - (y64["HLC_mit"] + y64["HLC_ext"])
    ulp = np.spacing(y32["HLC"].to_numpy())
    return pd.Series({
        "résidu nul en float32 (part)": float((res32 == 0).mean()),
        "|résidu| max en float32, en ulp de HLC": float((res32.abs() / ulp).max()),
        "|résidu| max (W/K)": float(res64.abs().max()),
        "|résidu| relatif max": float((res64 / y64["HLC"]).abs().max()),
    })


def time_constant_capacities(per_sim: pd.DataFrame) -> pd.DataFrame:
    """Capacités apparentes ``C = tau x HLC`` (MJ/K) et leur corrélation avec HLC.

    Pour un système du premier ordre, ``tau = C / H``. Une capacité apparente peu
    corrélée à HLC indique que la corrélation négative entre tau et HLC vient
    essentiellement du facteur ``1 / HLC``.
    """
    y = per_sim[list(TARGET_LABELS)].astype(np.float64)
    rows = []
    for tau in ("tau_b", "tau_n", "tau_inf"):
        c = y[tau] * 3600.0 * y["HLC"] / 1e6
        rows.append({"constante": tau, "C moyenne (MJ/K)": c.mean(),
                     "C écart-type (MJ/K)": c.std(),
                     "corr(C, HLC)": np.corrcoef(c, y["HLC"])[0, 1],
                     "corr(tau, HLC)": np.corrcoef(y[tau], y["HLC"])[0, 1],
                     "corr(tau, 1/HLC)": np.corrcoef(y[tau], 1.0 / y["HLC"])[0, 1]})
    return pd.DataFrame(rows).set_index("constante")


def ordering_checks(per_sim: pd.DataFrame) -> pd.Series:
    """Relations d'ordre et rapports entre constantes de temps."""
    y = per_sim[list(TARGET_LABELS)].astype(np.float64)
    return pd.Series({
        "tau_n < tau_inf (part)": float((y["tau_n"] < y["tau_inf"]).mean()),
        "tau_inf < tau_b (part)": float((y["tau_inf"] < y["tau_b"]).mean()),
        "tau_n / tau_inf : min": float((y["tau_n"] / y["tau_inf"]).min()),
        "tau_n / tau_inf : max": float((y["tau_n"] / y["tau_inf"]).max()),
        "tau_inf / tau_b : min": float((y["tau_inf"] / y["tau_b"]).min()),
        "tau_inf / tau_b : max": float((y["tau_inf"] / y["tau_b"]).max()),
        "part de HLC_mit dans HLC : min": float((y["HLC_mit"] / y["HLC"]).min()),
        "part de HLC_mit dans HLC : max": float((y["HLC_mit"] / y["HLC"]).max()),
    })


def excitation_target_correlation(per_sim: pd.DataFrame) -> pd.DataFrame:
    """Corrélation de Pearson entre paramètres d'excitation et cibles (indépendance attendue)."""
    cols = ["prbs_min_duration_hours", "prbs_probability_on", "duty", "n_switches"]
    data = per_sim[cols + list(TARGET_LABELS)].astype(np.float64)
    return data.corr().loc[cols, list(TARGET_LABELS)]


def batch_homogeneity(per_sim: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Écart de la moyenne de chaque batch à la moyenne globale, en erreurs-types (z-score).

    Si les batchs sont des tirages d'une même distribution, ces z-scores suivent
    approximativement une loi normale centrée réduite : sur 20 batchs, des valeurs
    absolues jusqu'à ~3 sont attendues.
    """
    data = per_sim[["batch", *columns]].astype({c: np.float64 for c in columns})
    grouped = data.groupby("batch")[list(columns)]
    means, counts = grouped.mean(), grouped.count()
    std = data[list(columns)].std()
    return (means - data[list(columns)].mean()) / (std / np.sqrt(counts))


def information_window(per_sim: pd.DataFrame, window_h: float) -> pd.DataFrame:
    """Rapport entre la durée d'observation et chaque constante de temps."""
    rows = []
    for tau in ("tau_b", "tau_n", "tau_inf"):
        ratio = window_h / per_sim[tau].astype(np.float64)
        rows.append({"constante": tau, "médiane de tau (h)": per_sim[tau].median(),
                     "fenêtre / tau : médiane": ratio.median(),
                     "fenêtre / tau : min": ratio.min(), "fenêtre / tau : max": ratio.max(),
                     "part avec tau < fenêtre": float((per_sim[tau] < window_h).mean())})
    return pd.DataFrame(rows).set_index("constante")
