"""Lecture et mise en forme des résultats d'entraînement (dossier ``runs/``).

Chaque exécution de :mod:`coenergy.train` écrit, dans ``runs/<modèle>/`` : ``metrics.json``,
``history.json``, ``run_info.json``, ``model.pt`` et ``predictions_<jeu>.npz``. Ce module
relit ces fichiers et construit les tableaux de comparaison présentés dans le notebook.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from coenergy.data import PROJECT_ROOT
from coenergy.dataset import TARGET_FAMILIES, TARGET_LABELS
from coenergy.evaluation import CAPACITY_LABELS

RUNS_DIR = PROJECT_ROOT / "runs"
MODEL_ORDER: tuple[str, ...] = ("mean", "ridge", "cnn", "cnn_gru", "cnn_gru_40ep",
                                 "cnn_gru_large", "cnn_free", "cnn_free_20ep", "cnn_hlc")
MODEL_ROLES: dict[str, str] = {
    "mean": "référence : moyenne d'entraînement",
    "ridge": "référence : régression linéaire (ridge)",
    "cnn": "CNN dilaté, tête physique",
    "cnn_gru": "CNN-GRU, tête physique (modèle retenu)",
    "cnn_gru_40ep": "CNN-GRU entraîné deux fois plus longtemps",
    "cnn_gru_large": "CNN-GRU quatre fois plus grand",
    "cnn_free": "ablation sans contraintes, 12 époques (essai initial)",
    "cnn_free_20ep": "ablation : CNN sans contraintes physiques",
    "cnn_hlc": "ablation : CNN entraîné sur le seul HLC",
}
MAIN_MODELS: tuple[str, ...] = ("mean", "ridge", "cnn", "cnn_gru", "cnn_free_20ep", "cnn_hlc")
"""Modèles comparés dans les tableaux principaux (tous entraînés 20 époques)."""
GRU_VARIANTS: tuple[str, ...] = ("cnn_gru", "cnn_gru_40ep", "cnn_gru_large")
NETWORKS: tuple[str, ...] = ("cnn", "cnn_gru", "cnn_gru_40ep", "cnn_gru_large", "cnn_free_20ep")
"""Réseaux multi-sorties, candidats au meilleur score."""
EVAL_SPLITS: tuple[str, ...] = ("val", "test_random", "test_new_u")


@dataclass
class Predictions:
    """Prédictions d'un modèle sur un jeu, alignées par simulation."""

    global_idx: npt.NDArray[np.int64]
    y_true: npt.NDArray[np.float64]
    y_pred: npt.NDArray[np.float64]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def available_models(runs_dir: Path = RUNS_DIR) -> list[str]:
    """Modèles présents dans ``runs_dir``, dans l'ordre de présentation."""
    return [m for m in MODEL_ORDER if (runs_dir / m / "metrics.json").exists()]


def load_metrics(model: str, runs_dir: Path = RUNS_DIR) -> dict:
    """Contenu de ``metrics.json``."""
    return _read_json(runs_dir / model / "metrics.json")


def load_history(model: str, runs_dir: Path = RUNS_DIR) -> dict:
    """Contenu de ``history.json``."""
    return _read_json(runs_dir / model / "history.json")


def load_run_info(model: str, runs_dir: Path = RUNS_DIR) -> dict:
    """Contenu de ``run_info.json`` (vide s'il est absent)."""
    return _read_json(runs_dir / model / "run_info.json")


def load_predictions(model: str, split: str, runs_dir: Path = RUNS_DIR) -> Predictions:
    """Prédictions d'un modèle sur un jeu (``val``, ``test_random`` ou ``test_new_u``)."""
    with np.load(runs_dir / model / f"predictions_{split}.npz") as f:
        return Predictions(f["global_idx"], f["y_true"].astype(np.float64),
                           f["y_pred"].astype(np.float64))


# --------------------------------------------------------------------------- tableaux
def metric_table(split: str, metric: str = "R²", models: list[str] | None = None,
                 with_capacities: bool = True, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Une métrique pour chaque modèle (lignes) et chaque cible (colonnes).

    Pour un modèle mono-cible, les cibles non entraînées sont masquées (NaN).
    """
    rows = {}
    for m in models or available_models(runs_dir):
        met = load_metrics(m, runs_dir)[split]
        row = {lab: met["per_target"][lab][metric] for lab in TARGET_LABELS}
        if with_capacities:
            row |= {c: met["capacities"][c][metric] for c in CAPACITY_LABELS}
        trained = load_run_info(m, runs_dir).get("trained_targets")
        if trained and len(trained) < len(TARGET_LABELS):
            row = {k: (v if k in trained else np.nan) for k, v in row.items()}
        rows[m] = row
    return pd.DataFrame(rows).T


def family_table(split: str, models: list[str] | None = None,
                 runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """R² moyen et erreur relative moyenne par famille de l'énoncé (HLC, Hb, CDT)."""
    models = models or available_models(runs_dir)
    columns = pd.MultiIndex.from_product([list(TARGET_FAMILIES), ["R²", "erreur relative (%)"]])
    rows = []
    for m in models:
        fam = load_metrics(m, runs_dir)[split]["per_family"]
        rows.append([fam[f][k] for f, k in columns])
    return pd.DataFrame(rows, index=models, columns=columns)


def physical_checks_table(split: str, models: list[str] | None = None,
                          runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Contrôles physiques des prédictions de chaque modèle."""
    return pd.DataFrame({m: load_metrics(m, runs_dir)[split]["physical_checks"]
                         for m in models or available_models(runs_dir)}).T


def training_table(models: list[str] | None = None, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Rôle, taille, durée et convergence de chaque entraînement."""
    rows = {}
    for m in models or available_models(runs_dir):
        info, hist = load_run_info(m, runs_dir), load_history(m, runs_dir)
        epochs = hist.get("epoch", [])
        rows[m] = {
            "rôle": MODEL_ROLES.get(m, ""),
            "paramètres": info.get("n_parameters"),
            "époques": len(epochs) if epochs else None,
            "meilleure époque": hist.get("best_epoch"),
            "durée d'ajustement (min)": round(info["fit_seconds"] / 60, 1) if info else None,
            "GPU": info.get("gpu"),
        }
    return pd.DataFrame(rows).T


def best_scores_table(split: str = "test_random", models: tuple[str, ...] = NETWORKS,
                      runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Pour chaque cible, le modèle qui obtient la plus faible erreur absolue moyenne,
    avec ses scores (MAE, erreur relative, P95, R²)."""
    rows = {}
    for lab in TARGET_LABELS:
        per_model = {m: load_metrics(m, runs_dir)[split]["per_target"][lab]
                     for m in models if (runs_dir / m / "metrics.json").exists()}
        best = min(per_model, key=lambda m: per_model[m]["MAE"])
        rows[lab] = {"meilleur modèle": best, **{k: per_model[best][k] for k in
                     ("MAE", "erreur relative (%)", "P95 |erreur|", "R²")}}
    return pd.DataFrame(rows).T


def training_gap_table(models: tuple[str, ...] = GRU_VARIANTS,
                       runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Pertes d'entraînement et de validation à la meilleure époque, et leur écart."""
    rows = {}
    for m in models:
        h, info = load_history(m, runs_dir), load_run_info(m, runs_dir)
        k = h["best_epoch"] - 1
        rows[m] = {"paramètres": info.get("n_parameters"), "époques": len(h["epoch"]),
                   "meilleure époque": h["best_epoch"],
                   "durée (min)": round(info["fit_seconds"] / 60, 1),
                   "perte d'entraînement": h["train_loss"][k], "perte de validation": h["val_loss"][k],
                   "écart validation − entraînement": h["val_loss"][k] - h["train_loss"][k]}
    out = pd.DataFrame(rows).T
    counts = ["paramètres", "époques", "meilleure époque"]
    out[counts] = out[counts].astype(int)
    return out


def error_by_true_value(pred: Predictions, target: str = "HLC", n_bins: int = 20) -> pd.DataFrame:
    """Erreur signée (prédit − vrai) selon la valeur vraie, par tranches de même effectif.

    Une erreur moyenne positive pour les petites valeurs et négative pour les grandes est
    la signature du resserrement des prédictions vers la moyenne.
    """
    j = TARGET_LABELS.index(target)
    yt, err = pred.y_true[:, j], pred.y_pred[:, j] - pred.y_true[:, j]
    edges = np.quantile(yt, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, yt, side="right") - 1, 0, n_bins - 1)
    rows = [{"valeur vraie (centre)": yt[idx == b].mean(), "erreur moyenne": err[idx == b].mean(),
             "écart-type de l'erreur": err[idx == b].std(), "simulations": int((idx == b).sum())}
            for b in range(n_bins)]
    return pd.DataFrame(rows)


def error_by_signal(model: str, descriptors: pd.DataFrame, min_count: int = 300,
                    target: str = "HLC", runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Erreur relative moyenne par signal U, sur la validation et les deux tests réunis.

    Args:
        descriptors: tableau ``per_sim`` de l'EDA (colonnes ``u_hash``, ``n_switches``,
            ``duty``), indexé par l'index global.
        min_count: effectif minimal d'un signal pour être classé.

    Returns:
        Une ligne par signal, triée de la plus faible à la plus forte erreur, avec son rang.
    """
    j = TARGET_LABELS.index(target)
    parts = []
    for split in EVAL_SPLITS:
        p = load_predictions(model, split, runs_dir)
        parts.append(pd.DataFrame({
            "u_hash": descriptors.loc[p.global_idx, "u_hash"].to_numpy(),
            "erreur relative (%)": 100 * np.abs(p.y_pred[:, j] - p.y_true[:, j]) / p.y_true[:, j]}))
    table = pd.concat(parts).groupby("u_hash").agg(
        simulations=("erreur relative (%)", "size"), **{"erreur relative (%)": ("erreur relative (%)", "mean")})
    table = table[table["simulations"] >= min_count].sort_values("erreur relative (%)")
    first = descriptors.drop_duplicates("u_hash").set_index("u_hash")
    table["commutations"] = first.loc[table.index, "n_switches"].to_numpy()
    table["taux d'allumage"] = first.loc[table.index, "duty"].to_numpy()
    table["rang"] = np.arange(1, len(table) + 1)
    return table


def shrinkage_table(pred: Predictions) -> pd.DataFrame:
    """Resserrement des prédictions vers la moyenne, par cible.

    - pente de la régression ``y_pred ~ y_true`` (1 : aucun resserrement ; 0 : le
      modèle prédit toujours la moyenne) ;
    - rapport des écarts-types prédit / vrai.

    Pour un prédicteur qui minimise l'erreur quadratique, la pente est proche du R² :
    moins l'information est disponible, plus la prédiction se rapproche de la moyenne.
    """
    yt, yp = pred.y_true, pred.y_pred
    slope = [np.polyfit(yt[:, i], yp[:, i], 1)[0] for i in range(yt.shape[1])]
    return pd.DataFrame({"pente prédit ~ vrai": slope,
                         "écart-type prédit / vrai": yp.std(axis=0) / yt.std(axis=0)},
                        index=list(TARGET_LABELS)).T


def worst_cases_profile(pred: Predictions, descriptors: pd.DataFrame, target: str = "HLC",
                        quantile: float = 0.99) -> pd.DataFrame:
    """Compare les simulations aux plus fortes erreurs relatives à l'ensemble.

    Args:
        descriptors: lignes de ``per_sim`` (EDA) alignées sur ``pred``.
        quantile: seuil d'erreur relative définissant les « pires cas ».
    """
    j = TARGET_LABELS.index(target)
    rel = np.abs(pred.y_pred[:, j] - pred.y_true[:, j]) / pred.y_true[:, j]
    worst = rel >= np.quantile(rel, quantile)
    d = descriptors.reset_index(drop=True)

    def profile(mask: npt.NDArray[np.bool_]) -> dict:
        return {
            "simulations": int(mask.sum()),
            "erreur relative moyenne (%)": 100 * rel[mask].mean(),
            f"{target} vrai moyen": pred.y_true[mask, j].mean(),
            f"{target} prédit moyen": pred.y_pred[mask, j].mean(),
            f"part avec {target} vrai < 180 ou > 290": float(
                ((pred.y_true[mask, j] < 180) | (pred.y_true[mask, j] > 290)).mean()),
            "part U ≡ 0": float(d.loc[mask, "all_off"].mean()),
            "taux d'allumage moyen": float(d.loc[mask, "duty"].mean()),
            "commutations (médiane)": float(d.loc[mask, "n_switches"].median()),
        }

    return pd.DataFrame({f"{int(100 * (1 - quantile))} % pires": profile(worst),
                         "toutes": profile(np.ones_like(worst))}).T
