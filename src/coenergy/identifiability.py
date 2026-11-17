"""Analyse d'identifiabilité : que peut-on, au mieux, déduire de 8 h de T_int ?

Ces analyses estiment le **plafond de performance** fixé par les données elles-mêmes,
indépendamment du modèle. Elles se placent dans le cadre le plus favorable, celui de
l'article de coEnergy : **une excitation U fixée**. Les milliers de simulations qui
partagent exactement le même signal U (EDA, section 2.3) ne diffèrent que par le
bâtiment et son état initial.

Trois questions :

1. **Problème direct** — quelle part de la variance de T(t) les six cibles expliquent-elles ?
   Si elle est proche de 1, T est bien déterminée par le bâtiment.
2. **Problème inverse** — avec quel R² peut-on prédire chaque cible à partir de T ?
   C'est le plafond empirique, à comparer aux performances du modèle.
3. **Corrélations canoniques** — combien de combinaisons indépendantes des paramètres
   du bâtiment T permet-elle de lire, et lesquelles ? Des corrélations canoniques qui
   décroissent vite indiquent que plusieurs bâtiments différents produisent la même
   courbe : les cibles ne sont pas identifiables séparément.

Les données utiles (simulations des signaux U non nuls les plus fréquents) sont
extraites en un passage sur les batchs puis mises en cache dans ``outputs/identifiability``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from coenergy.data import BATCH_NUMBERS, DATA_DIR, PROJECT_ROOT, iter_batches
from coenergy.dataset import SIMS_PER_BATCH, TARGET_LABELS, batch_targets
from coenergy.evaluation import CAPACITY_LABELS, effective_capacities

OUT_DIR = PROJECT_ROOT / "outputs" / "identifiability"
PARAMETER_LABELS: tuple[str, ...] = ("log HLC_mit", "log HLC_ext", "log tau_b", "log tau_n",
                                     "log tau_inf")
"""Cinq paramètres indépendants : HLC n'en est pas un, puisque HLC = HLC_mit + HLC_ext."""


@dataclass
class FixedExcitationGroup:
    """Toutes les simulations qui partagent un même signal U.

    Attributes:
        t: ``T - T(0)`` en K, après l'arrondi de l'énoncé, forme ``(n, 480)``.
        y: cibles physiques ``(n, 6)`` dans l'ordre ``TARGET_LABELS``.
        global_idx: index global des simulations ``(n,)``.
        u: le signal U commun (W), forme ``(480,)``.
    """

    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    global_idx: npt.NDArray[np.int64]
    u: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.y)


# --------------------------------------------------------------------------- données
def collect_fixed_excitation(per_sim: pd.DataFrame, n_patterns: int = 2,
                             data_dir: Path = DATA_DIR) -> dict[str, FixedExcitationGroup]:
    """Extrait, dans les 20 batchs, les simulations des ``n_patterns`` signaux U non nuls
    les plus fréquents (un passage, un batch en mémoire à la fois).

    Args:
        per_sim: tableau de l'EDA (colonnes ``u_hash``, ``all_off``), trié par index global.
    """
    counts = per_sim.loc[~per_sim["all_off"], "u_hash"].value_counts()
    hashes = counts.index[:n_patterns].to_numpy(np.uint64)
    u_hash = per_sim["u_hash"].to_numpy(np.uint64).reshape(-1, SIMS_PER_BATCH)
    parts: dict[str, dict[str, list]] = {f"pattern_{k + 1}": {"t": [], "y": [], "gi": [], "u": None}
                                         for k in range(n_patterns)}
    for b, batch in iter_batches(BATCH_NUMBERS, data_dir):
        for k, h in enumerate(hashes):
            rows = np.nonzero(u_hash[b - 1] == h)[0]
            if rows.size == 0:
                continue
            idx = torch.from_numpy(rows)
            t = batch.indoor_temperature[idx].double().round(decimals=2)
            p = parts[f"pattern_{k + 1}"]
            p["t"].append((t - t[:, :1]).numpy())
            p["y"].append(batch_targets(batch)[idx].double().numpy())
            p["gi"].append(rows + (b - 1) * SIMS_PER_BATCH)
            if p["u"] is None:
                p["u"] = batch.indoor_U[idx[0]].double().numpy()
    return {name: FixedExcitationGroup(np.concatenate(p["t"]), np.concatenate(p["y"]),
                                       np.concatenate(p["gi"]), p["u"])
            for name, p in parts.items()}


def ensure_fixed_excitation(per_sim: pd.DataFrame, n_patterns: int = 2,
                            data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR,
                            rerun: bool = False) -> dict[str, FixedExcitationGroup]:
    """Relit le cache des groupes à excitation fixée, ou le construit s'il est absent."""
    path = out_dir / f"fixed_excitation_{n_patterns}.npz"
    if path.exists() and not rerun:
        with np.load(path) as f:
            names = sorted({k.split("__")[0] for k in f.files})
            return {n: FixedExcitationGroup(f[f"{n}__t"], f[f"{n}__y"], f[f"{n}__gi"], f[f"{n}__u"])
                    for n in names}
    groups = collect_fixed_excitation(per_sim, n_patterns, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{n}__{k}": getattr(g, a) for n, g in groups.items()
                                 for k, a in (("t", "t"), ("y", "y"), ("gi", "global_idx"),
                                              ("u", "u"))})
    return groups


# --------------------------------------------------------------------------- outils
def physical_parameters(y: npt.NDArray) -> npt.NDArray[np.float64]:
    """Les cinq paramètres indépendants ``PARAMETER_LABELS`` (logarithmes)."""
    y = np.asarray(y, np.float64)
    cols = ["HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf"]
    return np.log(y[:, [TARGET_LABELS.index(c) for c in cols]])


def inverse_targets(y: npt.NDArray) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Grandeurs évaluées dans le problème inverse : les 6 cibles (log pour les tau) et
    les log des 3 capacités effectives."""
    y = np.asarray(y, np.float64)
    g = np.c_[y[:, :3], np.log(y[:, 3:]), np.log(effective_capacities(y))]
    labels = [*TARGET_LABELS[:3], *(f"log {t}" for t in TARGET_LABELS[3:]),
              *(f"log {c}" for c in CAPACITY_LABELS)]
    return g, labels


def r2_score(y: npt.NDArray, p: npt.NDArray) -> npt.NDArray[np.float64]:
    """R² par colonne."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    return 1 - ((p - y) ** 2).sum(axis=0) / ((y - y.mean(axis=0)) ** 2).sum(axis=0)


def split_indices(n: int, test_fraction: float = 0.2,
                  seed: int = 0) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Partition aléatoire (graine fixe) en indices d'ajustement et de test."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_fraction * n))
    return perm[n_test:], perm[:n_test]


def fit_mlp(x: npt.NDArray, y: npt.NDArray, x_eval: npt.NDArray, hidden: int = 512,
            epochs: int = 50, lr: float = 1e-3, batch_size: int = 256,
            seed: int = 0) -> npt.NDArray[np.float64]:
    """Régresseur MLP (3 couches cachées, PyTorch) ; entrées et sorties standardisées en
    interne. Renvoie les prédictions sur ``x_eval`` dans l'échelle de ``y``."""
    torch.manual_seed(seed)
    xt, xe, yt = (torch.as_tensor(a, dtype=torch.float32) for a in (x, x_eval, y))
    mx, sx = xt.mean(0), xt.std(0) + 1e-6
    my, sy = yt.mean(0), yt.std(0) + 1e-12
    xt, xe, zt = (xt - mx) / sx, (xe - mx) / sx, (yt - my) / sy
    net = nn.Sequential(nn.Linear(xt.shape[1], hidden), nn.GELU(), nn.Linear(hidden, hidden),
                        nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, yt.shape[1]))
    steps = max(1, len(xt) // batch_size)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(steps):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            loss = F.mse_loss(net(xt[idx]), zt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return (net(xe) * sy + my).double().numpy()


# --------------------------------------------------------------------------- analyses
def forward_explained_variance(group: FixedExcitationGroup, seed: int = 0) -> pd.Series:
    """Problème direct : part de la variance de T(t) expliquée par les six cibles.

    Un MLP prédit la courbe entière ``T(t)`` à partir des cibles ; la part expliquée est
    calculée sur un jeu de test. L'écart-type résiduel se compare à la dispersion de la
    réponse libre (≈ 0,2 K, EDA section 3.4), attribuée à l'état initial des parois.
    """
    fit, test = split_indices(len(group), seed=seed)
    pred = fit_mlp(group.y[fit], group.t[fit], group.y[test], seed=seed)
    t = group.t[test]
    explained = 1 - ((pred - t) ** 2).sum() / ((t - t.mean(axis=0)) ** 2).sum()
    return pd.Series({
        "part de la variance de T expliquée par les cibles": explained,
        "écart-type résiduel moyen (K)": float(np.sqrt(((pred - t) ** 2).mean(axis=0)).mean()),
        "écart-type de T entre bâtiments (K)": float(t.std(axis=0).mean()),
        "simulations": len(group),
    })


def inverse_r2(group: FixedExcitationGroup, seed: int = 0) -> pd.Series:
    """Problème inverse : R² de la prédiction des cibles (et des capacités) à partir de T.

    Plafond empirique : excitation fixée, ~11 700 simulations d'ajustement, MLP. Un
    modèle unique, entraîné sur des excitations mélangées, ne peut pas espérer faire
    mieux en moyenne.
    """
    g, labels = inverse_targets(group.y)
    fit, test = split_indices(len(group), seed=seed)
    pred = fit_mlp(group.t[fit], g[fit], group.t[test], seed=seed)
    return pd.Series(r2_score(g[test], pred), index=labels)


def canonical_correlations(group: FixedExcitationGroup, n_components: int = 20,
                           seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Corrélations canoniques entre T(t) et les cinq paramètres indépendants du bâtiment.

    T est d'abord réduite à ``n_components`` composantes principales (calculées sur le
    jeu d'ajustement) ; les directions canoniques sont estimées sur le jeu d'ajustement et
    les corrélations mesurées sur le jeu de test (pas de sur-ajustement).

    Returns:
        Les corrélations canoniques (test), et pour chaque direction les poids des
        paramètres standardisés qui composent la combinaison lue dans T (normalisés à 1).
    """
    params = physical_parameters(group.y)
    fit, test = split_indices(len(group), seed=seed)
    t_mean = group.t[fit].mean(axis=0)
    _, _, vt = np.linalg.svd(group.t[fit] - t_mean, full_matrices=False)
    scores = (group.t - t_mean) @ vt[:n_components].T

    def whitener(a: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        mu = a[fit].mean(axis=0)
        w, v = np.linalg.eigh(np.cov(a[fit] - mu, rowvar=False))
        return mu, v @ np.diag(w**-0.5) @ v.T

    mu_t, w_t = whitener(scores)
    mu_p, w_p = whitener(params)
    st, sp = (scores - mu_t) @ w_t, (params - mu_p) @ w_p
    left, _, right_t = np.linalg.svd(st[fit].T @ sp[fit] / (len(fit) - 1))
    k = params.shape[1]
    a, b = st @ left[:, :k], sp @ right_t.T[:, :k]
    corr = [float(np.corrcoef(a[test, i], b[test, i])[0, 1]) for i in range(k)]
    z = (params - params[fit].mean(axis=0)) / params[fit].std(axis=0)
    weights = []
    for i in range(k):
        coef = np.linalg.lstsq(z[fit], b[fit, i], rcond=None)[0]
        coef = coef / np.abs(coef).max()
        weights.append(coef * np.sign(coef[np.argmax(np.abs(coef))]))
    names = [f"direction {i + 1}" for i in range(k)]
    return (pd.Series(corr, index=names, name="corrélation canonique (test)"),
            pd.DataFrame(weights, index=names, columns=PARAMETER_LABELS))
