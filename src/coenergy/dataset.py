"""Découpages entraînement / validation / test, chargement et prétraitement.

Protocole (décidé à l'issue de l'EDA, ``notebooks/01_eda.ipynb``) :

- **test_new_u** : toutes les simulations, dans les 20 batchs, qui portent l'un des
  signaux U réservés (5 % des signaux distincts, tirés avec une graine fixe, hors signal
  U ≡ 0). Ces excitations n'apparaissent dans aucun autre jeu : ce test mesure la
  généralisation à des excitations jamais vues.
- **train** (batchs 1–16), **val** (17–18), **test_random** (19–20) : toutes les autres
  simulations, réparties par batch entier (les batchs sont interchangeables et chaque
  simulation est un bâtiment distinct).

Le découpage est calculé une fois à partir du cache de l'EDA puis figé dans
``splits/split_seed42.npz`` (un label par simulation, indexé par l'index global
``(batch - 1) * 50 000 + ligne``). L'entraînement relit ce fichier : aucun recalcul.

Prétraitement (identique pour tous les jeux) :

- arrondi de l'énoncé : T à 0,01 °C, U à 0,1 W ;
- entrées : ``(T - T(0)) / 10 K`` et ``U / 7000 W``, sans décalage temporel ;
- cibles : ordre officiel ``[HLC, *Hb_labels, *CDT]`` lu depuis le fichier ;
  standardisation (log pour les constantes de temps) ajustée sur l'entraînement seul.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from coenergy.data import BATCH_NUMBERS, DATA_DIR, PROJECT_ROOT, iter_batches

SIMS_PER_BATCH = 50_000
N_STEPS = 480
POWER_W = 7000.0
"""Puissance maximale de l'excitation (``max_power_wat``, constante sur le jeu)."""
T_SCALE = 10.0
"""Échelle (K) des écarts de température en entrée : ``T - T(0)`` reste entre 0 et ~27 K."""

TARGET_LABELS: tuple[str, ...] = ("HLC", "HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf")
"""Ordre des cibles : celui du code de l'énoncé, avec les labels Hb du fichier."""
EXPECTED_HB_LABELS: tuple[str, ...] = ("HLC_mit", "HLC_ext")
CDT_LABELS: tuple[str, ...] = ("tau_b", "tau_n", "tau_inf")
LOG_TARGETS: tuple[str, ...] = CDT_LABELS
TARGET_FAMILIES: dict[str, tuple[str, ...]] = {
    "HLC": ("HLC",), "Hb": EXPECTED_HB_LABELS, "CDT": CDT_LABELS,
}

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test_random", "test_new_u")
SPLIT_CODES: dict[str, int] = {name: code for code, name in enumerate(SPLIT_NAMES)}
BATCHES_BY_SPLIT: dict[str, tuple[int, ...]] = {
    "train": tuple(range(1, 17)), "val": (17, 18), "test_random": (19, 20),
}
SPLIT_PATH = PROJECT_ROOT / "splits" / "split_seed42.npz"


# --------------------------------------------------------------------------- découpages
def build_splits(per_sim: pd.DataFrame, held_fraction: float = 0.05,
                 seed: int = 42) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint64]]:
    """Construit le label de découpage de chaque simulation.

    Args:
        per_sim: tableau de l'EDA (une ligne par simulation, triée par ``global_idx``),
            avec les colonnes ``batch``, ``u_hash`` et ``all_off``.
        held_fraction: part des signaux U distincts réservés au test ``test_new_u``.
        seed: graine du tirage des signaux réservés.

    Returns:
        ``labels`` (codes de ``SPLIT_CODES``, un par simulation) et les empreintes des
        signaux réservés, triées.
    """
    if not (per_sim["global_idx"].to_numpy() == np.arange(len(per_sim))).all():
        raise ValueError("per_sim doit être trié par global_idx.")
    u_hash = per_sim["u_hash"].to_numpy(np.uint64)
    zero_hashes = np.unique(u_hash[per_sim["all_off"].to_numpy(bool)])
    all_hashes = np.unique(u_hash)
    candidates = np.setdiff1d(all_hashes, zero_hashes)
    n_held = int(round(held_fraction * all_hashes.size))
    rng = np.random.default_rng(seed)
    held = np.sort(rng.choice(candidates, size=n_held, replace=False))

    batch = per_sim["batch"].to_numpy()
    labels = np.full(len(per_sim), 255, dtype=np.uint8)
    for name, batches in BATCHES_BY_SPLIT.items():
        labels[np.isin(batch, batches)] = SPLIT_CODES[name]
    labels[np.isin(u_hash, held)] = SPLIT_CODES["test_new_u"]
    if (labels == 255).any():
        raise ValueError("Des simulations n'ont reçu aucun label.")
    return labels, held


def save_splits(labels: npt.NDArray[np.uint8], held_hashes: npt.NDArray[np.uint64],
                path: Path = SPLIT_PATH, held_fraction: float = 0.05, seed: int = 42) -> Path:
    """Écrit le découpage figé (compressé, ~1 Mo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels, held_u_hashes=held_hashes,
                        split_names=np.array(SPLIT_NAMES), held_fraction=held_fraction,
                        seed=seed)
    return path


def load_splits(path: Path = SPLIT_PATH) -> npt.NDArray[np.uint8]:
    """Relit les labels de découpage (un par simulation, indexés par l'index global)."""
    with np.load(path) as f:
        if tuple(f["split_names"]) != SPLIT_NAMES:
            raise ValueError(f"Noms de découpage inattendus dans {path}")
        return f["labels"]


def split_table(labels: npt.NDArray[np.uint8]) -> pd.DataFrame:
    """Effectif et part de chaque jeu, avec les batchs qu'il couvre."""
    batch = np.arange(labels.size) // SIMS_PER_BATCH + 1
    rows = []
    for name in SPLIT_NAMES:
        m = labels == SPLIT_CODES[name]
        b = np.unique(batch[m])
        rows.append({"jeu": name, "batchs": f"{b.min()}–{b.max()}" if b.size else "",
                     "simulations": int(m.sum()), "part (%)": 100 * m.mean()})
    out = pd.DataFrame(rows).set_index("jeu")
    out.loc["total"] = ["", int(out["simulations"].sum()), out["part (%)"].sum()]
    return out


# --------------------------------------------------------------------------- chargement
@dataclass
class SplitData:
    """Un jeu de données prétraité, en mémoire.

    Attributes:
        x: entrées ``(n, 2, N_STEPS)`` float32 : ``(T - T(0)) / T_SCALE`` et ``U / POWER_W``.
        y: cibles physiques ``(n, 6)`` float32, dans l'ordre ``TARGET_LABELS``.
        global_idx: index global de chaque simulation ``(n,)`` int64.
    """

    x: torch.Tensor
    y: torch.Tensor
    global_idx: torch.Tensor

    def __len__(self) -> int:
        return int(self.y.shape[0])


def preprocess_inputs(temperature: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """Applique l'arrondi de l'énoncé puis la mise à l'échelle des entrées.

    Args:
        temperature: ``indoor_temperature`` brute ``(n, N_STEPS)``, en °C.
        power: ``indoor_U`` brut ``(n, N_STEPS)``, en W.

    Returns:
        Tenseur ``(n, 2, N_STEPS)`` float32.
    """
    t = temperature.float().round(decimals=2)
    u = power.float().round(decimals=1)
    return torch.stack([(t - t[:, :1]) / T_SCALE, u / POWER_W], dim=1)


def batch_targets(batch) -> torch.Tensor:
    """Vecteur cible officiel ``[HLC, *Hb, *CDT]`` d'un batch, labels vérifiés."""
    if tuple(batch.Hb_labels) != EXPECTED_HB_LABELS:
        raise ValueError(f"Hb_labels inattendus : {batch.Hb_labels}")
    return torch.cat([batch.HLC.unsqueeze(1), batch.Hb, batch.CDT], dim=1).float()


def load_split_data(
    labels: npt.NDArray[np.uint8],
    splits: Iterable[str] = SPLIT_NAMES,
    data_dir: Path = DATA_DIR,
    batch_numbers: Sequence[int] = BATCH_NUMBERS,
    max_per_batch: int | None = None,
    seed: int = 0,
) -> dict[str, SplitData]:
    """Charge plusieurs jeux en un seul passage sur les batchs (un batch en mémoire à la fois).

    Args:
        labels: labels de découpage (voir :func:`load_splits`).
        splits: jeux à charger.
        data_dir: dossier contenant ``batch_1.pt ... batch_20.pt``.
        batch_numbers: batchs à parcourir (tous par défaut ; un sous-ensemble pour les
            tests rapides).
        max_per_batch: si fourni, nombre maximal de simulations retenues par jeu et par
            batch, tirées uniformément (graine ``seed``) — pour les tests rapides.
        seed: graine du sous-échantillonnage.

    Returns:
        Un :class:`SplitData` par jeu demandé (éventuellement vide).
    """
    splits = tuple(splits)
    rng = np.random.default_rng(seed)
    per_batch = labels.reshape(-1, SIMS_PER_BATCH)
    rows: dict[tuple[str, int], npt.NDArray[np.int64]] = {}
    for name in splits:
        for b in batch_numbers:
            r = np.nonzero(per_batch[b - 1] == SPLIT_CODES[name])[0]
            if max_per_batch is not None and r.size > max_per_batch:
                r = np.sort(rng.choice(r, size=max_per_batch, replace=False))
            rows[(name, b)] = r
    sizes = {name: sum(rows[(name, b)].size for b in batch_numbers) for name in splits}
    out = {name: SplitData(x=torch.empty((n, 2, N_STEPS)), y=torch.empty((n, len(TARGET_LABELS))),
                           global_idx=torch.empty(n, dtype=torch.int64))
           for name, n in sizes.items()}
    filled = dict.fromkeys(splits, 0)
    needed = [b for b in batch_numbers if any(rows[(s, b)].size for s in splits)]
    for b, batch in iter_batches(needed, data_dir):
        if batch.HLC.shape[0] != SIMS_PER_BATCH or batch.indoor_temperature.shape[1] != N_STEPS:
            raise ValueError(f"Forme inattendue pour le batch {b}")
        for name in splits:
            r = rows[(name, b)]
            if r.size == 0:
                continue
            idx = torch.from_numpy(r)
            k, n = filled[name], r.size
            out[name].x[k:k + n] = preprocess_inputs(batch.indoor_temperature[idx],
                                                     batch.indoor_U[idx])
            out[name].y[k:k + n] = batch_targets(batch)[idx]
            out[name].global_idx[k:k + n] = idx + (b - 1) * SIMS_PER_BATCH
            filled[name] += n
    return out


# --------------------------------------------------------------------------- cibles
class TargetScaler(nn.Module):
    """Transformation des cibles : log pour les constantes de temps, puis standardisation.

    Les moyennes et écarts-types sont ajustés sur l'entraînement seul et stockés comme
    buffers : ils sont sauvegardés avec les poids du modèle, ce qui garantit une
    transformation inverse identique à l'évaluation.
    """

    def __init__(self) -> None:
        super().__init__()
        log_mask = torch.tensor([lab in LOG_TARGETS for lab in TARGET_LABELS])
        self.register_buffer("log_mask", log_mask)
        self.register_buffer("mean", torch.zeros(len(TARGET_LABELS)))
        self.register_buffer("std", torch.ones(len(TARGET_LABELS)))

    def _forward_transform(self, y: torch.Tensor) -> torch.Tensor:
        # Le log n'est évalué que sur les colonnes concernées (les autres reçoivent 1) :
        # sinon la branche non retenue de torch.where peut produire des gradients NaN.
        safe = torch.where(self.log_mask, y, torch.ones_like(y)).clamp_min(1e-12)
        return torch.where(self.log_mask, torch.log(safe), y)

    def fit(self, y: torch.Tensor) -> "TargetScaler":
        """Ajuste moyennes et écarts-types sur des cibles physiques ``(n, 6)``."""
        g = self._forward_transform(y.double())
        self.mean.copy_(g.mean(dim=0).float())
        self.std.copy_(g.std(dim=0).float())
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Cibles physiques → espace standardisé."""
        return (self._forward_transform(y) - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Espace standardisé → cibles physiques."""
        g = z * self.std + self.mean
        safe = torch.where(self.log_mask, g, torch.zeros_like(g))  # évite exp(233) = inf
        return torch.where(self.log_mask, torch.exp(safe), g)
