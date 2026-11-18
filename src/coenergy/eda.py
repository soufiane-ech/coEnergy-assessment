"""Passe d'analyse exploratoire sur l'intégralité des batchs, et relecture de ses résultats.

La passe parcourt les 20 batchs un par un (un seul en mémoire) et produit, dans
``outputs/eda/`` :

- ``per_sim.npz`` : un enregistrement par simulation (1 000 000) — cibles, paramètres
  d'excitation variables, descripteurs de :mod:`coenergy.signals`, empreintes ;
- ``per_time.npz`` : sommes exactes par pas de temps et par groupe de simulations
  (moyennes, écarts-types, corrélations T(t)–cibles) ;
- ``curves.npz`` : une sélection de courbes individuelles pour les figures ;
- ``meta.json`` : inventaire par batch (formes, types, labels, bornes, contrôles).

Toutes les statistiques du notebook sont dérivées de ces fichiers, donc du million de
simulations ; seules les figures de courbes individuelles utilisent ``curves.npz``.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from coenergy.data import BATCH_NUMBERS, PROJECT_ROOT, ToSaveSimul, iter_batches
from coenergy.signals import excitation_features, row_hashes, temperature_features

OUT_DIR = PROJECT_ROOT / "outputs" / "eda"
CDT_LABELS: tuple[str, ...] = ("tau_b", "tau_n", "tau_inf")
EXPECTED_HB_LABELS: tuple[str, ...] = ("HLC_mit", "HLC_ext")
TARGET_LABELS: tuple[str, ...] = ("HLC", *EXPECTED_HB_LABELS, *CDT_LABELS)
VARYING_SOLICITATION: tuple[str, ...] = (
    "prbs_min_duration_hours", "prbs_max_duration_hours", "prbs_probability_on",
)

# Décalages constants appliqués avant accumulation (stabilité numérique des sommes).
T_REF = 24.624
Y_REF = np.array([233.0, 120.0, 112.0, 50.0, 11.0, 31.0])
T_HIST_EDGES = np.round(np.arange(20.0, 60.0001, 0.1), 1)


@dataclass
class GroupAccumulator:
    """Sommes exactes par pas de temps pour un groupe de simulations.

    Les températures et cibles sont décalées de ``T_REF`` et ``Y_REF`` avant sommation.
    """

    n_steps: int
    n_targets: int = len(TARGET_LABELS)
    n: int = 0
    s_t: npt.NDArray[np.float64] = field(init=False)
    s_tt: npt.NDArray[np.float64] = field(init=False)
    s_ty: npt.NDArray[np.float64] = field(init=False)
    s_y: npt.NDArray[np.float64] = field(init=False)
    s_yy: npt.NDArray[np.float64] = field(init=False)
    s_on: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.s_t = np.zeros(self.n_steps)
        self.s_tt = np.zeros(self.n_steps)
        self.s_ty = np.zeros((self.n_steps, self.n_targets))
        self.s_y = np.zeros(self.n_targets)
        self.s_yy = np.zeros(self.n_targets)
        self.s_on = np.zeros(self.n_steps)

    def update(self, t: npt.NDArray, y: npt.NDArray, on: npt.NDArray) -> None:
        """Ajoute un sous-ensemble ``t (m, n_steps)``, ``y (m, n_targets)``, ``on (m, n_steps)``."""
        if t.shape[0] == 0:
            return
        tc = t.astype(np.float64) - T_REF
        yc = y.astype(np.float64) - Y_REF
        self.n += t.shape[0]
        self.s_t += tc.sum(axis=0)
        self.s_tt += (tc * tc).sum(axis=0)
        self.s_ty += tc.T @ yc
        self.s_y += yc.sum(axis=0)
        self.s_yy += (yc * yc).sum(axis=0)
        self.s_on += on.sum(axis=0)

    def to_arrays(self, prefix: str) -> dict[str, npt.NDArray]:
        """Sérialise les sommes sous des clés ``{prefix}__{nom}``."""
        return {f"{prefix}__{k}": np.asarray(getattr(self, k))
                for k in ("n", "s_t", "s_tt", "s_ty", "s_y", "s_yy", "s_on")}


@dataclass(frozen=True)
class GroupStats:
    """Statistiques par pas de temps d'un groupe, dérivées des sommes exactes."""

    n: int
    t_mean: npt.NDArray[np.float64]
    t_std: npt.NDArray[np.float64]
    on_fraction: npt.NDArray[np.float64]
    corr_t_targets: npt.NDArray[np.float64]
    """Corrélation de Pearson entre T(t) et chaque cible, forme ``(n_steps, n_targets)``."""

    @classmethod
    def from_sums(cls, sums: dict[str, npt.NDArray]) -> "GroupStats":
        """Moyennes, écarts-types et corrélations par pas de temps, à partir des sommes exactes."""
        n = int(sums["n"])
        mean_t = sums["s_t"] / n
        var_t = sums["s_tt"] / n - mean_t**2
        mean_y = sums["s_y"] / n
        var_y = sums["s_yy"] / n - mean_y**2
        cov = sums["s_ty"] / n - np.outer(mean_t, mean_y)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.sqrt(np.outer(var_t, var_y))
        return cls(n=n, t_mean=mean_t + T_REF, t_std=np.sqrt(np.maximum(var_t, 0.0)),
                   on_fraction=sums["s_on"] / n, corr_t_targets=corr)


@dataclass
class EdaResults:
    """Résultats agrégés de la passe complète, relus depuis le disque."""

    per_sim: pd.DataFrame
    groups: dict[str, GroupStats]
    group_patterns: dict[str, npt.NDArray]
    t_hist_edges: npt.NDArray
    t_hist_counts: npt.NDArray
    curves: dict[str, npt.NDArray]
    meta: dict


def _batch_targets(batch: ToSaveSimul) -> npt.NDArray[np.float32]:
    hb_labels = tuple(batch.Hb_labels)
    if hb_labels != EXPECTED_HB_LABELS:
        raise ValueError(f"Hb_labels inattendus : {hb_labels}")
    return np.column_stack(
        [batch.HLC.numpy(), batch.Hb.numpy(), batch.CDT.numpy()]
    ).astype(np.float32)


def _batch_inventory(batch: ToSaveSimul, number: int, t: npt.NDArray, u: npt.NDArray,
                     s: npt.NDArray) -> dict:
    tensors = {name: getattr(batch, name) for name in
               ("HLC", "Hb", "CDT", "indoor_temperature", "indoor_U", "solicitation")}
    u_values = np.unique(u)
    return {
        "batch": number,
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
        "dtypes": {k: str(v.dtype) for k, v in tensors.items()},
        "hb_labels": list(batch.Hb_labels),
        "solicitation_field_names": list(batch.solicitation_field_names),
        "solicitation_min": s.min(axis=0).tolist(),
        "solicitation_max": s.max(axis=0).tolist(),
        "solicitation_n_unique": [int(np.unique(s[:, j]).size) for j in range(s.shape[1])],
        "u_unique_values": u_values[:20].tolist(),
        "u_n_unique_values": int(u_values.size),
        "n_nan": {"T": int(np.isnan(t).sum()), "U": int(np.isnan(u).sum())},
        "n_inf": {"T": int(np.isinf(t).sum()), "U": int(np.isinf(u).sum())},
        "t_min": float(t.min()),
        "t_max": float(t.max()),
    }


def run_eda_pass(
    out_dir: Path = OUT_DIR,
    batch_numbers: Sequence[int] = BATCH_NUMBERS,
    n_random_curves: int = 1000,
    n_pattern_groups: int = 8,
    n_group_curves: int = 400,
    seed: int = 42,
) -> Path:
    """Parcourt tous les batchs et écrit les résultats agrégés dans ``out_dir``.

    Groupes de simulations suivis par pas de temps :

    - ``all`` : toutes les simulations ;
    - ``free`` : simulations sans aucune puissance appliquée (U ≡ 0) ;
    - ``pattern_1`` … ``pattern_k`` : les ``k`` signaux U non nuls les plus fréquents du
      premier batch parcouru, retrouvés dans tous les batchs par leur empreinte exacte.
      À excitation identique, la variabilité de T ne vient que du bâtiment et de son
      état initial.

    Courbes conservées : ``n_random_curves`` simulations tirées uniformément sur
    l'ensemble (graine ``seed``), et ``n_group_curves`` simulations du premier batch pour
    ``free`` et ``pattern_1``.

    Returns:
        Le dossier de sortie.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    per_sim: dict[str, list[npt.NDArray]] = {}
    inventory: list[dict] = []
    accumulators: dict[str, GroupAccumulator] = {}
    pattern_hashes: dict[str, np.uint64] = {}
    pattern_curves: dict[str, npt.NDArray] = {}
    t_hist = np.zeros(T_HIST_EDGES.size + 1, dtype=np.int64)
    curves: dict[str, list[npt.NDArray]] = {k: [] for k in
                                            ("random_idx", "random_t", "random_u")}
    random_idx: npt.NDArray | None = None
    offset = 0
    started = time.time()

    for number, batch in iter_batches(batch_numbers):
        t = batch.indoor_temperature.numpy()
        u = batch.indoor_U.numpy()
        s = batch.solicitation.numpy()
        y = _batch_targets(batch)
        names = list(batch.solicitation_field_names)
        n, n_steps = t.shape
        inventory.append(_batch_inventory(batch, number, t, u, s))

        exc = excitation_features(u, s[:, names.index("max_power_wat")])
        tmp = temperature_features(t, u > 0)
        u_hash = row_hashes(u)
        record: dict[str, npt.NDArray] = {
            "global_idx": np.arange(offset, offset + n, dtype=np.int64),
            "batch": np.full(n, number, dtype=np.int16),
            "u_hash": u_hash,
            "t_hash": row_hashes(t),
            "y_hash": row_hashes(y),
        }
        record.update({lab: y[:, j] for j, lab in enumerate(TARGET_LABELS)})
        record.update({name: s[:, names.index(name)] for name in VARYING_SOLICITATION})
        record.update(exc)
        record.update(tmp)
        for key, value in record.items():
            if value.dtype == np.float64:
                value = value.astype(np.float32)
            per_sim.setdefault(key, []).append(value)

        if not accumulators:
            accumulators["all"] = GroupAccumulator(n_steps)
            accumulators["free"] = GroupAccumulator(n_steps)
            active = ~exc["all_off"]
            values, counts = np.unique(u_hash[active], return_counts=True)
            for rank, k in enumerate(np.argsort(-counts)[:n_pattern_groups], start=1):
                name = f"pattern_{rank}"
                pattern_hashes[name] = values[k]
                accumulators[name] = GroupAccumulator(n_steps)
                first = np.nonzero(u_hash == values[k])[0]
                pattern_curves[name] = u[first[0]].copy()
                if rank == 1:
                    sel = first[:n_group_curves]
                    curves["pattern_1_idx"] = [record["global_idx"][sel]]
                    curves["pattern_1_t"] = [t[sel].copy()]
            sel = np.nonzero(exc["all_off"])[0][:n_group_curves]
            curves["free_idx"] = [record["global_idx"][sel]]
            curves["free_t"] = [t[sel].copy()]
            total = n * len(batch_numbers)
            random_idx = np.sort(rng.choice(total, size=n_random_curves, replace=False))

        on = u > 0
        accumulators["all"].update(t, y, on)
        free = exc["all_off"]
        accumulators["free"].update(t[free], y[free], on[free])
        for name, h in pattern_hashes.items():
            m = u_hash == h
            accumulators[name].update(t[m], y[m], on[m])

        t_hist += np.bincount(np.searchsorted(T_HIST_EDGES, t.ravel(), side="right"),
                              minlength=t_hist.size)

        assert random_idx is not None
        local = random_idx[(random_idx >= offset) & (random_idx < offset + n)] - offset
        curves["random_idx"].append(local + offset)
        curves["random_t"].append(t[local].copy())
        curves["random_u"].append(u[local].copy())

        offset += n
        print(f"batch {number:2d} traité ({offset:,} simulations, "
              f"{time.time() - started:.0f} s)", flush=True)
        del t, u, s, y, on, exc, tmp

    np.savez(out_dir / "per_sim.npz", **{k: np.concatenate(v) for k, v in per_sim.items()})
    group_arrays: dict[str, npt.NDArray] = {}
    for name, acc in accumulators.items():
        group_arrays.update(acc.to_arrays(name))
    for name, pattern in pattern_curves.items():
        group_arrays[f"{name}__u"] = pattern
    np.savez(out_dir / "per_time.npz", t_hist_edges=T_HIST_EDGES, t_hist_counts=t_hist,
             group_names=np.array(list(accumulators)), **group_arrays)
    np.savez(out_dir / "curves.npz", **{k: np.concatenate(v) for k, v in curves.items()})
    meta = {
        "batch_numbers": list(batch_numbers),
        "n_simulations": offset,
        "seed": seed,
        "t_ref": T_REF,
        "y_ref": Y_REF.tolist(),
        "target_labels": list(TARGET_LABELS),
        "pattern_hashes": {k: int(v) for k, v in pattern_hashes.items()},
        "duration_seconds": round(time.time() - started, 1),
        "batches": inventory,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out_dir


def load_eda_results(out_dir: Path = OUT_DIR) -> EdaResults:
    """Relit les résultats écrits par :func:`run_eda_pass`."""
    with np.load(out_dir / "per_sim.npz") as f:
        per_sim = pd.DataFrame({k: f[k] for k in f.files})
    with np.load(out_dir / "per_time.npz") as f:
        names = [str(x) for x in f["group_names"]]
        groups = {
            g: GroupStats.from_sums({k.split("__", 1)[1]: f[k] for k in f.files
                                     if k.startswith(f"{g}__") and not k.endswith("__u")})
            for g in names
        }
        patterns = {g: f[f"{g}__u"] for g in names if f"{g}__u" in f.files}
        edges, counts = f["t_hist_edges"], f["t_hist_counts"]
    with np.load(out_dir / "curves.npz") as f:
        curves = {k: f[k] for k in f.files}
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return EdaResults(per_sim=per_sim, groups=groups, group_patterns=patterns,
                      t_hist_edges=edges, t_hist_counts=counts, curves=curves, meta=meta)


def ensure_eda_results(out_dir: Path = OUT_DIR, rerun: bool = False) -> EdaResults:
    """Relit les résultats en cache, ou lance la passe complète s'ils sont absents."""
    if rerun or not (out_dir / "meta.json").exists():
        run_eda_pass(out_dir)
    return load_eda_results(out_dir)
