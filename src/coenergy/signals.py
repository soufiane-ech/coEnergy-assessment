"""Descripteurs par simulation calculés sur les séries ``indoor_U`` et ``indoor_temperature``.

Toutes les fonctions sont vectorisées sur un batch : elles prennent des tableaux
``(n_simulations, n_steps)`` et renvoient un tableau par descripteur, de longueur
``n_simulations``. Les durées sont exprimées en **pas** ; la conversion en temps physique
relève de l'analyse (le pas de temps n'est pas fourni dans les données).
"""

import hashlib

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
BoolArray = npt.NDArray[np.bool_]

NO_VALUE = -1
"""Valeur sentinelle des descripteurs entiers non définis (ex. : aucun allumage)."""


def run_statistics(on: BoolArray) -> dict[str, IntArray]:
    """Statistiques des commutations d'un signal binaire.

    Une « série » (run) est une suite de pas consécutifs dans le même état. Les séries
    intérieures sont celles délimitées par deux commutations : la première et la dernière
    série de chaque simulation sont tronquées par la fenêtre et en sont exclues.

    Args:
        on: tableau booléen ``(n, n_steps)``, état allumé à chaque pas.

    Returns:
        ``n_switches`` (nombre de commutations), ``first_switch`` (indice du premier pas
        après la première commutation), ``min_run`` et ``max_run`` (longueurs extrêmes
        des séries intérieures, en pas). ``NO_VALUE`` si non défini.
    """
    n = on.shape[0]
    rows, cols = np.nonzero(np.diff(on.astype(np.int8), axis=1) != 0)
    switch_at = cols + 1
    n_switches = np.bincount(rows, minlength=n)

    first_switch = np.full(n, NO_VALUE, dtype=np.int64)
    min_run = np.full(n, NO_VALUE, dtype=np.int64)
    max_run = np.full(n, NO_VALUE, dtype=np.int64)
    if rows.size == 0:
        return {"n_switches": n_switches, "first_switch": first_switch,
                "min_run": min_run, "max_run": max_run}

    is_first = np.r_[True, rows[1:] != rows[:-1]]
    first_switch[rows[is_first]] = switch_at[is_first]

    same_row = ~is_first[1:]
    lengths = (switch_at[1:] - switch_at[:-1])[same_row]
    length_rows = rows[1:][same_row]
    big = np.iinfo(np.int64).max
    mins = np.full(n, big, dtype=np.int64)
    maxs = np.zeros(n, dtype=np.int64)
    np.minimum.at(mins, length_rows, lengths)
    np.maximum.at(maxs, length_rows, lengths)
    has_run = mins != big
    min_run[has_run] = mins[has_run]
    max_run[has_run] = maxs[has_run]
    return {"n_switches": n_switches, "first_switch": first_switch,
            "min_run": min_run, "max_run": max_run}


def excitation_features(u: FloatArray, max_power: FloatArray) -> dict[str, npt.NDArray]:
    """Descripteurs de l'excitation ``indoor_U``.

    Args:
        u: ``(n, n_steps)``, valeurs de ``indoor_U``.
        max_power: ``(n,)``, puissance maximale déclarée par simulation (``max_power_wat``).

    Returns:
        ``duty`` (fraction de pas allumés), ``all_off``, ``all_on``, ``first_on``
        (indice du premier pas allumé), ``n_off_grid`` (nombre de valeurs différentes de
        0 et de ``max_power``), ``u_max``, plus les sorties de :func:`run_statistics`.
    """
    on = u > 0
    any_on = on.any(axis=1)
    features: dict[str, npt.NDArray] = {
        "duty": on.mean(axis=1),
        "all_off": ~any_on,
        "all_on": on.all(axis=1),
        "first_on": np.where(any_on, on.argmax(axis=1), NO_VALUE),
        "n_off_grid": ((u != 0) & (u != max_power[:, None])).sum(axis=1),
        "u_max": u.max(axis=1),
    }
    features.update(run_statistics(on))
    return features


def temperature_features(
    t: FloatArray, on: BoolArray, quantum: float = 0.01
) -> dict[str, npt.NDArray]:
    """Descripteurs de la réponse ``indoor_temperature``.

    Args:
        t: ``(n, n_steps)``, température intérieure (°C).
        on: ``(n, n_steps)``, état allumé de l'excitation.
        quantum: résolution d'arrondi de la température prescrite par l'énoncé (°C).

    Returns:
        ``t0``, ``t_end``, ``t_min``, ``t_max``, ``t_mean`` ; ``max_abs_dt`` (plus grand
        incrément entre deux pas) ; ``frac_sub_quantum_dt`` (fraction des incréments
        inférieurs à un demi-quantum, donc effacés par l'arrondi) ; ``switch_on_jump``
        (accroissement de la pente au premier allumage, ``NaN`` si non défini, voir
        :func:`switch_on_jump`).
    """
    t64 = t.astype(np.float64)
    dt = np.diff(t64, axis=1)
    return {
        "t0": t64[:, 0],
        "t_end": t64[:, -1],
        "t_min": t64.min(axis=1),
        "t_max": t64.max(axis=1),
        "t_mean": t64.mean(axis=1),
        "max_abs_dt": np.abs(dt).max(axis=1),
        "frac_sub_quantum_dt": (np.abs(dt) < quantum / 2).mean(axis=1),
        "switch_on_jump": switch_on_jump(t64, on),
    }


def switch_on_jump(t: FloatArray, on: BoolArray) -> FloatArray:
    """Accroissement de l'incrément de température au premier allumage.

    Pour le premier pas allumé ``k`` (avec ``k >= 2``, précédé de deux pas éteints) :
    ``(T[k] - T[k-1]) - (T[k-1] - T[k-2])``. Retranchant la pente précédente, cette
    quantité isole l'effet de la puissance injectée pendant le pas ``k``.

    Returns:
        Tableau ``(n,)`` ; ``NaN`` si la simulation n'a pas d'allumage exploitable.
    """
    n = t.shape[0]
    any_on = on.any(axis=1)
    k = on.argmax(axis=1)
    valid = any_on & (k >= 2)
    out = np.full(n, np.nan)
    i = np.nonzero(valid)[0]
    kv = k[valid]
    out[i] = (t[i, kv] - t[i, kv - 1]) - (t[i, kv - 1] - t[i, kv - 2])
    return out


def row_hashes(a: npt.NDArray) -> npt.NDArray[np.uint64]:
    """Empreinte 64 bits exacte du contenu binaire de chaque ligne.

    Deux lignes ont la même empreinte si (à une collision près, de probabilité
    négligeable) elles sont identiques bit à bit.
    """
    a = np.ascontiguousarray(a)
    return np.array(
        [int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little")
         for row in a],
        dtype=np.uint64,
    )
