"""Chargement des batchs de simulations via le loader officiel coEnergy.

Le loader officiel (``provided/util_assessement.py``) n'est jamais modifié : il est
seulement rendu importable. Les batchs sont toujours parcourus dans l'ordre numérique
explicite ``batch_1 ... batch_20`` (et non dans l'ordre alphabétique de ``glob``).
"""

import gc
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "quentin_house"
BATCH_NUMBERS: tuple[int, ...] = tuple(range(1, 21))

sys.path.insert(0, str(PROJECT_ROOT / "provided"))

from util_assessement import ToSaveSimul, load_single_batch  # noqa: E402


def batch_path(batch_number: int) -> Path:
    """Chemin du fichier ``batch_{batch_number}.pt``."""
    return DATA_DIR / f"batch_{batch_number}.pt"


def load_batch(batch_number: int) -> ToSaveSimul:
    """Charge un seul batch avec le loader officiel."""
    return load_single_batch(batch_path(batch_number))


def iter_batches(
    batch_numbers: Sequence[int] = BATCH_NUMBERS,
) -> Iterator[tuple[int, ToSaveSimul]]:
    """Itère sur les batchs un par un, dans l'ordre donné, en libérant chacun après usage.

    Un seul batch (~200 Mo) est présent en mémoire à la fois.
    """
    for number in batch_numbers:
        batch = load_batch(number)
        yield number, batch
        del batch
        gc.collect()
