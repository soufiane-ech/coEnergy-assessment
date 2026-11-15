"""Chargement d'un batch de simulations via le loader officiel coEnergy."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "provided"))

from util_assessement import ToSaveSimul, load_single_batch  # noqa: E402


def load_batch(batch_number: int) -> ToSaveSimul:
    """Charge le fichier data/quentin_house/batch_{batch_number}.pt (un seul batch)."""
    path = PROJECT_ROOT / "data" / "quentin_house" / f"batch_{batch_number}.pt"
    return load_single_batch(path)
