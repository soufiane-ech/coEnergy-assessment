"""Modèles de régression multi-sorties (PyTorch).

Tous les modèles partagent la même interface : ``forward(x)`` reçoit les entrées
prétraitées ``(n, 2, N_STEPS)`` et renvoie les six cibles **physiques** ``(n, 6)`` dans
l'ordre ``TARGET_LABELS`` (W/K et h). Chaque modèle embarque son :class:`TargetScaler`,
sauvegardé avec ses poids.

- :class:`MeanBaseline` : prédit la moyenne d'entraînement (plancher de référence).
- :class:`LinearBaseline` : régression linéaire sur les 960 valeurs d'entrée, ajustée en
  forme close (ridge), dans l'espace standardisé des cibles.
- :class:`CNNRegressor` : encodeur convolutif 1D à convolutions dilatées, tronc commun et
  tête physique :class:`PhysicalHead` (contraintes garanties par construction), ou tête
  libre :class:`FreeHead` pour l'ablation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from coenergy.dataset import N_STEPS, TARGET_LABELS, TargetScaler

IDX = {label: i for i, label in enumerate(TARGET_LABELS)}
MODEL_NAMES: tuple[str, ...] = ("mean", "ridge", "cnn", "cnn_free")


# --------------------------------------------------------------------------- références
class MeanBaseline(nn.Module):
    """Prédit pour toute entrée la moyenne des cibles d'entraînement."""

    def __init__(self, scaler: TargetScaler) -> None:
        super().__init__()
        self.scaler = scaler
        self.register_buffer("y_mean", torch.zeros(len(TARGET_LABELS)))

    def fit(self, y_train: torch.Tensor) -> "MeanBaseline":
        """Mémorise la moyenne des cibles physiques d'entraînement."""
        self.y_mean.copy_(y_train.double().mean(dim=0).float())
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.y_mean.expand(x.shape[0], -1)


class LinearBaseline(nn.Module):
    """Régression linéaire des cibles standardisées sur les entrées aplaties (2 × 480)."""

    def __init__(self, scaler: TargetScaler, n_inputs: int = 2 * N_STEPS) -> None:
        super().__init__()
        self.scaler = scaler
        self.linear = nn.Linear(n_inputs, len(TARGET_LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler.inverse(self.linear(x.flatten(1)))


# --------------------------------------------------------------------------- têtes
def _inverse_softplus(v: float) -> float:
    """Valeur ``b`` telle que ``softplus(b) = v`` (``v > 0``)."""
    return math.log(math.expm1(v))


class PhysicalHead(nn.Module):
    """Tête de sortie qui garantit les contraintes physiques par construction.

    - ``HLC_mit = exp(.)`` et ``HLC_ext = exp(.)`` sont positifs, et
      ``HLC = HLC_mit + HLC_ext`` exactement (contrainte de l'énoncé).
    - ``log tau_inf = log tau_n + softplus(.)`` et ``log tau_b = log tau_inf + softplus(.)`` :
      l'ordre ``tau_n < tau_inf < tau_b``, vérifié dans 100 % des données, est garanti.

    Les sorties brutes sont centrées-réduites autour des statistiques d'entraînement
    (stockées comme buffers), de sorte qu'un réseau initialisé à zéro prédit l'ordre de
    grandeur moyen de chaque grandeur.
    """

    N_RAW = 5

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, self.N_RAW)
        # moyennes / écarts-types de log HLC_mit, log HLC_ext, log tau_n
        self.register_buffer("log_mean", torch.zeros(3))
        self.register_buffer("log_std", torch.ones(3))
        # biais tels que softplus(biais) = moyenne des écarts log(tau_inf/tau_n), log(tau_b/tau_inf)
        self.register_buffer("gap_bias", torch.zeros(2))
        self.register_buffer("gap_std", torch.ones(2))

    def init_from_targets(self, y_train: torch.Tensor) -> "PhysicalHead":
        """Initialise les statistiques de recentrage à partir des cibles d'entraînement."""
        y = y_train.double()
        logs = torch.log(y[:, [IDX["HLC_mit"], IDX["HLC_ext"], IDX["tau_n"]]])
        self.log_mean.copy_(logs.mean(0).float())
        self.log_std.copy_(logs.std(0).float())
        gaps = torch.stack([torch.log(y[:, IDX["tau_inf"]] / y[:, IDX["tau_n"]]),
                            torch.log(y[:, IDX["tau_b"]] / y[:, IDX["tau_inf"]])], dim=1)
        self.gap_bias.copy_(torch.tensor([_inverse_softplus(float(g)) for g in gaps.mean(0)]))
        self.gap_std.copy_(gaps.std(0).float())
        return self

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.linear(h.float())
        logs = self.log_mean + self.log_std * raw[:, :3]
        hlc_mit, hlc_ext = torch.exp(logs[:, 0]), torch.exp(logs[:, 1])
        log_tau_n = logs[:, 2]
        log_tau_inf = log_tau_n + F.softplus(self.gap_bias[0] + self.gap_std[0] * raw[:, 3])
        log_tau_b = log_tau_inf + F.softplus(self.gap_bias[1] + self.gap_std[1] * raw[:, 4])
        out = {"HLC": hlc_mit + hlc_ext, "HLC_mit": hlc_mit, "HLC_ext": hlc_ext,
               "tau_b": torch.exp(log_tau_b), "tau_n": torch.exp(log_tau_n),
               "tau_inf": torch.exp(log_tau_inf)}
        return torch.stack([out[label] for label in TARGET_LABELS], dim=1)


class FreeHead(nn.Module):
    """Tête sans contrainte : six sorties standardisées indépendantes (ablation)."""

    def __init__(self, in_features: int, scaler: TargetScaler) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, len(TARGET_LABELS))
        self.scaler = scaler

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scaler.inverse(self.linear(h.float()))


# --------------------------------------------------------------------------- encodeur
class DilatedResidualBlock(nn.Module):
    """Deux convolutions 1D dilatées (noyau 3) avec connexion résiduelle."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class CNNRegressor(nn.Module):
    """Encodeur convolutif dilaté + tronc commun + tête physique (ou libre).

    Choix motivés par l'EDA :

    - convolutions : l'information sur le bâtiment se concentre dans les minutes qui
      suivent chaque commutation (motifs locaux) ;
    - dilatations 1, 2, …, 128 : le champ réceptif (≈ 1 000 pas) couvre les 8 h, ce qui
      permet de relier chaque réponse à l'historique de l'excitation ;
    - un canal « temps » (0 → 1) est ajouté en entrée, car à excitation égale la position
      dans la fenêtre compte (état initial, dérive libre) ;
    - mise en commun moyenne + maximum sur le temps, puis tronc partagé par les six
      cibles (elles sont fortement liées : tau ≈ C / HLC).
    """

    def __init__(self, scaler: TargetScaler, channels: int = 64, hidden: int = 128,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
                 constrained: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        self.scaler = scaler
        self.register_buffer("time", torch.linspace(0.0, 1.0, N_STEPS).view(1, 1, -1))
        self.stem = nn.Sequential(nn.Conv1d(3, channels, 7, padding=3),
                                  nn.BatchNorm1d(channels), nn.GELU())
        self.blocks = nn.Sequential(*[DilatedResidualBlock(channels, d) for d in dilations])
        self.trunk = nn.Sequential(nn.Linear(2 * channels, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.head: nn.Module = PhysicalHead(hidden) if constrained else FreeHead(hidden, scaler)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = self.time.expand(x.shape[0], -1, -1).to(x.dtype)
        h = self.blocks(self.stem(torch.cat([x, time], dim=1)))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        h = self.trunk(h)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.head(h)


# --------------------------------------------------------------------------- fabrique
def build_model(name: str, scaler: TargetScaler, channels: int = 64,
                hidden: int = 128) -> nn.Module:
    """Construit un modèle par son nom (``MODEL_NAMES``), sans l'entraîner."""
    if name == "mean":
        return MeanBaseline(scaler)
    if name == "ridge":
        return LinearBaseline(scaler)
    if name in ("cnn", "cnn_free"):
        return CNNRegressor(scaler, channels=channels, hidden=hidden,
                            constrained=(name == "cnn"))
    raise ValueError(f"Modèle inconnu : {name} (attendu : {MODEL_NAMES})")


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model: nn.Module, name: str, config: dict) -> None:
    """Sauvegarde poids, buffers (dont le scaler) et configuration d'un modèle."""
    torch.save({"model_name": name, "config": config, "state_dict": model.state_dict(),
                "torch_version": torch.__version__}, path)


def load_checkpoint(path, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """Recharge un modèle sauvegardé par :func:`save_checkpoint`, en mode évaluation."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(ckpt["model_name"], TargetScaler(), channels=cfg.get("channels", 64),
                        hidden=cfg.get("hidden", 128))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), cfg
