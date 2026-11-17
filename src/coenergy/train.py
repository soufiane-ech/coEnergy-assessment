"""Entraînement et évaluation d'un modèle, en ligne de commande.

Exemples ::

    # test rapide sur CPU (3 batchs, 300 simulations par jeu et par batch)
    python -m coenergy.train --model cnn --batches 1 17 19 --max-per-batch 300 --epochs 2

    # entraînement complet sur Kaggle
    python -m coenergy.train --model cnn --data-dir /kaggle/input/<dataset>/quentin_house \\
        --out-dir /kaggle/working/cnn

Sorties dans ``--out-dir`` :

- ``model.pt`` : poids, scaler et configuration (voir :func:`coenergy.models.load_checkpoint`) ;
- ``history.json`` : perte d'entraînement et de validation par époque ;
- ``metrics.json`` : métriques physiques par jeu (validation et tests) ;
- ``predictions_<jeu>.npz`` : ``global_idx``, ``y_true``, ``y_pred`` pour l'analyse d'erreurs ;
- ``run_info.json`` : matériel, durées, effectifs, versions.

La perte est l'erreur quadratique moyenne dans l'espace standardisé des cibles (log
pour les constantes de temps), pondérée pour que chaque famille de l'énoncé (HLC, Hb,
CDT) compte autant.
"""

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coenergy.data import BATCH_NUMBERS, DATA_DIR
from coenergy.dataset import (SPLIT_NAMES, SPLIT_PATH, TARGET_FAMILIES, TARGET_LABELS,
                              SplitData, TargetScaler, load_split_data, load_splits)
from coenergy.evaluation import (capacity_metrics, family_metrics, physical_checks,
                                 regression_metrics)
from coenergy.models import (MODEL_NAMES, LinearBaseline, MeanBaseline, PhysicalHead,
                             build_model, count_parameters, save_checkpoint)

EVAL_SPLITS: tuple[str, ...] = ("val", "test_random", "test_new_u")


def log(message: str) -> None:
    """Affiche un message horodaté (visible immédiatement dans Kaggle)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def loss_weights() -> torch.Tensor:
    """Poids par cible : chaque famille pèse 1/3, répartie également entre ses cibles."""
    w = torch.zeros(len(TARGET_LABELS))
    for labels in TARGET_FAMILIES.values():
        for lab in labels:
            w[TARGET_LABELS.index(lab)] = 1.0 / (len(TARGET_FAMILIES) * len(labels))
    return w


def weighted_loss(scaler: TargetScaler, y_pred: torch.Tensor, y_true: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    """MSE pondérée dans l'espace standardisé des cibles."""
    diff = scaler.transform(y_pred.float()) - scaler.transform(y_true.float())
    return (diff.pow(2).mean(dim=0) * weights).sum()


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device,
            batch_size: int = 4096) -> torch.Tensor:
    """Prédictions physiques ``(n, 6)`` sur CPU, calculées par paquets."""
    model.eval()
    out = [model(x[i:i + batch_size].to(device, non_blocking=True)).float().cpu()
           for i in range(0, len(x), batch_size)]
    return torch.cat(out) if out else torch.empty((0, len(TARGET_LABELS)))


def evaluate_loss(model: nn.Module, data: SplitData, device: torch.device,
                  weights: torch.Tensor) -> float:
    """Perte pondérée sur un jeu complet."""
    y_pred = predict(model, data.x, device)
    scaler_cpu = copy.deepcopy(model.scaler).cpu()  # le scaler du modèle reste sur son device
    return float(weighted_loss(scaler_cpu, y_pred, data.y, weights.cpu()))


# --------------------------------------------------------------------------- ridge
def fit_ridge(model: LinearBaseline, train: SplitData, val: SplitData, device: torch.device,
              lambdas: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
              chunk: int = 16384) -> dict:
    """Ajuste la régression linéaire en forme close (équations normales, float64).

    Le coefficient de régularisation est choisi sur la validation parmi ``lambdas``.
    """
    scaler = model.scaler
    d = train.x[0].numel() + 1
    xtx = torch.zeros((d, d), dtype=torch.float64, device=device)
    xtz = torch.zeros((d, len(TARGET_LABELS)), dtype=torch.float64, device=device)
    for i in range(0, len(train), chunk):
        xb = train.x[i:i + chunk].to(device).flatten(1).double()
        xb = torch.cat([xb, torch.ones(len(xb), 1, dtype=xb.dtype, device=device)], dim=1)
        zb = scaler.to(device).transform(train.y[i:i + chunk].to(device)).double()
        xtx += xb.T @ xb
        xtz += xb.T @ zb
    n = len(train)
    reg = torch.eye(d, dtype=torch.float64, device=device)
    reg[-1, -1] = 0.0  # pas de pénalité sur le biais
    weights = loss_weights()
    results = {}
    best = None
    for lam in lambdas:
        w = torch.linalg.solve(xtx / n + lam * reg, xtz / n)
        model.linear.weight.data.copy_(w[:-1].T.float())
        model.linear.bias.data.copy_(w[-1].float())
        results[lam] = evaluate_loss(model, val, device, weights)
        log(f"ridge lambda={lam:.0e}  perte val={results[lam]:.5f}")
        if best is None or results[lam] < results[best[0]]:
            best = (lam, w.clone())
    lam, w = best
    model.linear.weight.data.copy_(w[:-1].T.float())
    model.linear.bias.data.copy_(w[-1].float())
    return {"val_loss_by_lambda": {str(k): v for k, v in results.items()}, "best_lambda": lam}


# --------------------------------------------------------------------------- réseau
def train_network(model: nn.Module, train: SplitData, val: SplitData, device: torch.device,
                  epochs: int, batch_size: int, lr: float, weight_decay: float,
                  patience: int, seed: int) -> dict:
    """Boucle d'entraînement : AdamW, schedule one-cycle, arrêt anticipé sur la validation.

    Sur GPU, le jeu d'entraînement est placé entièrement en mémoire GPU (≈ 3 Go pour
    l'ensemble complet), ce qui évite tout coût de transfert par mini-lot.
    """
    weights = loss_weights().to(device)
    scaler = model.scaler
    x_train, y_train = train.x.to(device), train.y.to(device)
    n = len(train)
    steps_per_epoch = max(1, n // batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    use_amp = device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "val_loss": [],
                                       "lr": [], "seconds": []}
    best_loss, best_state, best_epoch, waited = float("inf"), None, -1, 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        perm = torch.randperm(n, generator=generator)[: steps_per_epoch * batch_size]
        total = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size].to(device)
            xb, yb = x_train[idx], y_train[idx]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                y_pred = model(xb)
            loss = weighted_loss(scaler, y_pred, yb, weights)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            schedule.step()
            total += loss.item()
        val_loss = evaluate_loss(model, val, device, weights)
        history["epoch"].append(epoch)
        history["train_loss"].append(total / steps_per_epoch)
        history["val_loss"].append(val_loss)
        history["lr"].append(schedule.get_last_lr()[0])
        history["seconds"].append(time.time() - started)
        log(f"époque {epoch:3d}/{epochs}  perte train={total / steps_per_epoch:.5f}  "
            f"val={val_loss:.5f}  ({history['seconds'][-1]:.0f} s)")
        if val_loss < best_loss:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                log(f"arrêt anticipé : pas d'amélioration depuis {patience} époques")
                break
    if best_state is None:
        raise RuntimeError("Aucune époque valide : la perte de validation est NaN ou infinie.")
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_loss
    return history


# --------------------------------------------------------------------------- programme
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments de la ligne de commande."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--split-file", type=Path, default=SPLIT_PATH)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=list(BATCH_NUMBERS),
                   help="batchs à parcourir (sous-ensemble pour les tests rapides)")
    p.add_argument("--max-per-batch", type=int, default=None,
                   help="simulations max par jeu et par batch (tests rapides)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Charge les données, entraîne le modèle demandé, puis écrit poids et résultats."""
    args = parse_args(argv)
    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"modèle={args.model}  device={device}  données={args.data_dir}")

    labels = load_splits(args.split_file)
    data = load_split_data(labels, SPLIT_NAMES, args.data_dir, args.batches,
                           args.max_per_batch, seed=args.seed)
    sizes = {name: len(d) for name, d in data.items()}
    log(f"données chargées en {time.time() - started:.0f} s : {sizes}")
    train, val = data["train"], data["val"]
    if len(train) == 0 or len(val) == 0:
        raise ValueError("Les jeux train et val doivent être non vides.")

    scaler = TargetScaler().fit(train.y)
    model = build_model(args.model, scaler, args.channels, args.hidden)
    if isinstance(getattr(model, "head", None), PhysicalHead):
        model.head.init_from_targets(train.y)
    model.to(device)
    log(f"paramètres entraînables : {count_parameters(model):,}")

    fit_started = time.time()
    if isinstance(model, MeanBaseline):
        model.fit(train.y.to(device))
        history: dict = {}
    elif isinstance(model, LinearBaseline):
        history = fit_ridge(model, train, val, device)
    else:
        history = train_network(model, train, val, device, args.epochs, args.batch_size,
                                args.lr, args.weight_decay, args.patience, args.seed)
    fit_seconds = time.time() - fit_started

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    save_checkpoint(args.out_dir / "model.pt", model.cpu(), args.model, config)
    model.to(device)
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    metrics = {}
    for name in EVAL_SPLITS:
        split = data[name]
        if len(split) == 0:
            continue
        y_pred = predict(model, split.x, device).numpy()
        y_true = split.y.numpy()
        np.savez_compressed(args.out_dir / f"predictions_{name}.npz",
                            global_idx=split.global_idx.numpy(), y_true=y_true, y_pred=y_pred)
        per_target = regression_metrics(y_true, y_pred)
        capacities = capacity_metrics(y_true, y_pred)
        metrics[name] = {"per_target": per_target.drop(columns="unité").to_dict(orient="index"),
                         "per_family": family_metrics(per_target).to_dict(orient="index"),
                         "capacities": capacities.drop(columns="unité").to_dict(orient="index"),
                         "physical_checks": physical_checks(y_pred).to_dict()}
        log(f"{name} ({len(split):,}) — R² : "
            + "  ".join(f"{lab}={per_target.loc[lab, 'R²']:.3f}" for lab in TARGET_LABELS)
            + "  |  capacités : "
            + "  ".join(f"{c}={capacities.loc[c, 'R²']:.3f}" for c in capacities.index))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    run_info = {
        "model": args.model, "config": config, "sizes": sizes,
        "n_parameters": count_parameters(model), "fit_seconds": fit_seconds,
        "total_seconds": time.time() - started, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
    }
    (args.out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    log(f"terminé en {run_info['total_seconds']:.0f} s → {args.out_dir}")


if __name__ == "__main__":
    main()
