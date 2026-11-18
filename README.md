# Prédiction d'indicateurs thermiques à partir d'un essai court — assessment coEnergy

Modèle PyTorch qui prédit, à partir de 8 h de température intérieure $T_{int}(t)$ et de la
sollicitation de chauffage $U(t)$, les six indicateurs thermiques d'un bâtiment :
le coefficient de déperdition $HLC$, sa décomposition $HLC_{ext}$ et $HLC_{mit}$ (W/K), et
les constantes de temps $\tau_b$, $\tau_n$, $\tau_\infty$ (h).

## Résultats principaux

Modèle retenu : **CNN-GRU multi-sorties**, avec une tête de sortie qui garantit par
construction $HLC = HLC_{ext} + HLC_{mit}$ et $\tau_n < \tau_\infty < \tau_b$.

| Test (95 565 simulations) | HLC | HLC_ext | HLC_mit | τ_n | τ_b | τ_∞ |
|---|---|---|---|---|---|---|
| Erreur relative moyenne | 8,3 % | 9,1 % | 15,8 % | 12,2 % | 14,8 % | 17,9 % |
| R² | 0,43 | 0,47 | 0,28 | 0,26 | 0,13 | 0,10 |

- Les performances sont au moins aussi bonnes sur des **excitations jamais vues à l'entraînement**.
- L'analyse d'identifiabilité montre que la limite tient aux données : en 8 h, la
  température renseigne surtout sur la chaleur **stockée** (la capacité apparente
  $C_n = \tau_n \cdot HLC$ est prédite avec R² = 0,84), et peu sur la chaleur **perdue**.
  Même à excitation fixée, aucun modèle de référence ne dépasse R² ≈ 0,48 sur le HLC.

## Contenu

| Élément | Rôle |
|---|---|
| `notebooks/01_eda.ipynb` | Analyse exploratoire sur les 1 000 000 de simulations : signaux, pas de temps, cibles, information disponible, indépendance |
| `notebooks/02_modelisation.ipynb` | Protocole, modèles, résultats par cible et par famille, ablations, identifiabilité, analyse critique des erreurs |
| `src/coenergy/data.py` | Chargement des batchs avec le loader officiel, un à la fois |
| `src/coenergy/signals.py`, `eda.py`, `analysis.py` | Descripteurs et statistiques exactes de l'analyse exploratoire |
| `src/coenergy/dataset.py` | Découpage train / val / test, prétraitement, standardisation des cibles |
| `src/coenergy/models.py` | Modèles : références, CNN, CNN-GRU, tête physique |
| `src/coenergy/train.py` | Entraînement et évaluation en ligne de commande |
| `src/coenergy/evaluation.py` | Métriques physiques par cible, par famille, par type d'excitation |
| `src/coenergy/identifiability.py` | Performance de référence à excitation fixée, corrélations canoniques |
| `src/coenergy/results.py`, `plots.py` | Lecture des résultats d'entraînement, figures |
| `splits/split_seed42.npz` | Découpage figé (un label par simulation) |
| `runs/<modèle>/` | Poids, historique, métriques et prédictions de chaque modèle entraîné |
| `provided/util_assessement.py` | Loader fourni, utilisé sans modification |

## Installation

Environnement géré par [uv](https://docs.astral.sh/uv/) (Python ≥ 3.11) :

```bash
uv sync
```

Les données `quentin_house` (20 fichiers `batch_*.pt`) sont à placer dans
`data/quentin_house/`.

## Exécution

```bash
uv run jupyter lab
```

- `01_eda.ipynb` parcourt les 20 batchs un par un (≈ 2 min au premier lancement), puis
  relit un cache dans `outputs/`.
- `02_modelisation.ipynb` relit les résultats d'entraînement de `runs/` et s'exécute sur
  CPU (≈ 5 min). La section 6 relit quelques batchs bruts dans `data/`.

## Entraînement

Les modèles de `runs/` ont été entraînés sur GPU (Kaggle, Tesla T4). Pour en
entraîner un :

```bash
uv run python -m coenergy.train --model cnn_gru --epochs 20 --batch-size 1024 --lr 3e-3 \
    --data-dir data/quentin_house --out-dir runs/cnn_gru
```

Modèles disponibles : `mean`, `ridge`, `cnn`, `cnn_gru`, `cnn_free`, `cnn_hlc`. L'option
`--batches 1 17 19 --max-per-batch 300` permet un test rapide sur CPU.
