# Prédire les indicateurs thermiques d'un bâtiment à partir d'un essai court de 8 h

**Assessment technique coEnergy.** Un modèle PyTorch reçoit 8 heures de température intérieure
$T_{int}(t)$ et la puissance de chauffage appliquée $U(t)$. Il prédit les six indicateurs
thermiques du bâtiment : le coefficient de déperdition $HLC$, sa décomposition
$HLC_{ext}$ et $HLC_{mit}$ (W/K), et les constantes de temps $\tau_b$, $\tau_n$, $\tau_\infty$ (h).

---

## 1. Par où commencer

Les deux notebooks sont **déjà exécutés** : figures, tableaux et résultats se lisent
directement, sur GitHub ou dans VS Code, sans rien installer.

| Ordre | Document | Contenu | Lecture |
|---|---|---|---|
| 1 | Ce README | Résultats, démarche, organisation, instructions d'exécution | 5 min |
| 2 | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Analyse exploratoire du million de simulations. La **section 7 (Synthèse)** résume les constats, les hypothèses et les décisions de modélisation | 20 min |
| 3 | [`notebooks/02_modelisation.ipynb`](notebooks/02_modelisation.ipynb) | Protocole, modèles, résultats par cible et par famille, ablations, identifiabilité, analyse critique des erreurs. La **section 8 (Conclusion)** résume l'ensemble | 25 min |

Chaque section des notebooks suit le même schéma : une **question**, une **mesure**, une
**lecture** des résultats, puis la **conséquence** pour la suite. Les affirmations sont
étiquetées selon leur source : **[Assessment]** l'énoncé, **[Article]** l'article de
coEnergy, **[Mesuré]** un calcul sur les données, **[Hypothèse]** ce qui n'est pas vérifiable.

---

## 2. Résultats principaux

**Modèle retenu : CNN-GRU multi-sorties**, dont la tête de sortie garantit par construction
$HLC = HLC_{ext} + HLC_{mit}$ et $\tau_n < \tau_\infty < \tau_b$.

Sur le jeu de test (95 565 bâtiments jamais vus) :

| | HLC | HLC_ext | HLC_mit | τ_n | τ_b | τ_∞ |
|---|---|---|---|---|---|---|
| **Erreur relative moyenne** | **8,3 %** | 9,1 % | 15,8 % | 12,2 % | 14,8 % | 17,9 % |
| Erreur absolue moyenne | 18,9 W/K | 9,9 W/K | 18,0 W/K | 1,4 h | 7,4 h | 5,5 h |
| R² | 0,43 | 0,47 | 0,28 | 0,26 | 0,13 | 0,10 |
| Prédiction constante (référence) | 11,1 % | 12,8 % | 19,0 % | 14,6 % | 16,4 % | 19,7 % |

- **Contraintes physiques respectées exactement** pour toutes les prédictions.
- **Généralisation :** les scores sont au moins aussi bons sur 140 signaux d'excitation
  jamais vus à l'entraînement.
- **Meilleurs scores obtenus**, toutes variantes confondues : 8,22 % sur le HLC
  (`cnn_gru_large`), 9,02 % sur $HLC_{ext}$ (`cnn_gru_40ep`). L'écart avec le modèle retenu
  est inférieur à 0,1 point.
- **Ce qui limite la performance : les données, pas le modèle.** En 8 h, la température
  renseigne surtout sur la chaleur **stockée** par le bâtiment, et peu sur la chaleur
  **perdue**. La capacité apparente $C_n = \tau_n \cdot HLC$ est prédite avec R² = 0,84, alors
  que le HLC plafonne près de 0,48, même à excitation fixée. Doubler l'entraînement ou
  quadrupler la taille du réseau n'améliore pas les résultats.

---

## 3. Démarche

| Étape | Question | Où la lire |
|---|---|---|
| **1. Comprendre les données** | Que contiennent les signaux ? Quel pas de temps ? Comment les cibles sont-elles liées ? Où se trouve l'information ? Y a-t-il des risques de fuite ? | `01_eda`, sections 1 à 6 |
| **2. Définir le protocole** | Comment évaluer sans biais, y compris sur des excitations nouvelles ? Quelles métriques ? | `02_modelisation`, section 1 |
| **3. Concevoir et comparer les modèles** | Quelle architecture ? Les contraintes, le multi-sorties, la taille du modèle sont-ils justifiés ? | `02_modelisation`, sections 2 à 4 |
| **4. Analyser les erreurs et leurs limites** | Où et pourquoi le modèle se trompe-t-il ? Quelle performance les données permettent-elles ? | `02_modelisation`, sections 5 à 7 |

---

## 4. Organisation du dépôt

```
coEnergy-assessment/
├── README.md
├── pyproject.toml             dépendances du projet (uv)
├── uv.lock                    versions exactes, pour un environnement reproductible
├── notebooks/
│   ├── 01_eda.ipynb           analyse exploratoire
│   └── 02_modelisation.ipynb  modélisation, évaluation, analyse des erreurs
├── src/coenergy/              code du projet, importé par les notebooks
│   ├── data.py                chargement des batchs avec le loader fourni
│   ├── signals.py             descripteurs des signaux U et T
│   ├── eda.py                 passe complète sur les données, statistiques exactes
│   ├── analysis.py            tableaux de l'analyse exploratoire
│   ├── dataset.py             découpage train / val / test, prétraitement
│   ├── models.py              modèles : références, CNN, CNN-GRU, tête physique
│   ├── train.py               entraînement et évaluation, en ligne de commande
│   ├── evaluation.py          métriques par cible, par famille, par type d'excitation
│   ├── identifiability.py     performance de référence, corrélations canoniques
│   ├── results.py             lecture et comparaison des résultats d'entraînement
│   └── plots.py               style et fonctions de figures
├── provided/
│   └── util_assessement.py    loader fourni avec l'assessment, utilisé sans modification
├── splits/
│   └── split_seed42.npz       découpage figé : un label par simulation
└── runs/                      résultats d'entraînement (un dossier par modèle)
```

**Le dossier `runs/`.** Chaque modèle entraîné y a son dossier : `model.pt` (poids),
`metrics.json`, `history.json`, `run_info.json` et `predictions_<jeu>.npz`. Le notebook 02
relit ces fichiers ; il n'a pas besoin de GPU.

| Dossier | Rôle |
|---|---|
| `mean` | référence : prédiction constante (moyenne d'entraînement) |
| `ridge` | référence : régression linéaire |
| `cnn` | CNN à convolutions dilatées, tête physique |
| **`cnn_gru`** | **modèle retenu** : CNN-GRU, tête physique, 20 époques |
| `cnn_gru_40ep` | le même, entraîné deux fois plus longtemps |
| `cnn_gru_large` | le même, quatre fois plus de paramètres |
| `cnn_free_20ep` | ablation : le CNN sans contraintes physiques |
| `cnn_hlc` | ablation : le CNN entraîné sur le seul HLC |

---

## 5. Installation

**Prérequis :** [uv](https://docs.astral.sh/uv/getting-started/installation/) (il installe
lui-même la bonne version de Python, ≥ 3.11).

```bash
git clone https://github.com/soufiane-ech/coEnergy-assessment.git
cd coEnergy-assessment
uv sync
```

`uv sync` crée l'environnement `.venv/` à partir de `uv.lock` : PyTorch (version CPU),
NumPy, pandas, Matplotlib et Jupyter.

**Données.** Le dossier de simulations `quentin_house` fourni avec l'assessment n'est pas
versionné (3,9 Go). Il se place ainsi :

```
coEnergy-assessment/data/quentin_house/batch_1.pt … batch_20.pt
```

---

## 6. Exécuter les notebooks

```bash
uv run jupyter lab
```

Dans VS Code, il suffit d'ouvrir un notebook et de choisir le noyau `.venv` du projet.

| Notebook | Durée sur CPU | Ce qui se passe |
|---|---|---|
| `01_eda.ipynb` | ≈ 3 min la première fois, < 1 min ensuite | une passe sur les 20 batchs, un à la fois, puis mise en cache dans `outputs/` |
| `02_modelisation.ipynb` | ≈ 5 min | relit `runs/`, et la section 6 extrait deux groupes de simulations des données brutes |

Les deux notebooks ont besoin des données dans `data/quentin_house/`. Les caches de
`outputs/` sont recalculés automatiquement s'ils sont absents. Un seul batch est chargé en
mémoire à la fois : 8 Go de RAM suffisent.

---

## 7. Réentraîner un modèle

Le point d'entrée est `coenergy.train`. **Test rapide sur CPU** (≈ 1 min, trois batchs) :

```bash
uv run python -m coenergy.train --model cnn_gru --batches 1 17 19 --max-per-batch 300 \
    --epochs 2 --out-dir runs_test/cnn_gru
```

**Entraînement complet**, sur GPU (≈ 20 min pour le CNN-GRU sur un Tesla T4) :

```bash
python -m coenergy.train --model cnn_gru --epochs 20 --batch-size 1024 --lr 3e-3 \
    --data-dir data/quentin_house --out-dir runs_new/cnn_gru
```

- **GPU :** `uv.lock` installe PyTorch en version CPU, pour que l'environnement soit léger
  et identique partout. Pour entraîner sur GPU, il faut un PyTorch compatible CUDA. Les
  modèles de `runs/` ont été entraînés sur Kaggle, dont l'environnement fournit déjà
  PyTorch CUDA : le code y est lancé avec `PYTHONPATH=src python -m coenergy.train …`, sans
  autre installation.
- **Modèles disponibles** (`--model`) : `mean`, `ridge`, `cnn`, `cnn_gru`, `cnn_free`,
  `cnn_hlc`. La taille se règle avec `--channels` et `--hidden` (par exemple
  `--channels 128 --hidden 256` pour `cnn_gru_large`), la durée avec `--epochs`.
- **Découpage :** il est relu dans `splits/split_seed42.npz`, ce qui rend les résultats
  comparables d'un entraînement à l'autre.

---

## 8. Correspondance avec l'assessment

| Exigence de l'assessment | Où la trouver |
|---|---|
| Notebook-rapport exécutable : démarche, EDA, choix méthodologiques, résultats | `notebooks/01_eda.ipynb` et `notebooks/02_modelisation.ipynb` |
| Modules Python propres, typés, documentés, importés dans le notebook | `src/coenergy/` |
| Environnement reproductible avec uv | `pyproject.toml`, `uv.lock` |
| Modèles implémentés exclusivement en PyTorch | `src/coenergy/models.py`, `src/coenergy/train.py` |
| Contrainte $HLC = HLC_{ext} + HLC_{mit}$ | tête physique, `models.PhysicalHead` ; vérification dans `02_modelisation`, section 4.1 |
| Métriques par famille d'indicateurs | `02_modelisation`, section 3.2 |
| Rigueur du protocole d'évaluation | `02_modelisation`, section 1 ; découpage dans `dataset.py` |
| Analyse critique des erreurs | `02_modelisation`, sections 5 à 7 |
