# Projet : assessment technique coEnergy (stage PFE)

## Contexte
- Candidat : étudiant en Industrial Data Science (EMINES, UM6P). À l'aise en Python/ML/PyTorch,
  débutant en thermique du bâtiment : expliquer les notions physiques quand elles apparaissent.
- Date limite de rendu : 18/11/2026 à 23h59 (envoi à reda.nassif@coenergy.fr).
- Langue : répondre en français, termes techniques en anglais quand c'est l'usage.

## Sources (dans docs/ et provided/)
- docs/Assessment_de_stage_coEnergy.pdf : SPÉCIFICATION OFFICIELLE. Prioritaire sur tout le reste.
- docs/paper_coEnergy_HLC.pdf : article de référence scientifique (pas une spécification à reproduire).
- docs/emails_coEnergy.md : précisions de coEnergy (Kaggle autorisé, GPU gratuits).
- provided/util_assessement.py : loader officiel des données. Ne jamais le modifier.
- data/quentin_house/ : données brutes (1 000 000 de simulations, batchs de 50 000). Lecture seule.

## Le problème
Régression multi-sortie sur séries temporelles, en PyTorch exclusivement.
- Entrées : indoor_temperature T_int(t) et indoor_U U(t).
- Sorties (6) : HLC, HLC_ext, HLC_mit (W/K) ; tau_b, tau_n, tau_inf (h).
- Contrainte physique exacte : HLC = HLC_ext + HLC_mit.
- Livrables : notebook .ipynb (rapport exécutable), modules .py typés et documentés,
  environnement reproductible avec uv (pyproject.toml + uv.lock).

## Points à ne PAS supposer (à vérifier dans les données et le loader)
- indoor_U n'est pas forcément identique à P_int de l'article (signe, unité, brute/nette).
- Le pas de temps, la longueur des séries, l'ordre des colonnes CDT, les labels de Hb.
- L'indépendance des simulations (bâtiments ou excitations partagés : risque de fuite au split).
- L'article prédit seulement le HLC avec un CNN-GRU ; l'assessment demande 6 sorties.
  Ne pas reproduire l'article par défaut.

## Organisation
- Développement et EDA en local (VS Code). Entraînement complet sur Kaggle (GPU), le code étant
  récupéré via GitHub. Le code ne se modifie qu'en local.
- data/ ne doit jamais être versionné dans git.
- Mémoire : ne jamais charger tout le dataset sans avoir vérifié sa taille et la RAM disponible.

## Règles de travail (OBLIGATOIRES)
1. Avant chaque étape : expliquer ce que tu comptes faire et pourquoi, puis ATTENDRE ma validation.
2. Ne jamais créer ni modifier plusieurs fichiers d'un coup sans plan validé.
3. Ne jamais inventer de résultat, de dimension ou de définition : inspecter d'abord.
   Si une information manque, t'arrêter et me le dire.
4. Distinguer toujours : ce que dit l'assessment / ce que dit l'article / ce qui est mesuré
   dans les données / ce qui est une hypothèse.
5. Chaque analyse de l'EDA doit répondre à une question précise et mener à une décision
   pour la suite. Pas d'EDA générique.
6. Une étape à la fois. Résumer ce qui a été fait à la fin de chaque étape.
7. Ne jamais exécuter de commande git (init, add, commit, push, etc.). Je gère git moi-même.
   Tu peux créer ou modifier le fichier .gitignore.