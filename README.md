# Projet Machine Learning — Réorganisation

Résumé des changements effectués :

- Nouveau package `src/` avec :
  - `src/data/` : code dataset (copie de `Dataset.py` → `src/data/dataset.py`)
  - `src/augment/` : scripts d'augmentation (copies de `Adaptability/`)
  - `src/models/`, `src/utils/` : emplacements prêts pour wrappers/utilitaires
- Notebooks déplacés vers `notebooks/` (`pipelinemodel.ipynb`, `comparaison.ipynb`, `data_analysis.ipynb`) et imports mis à jour pour utiliser `src`.
- Ajout d'un fichier compatibilité `main.py` à la racine qui ré-exporte `SARD2YOLODataset` et `collate_fn`.
- Ajout d'un test smoke simple `tests/test_smoke.py` qui vérifie l'import du dataset.

Remarques :
- Je n'ai pas modifié les dossiers `mlruns/` et `runs/` (pas d'intervention sur MLflow comme demandé).
- Aucun `.env` créé (non nécessaire pour l'instant).

Comment utiliser :
- Optionnel : installer en editable pour faciliter les imports dans les notebooks :
  - `pip install -e .` (si tu ajoutes un `setup.py` / `pyproject.toml` ultérieurement)
- Ou exécuter les notebooks après avoir ajouté le dossier racine au PYTHONPATH ou en important depuis `src`.

Si tu veux, j'ajoute :
- suppression des anciens fichiers à la racine (ou création de symlinks),
- installation editable (`pyproject.toml` / `setup.cfg`) et tests unitaires plus complets.

---

## Quickstart

1. Créer un environnement virtuel :
   - `python -m venv .venv` et activer (`.venv\Scripts\activate` sur Windows)
2. Installer les dépendances :
   - `pip install -r requirements.txt`
3. Lancer les notebooks dans `notebooks/`.

**Note**: les fichiers originaux ont été déplacés vers `archive/` pour conserver l'historique.

Licence: MIT

