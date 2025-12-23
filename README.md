# Projet Machine Learning — Repository overview 🚀

**Short description:** This repository contains code, data and notebooks for an object-detection project (YOLO-based) using custom datasets and training pipelines.

## Project layout 📁

- `dataset/` — dataset source: `data.yaml`, `images/` (train/valid/test) and `labels/` (matching txt annotations).
- `data/` — data processing folders: `raw/`, `interim/`, `processed/`.
- `src/` — project source code (modules for `data`, `models`, `utils`, etc.). Use this for imports in notebooks and scripts.
- `notebooks/` — analysis and experiments (`pipelinemodel.ipynb`, `comparaison.ipynb`, `data_analysis.ipynb`). There are also notebooks at the repo root (e.g., `pipelinemodel.ipynb`).
- `third_party/yolov5/` — YOLOv5 training/eval scripts (`train.py`, `detect.py`, `val.py`) and its `requirements.txt`.
- `runs/` — outputs from training & detection (`runs/train/`, `runs/detect/`, etc.).
- `mlruns/` and `mlflow/` — MLflow experiment logs and artifacts.
- `models/` — saved checkpoints and registry of models (note: large checkpoint files are typically not committed to GitHub; see note below).
- `archive/` — historical copies of removed/archived files.
- `Adaptability/` — augmentation and image transformation scripts.
- Root-level model files (examples): `ssd300_vgg16_final.pth`, `yolo11n.pt`.

## Quickstart — setup & common commands ⚡

1. Create and activate a virtual environment (Windows):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   - If present, install root requirements: `pip install -r requirements.txt`
   - Install YOLOv5 deps: `pip install -r third_party/yolov5/requirements.txt`

3. Run training (example using YOLOv5):
   ```bash
   python third_party/yolov5/train.py --data dataset/data.yaml --cfg yolov5s.yaml --weights '' --project runs/train --name experiment1
   ```
   - Training outputs are saved under `runs/train/<timestamp>/`.

4. Run inference / detection (example):
   ```bash
   python third_party/yolov5/detect.py --weights runs/train/<exp>/weights/best.pt --source dataset/images/test --project runs/detect --name exp_detect
   ```

5. Inspect MLflow runs: open `mlruns/` or start an MLflow UI pointing at the `mlruns/` folder.

## Notebooks & code usage 🧪

- Prefer importing from `src/` (e.g., `from src.data.dataset import ...`) for reproducibility.
- The main model development, training and evaluation worklive in `pipelinemodel.ipynb` (see that notebook for details and example training/eval cells).
- Notebooks that demonstrate pipeline and evaluation live in `notebooks/` and at the repo root.

> **Note:** Model checkpoint files (trained weights) are not included in this repository because they are too large for a standard GitHub push/pull; keep checkpoints in `models/` locally or store them in external storage (cloud bucket or artifact store) and download them when needed.

## Notes & contributions ✍️

- If you want to run notebooks easily, consider installing the package in editable mode (after adding `pyproject.toml` / `setup.cfg`):
  ```bash
  pip install -e .
  ```
- Add tests under `tests/` and CI as needed. Archive or clean large artifacts (models, mlruns) if sharing the repo.

---

**Licence:** MIT