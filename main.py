"""Compatibility wrapper for scripts/notebooks that used `from main import ...`"""

# Lazy import dataset utilities so `import main` works even without torch installed

def get_dataset_classes():
    try:
        from src.data.dataset import SARD2YOLODataset, collate_fn
        return SARD2YOLODataset, collate_fn
    except Exception as e:
        raise RuntimeError('Failed to import dataset utilities; ensure dependencies are installed') from e

# Optional helpers (placeholders)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def detect_image_yolo(img_path, weights=None):
    if YOLO is None:
        raise RuntimeError('ultralytics YOLO not available; install ultralytics')
    model = YOLO(weights) if weights else YOLO()
    return model.predict(img_path)


def detect_image_maskrcnn(*args, **kwargs):
    raise NotImplementedError('Mask R-CNN detection helper not implemented in main; use src.models wrappers')

__all__ = ["get_dataset_classes", "detect_image_yolo", "detect_image_maskrcnn"]
