import pytest


def test_import_dataset():
    try:
        import torch  # dataset uses torch; if missing, skip
    except Exception:
        pytest.skip('torch not installed in environment')

    from src.data.dataset import SARD2YOLODataset, collate_fn
    ds = SARD2YOLODataset('dataset/images/train', 'dataset/labels/train')
    assert hasattr(ds, '__len__')
