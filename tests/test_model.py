import torch

from ml.model import MobileNetMILRQSA


def test_forward_shapes():
    model = MobileNetMILRQSA()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    iou = torch.full((2, 1), 0.5)
    with torch.no_grad():
        logits, masks = model(x, iou)
    assert logits.shape == (2, 1)
    assert masks.shape == (2, 1, 7, 7)


def test_score_regions_probability_range():
    model = MobileNetMILRQSA()
    model.eval()
    x = torch.randn(3, 3, 224, 224)
    probs = model.score_regions(x)
    assert probs.shape == (3,)
    assert bool(((probs >= 0) & (probs <= 1)).all())
