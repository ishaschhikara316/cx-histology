import torch

from workflow.scripts.loss import MultiTaskHuber, fit_label_transform, apply_label_transform


def test_multitask_huber_zero_when_perfect_prediction():
    pred = torch.tensor([[0.0, 1.0, -1.0]])
    tgt  = torch.tensor([[0.0, 1.0, -1.0]])
    loss = MultiTaskHuber(n_out=3, delta=1.0)(pred, tgt)
    assert loss.item() == 0.0


def test_multitask_huber_respects_per_signature_scale():
    loss_fn = MultiTaskHuber(n_out=2, delta=1.0)
    pred = torch.tensor([[2.0, 0.0]])
    tgt  = torch.tensor([[0.0, 0.0]])
    # huber(2, delta=1) = 1.5 and huber(0) = 0, averaged across 2 sigs => 0.75.
    baseline = loss_fn(pred, tgt).item()
    assert abs(baseline - 0.75) < 1e-6
    loss_fn.task_scale.data = torch.tensor([2.0, 1.0])
    # CX0 weighted by 1/2 => (0.75 + 0) / 2 = 0.375
    assert loss_fn(pred, tgt).item() < baseline


def test_label_transform_is_invertible_on_reasonable_values():
    y = torch.tensor([[0.01, 0.5, 2.0],
                      [0.02, 0.4, 3.0],
                      [0.03, 0.6, 5.0]])
    mu, sd = fit_label_transform(y)
    z = apply_label_transform(y, mu, sd)
    assert z.shape == y.shape
    assert torch.allclose(z.mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(z.std(dim=0, unbiased=False), torch.ones(3), atol=1e-5)
