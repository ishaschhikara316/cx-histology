import torch

from workflow.scripts.abmil import GatedABMIL


def test_abmil_forward_output_shapes():
    model = GatedABMIL(in_dim=1024, hidden=256, n_out=17, dropout=0.25)
    x = torch.randn(3, 400, 1024)           # batch=3, tiles=400
    pred, attn = model(x)
    assert pred.shape == (3, 17)
    assert attn.shape == (3, 400)
    assert torch.allclose(attn.sum(dim=1), torch.ones(3), atol=1e-5)


def test_abmil_handles_variable_bag_sizes():
    model = GatedABMIL(in_dim=1024, hidden=256, n_out=17)
    # Simulate padded bag via a mask.
    x = torch.randn(2, 500, 1024)
    mask = torch.zeros(2, 500, dtype=torch.bool)
    mask[0, :300] = True       # slide 0 has 300 valid tiles
    mask[1, :500] = True       # slide 1 has 500
    pred, attn = model(x, mask=mask)
    assert pred.shape == (2, 17)
    assert attn[0, 300:].abs().sum() == 0       # masked positions zeroed
    assert torch.allclose(attn[0].sum(), torch.tensor(1.0), atol=1e-5)


def test_abmil_is_deterministic_with_fixed_seed():
    torch.manual_seed(0)
    m1 = GatedABMIL(1024, 256, 17)
    torch.manual_seed(0)
    m2 = GatedABMIL(1024, 256, 17)
    x = torch.randn(1, 100, 1024)
    torch.manual_seed(99)
    p1, _ = m1(x)
    torch.manual_seed(99)
    p2, _ = m2(x)
    assert torch.allclose(p1, p2)
