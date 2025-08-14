import torch
from torch import nn
from src.instrument.hvp import hvp
class Quad(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, -2.0]))
    def forward(self, x):
        return 0.5*(self.w @ self.w)
def loss_fn(m, batch): return m(None)
def test_hvp_symmetry():
    m = Quad()
    v = torch.tensor([0.7, -0.3])
    w = torch.tensor([-0.2, 0.9])
    Hv_v = hvp(loss_fn, m, None, v)
    Hv_w = hvp(loss_fn, m, None, w)
    assert torch.allclose(Hv_v, v, atol=1e-6)
    assert torch.allclose(Hv_w, w, atol=1e-6)
    assert abs(v @ Hv_w - w @ Hv_v) < 1e-6
