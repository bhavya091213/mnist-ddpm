import pytest
import torch
from torch import nn

from mnist_flow.config import Config
from mnist_flow.unet import UNet
from mnist_flow.utils import count_params, get_device, set_seed


@pytest.mark.parametrize("cfg", [Config(), Config.tiny()], ids=["default", "tiny"])
def test_groupnorm_divisibility(cfg):
    for m in UNet(cfg).modules():
        if isinstance(m, nn.GroupNorm):
            assert m.num_channels % m.num_groups == 0


def test_default_param_count():
    # Spec asked for 1.5M-2.5M, but the specified architecture (base 64, mults
    # (1,2,2), 2 res blocks/level, 3 up blocks/level with skip concat) gives ~5.9M.
    n = count_params(UNet(Config()))
    print("default params:", n)
    assert 5_000_000 <= n <= 7_000_000


def test_base32_param_count_is_in_spec_range():
    import dataclasses

    n = count_params(UNet(dataclasses.replace(Config(), base_channels=32)))
    print("base32 params:", n)
    assert 1_300_000 <= n <= 2_500_000


def test_forward_shape(tiny_cfg):
    net = UNet(tiny_cfg)
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    assert net(x, t).shape == x.shape


def test_grads_flow_to_all_params(tiny_cfg):
    net = UNet(tiny_cfg)
    out = net(torch.randn(4, 1, 28, 28), torch.rand(4))
    out.pow(2).mean().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing


def test_time_conditioning_matters(tiny_cfg):
    set_seed(0)
    net = UNet(tiny_cfg).eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        a = net(x, torch.zeros(2))
        b = net(x, torch.ones(2))
    assert not torch.allclose(a, b)


def test_get_device_cpu_and_auto():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in ("cuda", "mps", "cpu")


def test_set_seed_reproducible():
    set_seed(3)
    a = torch.randn(3)
    set_seed(3)
    assert torch.equal(a, torch.randn(3))
