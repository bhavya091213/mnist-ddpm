import pytest
import torch

from mnist_flow.config import Config


@pytest.fixture
def tiny_cfg() -> Config:
    return Config.tiny()


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")
