"""CUDA smoke test for the Blackwell (sm_120) laptop GPU; run with `uv run pytest -m gpu`."""

import pytest

pytestmark = pytest.mark.gpu


def test_torch_is_a_cuda_wheel_that_sees_the_blackwell_gpu():
    import torch

    assert "+cpu" not in torch.__version__, "CPU-only wheel: check [tool.uv.sources] torch index"
    assert torch.cuda.is_available()
    # RTX 5070 Laptop = Blackwell, compute capability 12.0; cu130 wheels ship sm_120 SASS.
    assert torch.cuda.get_device_capability(0) == (12, 0)
    ones = torch.ones(64, 64, device="cuda")
    assert float((ones @ ones).sum()) == 64**3  # a real kernel launch, not just device queries
