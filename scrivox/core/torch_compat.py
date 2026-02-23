"""PyTorch 2.6+ compatibility: context manager for weights_only=False loading."""

import contextlib
import torch

_original_torch_load = torch.load


@contextlib.contextmanager
def _allow_unsafe_torch_load():
    """Temporarily force weights_only=False for pyannote model loading."""
    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched
    try:
        yield
    finally:
        torch.load = _original_torch_load
