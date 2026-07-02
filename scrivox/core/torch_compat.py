"""PyTorch 2.6+ compatibility: context manager for weights_only=False loading."""

import contextlib
import threading

import torch

_original_torch_load = torch.load
_patch_lock = threading.Lock()
_patch_depth = 0


@contextlib.contextmanager
def _allow_unsafe_torch_load():
    """Temporarily force weights_only=False for pyannote model loading.

    Reference-counted so overlapping uses from concurrent threads don't
    restore the original torch.load while another load is still in flight.
    """
    global _patch_depth

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    with _patch_lock:
        if _patch_depth == 0:
            torch.load = patched
        _patch_depth += 1
    try:
        yield
    finally:
        with _patch_lock:
            _patch_depth -= 1
            if _patch_depth == 0:
                torch.load = _original_torch_load
