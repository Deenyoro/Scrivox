"""Runtime feature detection — single source of truth for build variant capabilities."""


def has_diarization() -> bool:
    """False in Lite builds (pyannote not bundled)."""
    try:
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def has_advanced_features() -> bool:
    """Vision/summary gated to Regular/Full (Lite = transcription only)."""
    return has_diarization()


def get_variant_name() -> str:
    """'Lite', 'Regular', or 'Full' for display."""
    if not has_diarization():
        return "Lite"
    from .diarizer import _get_bundled_models_dir
    return "Full" if _get_bundled_models_dir() else "Regular"
