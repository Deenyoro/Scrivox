# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Scrivox — onedir build with CUDA support.

Supports three variants via SCRIVOX_VARIANT env var:
  - Lite:    Transcription only (no pyannote/speechbrain)
  - Regular: All features, no bundled models
  - Full:    All features + bundled diarization models

Set SCRIVOX_DIST_NAME env var to change output directory name (default: Scrivox).
"""

import glob
import os
import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

block_cipher = None
dist_name = os.environ.get('SCRIVOX_DIST_NAME', 'Scrivox')
variant = os.environ.get('SCRIVOX_VARIANT', 'regular').lower()
is_lite = variant == 'lite'

# ── Always collected (all variants) ──
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
ct2_datas, ct2_binaries, ct2_hiddenimports = collect_all('ctranslate2')
fw_datas, fw_binaries, fw_hiddenimports = collect_all('faster_whisper')

all_datas = torch_datas + ct2_datas + fw_datas
all_binaries = torch_binaries + ct2_binaries + fw_binaries
all_hiddenimports = torch_hiddenimports + ct2_hiddenimports + fw_hiddenimports

# ── Ensure CUDA DLLs from torch/lib are included ──
# collect_all('torch') should get these, but we explicitly add them as a safety net
import torch as _torch
_torch_lib = os.path.join(os.path.dirname(_torch.__file__), 'lib')
for _dll in glob.glob(os.path.join(_torch_lib, '*.dll')):
    _dll_name = os.path.basename(_dll)
    # Check if already in binaries
    _already = any(_dll_name == os.path.basename(b[0]) for b in all_binaries)
    if not _already:
        all_binaries.append((_dll, 'torch/lib'))

# ── Collect nvidia CUDA runtime packages if present ──
for _nvidia_pkg in [
    'nvidia.cuda_runtime', 'nvidia.cublas', 'nvidia.cufft',
    'nvidia.curand', 'nvidia.cusolver', 'nvidia.cusparse',
    'nvidia.cudnn', 'nvidia.nccl', 'nvidia.nvtx',
]:
    try:
        _d, _b, _h = collect_all(_nvidia_pkg)
        all_datas += _d
        all_binaries += _b
        all_hiddenimports += _h
    except Exception:
        pass

# ── Regular/Full only (diarization + advanced features) ──
if not is_lite:
    pa_datas, pa_binaries, pa_hiddenimports = collect_all('pyannote.audio')
    sb_datas, sb_binaries, sb_hiddenimports = collect_all('speechbrain')
    ta_datas, ta_binaries, ta_hiddenimports = collect_all('torchaudio')
    pc_datas, pc_binaries, pc_hiddenimports = collect_all('pyannote.core')
    pd_datas, pd_binaries, pd_hiddenimports = collect_all('pyannote.database')
    pp_datas, pp_binaries, pp_hiddenimports = collect_all('pyannote.pipeline')
    tm_datas, tm_binaries, tm_hiddenimports = collect_all('torchmetrics')

    all_datas += pa_datas + sb_datas + ta_datas + pc_datas + pd_datas + pp_datas + tm_datas
    all_binaries += pa_binaries + sb_binaries + ta_binaries + pc_binaries + pd_binaries + pp_binaries + tm_binaries
    all_hiddenimports += (
        pa_hiddenimports + sb_hiddenimports + ta_hiddenimports
        + pc_hiddenimports + pd_hiddenimports + pp_hiddenimports + tm_hiddenimports
        + [
            'sklearn', 'sklearn.cluster', 'sklearn.utils', 'sklearn.utils._cython_blas',
            'sklearn.neighbors', 'sklearn.neighbors._typedefs',
            'sklearn.metrics', 'sklearn.metrics.pairwise', 'sklearn.preprocessing',
            'scipy', 'scipy.signal', 'scipy.spatial', 'scipy.sparse',
            'scipy.linalg', 'scipy.special', 'scipy.stats',
            'scipy.fft', 'scipy.interpolate',
            'asteroid_filterbanks',
            'pyannote.audio.pipelines',
            'pyannote.audio.pipelines.speaker_diarization',
        ]
    )

# Common hidden imports for all variants
all_hiddenimports += [
    'sounddevice', 'pynput', 'pynput.keyboard', 'pynput.keyboard._win32',
    'keyboard', 'pyperclip', 'dotenv',
]

# Metadata needed for version detection
metadata_packages = [
    'torch', 'transformers', 'huggingface_hub', 'tokenizers',
    'safetensors', 'tqdm', 'packaging', 'filelock', 'numpy',
    'requests', 'certifi', 'charset_normalizer', 'urllib3', 'idna',
    'faster_whisper', 'ctranslate2',
]
if not is_lite:
    metadata_packages += ['pyannote.audio', 'speechbrain', 'torchaudio']

extra_datas = []
for pkg in metadata_packages:
    try:
        extra_datas += copy_metadata(pkg)
    except Exception:
        pass

# Include .env.example
extra_datas.append(('.env.example', '.'))

# Icon
icon_file = 'assets/scrivox.ico' if os.path.isfile('assets/scrivox.ico') else None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas + extra_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_cuda.py'],
    excludes=[
        'IPython', 'pytest',
        'tkinter.test',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Scrivox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX breaks CUDA DLLs
    console=False,  # Windowed mode — no CMD window for GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=dist_name,
)
