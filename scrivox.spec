# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Scrivox — onedir build with CUDA support.

Supports two variants:
  - Full: includes bundled diarization models in models/ directory
  - Lite: no bundled models, users provide their own HF token

Set SCRIVOX_DIST_NAME env var to change output directory name (default: Scrivox).
"""

import os
import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

block_cipher = None
dist_name = os.environ.get('SCRIVOX_DIST_NAME', 'Scrivox')

# Collect large packages — every package that is imported at runtime
# must be collected here so PyInstaller bundles all its files.
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
ct2_datas, ct2_binaries, ct2_hiddenimports = collect_all('ctranslate2')
fw_datas, fw_binaries, fw_hiddenimports = collect_all('faster_whisper')
pa_datas, pa_binaries, pa_hiddenimports = collect_all('pyannote.audio')
sb_datas, sb_binaries, sb_hiddenimports = collect_all('speechbrain')
ta_datas, ta_binaries, ta_hiddenimports = collect_all('torchaudio')
pc_datas, pc_binaries, pc_hiddenimports = collect_all('pyannote.core')
pd_datas, pd_binaries, pd_hiddenimports = collect_all('pyannote.database')
pp_datas, pp_binaries, pp_hiddenimports = collect_all('pyannote.pipeline')
tm_datas, tm_binaries, tm_hiddenimports = collect_all('torchmetrics')

# Metadata needed for version detection
metadata_packages = [
    'torch', 'transformers', 'huggingface_hub', 'tokenizers',
    'safetensors', 'tqdm', 'packaging', 'filelock', 'numpy',
    'requests', 'certifi', 'charset_normalizer', 'urllib3', 'idna',
    'faster_whisper', 'ctranslate2', 'pyannote.audio', 'speechbrain',
    'torchaudio',
]

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
    binaries=(
        torch_binaries + ct2_binaries + fw_binaries + pa_binaries + sb_binaries
        + ta_binaries + pc_binaries + pd_binaries + pp_binaries + tm_binaries
    ),
    datas=(
        torch_datas + ct2_datas + fw_datas + pa_datas + sb_datas
        + ta_datas + pc_datas + pd_datas + pp_datas + tm_datas
        + extra_datas
    ),
    hiddenimports=(
        torch_hiddenimports + ct2_hiddenimports + fw_hiddenimports
        + pa_hiddenimports + sb_hiddenimports
        + ta_hiddenimports + pc_hiddenimports + pd_hiddenimports
        + pp_hiddenimports + tm_hiddenimports
        + [
            'sounddevice', 'pynput', 'pynput.keyboard', 'pynput.keyboard._win32',
            'keyboard', 'pyperclip', 'dotenv',
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
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    console=True,  # Dual mode: GUI when double-clicked, CLI from terminal
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
