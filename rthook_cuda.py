"""PyInstaller runtime hook — set up CUDA DLL search paths.

Runs before the main script. Ensures CUDA DLLs bundled in torch/lib/
are findable by all native extensions (torch, ctranslate2, etc.).

Two DLL loading mechanisms are used by our dependencies:
  - torch uses LoadLibraryExW with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    -> needs os.add_dll_directory()
  - ctranslate2 uses LoadLibraryA (plain)
    -> needs PATH or SetDllDirectoryA

We handle both by adding torch/lib to os.add_dll_directory AND prepending
to PATH. We also clear CUDA_PATH to prevent ctranslate2's C++ code from
calling SetDllDirectoryA with a system CUDA path, which would cause it to
find system DLLs before our bundled ones.

If the user opts to use their system CUDA installation (via the
``use_system_cuda`` flag file next to the exe, or the
SCRIVOX_USE_SYSTEM_CUDA=1 env var), we skip forcing bundled CUDA paths
and only add the ctranslate2 dir for libiomp5md.dll.
"""
import os
import sys

_base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Check for system CUDA override: flag file next to exe or env var
_use_system = (os.environ.get('SCRIVOX_USE_SYSTEM_CUDA', '0') == '1'
               or os.path.isfile(os.path.join(_base, '..', 'use_system_cuda')))

# ctranslate2 package dir contains ctranslate2.dll, cudnn64_9.dll shim, libiomp5md.dll
_ct2_dir = os.path.join(_base, 'ctranslate2')

if _use_system:
    # System CUDA mode: only add ctranslate2 dir for libiomp5md.dll
    # Leave CUDA_PATH and PATH alone so system CUDA is found
    if os.path.isdir(_ct2_dir):
        os.add_dll_directory(_ct2_dir)
else:
    # Bundled CUDA mode (default): force bundled DLLs
    # torch/lib contains all CUDA DLLs: cuBLAS, cuDNN, cudart, cusolver, etc.
    _torch_lib = os.path.join(_base, 'torch', 'lib')
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')

    if os.path.isdir(_ct2_dir):
        os.add_dll_directory(_ct2_dir)

    # Prevent ctranslate2's C++ cublas_stub from calling SetDllDirectoryA
    # with a system CUDA path (CUDA_PATH/bin), which would override our
    # bundled DLLs in the DLL search order.
    os.environ.pop('CUDA_PATH', None)
