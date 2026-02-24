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
"""
import os
import sys

_base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# torch/lib contains all CUDA DLLs: cuBLAS, cuDNN, cudart, cusolver, etc.
_torch_lib = os.path.join(_base, 'torch', 'lib')
if os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
    os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')

# ctranslate2 package dir contains ctranslate2.dll, cudnn64_9.dll shim, libiomp5md.dll
_ct2_dir = os.path.join(_base, 'ctranslate2')
if os.path.isdir(_ct2_dir):
    os.add_dll_directory(_ct2_dir)

# Prevent ctranslate2's C++ cublas_stub from calling SetDllDirectoryA
# with a system CUDA path (CUDA_PATH/bin), which would override our
# bundled DLLs in the DLL search order.
os.environ.pop('CUDA_PATH', None)
