"""PyInstaller runtime hook — add torch/lib to DLL search path for CUDA.

Runs before the main script. Ensures CUDA DLLs bundled in torch/lib/
are findable by ctranslate2 and other native extensions that need CUDA.
"""
import os
import sys

_base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
_torch_lib = os.path.join(_base, 'torch', 'lib')
if os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
    os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')
