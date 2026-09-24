"""Build gates for the Windows GitLab CI job (see .gitlab-ci.yml).

  python ci/check_build.py torch
      Before PyInstaller: the installed torch must be a CUDA build that
      carries its CUDA DLLs (same check as the old GitHub Actions workflow).

  python ci/check_build.py dist <dist-dir> <lite|regular|full>
      After build.py: the onedir output must contain what the release
      needs, so a half-collected bundle fails the job instead of shipping.

Standard library only; torch is imported only by the `torch` check.
"""

import glob
import os
import sys

CUDA_KEYWORDS = ("cublas", "cufft", "cudnn", "cusolver", "cusparse",
                 "curand", "nvrtc", "cudart", "nvjitlink")
MIN_CUDA_DLLS = 5


def _cuda_dlls(lib_dir):
    dlls = glob.glob(os.path.join(lib_dir, "*.dll"))
    return dlls, sorted(os.path.basename(d) for d in dlls
                        if any(k in os.path.basename(d).lower() for k in CUDA_KEYWORDS))


def check_torch():
    import torch

    version = torch.__version__
    if "cu" not in version:
        return [f"expected a CUDA torch build, got {version}"]
    print(f"torch {version} OK")
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    dlls, cuda = _cuda_dlls(lib)
    print(f"Total DLLs: {len(dlls)}, CUDA DLLs: {len(cuda)}")
    for name in cuda:
        print(f"  {name}")
    if len(cuda) < MIN_CUDA_DLLS:
        return [f"expected >={MIN_CUDA_DLLS} CUDA DLLs in {lib}, found {len(cuda)}"]
    return []


def _find_dirs(root, rel):
    """Every directory under root whose path ends with rel (e.g. torch/lib)."""
    parts = tuple(p.lower() for p in rel.split("/"))
    found = []
    for dirpath, _, _ in os.walk(root):
        tail = tuple(p.lower() for p in os.path.normpath(dirpath).split(os.sep)[-len(parts):])
        if tail == parts:
            found.append(dirpath)
    return found


def _size_mb(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(dirpath, name)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def check_dist(dist, variant):
    variant = variant.lower()
    if variant not in ("lite", "regular", "full"):
        return [f"unknown variant {variant!r}"]
    if not os.path.isdir(dist):
        return [f"{dist} does not exist"]
    errors = []

    for name in ("Scrivox.exe", ".env.example"):
        if not os.path.isfile(os.path.join(dist, name)):
            errors.append(f"{name} missing from {dist}")

    torch_libs = _find_dirs(dist, "torch/lib")
    if not torch_libs:
        errors.append("torch/lib missing from the bundle")
    else:
        _, cuda = _cuda_dlls(torch_libs[0])
        print(f"bundled CUDA DLLs in {torch_libs[0]}: {len(cuda)}")
        if len(cuda) < MIN_CUDA_DLLS:
            errors.append(f"expected >={MIN_CUDA_DLLS} CUDA DLLs in the bundle's torch/lib, found {len(cuda)}")

    # scrivox.spec drops nvcuda.dll on purpose: it must come from the user's
    # NVIDIA driver ("forward compatibility was attempted" otherwise).
    for dirpath, _, files in os.walk(dist):
        if any(f.lower() == "nvcuda.dll" for f in files):
            errors.append(f"nvcuda.dll was bundled ({dirpath}); it must come from the driver")

    # Tkinter GUI: the bundle needs _tkinter and the Tcl/Tk script libraries.
    has_tkinter = any(glob.glob(os.path.join(d, "_tkinter*.pyd"))
                      for d, _, _ in os.walk(dist))
    if not has_tkinter:
        errors.append("_tkinter.pyd missing from the bundle")
    # tk.tcl is the core of the Tk script library (PyInstaller 6 puts it in
    # _internal/_tk_data); without it Tk() fails in the frozen app.
    if not any("tk.tcl" in files for _, _, files in os.walk(dist)):
        errors.append("Tk script library (tk.tcl) missing from the bundle")
    if not _find_dirs(dist, "tkinterdnd2"):
        # Optional at runtime (the GUI falls back to browse-only), as in scrivox.spec.
        print("warning: tkinterdnd2 data not bundled; drag-and-drop will be unavailable")

    has_pyannote = bool(_find_dirs(dist, "pyannote/audio")
                        or glob.glob(os.path.join(dist, "**", "pyannote_audio-*.dist-info"), recursive=True))
    if variant == "lite" and has_pyannote:
        errors.append("Lite bundle contains pyannote.audio; scrivox.spec should exclude it")
    if variant != "lite" and not has_pyannote:
        errors.append(f"{variant} bundle is missing pyannote.audio")

    if variant == "full":
        hub = os.path.join(dist, "models", "hub")
        if not (os.path.isdir(hub) and any(e.startswith("models--") for e in os.listdir(hub))):
            errors.append("Full bundle is missing models/hub/models--* (bundled diarization models)")
    elif os.path.isdir(os.path.join(dist, "models", "hub")):
        errors.append(f"{variant} bundle unexpectedly contains models/hub")

    print(f"{dist}: {_size_mb(dist):.0f} MB ({variant})")
    return errors


def main(argv):
    if len(argv) == 1 and argv[0] == "torch":
        errors = check_torch()
    elif len(argv) == 3 and argv[0] == "dist":
        errors = check_dist(argv[1], argv[2])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
