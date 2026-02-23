"""PyInstaller build wrapper for Scrivox.

Build variants:
  python build.py --full     Downloads diarization models, bundles into exe
  python build.py --lite     No models bundled, smaller download
  python build.py            Builds both variants
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


def get_size_mb(path):
    """Get total size of a directory in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def download_diarization_models(models_dir):
    """Download pyannote diarization models into the bundled models directory.

    Requires HF_TOKEN environment variable to be set.
    The token is only used for downloading — it is NOT stored in the output.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable required to download models.", file=sys.stderr)
        print("Set it with: set HF_TOKEN=hf_your_token_here", file=sys.stderr)
        sys.exit(1)

    hub_dir = os.path.join(models_dir, "hub")
    os.makedirs(hub_dir, exist_ok=True)

    print("Downloading diarization models (this may take a few minutes)...")
    print(f"  Cache directory: {models_dir}")

    # Use huggingface_hub to download the models into our custom cache
    os.environ["HF_HOME"] = models_dir
    os.environ["HF_HUB_CACHE"] = hub_dir

    try:
        import warnings
        warnings.filterwarnings("ignore")
        import contextlib
        import torch

        _orig = torch.load

        @contextlib.contextmanager
        def _unsafe():
            def p(*a, **kw):
                kw["weights_only"] = False
                return _orig(*a, **kw)
            torch.load = p
            try:
                yield
            finally:
                torch.load = _orig

        from pyannote.audio import Pipeline
        with _unsafe():
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        size = get_size_mb(models_dir)
        print(f"  Models downloaded successfully ({size:.0f} MB)")
        return True

    except Exception as e:
        print(f"Error downloading models: {e}", file=sys.stderr)
        return False


def run_build(project_dir, spec_file, variant_name, dist_name):
    """Run PyInstaller build."""
    print()
    print("=" * 60)
    print(f"  Building {variant_name}")
    print("=" * 60)
    print()

    t0 = time.time()
    env = os.environ.copy()
    env["SCRIVOX_DIST_NAME"] = dist_name

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", spec_file, "--noconfirm"],
        cwd=project_dir,
        env=env,
    )

    if result.returncode != 0:
        print(f"\nBuild FAILED (exit code {result.returncode})", file=sys.stderr)
        return False

    elapsed = time.time() - t0
    dist_dir = os.path.join(project_dir, "dist", dist_name)

    # Copy .env.example
    env_example_src = os.path.join(project_dir, ".env.example")
    env_example_dst = os.path.join(dist_dir, ".env.example")
    if os.path.isfile(env_example_src) and not os.path.isfile(env_example_dst):
        shutil.copy2(env_example_src, env_example_dst)

    print()
    print("=" * 60)
    print(f"  {variant_name} — Build SUCCEEDED")
    print("=" * 60)
    print(f"  Time:   {elapsed:.0f}s")
    if os.path.isdir(dist_dir):
        size = get_size_mb(dist_dir)
        print(f"  Output: {dist_dir}")
        print(f"  Size:   {size:.0f} MB")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Build Scrivox executable")
    parser.add_argument("--clean", action="store_true",
                        help="Remove build/ and dist/ before building")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true",
                       help="Build Full variant only (with bundled diarization models)")
    group.add_argument("--lite", action="store_true",
                       help="Build Lite variant only (no bundled models)")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    spec_file = os.path.join(project_dir, "scrivox.spec")
    models_dir = os.path.join(project_dir, "models")

    if not os.path.isfile(spec_file):
        print(f"Error: {spec_file} not found", file=sys.stderr)
        sys.exit(1)

    # Clean
    if args.clean:
        for d in ("build", "dist"):
            path = os.path.join(project_dir, d)
            if os.path.isdir(path):
                print(f"Removing {d}/...")
                shutil.rmtree(path)

    build_full = args.full or (not args.full and not args.lite)
    build_lite = args.lite or (not args.full and not args.lite)

    # ── Full build (with models) ──
    if build_full:
        # Download models if not already present
        if not os.path.isdir(models_dir):
            if not download_diarization_models(models_dir):
                print("Skipping Full build due to model download failure.", file=sys.stderr)
                build_full = False

        if build_full:
            if not run_build(project_dir, spec_file, "Scrivox Full", "Scrivox"):
                sys.exit(1)

            # Copy models into dist
            dist_models = os.path.join(project_dir, "dist", "Scrivox", "models")
            if os.path.isdir(models_dir) and not os.path.isdir(dist_models):
                print("Copying bundled models into dist...")
                shutil.copytree(models_dir, dist_models)

    # ── Lite build (no models) ──
    if build_lite:
        # Temporarily hide models dir so the spec doesn't pick it up
        models_backup = None
        if os.path.isdir(models_dir):
            models_backup = models_dir + "_backup"
            os.rename(models_dir, models_backup)

        try:
            if not run_build(project_dir, spec_file, "Scrivox Lite", "Scrivox-Lite"):
                sys.exit(1)
        finally:
            if models_backup and os.path.isdir(models_backup):
                os.rename(models_backup, models_dir)


if __name__ == "__main__":
    main()
