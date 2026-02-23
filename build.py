"""PyInstaller build wrapper for Scrivox.

Build variants:
  python build.py --lite      Transcription only, no pyannote (smallest)
  python build.py --regular   All features, no bundled models
  python build.py --full      All features + bundled diarization models
  python build.py             Builds all 3 variants
  python build.py --clean     Clean build/ and dist/ before building
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

    Uses huggingface_hub.snapshot_download instead of loading models into memory,
    so no torch/GPU is needed — just file downloads.
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

    os.environ["HF_HOME"] = models_dir
    os.environ["HF_HUB_CACHE"] = hub_dir

    # Model repos match defaults in scrivox/core/constants.py
    model_repos = [
        "pyannote/speaker-diarization-3.1",   # DEFAULT_DIARIZATION_MODEL
        "pyannote/segmentation-3.0",           # DEFAULT_SEGMENTATION_MODEL
        "speechbrain/spkrec-ecapa-voxceleb",   # DEFAULT_SPEAKER_EMBEDDING_MODEL
    ]

    try:
        from huggingface_hub import snapshot_download

        for repo in model_repos:
            print(f"  Downloading {repo}...")
            snapshot_download(
                repo,
                token=hf_token,
                cache_dir=hub_dir,
            )

        size = get_size_mb(models_dir)
        print(f"  Models downloaded successfully ({size:.0f} MB)")
        return True

    except Exception as e:
        print(f"Error downloading models: {e}", file=sys.stderr)
        return False


def _clean_build_dir(project_dir):
    """Clear build/ directory between variants (different package sets)."""
    build_dir = os.path.join(project_dir, "build")
    if os.path.isdir(build_dir):
        print("Clearing build/ for next variant...")
        shutil.rmtree(build_dir)


def run_build(project_dir, spec_file, variant_name, dist_name, variant):
    """Run PyInstaller build for a specific variant."""
    print()
    print("=" * 60)
    print(f"  Building {variant_name}")
    print("=" * 60)
    print()

    t0 = time.time()
    env = os.environ.copy()
    env["SCRIVOX_DIST_NAME"] = dist_name
    env["SCRIVOX_VARIANT"] = variant

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
    parser.add_argument("--lite", action="store_true",
                        help="Build Lite variant (transcription only, smallest)")
    parser.add_argument("--regular", action="store_true",
                        help="Build Regular variant (all features, no bundled models)")
    parser.add_argument("--full", action="store_true",
                        help="Build Full variant (all features + bundled diarization models)")
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

    # Default: build all 3 if no flags specified
    build_lite = args.lite or (not args.lite and not args.regular and not args.full)
    build_regular = args.regular or (not args.lite and not args.regular and not args.full)
    build_full = args.full or (not args.lite and not args.regular and not args.full)

    # ── Lite build (transcription only) ──
    if build_lite:
        if not run_build(project_dir, spec_file, "Scrivox Lite", "Scrivox-Lite", "lite"):
            sys.exit(1)
        _clean_build_dir(project_dir)

    # ── Regular build (all features, no bundled models) ──
    if build_regular:
        if not run_build(project_dir, spec_file, "Scrivox", "Scrivox", "regular"):
            sys.exit(1)
        _clean_build_dir(project_dir)

    # ── Full build (with models) ──
    full_explicitly_requested = args.full
    if build_full:
        # Download models if not already present
        if not os.path.isdir(models_dir):
            if not download_diarization_models(models_dir):
                if full_explicitly_requested:
                    print("Full build failed: model download failed.", file=sys.stderr)
                    sys.exit(1)
                print("Skipping Full build due to model download failure.", file=sys.stderr)
                build_full = False

        if build_full:
            if not run_build(project_dir, spec_file, "Scrivox Full", "Scrivox-Full", "full"):
                sys.exit(1)

            # Copy models into dist
            dist_models = os.path.join(project_dir, "dist", "Scrivox-Full", "models")
            if os.path.isdir(models_dir) and not os.path.isdir(dist_models):
                print("Copying bundled models into dist...")
                shutil.copytree(models_dir, dist_models)


if __name__ == "__main__":
    main()
