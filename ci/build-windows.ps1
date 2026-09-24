# Builds one Scrivox variant on the Windows GitLab runner: PyInstaller onedir
# (build.py), the portable .7z and the Inno Setup installer. Called from
# .gitlab-ci.yml after ci\tools-windows.ps1 has put Python 3.11 + Tk,
# $env:ISCC and $env:SEVENZIP in place and $env:VERSION is set.
#
#   .\ci\build-windows.ps1 -Variant Lite|Regular|Full
#
# Mirrors the steps of the old GitHub Actions workflow
# (.github/workflows/release.yml): CUDA 12.6 torch, requirements, forced
# CUDA torch reinstall, CUDA DLL check, build.py --clean --<variant>, 7z of
# the dist dir (dotfiles included), ISCC. Output lands in out\ with the same
# file names the GitHub releases used.
param(
  [Parameter(Mandatory = $true)][ValidateSet('Lite', 'Regular', 'Full')][string]$Variant
)
$ErrorActionPreference = 'Stop'

function Invoke-Checked([string]$What, [scriptblock]$Block) {
  $global:LASTEXITCODE = 0
  & $Block
  if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

if (-not $env:VERSION) { throw "VERSION is not set" }
$root = (Get-Location).Path
$tag = "v$env:VERSION"
$distName = @{ Lite = 'Scrivox-Lite'; Regular = 'Scrivox'; Full = 'Scrivox-Full' }[$Variant]
$flag = "--$($Variant.ToLowerInvariant())"

# A fresh venv per job (the runner's git clean removes the previous one);
# the CUDA wheels come from the shared pip cache after the first run.
Invoke-Checked 'venv' { python -m venv .venv-ci }
$py = Join-Path $root '.venv-ci\Scripts\python.exe'
# The venv's _tkinter finds Tcl/Tk through the base NuGet Python's tcl\
# folder (from tcltk.msi). Check that here, before the ~2.5 GB CUDA torch
# install and the PyInstaller run, not only in the bundle check at the end.
Invoke-Checked 'venv tkinter' { & $py -c "import tkinter; tkinter.Tcl(); print('venv Tk', tkinter.TkVersion)" }
$cudaIndex = 'https://download.pytorch.org/whl/cu126'
Invoke-Checked 'pip install CUDA torch' { & $py -m pip install -q torch torchaudio --index-url $cudaIndex }
Invoke-Checked 'pip install requirements' { & $py -m pip install -q -r requirements.txt pyinstaller }
# Transitive dependencies can pull the CPU torch from PyPI over the CUDA one.
Invoke-Checked 'pip force CUDA torch' { & $py -m pip install -q torch torchaudio --index-url $cudaIndex --force-reinstall --no-deps }
Invoke-Checked 'CUDA torch check' { & $py ci\check_build.py torch }
# The Windows-path helpers only take their Windows branches on Windows; the
# Linux test job can only run their no-op side.
Invoke-Checked 'tests.test_winnative' { & $py -m unittest -v tests.test_winnative }

# build.py --clean wipes build\ and dist\ first; Full downloads the
# pyannote models with HF_TOKEN (the job only exists when HF_TOKEN is set).
Invoke-Checked "build.py $flag" { & $py build.py --clean $flag }
$dist = Join-Path $root "dist\$distName"
Invoke-Checked 'bundle check' { & $py ci\check_build.py dist $dist $Variant }

$out = Join-Path $root 'out'
New-Item -ItemType Directory -Force $out | Out-Null

# Archive from inside the dir so dotfiles (.env.example) are included.
$archive = Join-Path $out "$distName-$tag-win64.7z"
Push-Location $dist
try {
  Invoke-Checked '7z' { & $env:SEVENZIP a -bso0 -bsp0 $archive . }
} finally {
  Pop-Location
}

Invoke-Checked 'ISCC' { & $env:ISCC "/DMyAppVersion=$env:VERSION" "/DVariant=$Variant" installer\windows.iss }
$setupName = "$distName-$env:VERSION-win64-setup.exe"
$setup = Join-Path $root "installer\installer_out\$setupName"
if (-not (Test-Path $setup)) { throw "installer was not produced: $setup" }
Copy-Item $setup $out -Force

# GitHub release assets must be under 2 GiB, and gitlab-release-mirror copies
# Scrivox's GitLab releases to GitHub, so that is the limit that binds (this
# GitLab instance sets no generic package file size limit). As in the old
# workflow (.github/check_asset_size.ps1), an oversized installer does not
# fail the variant: it is moved aside to out\oversize\ (kept, not deleted),
# left out of SHA256SUMS and therefore not published, and the .7z portable
# still ships. An oversized .7z would leave the variant with nothing to
# publish, so that fails the job.
$limit = 2147483648
$oversize = Join-Path $out 'oversize'
$ship = @()
foreach ($f in @((Get-Item $archive), (Get-Item (Join-Path $out $setupName)))) {
  $gb = [math]::Round($f.Length / 1GB, 2)
  if ($f.Length -lt $limit) {
    Write-Host "$($f.Name): $gb GiB (under the 2 GiB GitHub release asset limit)"
    $ship += $f
  } elseif ($f.Extension -eq '.7z') {
    throw "$($f.Name) is $gb GiB, at or over the 2 GiB GitHub release asset limit (the release mirror could not copy it); the variant has nothing publishable"
  } else {
    New-Item -ItemType Directory -Force $oversize | Out-Null
    Move-Item -Force $f.FullName $oversize
    Write-Warning ("{0} is {1} GiB, at or over the 2 GiB GitHub release asset limit." -f $f.Name, $gb)
    Write-Warning "It is NOT published for this release (moved to out\oversize\). The .7z portable remains."
  }
}

# Checksums of what ships (published with them on releases; the publish
# step uploads exactly the files listed here).
$sums = Join-Path $out "SHA256SUMS-$distName.txt"
$lines = foreach ($f in $ship) {
  "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 $f.FullName).Hash.ToLowerInvariant(), $f.Name
}
# ASCII + LF so `sha256sum -c` reads it on any platform.
[IO.File]::WriteAllText($sums, (($lines -join "`n") + "`n"), [Text.Encoding]::ASCII)
Get-Content $sums
Get-ChildItem $out
