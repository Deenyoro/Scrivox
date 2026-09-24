# Toolchain bootstrap for the Windows GitLab runner (shell executor).
# Dot-source from before_script. Installs, once per machine, into the
# runner's per-machine tools directory (no admin, no winget, no choco):
#   - CPython 3.11 (NuGet package) + Tcl/Tk from python.org's tcltk.msi
#   - Inno Setup 6 (per-user, silent)
#   - 7-Zip's standalone 7zr.exe (the .7z portable archives)
# and puts them on PATH / in $env:ISCC / $env:SEVENZIP for the job. Later
# jobs find them already present. Every download is pinned and SHA-256
# checked; a download that fails the check is moved aside to
# <name>.bad-<timestamp> (never deleted) and fetched once more. No other
# existing file in the tools dir is modified, moved or removed; other
# projects' Python dirs (python-3.12.10-tk for CameraMeasurementTool,
# Tagestry's plain python-<ver>) are left alone.
# Same layout and approach as CameraMeasurementTool's ci/tools-windows.ps1.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Python 3.11 to match the GitHub Actions build (.github/workflows/release.yml)
# and the Tk 8.6 the released .exe ships. 3.11.9 is the last 3.11 release
# python.org and NuGet publish Windows binaries for.
$PythonVersion      = '3.11.9'
$PythonNupkgSha256  = '9283876d58c017e0e846f95b490da3bca0fc0a6ee1134b2870677cfb7eec3c67'
$TclTkMsiSha256     = 'c8845743fb77abec0f01faa823bf3300e2113ef344758fc36360fee3dd71a8a8'
$InnoVersion        = '6.7.3'
$InnoSha256         = '9c73c3bae7ed48d44112a0f48e66742c00090bdb5bef71d9d3c056c66e97b732'
$SevenZipVersion    = '25.01'
$SevenZipSha256     = '27cbe3d5804ad09e90bbcaa916da0d5c3b0be9462d0e0fb6cb54be5ed9030875'

$Tools = Join-Path $env:ProgramData 'gitlab-runner-tools'
$Downloads = Join-Path $Tools 'downloads'
New-Item -ItemType Directory -Force $Tools, $Downloads | Out-Null

# Download $Url to $Downloads\$Name (reused if already there) and fail unless
# its SHA-256 matches. A cached copy that fails the check (truncated or
# corrupt) is moved aside to <name>.bad-<timestamp> (kept for inspection,
# never deleted) and downloaded once more; only a second mismatch fails.
# Returns the local path.
function Get-VerifiedDownload([string]$Url, [string]$Name, [string]$Sha256) {
  $out = Join-Path $Downloads $Name
  foreach ($attempt in 1, 2) {
    if (-not (Test-Path $out)) {
      Write-Host "Downloading $Url"
      Invoke-WebRequest $Url -OutFile "$out.part" -UseBasicParsing
      Move-Item "$out.part" $out -Force
    }
    $actual = (Get-FileHash -Algorithm SHA256 $out).Hash
    if ($actual -eq $Sha256.ToUpperInvariant()) { return $out }
    if ($attempt -eq 2) {
      throw "SHA-256 mismatch for ${Name} after a fresh download: expected $Sha256, got $actual"
    }
    $bad = "$out.bad-$(Get-Date -Format 'yyyyMMddHHmmss')"
    Move-Item $out $bad
    Write-Warning "SHA-256 mismatch for ${Name} (got $actual, expected $Sha256); moved aside to $bad (not deleted), downloading once more"
  }
}

# ---- Python + Tk ---------------------------------------------------------
# python.org's bundle installer will not run under the runner service, so
# use the NuGet package: the same CPython as a plain zip, with pip. It has
# NO Tcl/Tk/tkinter, and Scrivox's GUI is Tkinter, so add Tk from
# python.org's own per-component MSI for the SAME version. `msiexec /a` is an
# administrative extract: it unpacks the files into TARGETDIR and
# installs/registers nothing. Layout of the 3.11.9 tcltk.msi (verified by
# extracting it on Linux; unlike 3.12's it has no zlib1.dll):
#   DLLs\_tkinter.pyd, tcl86t.dll, tk86t.dll              -> <python>\DLLs\
#   Lib\tkinter\                                        -> <python>\Lib\tkinter\
#   tcl\  (tcl8.6, tk8.6, tcl8, tix8.4.3, reg1.3, dde1.4) -> <python>\tcl\
# (Lib\idlelib, Lib\turtledemo and the Start-menu entry are skipped.)
$python = Join-Path $Tools "python-$PythonVersion-tk"
$pythonReady = ((Test-Path (Join-Path $python 'python.exe')) -and
                (Test-Path (Join-Path $python 'DLLs\_tkinter.pyd')) -and
                (Test-Path (Join-Path $python 'Lib\tkinter\__init__.py')) -and
                (Test-Path (Join-Path $python 'tcl\tk8.6')))
if (-not $pythonReady) {
  Write-Host "Installing Python $PythonVersion with Tcl/Tk into $python"
  $pkg = Get-VerifiedDownload "https://www.nuget.org/api/v2/package/python/$PythonVersion" `
                              "python.$PythonVersion.nupkg.zip" $PythonNupkgSha256
  $nuget = Join-Path $Tools "staging\python-$PythonVersion-nuget"
  Expand-Archive $pkg -DestinationPath $nuget -Force
  New-Item -ItemType Directory -Force $python | Out-Null
  Copy-Item (Join-Path $nuget 'tools\*') $python -Recurse -Force

  $msi = Get-VerifiedDownload "https://www.python.org/ftp/python/$PythonVersion/amd64/tcltk.msi" `
                              "tcltk-$PythonVersion-amd64.msi" $TclTkMsiSha256
  $tk = Join-Path $Tools "staging\tcltk-$PythonVersion"
  New-Item -ItemType Directory -Force $tk | Out-Null
  $p = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @(
    '/a', "`"$msi`"", '/qn', "TARGETDIR=`"$tk`"")
  if ($p.ExitCode -ne 0) { throw "msiexec /a tcltk.msi failed (exit $($p.ExitCode))" }
  foreach ($f in '_tkinter.pyd', 'tcl86t.dll', 'tk86t.dll') {
    $src = Join-Path $tk "DLLs\$f"
    if (-not (Test-Path $src)) { throw "tcltk.msi extract is missing DLLs\$f" }
    Copy-Item $src (Join-Path $python 'DLLs') -Force
  }
  foreach ($d in 'Lib\tkinter', 'tcl') {
    $src = Join-Path $tk $d
    if (-not (Test-Path $src)) { throw "tcltk.msi extract is missing $d" }
    $dst = Join-Path $python $d
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item (Join-Path $src '*') $dst -Recurse -Force
  }
}

# ---- Inno Setup ----------------------------------------------------------
$inno = Join-Path $Tools "innosetup-$InnoVersion"
if (-not (Test-Path (Join-Path $inno 'ISCC.exe'))) {
  Write-Host "Installing Inno Setup $InnoVersion into $inno"
  $setup = Get-VerifiedDownload "https://github.com/jrsoftware/issrc/releases/download/is-$($InnoVersion -replace '\.','_')/innosetup-$InnoVersion.exe" `
                                "innosetup-$InnoVersion.exe" $InnoSha256
  $p = Start-Process $setup -Wait -PassThru -ArgumentList @(
    '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/CURRENTUSER',
    "/DIR=`"$inno`"", '/NOICONS')
  if ($p.ExitCode -ne 0) { throw "Inno Setup installer failed (exit $($p.ExitCode))" }
  if (-not (Test-Path (Join-Path $inno 'ISCC.exe'))) { throw "ISCC.exe not found in $inno after install" }
}

# ---- 7-Zip ---------------------------------------------------------------
# The GitHub runner image had 7z preinstalled; this laptop may not. 7zr.exe
# is 7-Zip's standalone console build: it only handles .7z, which is all the
# portable archives need, and it needs no install.
$sevenZipDir = Join-Path $Tools "7zr-$SevenZipVersion"
$sevenZip = Join-Path $sevenZipDir '7zr.exe'
if (-not (Test-Path $sevenZip)) {
  $dl = Get-VerifiedDownload "https://github.com/ip7z/7zip/releases/download/$SevenZipVersion/7zr.exe" `
                             "7zr-$SevenZipVersion.exe" $SevenZipSha256
  New-Item -ItemType Directory -Force $sevenZipDir | Out-Null
  Copy-Item $dl $sevenZip -Force
}

$env:PATH = "$python;$python\Scripts;$env:PATH"
$env:ISCC = Join-Path $inno 'ISCC.exe'
$env:SEVENZIP = $sevenZip
# The CUDA torch wheels are ~2.5 GB; the shared pip cache means only the
# first job on this machine downloads them.
$env:PIP_CACHE_DIR = Join-Path $Tools 'pip-cache'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

# Gate: this is a Tkinter app; refuse to build with a Python that lacks Tk 8.6.
$tkv = & (Join-Path $python 'python.exe') -c "import tkinter; tkinter.Tcl(); print(tkinter.TkVersion)"
if ($LASTEXITCODE -ne 0 -or "$tkv".Trim() -ne '8.6') { throw "tkinter check failed: got '$tkv' (need 8.6)" }

Write-Host ("python {0} (Tk {1}) / ISCC {2} / 7zr {3}" -f (& python --version), "$tkv".Trim(), $env:ISCC, $SevenZipVersion)
