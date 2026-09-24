# Uploads one variant's builds from out\ straight to the project's Generic
# Package Registry (release runs only), and writes out\links-<dist>.ndjson
# for the release job, which creates the GitLab Release from those links.
#
#   .\ci\publish-windows.ps1 -DistName Scrivox-Lite|Scrivox|Scrivox-Full
#
# Why upload from here instead of passing job artifacts to the release job:
# the CUDA builds are 0.3-2 GB per file (two files per variant), far over
# GitLab's default 100 MB maximum job artifact size. The generic package
# registry's default per-file limit is 5 GiB. Uploads go through curl.exe
# (built into Windows 10 1803+), which streams the file; Windows
# PowerShell 5.1's Invoke-WebRequest -InFile would buffer it in memory.
param(
  [Parameter(Mandatory = $true)][ValidateSet('Scrivox-Lite', 'Scrivox', 'Scrivox-Full')][string]$DistName
)
$ErrorActionPreference = 'Stop'

if (-not $env:VERSION) { throw "VERSION is not set" }
$tag = "v$env:VERSION"
# The test job already refuses RELEASE_VERSION runs that are not on the tag;
# re-check here because this job publishes before the release job runs.
if ($env:CI_COMMIT_TAG -ne $tag) {
  throw "Refusing to publish ${tag}: this pipeline runs on '$env:CI_COMMIT_TAG' (ref $env:CI_COMMIT_REF_NAME), not the tag $tag"
}

$curl = Join-Path $env:SystemRoot 'System32\curl.exe'
if (-not (Test-Path $curl)) { throw "curl.exe not found at $curl" }

$pkg = "$env:CI_API_V4_URL/projects/$env:CI_PROJECT_ID/packages/generic/Scrivox/$env:VERSION"
# Exact names (the Regular variant's prefix "Scrivox-" also starts the others').
$files = foreach ($n in "$DistName-$tag-win64.7z", "$DistName-$env:VERSION-win64-setup.exe", "SHA256SUMS-$DistName.txt") {
  $p = Join-Path out $n
  if (-not (Test-Path $p -PathType Leaf)) { throw "out\$n is missing" }
  Get-Item $p
}

$links = foreach ($f in $files) {
  Write-Host ("uploading {0} ({1:N2} GiB)" -f $f.Name, ($f.Length / 1GB))
  $global:LASTEXITCODE = 0
  & $curl -fsS --retry 3 -o NUL -H "JOB-TOKEN: $env:CI_JOB_TOKEN" --upload-file $f.FullName "$pkg/$($f.Name)"
  if ($LASTEXITCODE -ne 0) { throw "upload of $($f.Name) failed (curl exit $LASTEXITCODE)" }
  # Link the API download URL (what the release mirror accepts), not the web one.
  [pscustomobject]@{ name = $f.Name; url = "$pkg/$($f.Name)"; link_type = 'package' } | ConvertTo-Json -Compress
}
$ndjson = Join-Path (Resolve-Path out).Path "links-$DistName.ndjson"
[IO.File]::WriteAllText($ndjson, (($links -join "`n") + "`n"), [Text.Encoding]::ASCII)
Get-Content $ndjson
