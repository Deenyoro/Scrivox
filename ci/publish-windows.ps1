# Uploads one variant's builds from out\ straight to the project's Generic
# Package Registry (release runs only), and writes out\links-<dist>.ndjson
# for the release job, which creates the GitLab Release from those links.
#
#   .\ci\publish-windows.ps1 -DistName Scrivox-Lite|Scrivox|Scrivox-Full
#
# Why upload from here instead of passing job artifacts to the release job:
# the CUDA builds are 0.2-2 GB per file (two files per variant), far over
# GitLab's default 100 MB maximum job artifact size. This instance sets no
# generic package file size limit (0 = unlimited); the limit that binds is
# GitHub's 2 GiB per release asset, because gitlab-release-mirror copies the
# release to GitHub, and build-windows.ps1 already kept anything that size
# out of SHA256SUMS-<dist>.txt. Only the files listed there (plus the list
# itself) are uploaded, so a variant whose installer was too big ships just
# its .7z. Uploads go through curl.exe (built into Windows 10 1803+), which
# streams the file; Windows PowerShell 5.1's Invoke-WebRequest -InFile would
# buffer it in memory.
#
# Builds are not reproducible, so re-uploading a name that already exists in
# this package version would add a second file with a different SHA-256
# under the same name; once the release mirror has copied the release to
# GitHub, that makes the mirror fail for this tag. This script therefore
# refuses to publish over an existing file: release a new patch version
# instead.
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
$sumsName = "SHA256SUMS-$DistName.txt"
$sumsPath = Join-Path out $sumsName
if (-not (Test-Path $sumsPath -PathType Leaf)) { throw "out\$sumsName is missing" }
# "<sha256>  <name>" lines written by build-windows.ps1.
$names = @(Get-Content $sumsPath | Where-Object { $_.Trim() } | ForEach-Object {
  if ($_ -notmatch '^[0-9a-f]{64}  (\S+)$') { throw "unexpected line in ${sumsName}: $_" }
  $Matches[1]
})
$archiveName = "$DistName-$tag-win64.7z"
if ($names -notcontains $archiveName) { throw "$sumsName does not list $archiveName" }
$setupName = "$DistName-$env:VERSION-win64-setup.exe"
if ($names -notcontains $setupName) { Write-Warning "$setupName is not published for this release (over the 2 GiB limit; see the build log). The .7z portable is." }
$files = foreach ($n in @($names) + $sumsName) {
  $p = Join-Path out $n
  if (-not (Test-Path $p -PathType Leaf)) { throw "out\$n is missing" }
  Get-Item $p
}

# Refuse before uploading anything if a file of this variant is already in
# the package version (see above). HEAD on the download URL: 200 = exists,
# 404 = not yet. Any other answer is reported and the upload goes ahead.
foreach ($f in $files) {
  $global:LASTEXITCODE = 0
  $code = & $curl -sS -I -o NUL -w '%{http_code}' -H "JOB-TOKEN: $env:CI_JOB_TOKEN" "$pkg/$($f.Name)"
  if ($code -eq '200') {
    throw "$($f.Name) is already in package Scrivox/$env:VERSION. Builds are not reproducible, so publishing it again would add a second file with a different SHA-256 under the same name (and break the GitHub release mirror if it has copied this release). Release a new patch version instead."
  } elseif ($code -ne '404') {
    Write-Warning "Could not check whether $($f.Name) is already published (HTTP '$code', curl exit $LASTEXITCODE); uploading."
  }
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
