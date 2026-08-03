# GitHub Releases reject assets of 2 GiB or more. An oversized installer must
# not fail the whole release (the .7z portable for that variant still ships),
# so it is dropped here with a loud warning instead of erroring at upload time.
param([Parameter(Mandatory = $true)][string]$Pattern)

$limit = 2147483648
$found = Get-ChildItem $Pattern -ErrorAction SilentlyContinue
if (-not $found) { throw "no installer matched: $Pattern" }
foreach ($f in $found) {
    $gb = [math]::Round($f.Length / 1e9, 2)
    if ($f.Length -ge $limit) {
        Write-Warning ("{0} is {1} GB, over the 2 GiB GitHub release asset " -f $f.Name, $gb)
        Write-Warning "limit - REMOVING it from this release. The .7z portable remains."
        Remove-Item $f.FullName -Force
    } else {
        Write-Host "$($f.Name): $gb GB (within the 2 GiB asset limit)"
    }
}
