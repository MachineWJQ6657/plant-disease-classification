param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    foreach ($seed in @(664,665,666)) {
        & $Python verify.py --out "results/fresh_views/seed_$seed"
        if ($LASTEXITCODE -ne 0) { throw "Artifact verification failed: seed $seed" }
    }
    & $Python demo.py examples/field.jpg
    if ($LASTEXITCODE -ne 0) { throw 'First demo failed' }
    & $Python demo.py examples/field_second.jpg
    if ($LASTEXITCODE -ne 0) { throw 'Second demo failed' }
} finally { Pop-Location }
