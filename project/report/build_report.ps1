$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Path ../.build/report -Force | Out-Null
    $buildPath=(Resolve-Path -LiteralPath ../.build/report).Path
    & pdflatex '-interaction=nonstopmode' '-halt-on-error' "-output-directory=$buildPath" 'main.tex'
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed' }
    & pdflatex '-interaction=nonstopmode' '-halt-on-error' "-output-directory=$buildPath" 'main.tex'
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX reference pass failed' }
    Copy-Item -LiteralPath ../.build/report/main.pdf -Destination ../../submission/Tomato_Report.pdf -Force
} finally { Pop-Location }
