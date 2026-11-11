#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../.build/report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../.build/report main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../.build/report main.tex
cp ../.build/report/main.pdf ../../submission/Tomato_Report.pdf
