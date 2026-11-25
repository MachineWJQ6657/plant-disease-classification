# Overleaf project: COMP 664 Topic 2 report

## Upload
1. Overleaf → **New Project → Upload Project** → choose the zip.
2. Menu → Compiler: **pdfLaTeX**; Main document: **main.tex**. (These are Overleaf's defaults.)

## Where to edit
- **Team names:** top of `main.tex`, the block `EDIT HERE` (`\TeamMembers`).
- **Text:** `main.tex`. Appendix A is the earlier single-seed pilot; Appendix B is provenance and commands.
- **Numbers and tables:** `figures/fresh_*.tex` and `figures/counts.tex` are *generated* from
  `project/results/fresh_views/` by `project/make_fresh_figures.py`. Edit them here only for typos;
  if results change, regenerate them in the repository and copy the files over.
  - `fresh_numbers.tex`: macros used in the abstract and captions (`\FreshOrdFieldAcc`, ...)
  - `fresh_interpretation.tex`: the one-sentence conclusion `\FreshConclusion`
  - `fresh_demo.tex`: predictions shown under the two example photographs
- **Figures:** `figures/fresh_performance.pdf`, `fresh_history.pdf`, `fresh_confusions.pdf` (vector PDFs from the
  same script). Example photographs and credits: `figures/examples/`.

## Not included
Code, weights, raw data and the Notebook stay in the repository
(https://github.com/MachineWJQ6657/plant-disease-classification); this project holds only the report.
Figures from the earlier pilot (`performance`, `validation`, `field_confusions`) are not used and were left out.
