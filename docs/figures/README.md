# Architecture Figures

The PNG and PDF files are compiled from the standalone TikZ sources in this directory. The figures reproduce the architecture comparison and recommended MA-TRM-Lite design from the paper in `paper/`.

Rebuild with:

```bash
cd docs/figures
latexmk -pdf -interaction=nonstopmode -halt-on-error trm_vs_ma_trm.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ma_trm_lite.tex
pdftoppm -png -r 180 -singlefile trm_vs_ma_trm.pdf trm_vs_ma_trm
pdftoppm -png -r 180 -singlefile ma_trm_lite.pdf ma_trm_lite
```
