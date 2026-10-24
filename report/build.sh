#!/bin/sh
# Build report/main.pdf with Docker (no local LaTeX needed). Run from the report/ folder.
# On Windows Git Bash prefix with MSYS_NO_PATHCONV=1 and use $(pwd -W) instead of $(pwd).
docker run --rm -v "$(pwd):/work" -w /work texlive/texlive:latest-medium \
  sh -c "pdflatex -interaction=nonstopmode -halt-on-error main && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main && pdflatex -interaction=nonstopmode -halt-on-error main"
