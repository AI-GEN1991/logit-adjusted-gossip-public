#!/usr/bin/env bash
# Recompute every reported statistic from the stored results and redraw Figs. 3-16 (CPU only, a few minutes).
set -euo pipefail
cd "$(dirname "$0")"
for s in analyse_round2A analyse_la_followup analyse_sweep5 analyse_round2C analyse_final_exp analyse_round3D analyse_round3E table_round3E analyse_negative_controls kappa_prop2; do
  echo "================ analysis/$s.py"; python3 analysis/$s.py
done
echo "================ theory/theory_numerical_check.py"; python3 theory/theory_numerical_check.py
echo "================ plotting/make_figures.py"; python3 plotting/make_figures.py
echo "================ plotting/make_figures_results.py"; python3 plotting/make_figures_results.py
