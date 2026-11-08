#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fourier vs depth: full study -> NTK -> animations -> figures and summary table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) train.py   every experiment grid, appended to results/<experiment>.jsonl
# 2) ntk.py     NTK spectra at init           -> results/ntk.json
# 3) animate.py training GIFs + spectra       -> figures/*.gif, results/spectral.npz
# 4) plots.py   all figures + summary table   -> figures/*.png, results/summary.md
#
# - Resumable: train.py skips configurations already in results/*.jsonl, so re-running after an
#   interruption only trains what is left. Steps 2-4 are cheap and always re-run.
# - Uses the `python` of the active environment (MLX on Apple silicon, mlx[cuda] on Linux).
# - Logs are appended to logs/.
#
# Example:
#   micromamba activate research
#   caffeinate -i ./run_all.sh      # caffeinate keeps macOS awake during the multi-hour run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

python train.py --experiment sines_freq sines_dim sawtooth teacher images shapes tabular ablate_residual ablate_m 2>&1 | tee -a logs/train.log
python ntk.py 2>&1 | tee -a logs/ntk.log
python animate.py 2>&1 | tee -a logs/animate.log
python plots.py 2>&1 | tee -a logs/plots.log
