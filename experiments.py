"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment definitions: tasks, encodings and capacity grid of each named study
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each experiment first tunes sigma for every bandwidth encoding at the SWEEP reference config,
then trains the full grid (encodings x m x depths x widths x residual) with the selected sigma.

- Raw MLPs also train a plain (non-residual) body at the same depths, so depth gets its best shot:
  the plain body converts depth into frequency better, but collapses to a constant beyond ~16 layers
  (ablate_residual runs both bodies to depth 64), while the residual body trains at any depth.
- SIREN trains at its paper learning rate (`siren_lr`); every other model shares `lr`.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from dataclasses import dataclass

SWEEP = dict(m=256, depth=2, width=128, residual=True)  # reference config for sigma selection
MAIN_WIDTHS = (32, 128, 512)


@dataclass(frozen=True)
class Experiment:
    # ── Data ──────────────────────────────────────────
    tasks: tuple

    # ── Grid ──────────────────────────────────────────
    encodings: tuple = ("raw", "pe", "rff", "lff", "periodic", "siren")
    depths: tuple = (1, 2, 4, 8, 16)
    plain_raw_depths: tuple = (1, 2, 4, 8, 16)  # extra plain-body grid for raw MLPs
    widths: tuple = (128,)
    ms: tuple = (256,)
    residuals: tuple = (True,)
    sigmas: tuple = (1, 2, 4, 8, 16, 32, 64)

    # ── Optimization ──────────────────────────────────
    steps: int = 2000
    batch: int = 8192
    lr: float = 1e-3
    siren_lr: float = 1e-4  # Sitzmann et al. 2020; the shared 1e-3 collapses SIREN at depth >= 8
    eval_every: int = 100


EXPERIMENTS = {
    # Synthetic sweeps: one capacity axis (depth), conditions vary across tasks.
    "sines_freq": Experiment(tasks=tuple(f"sines_d1_f{f}" for f in (2, 8, 32, 128)), sigmas=(1, 2, 4, 8, 16, 32, 64, 128, 256)),
    "sines_dim": Experiment(tasks=tuple(f"sines_d{d}_f8" for d in (1, 2, 4, 8, 16)), sigmas=(0.5, 1, 2, 4, 8, 16, 32)),
    "sawtooth": Experiment(tasks=tuple(f"sawtooth_k{k}" for k in (1, 3, 5, 7)), sigmas=(1, 2, 4, 8, 16, 32, 64, 128)),
    "teacher": Experiment(tasks=tuple(f"teacher_d{d}" for d in (2, 8, 32)), sigmas=(0.125, 0.25, 0.5, 1, 2, 4, 8)),

    # Main domains: full depth x width grid.
    "images": Experiment(tasks=("astronaut", "text"), widths=MAIN_WIDTHS, batch=16384),
    "shapes": Experiment(tasks=("gyroid", "bumpy_sphere"), widths=MAIN_WIDTHS, sigmas=(1, 2, 4, 8, 16, 32), steps=3000, batch=16384),
    "tabular": Experiment(tasks=("california", "covertype"), widths=MAIN_WIDTHS, sigmas=(0.03, 0.1, 0.3, 1, 3, 10),
                          steps=4000, batch=512, eval_every=200),

    # Ablations.
    "ablate_residual": Experiment(tasks=("sines_d1_f32", "astronaut"), encodings=("raw", "rff"), depths=(1, 2, 4, 8, 16, 32, 64),
                                  residuals=(True, False), batch=16384),
    "ablate_m": Experiment(tasks=("sines_d1_f32", "astronaut"), encodings=("rff",), depths=(2,), ms=(16, 64, 256, 1024), batch=16384),
}
