"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Empirical NTK at initialization: does depth reshape the kernel like Fourier features do?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Computes the empirical neural tangent kernel K = J J^T on evenly spaced points in [0, 1] for
1D-input, 1-output models, where J [N, P] stacks the per-point gradients of the output w.r.t. all
trainable parameters. Writes normalized eigenvalue spectra and kernel slices K(x, 0.5) to results/ntk.json.

1) Build each configuration with seed 0 (same bodies as train.py: residual and plain raw MLPs, width 256).
2) N per-point backward passes -> J -> K -> eigenvalues in float64.

- Kernel-regime theory predicts deep ReLU NTKs keep the asymptotic spectral decay of shallow ones,
  while sigma directly widens the spectrum of a Fourier-feature kernel; this measures both at finite width.
- One initialization per configuration; the spectrum is a property of the architecture at init, not of training.

Example:
  python ntk.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import json
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from models import build_model
from train import RESULTS, SEED


@dataclass
class NTKConfig:
    n_points: int = 128
    width: int = 256
    m: int = 256
    models: tuple = (("raw", None, 1, True), ("raw", None, 4, True), ("raw", None, 16, True),
                     ("raw", None, 4, False), ("raw", None, 16, False),
                     ("rff", 1, 2, True), ("rff", 4, 2, True), ("rff", 16, 2, True), ("rff", 64, 2, True),
                     ("siren", None, 2, False), ("siren", None, 4, False))  # (encoding, sigma, depth, residual)


def jacobian(model, x):
    """[N, P]: gradient of the scalar output at each point w.r.t. every trainable parameter."""
    grad_fn = nn.value_and_grad(model, lambda xi: model(xi).sum())
    rows = []
    for i in range(len(x)):
        _, grads = grad_fn(x[i:i + 1])
        rows.append(mx.concatenate([g.reshape(-1) for _, g in tree_flatten(grads)]))
    return mx.stack(rows)


def main():
    """Compute and save NTK spectra for every configuration."""
    cfg = NTKConfig()
    x = mx.linspace(0, 1, cfg.n_points)[:, None]  # [N, 1]
    mid = cfg.n_points // 2
    out = []
    for encoding, sigma, depth, residual in cfg.models:
        mx.random.seed(SEED)
        model = build_model(encoding, 1, 1, depth, cfg.width, cfg.m, sigma, residual)
        J = jacobian(model, x)  # [N, P]
        K = np.array(J @ J.T).astype(np.float64)  # [N, N]
        eig = np.linalg.eigvalsh(K)[::-1]
        out.append(dict(encoding=encoding, sigma=sigma, depth=depth, residual=residual, eigvals=(eig / eig[0]).tolist(),
                        slice=(K[:, mid] / K[mid, mid]).tolist()))
        print(f"[INFO] ntk | {encoding} sigma={sigma} depth={depth} res={residual} | P={J.shape[1]} "
              f"lambda_10/lambda_1={eig[9] / eig[0]:.2e}", flush=True)
    (RESULTS / "ntk.json").write_text(json.dumps(dict(x=np.array(x[:, 0]).tolist(), models=out)))


if __name__ == "__main__":
    main()
