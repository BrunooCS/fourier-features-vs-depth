"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tasks: synthetic spectra, images, analytic 3D shapes and tabular data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every task has fixed train/val/test splits of inputs x [N, d] and targets. Coordinates live in
[0, 1]^d; tabular numeric columns are quantile-normalized with train statistics.

  synthetic  sines_d{d}_f{f}    sum of 16 sinusoids, |freq| spread over [1, f] cycles/unit   MSE  nmse
             sawtooth_k{k}      tent map composed k times -> 2^(k-1) teeth                  MSE  nmse
             teacher_d{d}       random 8-layer ReLU network                                 MSE  nmse
  image      astronaut, text    256x256 RGB, train on the [::2, ::2] grid (Tancik et al.)   MSE  PSNR
  shape      gyroid, bumpy_sphere   occupancy in [0, 1]^3, half samples near the surface    BCE  IoU
  tabular    california (8 features, regression)                                         MSE  RMSE
             covertype (54 features, 7 classes, 50k subsample)                            CE   accuracy

All generation is seeded; California Housing and Covertype are downloaded once into data/.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data as skdata
from skimage.transform import resize
from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.preprocessing import QuantileTransformer

DATA_DIR = Path(__file__).parent / "data"
SEED = 0


@dataclass
class Task:
    name: str
    domain: str     # synthetic | image | shape | tabular
    objective: str  # mse | bce | ce
    metric: str     # nmse | psnr | iou | rmse | acc
    splits: dict    # split -> (x [N, d] float32, y [N, out] float32 or [N] int32 labels)
    out_dim: int
    y_scale: float = 1.0          # std of the raw regression target, to report RMSE in original units
    render_x: np.ndarray = None   # dense query points for figures (1D curves, images, 3D grids)
    render_y: np.ndarray = None   # ground truth at render_x
    render_shape: tuple = None


# ── Synthetic ────────────────────────────────────────

N_SYNTH = (8192, 4096, 4096)  # train, val, test


def synthetic_task(name, f, d):
    """Uniform samples in [0, 1]^d; targets standardized with train statistics."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(size=(sum(N_SYNTH), d)).astype(np.float32)
    grid = np.linspace(0, 1, 2048, dtype=np.float32)[:, None]  # only rendered for d = 1
    y = f(x)
    mu, sd = y[:N_SYNTH[0]].mean(), y[:N_SYNTH[0]].std()
    y = ((y - mu) / sd).astype(np.float32)[:, None]  # [N, 1]
    a, b = N_SYNTH[0], N_SYNTH[0] + N_SYNTH[1]
    splits = {"train": (x[:a], y[:a]), "val": (x[a:b], y[a:b]), "test": (x[b:], y[b:])}
    if d > 1:
        return Task(name, "synthetic", "mse", "nmse", splits, 1)
    return Task(name, "synthetic", "mse", "nmse", splits, 1, render_x=grid,
                render_y=((f(grid) - mu) / sd).astype(np.float32), render_shape=(len(grid),))


def sines(d, f_max, n_terms=16):
    rng = np.random.default_rng(SEED + 1)
    u = rng.normal(size=(n_terms, d))
    w = u / np.linalg.norm(u, axis=1, keepdims=True) * np.linspace(1, f_max, n_terms)[:, None]  # [K, d], |w| in [1, f_max]
    phase = rng.uniform(0, 2 * np.pi, n_terms)
    return synthetic_task(f"sines_d{d}_f{f_max}", lambda x: np.sin(2 * np.pi * x @ w.T + phase).sum(1), d)


def sawtooth(k):
    def f(x):
        y = x[:, 0]
        for _ in range(k):
            y = 1 - np.abs(2 * y - 1)  # tent map: exactly 2 ReLUs per composition
        return y
    return synthetic_task(f"sawtooth_k{k}", f, 1)


def teacher(d, depth=8, width=64):
    """Random He-initialized ReLU network: a target that is 'MLP-shaped' by construction."""
    rng = np.random.default_rng(SEED + 2)
    dims = [d] + [width] * depth
    layers = [(rng.normal(0, np.sqrt(2 / a), (a, b)), rng.normal(0, 0.1, b)) for a, b in zip(dims[:-1], dims[1:])]
    w_out = rng.normal(0, np.sqrt(1 / width), width)

    def f(x):
        h = 2 * x - 1  # [N, d] in [-1, 1]
        for W, b in layers:
            h = np.maximum(h @ W + b, 0)  # [N, width]
        return h @ w_out  # [N]
    return synthetic_task(f"teacher_d{d}", f, d)


# ── Images ───────────────────────────────────────────

IMG = 256


def image_task(name, img):
    """Fit the [::2, ::2] grid (1/4 of pixels); the other 3/4 are split into val and test."""
    ii, jj = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
    x = np.stack([ii / IMG, jj / IMG], -1).reshape(-1, 2).astype(np.float32)  # [H*W, 2]
    y = img.reshape(-1, 3).astype(np.float32)  # [H*W, 3]
    train = ((ii % 2 == 0) & (jj % 2 == 0)).ravel()
    rest = np.random.default_rng(SEED).permutation(np.flatnonzero(~train))
    val, test = np.array_split(rest, 2)
    splits = {"train": (x[train], y[train]), "val": (x[val], y[val]), "test": (x[test], y[test])}
    return Task(name, "image", "mse", "psnr", splits, 3, render_x=x, render_y=y, render_shape=(IMG, IMG, 3))


def astronaut():
    return image_task("astronaut", resize(skdata.astronaut(), (IMG, IMG), anti_aliasing=True))  # floats in [0, 1]


def text():
    """Colored text on white: sharp edges make it the high-frequency image (as in Tancik et al.)."""
    canvas = Image.new("RGB", (IMG, IMG), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=24)
    lines = ["Fourier features", "let networks", "learn high", "frequency", "functions in", "low dimensional", "domains", "sin(2pi Bv)"]
    colors = [(200, 30, 30), (30, 90, 200), (20, 140, 60), (150, 40, 160), (220, 120, 0), (0, 130, 150), (90, 90, 90), (180, 20, 110)]
    for i, (line, color) in enumerate(zip(lines, colors)):
        draw.text((8, 6 + 31 * i), line, fill=color, font=font)
    return image_task("text", np.asarray(canvas, np.float32) / 255)


# ── 3D shapes ────────────────────────────────────────

RES = 128
N_SHAPE = (65536, 16384, 32768)  # train, val, test


def gyroid_occ(p, cells=3, thickness=0.35):
    q = 2 * np.pi * cells * p
    g = np.sin(q[:, 0]) * np.cos(q[:, 1]) + np.sin(q[:, 1]) * np.cos(q[:, 2]) + np.sin(q[:, 2]) * np.cos(q[:, 0])
    return (np.abs(g) < thickness) & (np.linalg.norm(p - 0.5, axis=1) < 0.45)  # gyroid shell clipped to a ball


def bumpy_occ(p, freq=14, amp=0.04):
    v = p - 0.5
    r = np.linalg.norm(v, axis=1)
    theta, phi = np.arccos(np.clip(v[:, 2] / (r + 1e-12), -1, 1)), np.arctan2(v[:, 1], v[:, 0])
    return r < 0.3 + amp * np.sin(freq * theta) * np.sin(freq * phi)


def shape_task(name, occ):
    """Half uniform points, half points whose 0.02-jitter crosses the surface (where the detail lives)."""
    rng = np.random.default_rng(SEED)
    n = sum(N_SHAPE)
    cand = rng.uniform(size=(50 * n, 3))
    near = cand[occ(cand) != occ(cand + rng.normal(scale=0.02, size=cand.shape))][: n // 2]
    x = rng.permutation(np.concatenate([rng.uniform(size=(n - len(near), 3)), near])).astype(np.float32)
    y = occ(x).astype(np.float32)[:, None]  # [N, 1]
    a, b = N_SHAPE[0], N_SHAPE[0] + N_SHAPE[1]
    splits = {"train": (x[:a], y[:a]), "val": (x[a:b], y[a:b]), "test": (x[b:], y[b:])}
    c = (np.arange(RES) + 0.5) / RES
    grid = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)  # [RES^3, 3]
    return Task(name, "shape", "bce", "iou", splits, 1, render_x=grid, render_y=occ(grid), render_shape=(RES, RES, RES))


# ── Tabular ──────────────────────────────────────────

def tabular_task(name, X, y, n_numeric, objective, metric, out_dim):
    """70/15/15 split; quantile-normalize numeric columns on train (Gorishniy et al. 2022 protocol)."""
    idx = np.random.default_rng(SEED).permutation(len(X))
    parts = dict(zip(("train", "val", "test"), np.split(idx, [int(0.7 * len(X)), int(0.85 * len(X))])))
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED).fit(X[parts["train"], :n_numeric])
    X = np.concatenate([qt.transform(X[:, :n_numeric]), X[:, n_numeric:]], axis=1).astype(np.float32)
    return Task(name, "tabular", objective, metric, {k: (X[v], y[v]) for k, v in parts.items()}, out_dim)


def california():
    X, y = fetch_california_housing(data_home=DATA_DIR, return_X_y=True)
    task = tabular_task("california", X, ((y - y.mean()) / y.std()).astype(np.float32)[:, None], 8, "mse", "rmse", 1)
    task.y_scale = float(y.std())  # RMSE reported in units of $100k, as in the tabular DL literature
    return task


def covertype(n=50_000):
    X, y = fetch_covtype(data_home=DATA_DIR, return_X_y=True)
    idx = np.random.default_rng(SEED).choice(len(X), n, replace=False)
    return tabular_task("covertype", X[idx], (y[idx] - 1).astype(np.int32), 10, "ce", "acc", 7)  # 10 numeric + 44 binary


# ── Registry ─────────────────────────────────────────

TASKS = {
    **{f"sines_d{d}_f{f}": partial(sines, d, f) for d, f in [(1, 2), (1, 8), (1, 32), (1, 128), (2, 8), (4, 8), (8, 8), (16, 8)]},
    **{f"sawtooth_k{k}": partial(sawtooth, k) for k in (1, 3, 5, 7)},
    **{f"teacher_d{d}": partial(teacher, d) for d in (2, 8, 32)},
    "astronaut": astronaut, "text": text,
    "gyroid": partial(shape_task, "gyroid", gyroid_occ), "bumpy_sphere": partial(shape_task, "bumpy_sphere", bumpy_occ),
    "california": california, "covertype": covertype,
}


def make_task(name):
    return TASKS[name]()
