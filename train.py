"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fourier vs depth: train the grid of one or more named experiments (MLX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Appends one JSON line per run to results/<experiment>.jsonl and saves renders to results/renders/.

1) Bandwidth sweep: each pe/rff/lff/periodic encoding trains every sigma at the SWEEP config
   (depth 2, width 128, m 256); the best sigma on val is kept for that task.
2) Grid: every encoding x m x depth x width x residual configuration, with the selected sigma;
   raw MLPs also train a plain body (`plain_raw_depths`).
3) Each run records the test metric at the best-val evaluation step, trainable params, forward
   FLOPs/sample, training FLOPs, train seconds and inference latency for 64k points.
   Tabular tasks also get a HistGradientBoosting reference row.

- Resumable: configurations already in the JSONL are skipped.
- Single seed (0) for init and minibatch order; AdamW with linear warmup + cosine decay, SIREN at its paper lr.
- No device code: MLX runs on the Metal GPU on macOS and on CUDA with `mlx[cuda]` on Linux.
- Renders (image PNG, packed 3D occupancy, 1D curves) are saved for width 128 at depths 1, 4, 16 (not committed).

Example:
  python train.py --experiment sines_freq images
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import itertools
import json
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from data import make_task
from experiments import EXPERIMENTS, SWEEP
from models import BANDWIDTH, build_model, count_params, flops_per_sample

RESULTS = Path(__file__).parent / "results"
SEED = 0
RENDER_WIDTH, RENDER_DEPTHS = 128, (1, 4, 16)


# ── Loss / Evaluation ────────────────────────────────

LOSSES = {
    "mse": lambda out, y: mx.mean(mx.square(out - y)),
    "bce": lambda out, y: nn.losses.binary_cross_entropy(out, y, with_logits=True, reduction="mean"),
    "ce": lambda out, y: nn.losses.cross_entropy(out, y, reduction="mean"),
}


def predict(model, x, chunk=65536):
    """Chunked forward pass so 128^3 render grids fit in memory."""
    return np.concatenate([np.array(model(mx.array(x[i:i + chunk]))) for i in range(0, len(x), chunk)])


def evaluate(task, pred, y):
    """Return (error, metric). error is lower-is-better: nmse, 1 - IoU or 1 - accuracy."""
    if task.objective == "mse":
        mse = float(np.mean((pred - y) ** 2))
        metric = {"nmse": mse / float(y.var()), "psnr": -10 * np.log10(mse), "rmse": np.sqrt(mse) * task.y_scale}[task.metric]
        return mse / float(y.var()), float(metric)
    if task.objective == "bce":
        p, t = pred[:, 0] > 0, y[:, 0] > 0.5
        iou = float((p & t).sum() / (p | t).sum())
        return 1 - iou, iou
    acc = float((pred.argmax(1) == y).mean())
    return 1 - acc, acc


def latency_ms(model, d, n=65536, reps=5):
    """Median forward time for n points after one warm-up pass."""
    x = mx.random.uniform(shape=(n, d))
    mx.eval(model(x))
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(model(x))
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


# ── Execution / Training ─────────────────────────────

def fit(model, task, exp, callback=None):
    """Train with AdamW + warmup/cosine; return the best-val record and pure training seconds."""
    # ── Optimizer and compiled step ───────────────────
    warm = exp.steps // 20
    schedule = optim.join_schedules([optim.linear_schedule(0.0, exp.lr, warm), optim.cosine_decay(exp.lr, exp.steps - warm)], [warm])
    opt = optim.AdamW(learning_rate=schedule)
    opt.init(model.trainable_parameters())  # materialize optimizer state before compile captures it
    loss_and_grad = nn.value_and_grad(model, lambda x, y: LOSSES[task.objective](model(x), y))
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = loss_and_grad(x, y)
        opt.update(model, grads)
        return loss

    # ── Train, selecting the test score at the best val step ──
    x_tr, y_tr = (mx.array(a) for a in task.splits["train"])
    (x_val, y_val), (x_te, y_te) = task.splits["val"], task.splits["test"]
    best = dict(val_error=float("inf"), test_error=float("nan"), test_metric=float("nan"), best_step=0)
    train_s = 0.0
    for s in range(1, exp.steps + 1):
        t = time.perf_counter()
        idx = mx.random.permutation(len(x_tr))[:exp.batch]  # without replacement; full batch when batch >= N
        loss = step(x_tr[idx], y_tr[idx])
        mx.eval(loss, state)
        train_s += time.perf_counter() - t
        if s % exp.eval_every == 0:
            val_error, _ = evaluate(task, predict(model, x_val), y_val)
            if val_error < best["val_error"]:
                test_error, test_metric = evaluate(task, predict(model, x_te), y_te)
                best = dict(val_error=val_error, test_error=test_error, test_metric=test_metric, best_step=s)
            if callback is not None:
                callback(s, model)
    return best, train_s


def train_one(task, exp, encoding, sigma, m, depth, width, residual):
    mx.random.seed(SEED)
    exp = replace(exp, lr=exp.siren_lr) if encoding == "siren" else exp
    d = task.splits["train"][0].shape[1]
    model = build_model(encoding, d, task.out_dim, depth, width, m, sigma, residual)
    mx.eval(model.parameters())
    best, train_s = fit(model, task, exp)
    flops = flops_per_sample(model)
    record = dict(task=task.name, domain=task.domain, metric=task.metric, encoding=encoding, sigma=sigma, m=m, depth=depth,
                  width=width, residual=residual, lr=exp.lr, params=count_params(model), flops=flops,
                  train_flops=3 * flops * min(exp.batch, len(task.splits["train"][0])) * exp.steps,
                  **best, train_s=train_s, infer_ms=latency_ms(model, d), device=str(mx.default_device()), mlx=mx.__version__)
    return model, record


def gbdt_record(task):
    """HistGradientBoosting reference with sklearn defaults and internal early stopping."""
    (x, y), (x_te, y_te) = task.splits["train"], task.splits["test"]
    t = time.perf_counter()
    if task.objective == "mse":
        pred = HistGradientBoostingRegressor(max_iter=1000, early_stopping=True, random_state=SEED).fit(x, y[:, 0]).predict(x_te)[:, None]
    else:
        pred = HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, random_state=SEED).fit(x, y).predict_proba(x_te)
    test_error, test_metric = evaluate(task, pred, y_te)
    return dict(task=task.name, domain=task.domain, metric=task.metric, encoding="gbdt", sigma=None, m=None, depth=None, width=None,
                residual=None, test_error=test_error, test_metric=test_metric, train_s=time.perf_counter() - t)


# ── Output / IO ──────────────────────────────────────

def save_render(model, task, path):
    """Dense prediction for figures: PNG for images, packed occupancy bits for shapes, curve for 1D."""
    pred = predict(model, task.render_x)
    if task.domain == "image":
        Image.fromarray((np.clip(pred, 0, 1) * 255).round().astype(np.uint8).reshape(task.render_shape)).save(path.with_suffix(".png"))
    elif task.domain == "shape":
        np.save(path.with_suffix(".npy"), np.packbits(pred[:, 0] > 0))
    else:
        np.save(path.with_suffix(".npy"), pred[:, 0].astype(np.float32))


def run_key(r):
    return (r["task"], r["encoding"], r["sigma"], r["m"], r["depth"], r["width"], r["residual"])


def best_sigma(records, task, encoding):
    """Sigma with the lowest val error at the SWEEP reference config."""
    sweep = [r for r in records if r["task"] == task and r["encoding"] == encoding
             and all(r[k] == v for k, v in SWEEP.items())]
    return min(sweep, key=lambda r: r["val_error"])["sigma"]


def run_experiment(name):
    """Sigma sweep then full grid for every task of one experiment; resumes from the JSONL."""
    exp = EXPERIMENTS[name]
    out = RESULTS / f"{name}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    out.touch()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    done = {run_key(r) for r in records}

    def run(task, encoding, sigma, m, depth, width, residual):
        # raw/siren have no bandwidth or m; siren is always a plain (non-residual) sine stack
        sigma, m = (sigma, m) if encoding in BANDWIDTH else (None, None)
        residual = residual and encoding != "siren"
        key = (task.name, encoding, sigma, m, depth, width, residual)
        if key in done:
            return
        model, record = train_one(task, exp, encoding, sigma, m, depth, width, residual)
        if task.render_x is not None and width == RENDER_WIDTH and depth in RENDER_DEPTHS:
            path = RESULTS / "renders" / task.name / f"{encoding}_d{depth}_w{width}_m{m}_{'res' if residual else 'plain'}"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_render(model, task, path)
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        records.append(record)
        done.add(key)
        print(f"[INFO] {name} | {task.name} {encoding} sigma={sigma} m={m} depth={depth} width={width} res={residual} | "
              f"{task.metric}={record['test_metric']:.4f} params={record['params']} {record['train_s']:.1f}s", flush=True)

    for task_name in exp.tasks:
        task = make_task(task_name)

        # 1) Bandwidth sweep at the reference config
        for encoding, sigma in itertools.product([e for e in exp.encodings if e in BANDWIDTH], exp.sigmas):
            run(task, encoding, sigma, **SWEEP)

        # 2) Grid with the selected sigma, plus a deeper plain body for raw MLPs
        for encoding in exp.encodings:
            sigma = best_sigma(records, task.name, encoding) if encoding in BANDWIDTH else None
            for m, depth, width, residual in itertools.product(exp.ms, exp.depths, exp.widths, exp.residuals):
                run(task, encoding, sigma, m, depth, width, residual)
        if "raw" in exp.encodings:
            for depth, width in itertools.product(exp.plain_raw_depths, exp.widths):
                run(task, "raw", None, None, depth, width, False)

        # 3) Tree-ensemble reference for tabular data
        if task.domain == "tabular" and (task.name, "gbdt", None, None, None, None, None) not in done:
            record = gbdt_record(task)
            with out.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[INFO] {name} | {task.name} gbdt | {task.metric}={record['test_metric']:.4f}", flush=True)


# ── Main ─────────────────────────────────────────────

def main():
    """Run the named experiments sequentially."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", nargs="+", choices=list(EXPERIMENTS), required=True)
    for name in parser.parse_args().experiment:
        run_experiment(name)


if __name__ == "__main__":
    main()
