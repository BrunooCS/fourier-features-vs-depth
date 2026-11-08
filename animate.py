"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training animations and spectral-bias dynamics for showcase configurations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retrains a few grid runs of the main study (seed 0, the sigma and body selected on val by train.py)
while recording dense predictions, then writes:

  figures/image_training.gif   astronaut over training: 1-layer Fourier MLPs vs the best raw MLP of the grid
  figures/sines_training.gif   sines_d1_f32: fit and residual spectrum, best raw MLP vs RFF depth 2
  results/spectral.npz         residual amplitude per frequency vs step for the 1D runs (plots.py spectral_bias)

Requires results/images.jsonl and results/sines_freq.jsonl for the selected sigmas.

Example:
  python animate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import json
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from PIL import Image

from data import make_task
from experiments import EXPERIMENTS
from models import BANDWIDTH, build_model
from plots import COLOR, FIGURES, FOG, GRAYS, LABEL, MUTED, fmt_params, set_style
from train import RESULTS, SEED, best_sigma, fit, predict


@dataclass
class AnimateConfig:
    image_task: str = "astronaut"
    image_models: tuple = (("raw", 1, 128), ("raw", None, None), ("pe", 1, 128), ("rff", 1, 128), ("lff", 1, 128))  # (encoding, depth, width); None: best of grid
    image_every: int = 50
    curve_task: str = "sines_d1_f32"
    curve_models: tuple = (("raw", None, None), ("rff", 2, 128))
    curve_every: int = 20
    fps: int = 10


# ── Execution / Training ─────────────────────────────

def grid_run(experiment, task, encoding, depth, width):
    """Lowest-val run of one encoding at its selected sigma, at (depth, width) or over the whole grid when depth is None."""
    records = [json.loads(line) for line in (RESULTS / f"{experiment}.jsonl").read_text().splitlines()]
    sigma = best_sigma(records, task, encoding) if encoding in BANDWIDTH else None
    runs = [r for r in records if r["task"] == task and r["encoding"] == encoding and r["sigma"] == sigma
            and (depth is None or (r["depth"], r["width"]) == (depth, width))]
    return min(runs, key=lambda r: r["val_error"])


def snapshots(task, experiment, run, every):
    """Retrain one grid run (same seed, so the same trajectory) and return dense predictions every `every` steps: [T, R, out]."""
    mx.random.seed(SEED)
    model = build_model(run["encoding"], task.splits["train"][0].shape[1], task.out_dim, run["depth"], run["width"], run["m"], run["sigma"], run["residual"])
    frames = []
    fit(model, task, replace(EXPERIMENTS[experiment], eval_every=every), callback=lambda step, m: frames.append(predict(m, task.render_x)))
    return np.stack(frames)


# ── Output / IO ──────────────────────────────────────

def write_gif(fig, update, n, path, fps):
    """Render n frames of a fixed figure through `update(i)`; the last frame is held for 2 s."""
    frames = []
    for i in range(n):
        update(i)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=192))
    durations = [1000 // fps] * (n - 1) + [2000]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    plt.close(fig)


def image_gif(cfg):
    """2x3 grid: ground truth + each model's full-image prediction, with held-out PSNR per frame."""
    task = make_task(cfg.image_task)
    shape = task.render_shape
    gt = task.render_y.reshape(shape)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    held_out = ~((ii % 2 == 0) & (jj % 2 == 0))  # pixels never seen in training
    models = [grid_run("images", task.name, enc, depth, width) for enc, depth, width in cfg.image_models]
    runs = [snapshots(task, "images", run, cfg.image_every).reshape(-1, *shape) for run in models]
    psnr = [-10 * np.log10(((r - gt) ** 2)[:, held_out].mean(axis=(1, 2))) for r in runs]  # [T] per model

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.6))
    axes[0, 0].imshow(gt)
    axes[0, 0].set_title("ground truth", fontsize=12)
    panels = [axes.flat[k].imshow(np.clip(r[0], 0, 1)) for k, r in zip(range(1, 6), runs)]
    for ax in axes.flat:
        ax.set_axis_off()
    step_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.04, hspace=0.18)

    def update(i):
        for k, (panel, r, run) in enumerate(zip(panels, runs, models)):
            panel.set_data(np.clip(r[i], 0, 1))
            layers = f"{run['depth']} layer" + ("s" if run["depth"] > 1 else "")
            axes.flat[k + 1].set_title(f"{LABEL[run['encoding']]}, {layers}\n{fmt_params(run['params'])} params · {psnr[k][i]:.1f} dB",
                                       fontsize=10, color=COLOR[run["encoding"]])
        step_text.set_text(f"step {(i + 1) * cfg.image_every}")

    write_gif(fig, update, len(runs[0]), FIGURES / "image_training.gif", cfg.fps)


def curve_gif(cfg):
    """Fit vs target (top) and residual amplitude spectrum (bottom) for the 1D showcase runs."""
    task = make_task(cfg.curve_task)
    x, truth = task.render_x[:, 0], task.render_y
    models = [grid_run("sines_freq", task.name, enc, depth, width) for enc, depth, width in cfg.curve_models]
    runs = {f"{run['encoding']}_d{run['depth']}": snapshots(task, "sines_freq", run, cfg.curve_every)[..., 0] for run in models}  # [T, R]
    spectra = {k: np.abs(np.fft.rfft(r - truth, axis=-1)) / len(truth) for k, r in runs.items()}  # [T, F]
    steps = np.arange(1, len(next(iter(runs.values()))) + 1) * cfg.curve_every
    np.savez(RESULTS / "spectral.npz", steps=steps, freqs=np.arange(len(truth) // 2 + 1), truth=np.abs(np.fft.rfft(truth)) / len(truth), **spectra)

    fmax = 48
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 5.2), gridspec_kw=dict(height_ratios=(1.3, 1)))
    a.plot(x, truth, color=FOG, lw=4, label="target")
    lines = {k: a.plot(x, r[0], color=COLOR[k.split("_d")[0]], lw=1.3, label=f"{LABEL[k.split('_d')[0]]} · depth {k.split('_d')[1]}")[0] for k, r in runs.items()}
    bars = {k: b.plot(np.arange(fmax), s[0, :fmax], color=COLOR[k.split("_d")[0]], lw=1.6)[0] for k, s in spectra.items()}
    a.set(xlim=(0, 1), ylim=(truth.min() - 0.5, truth.max() + 0.5), xlabel="x", ylabel="f(x)")
    a.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    b.set(yscale="log", ylim=(1e-5, 1), xlim=(0, fmax - 1), xlabel="frequency (cycles per unit)", ylabel="|residual| spectrum")
    b.axvline(32, color=GRAYS[1], ls=":", lw=1)
    step_text = fig.text(0.98, 0.01, "", ha="right", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    def update(i):
        for k in runs:
            lines[k].set_ydata(runs[k][i])
            bars[k].set_ydata(spectra[k][i, :fmax])
        step_text.set_text(f"step {steps[i]}")

    write_gif(fig, update, len(steps), FIGURES / "sines_training.gif", cfg.fps)


# ── Main ─────────────────────────────────────────────

def main():
    """Write both GIFs and the spectral-dynamics arrays."""
    cfg = AnimateConfig()
    set_style()
    FIGURES.mkdir(exist_ok=True)
    curve_gif(cfg)
    image_gif(cfg)


if __name__ == "__main__":
    main()
