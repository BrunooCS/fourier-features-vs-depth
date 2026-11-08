"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Figures for the Fourier-vs-depth study (README-ready PNGs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads results/*.jsonl (train.py), results/ntk.json (ntk.py), results/spectral.npz (animate.py) and
results/renders/, and writes figures/<name>.png with a minimal built-in style (default matplotlib font).

  exchange_rate   raw-MLP params needed to match the best depth <= 2 Fourier model (task x encoding)
  pareto          test metric vs trainable params and vs inference latency (images, shapes, tabular)
  depth           test metric vs depth at width 128 on the image and 3D shape tasks
  frequency       sines_freq: error vs target bandwidth
  dimension       sines_dim and teacher: error vs input dimension
  sawtooth        error vs number of teeth, plus example fits
  bandwidth       sigma sweeps at the reference config (depth 2, width 128)
  ablations       number of Fourier features m; residual vs plain bodies
  ntk             NTK eigenvalue decay and kernel slices at init
  spectral_bias   residual energy per frequency band over training
  images          reconstructions at depth 1 and 16, width 128
  shapes          marching-cubes renders of predicted occupancy
  summary         results/summary.md: best raw MLP vs best shallow model of each encoding, per task

Style: white ground, near-black ink, slate grays for context, one blue family for Fourier encodings,
violet for the per-feature variant, names written on the lines instead of legend boxes.
Raw MLPs train plain and residual bodies; every main figure keeps the lower-val body per configuration.
The exchange-rate heatmap uses whichever main experiments are present in results/.

Example:
  python plots.py                      # every figure
  python plots.py frequency depth      # a subset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy import ndimage
from skimage import measure

from data import RES, make_task
from experiments import EXPERIMENTS, SWEEP
from models import BANDWIDTH

ROOT = Path(__file__).parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
MAIN = ("sines_freq", "sines_dim", "sawtooth", "teacher", "images", "shapes", "tabular")
TOL = 0.05  # a raw MLP "matches" when its test error is within 5% of the Fourier reference
WIDTH = 7.5  # figure width in inches; GitHub scales the 200-dpi PNG to the README column

# ── Style ────────────────────────────────────────────

INK, MUTED, FOG, GRIDLINE = "#111827", "#6b7280", "#e5e7eb", "#eef0f3"
GRAYS = ["#d1d5db", "#9ca3af", "#6b7280", "#374151", "#111827"]  # light -> dark: depth families
BLUES = ["#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a"]
VIOLETS = ["#a78bfa", "#6d28d9"]
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#475569", "#cbd5e1", "#ffffff", "#93c5fd", "#1d4ed8"])

ENCODINGS = ("raw", "pe", "rff", "lff", "periodic", "siren")
FOURIER = ENCODINGS[1:]
LABEL = {"raw": "raw", "pe": "positional", "rff": "Gaussian RFF", "lff": "learned FF", "periodic": "periodic", "siren": "SIREN"}
COLOR = {"raw": INK, "pe": "#60a5fa", "rff": "#2563eb", "lff": "#1e3a8a", "periodic": "#7c3aed", "siren": "#94a3b8"}
MARKER = {"raw": "o", "pe": "s", "rff": "D", "lff": "^", "periodic": "v", "siren": "P"}
HIGHER = {"psnr": True, "iou": True, "acc": True, "rmse": False, "nmse": False}
METRIC_LABEL = {"psnr": "test PSNR (dB)", "iou": "test IoU", "acc": "test accuracy", "rmse": "test RMSE ($100k)", "nmse": "test NMSE"}


def set_style():
    """Quiet axes: open frame, faint y-grid, left-aligned titles, no legend frames, default font."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAYS[1], "axes.linewidth": 0.8,
        "xtick.color": GRAYS[3], "ytick.color": GRAYS[3], "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIDLINE, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
    })


# ── Inputs / Data ────────────────────────────────────

def load_results():
    """All runs from results/*.jsonl as one DataFrame with an `experiment` column."""
    rows = [dict(json.loads(line), experiment=p.stem) for p in sorted(RESULTS.glob("*.jsonl")) for line in p.read_text().splitlines()]
    return pd.DataFrame(rows)


def grid_runs(df, experiment):
    """Runs of one experiment at the selected sigma (drops the other sigma-sweep runs)."""
    e = df[df.experiment == experiment]
    sweep = e[e.encoding.isin(BANDWIDTH) & (e.depth == SWEEP["depth"]) & (e.width == SWEEP["width"]) & (e.m == SWEEP["m"]) & (e.residual == True)]  # noqa: E712
    best = sweep.loc[sweep.groupby(["task", "encoding"]).val_error.idxmin(), ["task", "encoding", "sigma"]]
    e = e.merge(best, on=["task", "encoding"], how="left", suffixes=("", "_best"))
    return e[e.sigma_best.isna() | (e.sigma == e.sigma_best)]


def main_runs(df, experiment):
    """grid_runs with one body per configuration, the lower-val one (raw MLPs train plain and residual bodies)."""
    return grid_runs(df, experiment).sort_values("val_error").drop_duplicates(["task", "encoding", "m", "depth", "width"])


def pretty(task):
    if task.startswith("sines"):
        d, f = task.split("_")[1:]
        return f"sines {d[1:]}D, f ≤ {f[1:]}"
    if task.startswith("sawtooth"):
        return f"sawtooth, {2 ** (int(task.split('_k')[1]) - 1)} teeth"
    if task.startswith("teacher"):
        return f"ReLU teacher {task.split('_d')[1]}D"
    return task.replace("_", " ")


def render_path(task, encoding, depth, width=128, m=256, residual=True):
    """Mirror of train.py's render naming (raw/siren have no m; siren is always plain)."""
    m, residual = (m, residual) if encoding in BANDWIDTH else (None, residual and encoding != "siren")
    return RESULTS / "renders" / task / f"{encoding}_d{depth}_w{width}_m{m}_{'res' if residual else 'plain'}"


def render_of(g, task, encoding, depth, width=128):
    """Selected run of a main grid at (depth, width) and the render path of its body."""
    row = g[(g.task == task) & (g.encoding == encoding) & (g.depth == depth) & (g.width == width)].iloc[0]
    return row, render_path(task, encoding, depth, width, residual=row.residual)


# ── Plot helpers ─────────────────────────────────────

def figure(nrows=1, ncols=1, aspect=0.42, **kw):
    return plt.subplots(nrows, ncols, figsize=(WIDTH, WIDTH * aspect), **kw)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def note(fig, text):
    fig.text(0.01, -0.01, textwrap.fill(text, 150), ha="left", va="top", fontsize=8, color=MUTED)


def log2_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values, [str(v) for v in values])
    ax.minorticks_off()


def direct_labels(ax, gap_pt=11):
    """Name each labelled line at its last finite point, in its color, spreading labels so none overlap."""
    ax.figure.canvas.draw()  # finalize limits so data -> display transforms are exact
    items = []
    for line in ax.get_lines():
        x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        ok = np.isfinite(x) & np.isfinite(y)
        if line.get_label().startswith("_") or not ok.any():  # unlabelled helper lines / fully diverged runs
            continue
        xy = (x[ok][-1], y[ok][-1])
        items.append([ax.transData.transform(xy)[1], xy, line.get_label(), line.get_color()])
    items.sort(key=lambda it: it[0])
    gap = gap_pt * ax.figure.dpi / 72
    for prev, cur in zip(items, items[1:]):  # push up to open gaps ...
        cur[0] = max(cur[0], prev[0] + gap)
    top = ax.bbox.y1
    for cur, nxt in zip(items[::-1], [None] + items[::-1][:-1]):  # ... then pull back down below the axes top
        cur[0] = min(cur[0], top if nxt is None else nxt[0] - gap)
    for target, xy, label, color in items:
        dy = (target - ax.transData.transform(xy)[1]) * 72 / ax.figure.dpi
        ax.annotate(label, xy=xy, xytext=(6, dy), textcoords="offset points", va="center", fontsize=9, color=color, annotation_clip=False)


def legend_top(fig, ax, ncol=6):
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.02), handlelength=1.6)


def pareto_front(e, x, higher):
    """Points that beat every cheaper point."""
    e = e.sort_values(x)
    best = e.test_metric.cummax() if higher else e.test_metric.cummin()
    return e[e.test_metric == best]


def exchange_table(g):
    """task x encoding: params of the smallest matching raw MLP / params of the best depth <= 2 Fourier model."""
    rows = []
    for task, t in g.groupby("task"):
        raw = t[t.encoding == "raw"]
        for enc in FOURIER:
            ff = t[(t.encoding == enc) & (t.depth <= 2)]
            ref = ff.loc[ff.test_error.idxmin()]
            match = raw[raw.test_error <= ref.test_error * (1 + TOL)]
            rows.append(dict(task=task, encoding=enc, ratio=match.params.min() / ref.params if len(match) else np.inf))
    return pd.DataFrame(rows).pivot(index="task", columns="encoding", values="ratio")


# ── Figures: capacity exchange ───────────────────────

def fig_exchange_rate(df):
    tables = [exchange_table(main_runs(df, name)).reindex(EXPERIMENTS[name].tasks) for name in MAIN if name in set(df.experiment)]
    table = pd.concat(tables)
    table = table[~table.index.duplicated()][list(FOURIER)]
    ratio = table.to_numpy(float)
    fig, ax = figure(aspect=0.16 + 0.04 * len(table))
    ax.imshow(np.clip(np.log10(np.where(np.isinf(ratio), np.nan, ratio)), -2, 2), cmap=CMAP_DIV, vmin=-2, vmax=2, aspect="auto")
    for i, j in np.ndindex(ratio.shape):
        never = np.isinf(ratio[i, j])
        if never:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color=BLUES[3]))
        dark = never or abs(np.log10(ratio[i, j])) > 1.2
        ax.text(j, i, "never" if never else f"{ratio[i, j]:.2g}×", ha="center", va="center", fontsize=9, color="white" if dark else INK)
    ax.set_xticks(range(len(FOURIER)), [LABEL[e] for e in FOURIER])
    ax.set_yticks(range(len(table)), [pretty(t) for t in table.index])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    note(fig, f"Cell: params of the smallest raw MLP within {TOL:.0%} of the best depth ≤ 2 Fourier model's test error, divided by that "
              "model's params. >1 (blue): the encoding saves capacity; <1 (gray): raw is cheaper; never: no raw MLP in the grid matches.")
    save(fig, "exchange_rate")


def fig_pareto(df):
    for experiment in ("images", "shapes", "tabular"):
        g, tasks = main_runs(df, experiment), EXPERIMENTS[experiment].tasks
        fig, axes = figure(nrows=2, ncols=len(tasks), aspect=0.78, squeeze=False)
        for j, task in enumerate(tasks):
            t = g[g.task == task]
            metric = t.metric.iloc[0]
            for i, x in enumerate(("params", "infer_ms")):
                ax = axes[i, j]
                for enc in ENCODINGS:
                    e = t[t.encoding == enc]
                    ax.scatter(e[x], e.test_metric, s=10, color=COLOR[enc], alpha=0.3, lw=0, zorder=2)
                    front = pareto_front(e, x, HIGHER[metric])
                    ax.plot(front[x], front.test_metric, color=COLOR[enc], marker=MARKER[enc], label=LABEL[enc], zorder=3)
                for value in t[t.encoding == "gbdt"].test_metric:
                    ax.axhline(value, color=INK, ls=":", lw=1.2)
                    ax.annotate("GBDT", (0.98, value), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=9)
                ax.set_xscale("log")
                ax.set_xlabel("trainable params" if x == "params" else "inference ms per 64k points")
                ax.set_ylabel(METRIC_LABEL[metric])
            axes[0, j].set_title(pretty(task))
        legend_top(fig, axes[0, 0])
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save(fig, f"pareto_{experiment}")


def fig_depth(df):
    tasks = [(name, t) for name in ("images", "shapes") for t in EXPERIMENTS[name].tasks]  # tabular: pareto_tabular
    fig, axes = figure(nrows=2, ncols=2, aspect=0.72)
    for ax, (name, task) in zip(axes.flat, tasks):
        t = main_runs(df, name)
        t = t[(t.task == task) & (t.width == 128)]
        for enc in ENCODINGS:
            e = t[t.encoding == enc].sort_values("depth")
            ax.plot(e.depth, e.test_metric, color=COLOR[enc], marker=MARKER[enc], label=LABEL[enc])
        log2_axis(ax, (1, 2, 4, 8, 16))
        ax.set(title=pretty(task), ylabel=METRIC_LABEL[t.metric.iloc[0]], xlabel="depth (hidden layers)")
    legend_top(fig, axes[0, 0])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "depth")


# ── Figures: conditions ──────────────────────────────

def raw_depth_family(ax, g, x):
    """Raw MLP at every depth as gray lightness steps (darker = deeper)."""
    for depth, color in zip((1, 2, 4, 8, 16), GRAYS):
        r = g[(g.encoding == "raw") & (g.depth == depth)].sort_values(x)
        ax.plot(r[x], r.test_error, color=color, marker="o", label=f"raw d{depth}")


def fig_frequency(df):
    g = main_runs(df, "sines_freq")
    g["f"] = g.task.str.extract(r"_f(\d+)", expand=False).astype(int)
    fig, (a, b) = figure(ncols=2)
    raw_depth_family(a, g, "f")
    r = g[(g.encoding == "rff") & (g.depth == 1)].sort_values("f")
    a.plot(r.f, r.test_error, color=COLOR["rff"], marker="D", lw=2.4, label="RFF d1")
    a.set_title("Raw MLP: add depth")
    for enc in ENCODINGS:
        e = g[(g.encoding == enc) & (g.depth == 2)].sort_values("f")
        b.plot(e.f, e.test_error, color=COLOR[enc], marker=MARKER[enc], label=LABEL[enc])
    b.set_title("Every encoding at depth 2")
    for ax in (a, b):
        log2_axis(ax, sorted(g.f.unique()))
        ax.set(yscale="log", xlabel="target bandwidth (cycles per unit)")
        direct_labels(ax)
    a.set_ylabel("test NMSE")
    fig.tight_layout(w_pad=4)
    save(fig, "frequency")


def fig_dimension(df):
    fig, axes = figure(ncols=2)
    for ax, name, title in zip(axes, ("sines_dim", "teacher"), ("Sum of sines (Fourier-shaped)", "Random ReLU teacher (MLP-shaped)")):
        g = main_runs(df, name)
        g["d"] = g.task.str.extract(r"_d(\d+)", expand=False).astype(int)
        best = g.loc[g.groupby(["task", "encoding"]).test_error.idxmin()]  # best depth per encoding and task
        for enc in ENCODINGS:
            e = best[best.encoding == enc].sort_values("d")
            ax.plot(e.d, e.test_error, color=COLOR[enc], marker=MARKER[enc], label=LABEL[enc])
        log2_axis(ax, sorted(g.d.unique()))
        ax.set(yscale="log", xlabel="input dimension d", title=title)
    axes[0].set_ylabel("test NMSE (best depth)")
    legend_top(fig, axes[0])  # lines converge at high d, so a shared legend beats end labels
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=3)
    save(fig, "dimension")


def fig_sawtooth(df):
    g = main_runs(df, "sawtooth")
    g["teeth"] = 2 ** (g.task.str.extract(r"_k(\d+)", expand=False).astype(int) - 1)
    fig, (a, b) = figure(ncols=2)
    raw_depth_family(a, g, "teeth")
    r = g[(g.encoding == "rff") & (g.depth == 1)].sort_values("teeth")
    a.plot(r.teeth, r.test_error, color=COLOR["rff"], marker="D", lw=2.4, label="RFF d1")
    log2_axis(a, sorted(g.teeth.unique()))
    a.set(yscale="log", xlabel="teeth (tent map composed k times)", ylabel="test NMSE", title="Representable is not learnable")
    direct_labels(a)

    task = make_task("sawtooth_k5")
    x = task.render_x[:, 0]
    b.plot(x, task.render_y, color=FOG, lw=4, label="target")
    for enc, depth in (("raw", 16), ("rff", 1)):
        _, path = render_of(g, "sawtooth_k5", enc, depth)
        b.plot(x, np.load(path.with_suffix(".npy")), color=COLOR[enc], lw=1.3, label=f"{LABEL[enc]} d{depth}")
    b.set(xlabel="x", ylabel="f(x), standardized", title="16 teeth, width 128", ylim=(task.render_y.min() - 0.3, task.render_y.max() + 1.2))
    b.legend(loc="upper center", ncol=3)
    fig.tight_layout(w_pad=4)
    save(fig, "sawtooth")


def fig_bandwidth(df):
    panels = [(name, t) for name, t in (("sines_freq", "sines_d1_f32"), ("images", "astronaut"), ("images", "text"),
                                         ("shapes", "gyroid"), ("tabular", "california"), ("tabular", "covertype")) if name in set(df.experiment)]
    fig, axes = figure(nrows=2, ncols=3, aspect=0.62)
    for ax, (name, task) in zip(axes.flat, panels):
        e = df[(df.experiment == name) & (df.task == task) & df.encoding.isin(BANDWIDTH) & (df.depth == SWEEP["depth"])
               & (df.width == SWEEP["width"]) & (df.m == SWEEP["m"]) & (df.residual == True)]  # noqa: E712
        for enc in BANDWIDTH:
            s = e[e.encoding == enc].sort_values("sigma")
            ax.plot(s.sigma, s.test_metric, color=COLOR[enc], marker=MARKER[enc], label=LABEL[enc])
            pick = s.loc[s.val_error.idxmin()]
            ax.scatter([pick.sigma], [pick.test_metric], s=70, facecolor="none", edgecolor=COLOR[enc], lw=1.4, zorder=4)
        ax.set(xscale="log", xlabel="σ (frequency scale)", ylabel=METRIC_LABEL[e.metric.iloc[0]], title=pretty(task))
        if e.metric.iloc[0] == "nmse":
            ax.set_yscale("log")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    legend_top(fig, axes.flat[0], ncol=4)
    note(fig, "Depth 2, width 128, m = 256. Circles mark the σ selected on the validation split and reused for the whole grid.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "bandwidth")


def fig_ablations(df):
    fig, axes = figure(nrows=2, ncols=2, aspect=0.7)
    gm, gr = grid_runs(df, "ablate_m"), grid_runs(df, "ablate_residual")
    for j, task in enumerate(EXPERIMENTS["ablate_m"].tasks):
        e = gm[gm.task == task].sort_values("m")
        metric = e.metric.iloc[0]
        axes[0, j].plot(e.m, e.test_metric, color=COLOR["rff"], marker="D")
        log2_axis(axes[0, j], sorted(e.m.astype(int).unique()))
        axes[0, j].set(xlabel="Fourier features m (RFF, depth 2)", ylabel=METRIC_LABEL[metric], title=pretty(task))
        if metric == "nmse":  # a decade of margin so a flat curve reads as flat
            axes[0, j].set(yscale="log", ylim=(e.test_metric.min() / 10, e.test_metric.max() * 10))
        t = gr[gr.task == task]
        for enc in ("raw", "rff"):
            for residual, ls in ((True, "-"), (False, "--")):
                r = t[(t.encoding == enc) & (t.residual == residual)].sort_values("depth")
                axes[1, j].plot(r.depth, r.test_metric, color=COLOR[enc], ls=ls, marker=MARKER[enc], label=f"{LABEL[enc]} {'residual' if residual else 'plain'}")
        log2_axis(axes[1, j], sorted(t.depth.astype(int).unique()))
        axes[1, j].set(xlabel="depth (hidden layers)", ylabel=METRIC_LABEL[metric], yscale="log" if metric == "nmse" else "linear")
        direct_labels(axes[1, j])
    fig.tight_layout(w_pad=6)
    save(fig, "ablations")


# ── Figures: mechanism ───────────────────────────────

def fig_ntk(df):
    ntk = json.loads((RESULTS / "ntk.json").read_text())
    colors = {("raw", True): iter(GRAYS[1:]), ("raw", False): iter(GRAYS[2:]), ("rff", True): iter(BLUES), ("siren", False): iter(VIOLETS)}
    fig, (a, b) = figure(ncols=2)
    for r in ntk["models"]:
        color = next(colors[r["encoding"], r["residual"]])
        plain = r["encoding"] == "raw" and not r["residual"]
        label = f"RFF σ={r['sigma']}" if r["encoding"] == "rff" else f"{LABEL[r['encoding']]} d{r['depth']}" + (" plain" if plain else "")
        eig = np.maximum(np.array(r["eigvals"]), 1e-12)  # float round-off can leave tiny negative eigenvalues
        a.plot(np.arange(1, len(eig) + 1), eig, color=color, ls="--" if plain else "-", label=label)
        b.plot(ntk["x"], r["slice"], color=color, ls="--" if plain else "-", label=label)
    a.set(xscale="log", yscale="log", ylim=(1e-8, 2), xlabel="eigenvalue index", ylabel="λᵢ / λ₁", title="NTK spectrum at init")
    b.set(xlabel="x", ylabel="K(x, 0.5) / K(0.5, 0.5)", title="Kernel slice")
    fig.tight_layout(w_pad=3)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.01), fontsize=8, handlelength=1.8)
    save(fig, "ntk")


BANDS = (("1–8 cycles", 1, 8), ("8–16", 8, 16), ("16–32", 16, 33))  # the f <= 32 target has 16 sinusoids in [1, 32] cycles


def fig_spectral_bias(df):
    s = np.load(RESULTS / "spectral.npz")
    models = [k for k in s.files if k not in ("steps", "freqs", "truth")]
    fig, axes = figure(ncols=len(models), aspect=0.36, sharey=True)
    for ax, key in zip(axes, models):
        for (label, lo, hi), color in zip(BANDS, (GRAYS[1], GRAYS[2], INK)):
            band = (s["freqs"] >= lo) & (s["freqs"] < hi)
            rel = (s[key][:, band] ** 2).sum(1) / (s["truth"][band] ** 2).sum()  # [T] residual energy / target energy in the band
            ax.plot(s["steps"], rel, color=color, label=label)
        enc, depth = key.split("_d")
        ax.set(title=f"{LABEL[enc]}, depth {depth}", xlabel="training step", yscale="log")
        direct_labels(ax)
    axes[0].set_ylabel("residual / target energy")
    fig.tight_layout(w_pad=5)
    save(fig, "spectral_bias")


# ── Figures: qualitative ─────────────────────────────

def fig_images(df):
    g = main_runs(df, "images")
    for task in EXPERIMENTS["images"].tasks:
        t = make_task(task)
        fig, axes = figure(nrows=2, ncols=7, aspect=0.36)
        axes[0, 0].imshow(t.render_y.reshape(t.render_shape))
        axes[0, 0].set_title("ground truth", fontsize=8)
        axes[1, 0].text(0.5, 0.5, "top: 1 layer\nbottom: 16 layers", transform=axes[1, 0].transAxes, ha="center", va="center", fontsize=8, color=MUTED)
        for i, depth in enumerate((1, 16)):
            for j, enc in enumerate(ENCODINGS, 1):
                row, path = render_of(g, task, enc, depth)
                axes[i, j].imshow(Image.open(path.with_suffix(".png")))
                axes[i, j].set_title(f"{LABEL[enc]}\n{row.test_metric:.1f} dB" if i == 0 else f"{row.test_metric:.1f} dB", fontsize=8, color=COLOR[enc])
        for ax in axes.flat:
            ax.set_axis_off()
        note(fig, f"Width 128; top row depth 1, bottom row depth 16. PSNR on held-out pixels (training sees 1/4 of the {t.render_shape[0]}² grid).")
        save(fig, f"images_{task}")


LIGHT = np.array([0.4, -0.5, 0.77])


def draw_occupancy(ax, occ):
    """Marching-cubes surface of a [R, R, R] occupancy grid with two-tone Lambert shading."""
    ax.set_axis_off()
    ax.view_init(elev=22, azim=40)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlim=(0, RES), ylim=(0, RES), zlim=(0, RES))
    vol = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)  # display-only smoothing of the voxel staircase
    if vol.max() < 0.5:  # the model predicts an empty volume: nothing to draw
        return
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5, step_size=2)
    tri = verts[faces]  # [F, 3 vertices, 3 coords]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lam = np.abs(n @ LIGHT) / (np.linalg.norm(n, axis=1) + 1e-12)  # [F]
    lit, shadow = np.array(to_rgb("#bfdbfe")), np.array(to_rgb(BLUES[3]))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=shadow + (lit - shadow) * lam[:, None], edgecolor="none"))


def fig_shapes(df):
    g = main_runs(df, "shapes")
    columns = (("raw", 1), ("raw", 16), ("rff", 1), ("periodic", 1), ("siren", 4))
    tasks = EXPERIMENTS["shapes"].tasks
    fig = plt.figure(figsize=(WIDTH, 0.36 * WIDTH))
    for i, task in enumerate(tasks):
        t = make_task(task)
        ax = fig.add_subplot(len(tasks), len(columns) + 1, i * (len(columns) + 1) + 1, projection="3d")
        draw_occupancy(ax, t.render_y.reshape(t.render_shape))
        ax.set_title(f"{pretty(task)}\nground truth", fontsize=9, loc="center")
        for j, (enc, depth) in enumerate(columns, 2):
            row, path = render_of(g, task, enc, depth)
            occ = np.unpackbits(np.load(path.with_suffix(".npy")))[:RES ** 3].reshape(t.render_shape)
            ax = fig.add_subplot(len(tasks), len(columns) + 1, i * (len(columns) + 1) + j, projection="3d")
            draw_occupancy(ax, occ)
            ax.set_title(f"{LABEL[enc]} d{depth}\nIoU {row.test_metric:.3f}", fontsize=9, color=COLOR[enc], loc="center")
    fig.subplots_adjust(wspace=0, hspace=0.15)
    save(fig, "shapes")


# ── Text summary ─────────────────────────────────────

def fmt_params(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e4 else f"{n / 1e3:.1f}k"


def fig_summary(df):
    """results/summary.md: per task, the best raw MLP (any size) vs the best depth <= 2 model of each encoding."""
    cell = lambda r: f"{r.test_metric:.4g} ({fmt_params(r.params)})"
    best = lambda t: t.loc[t.test_error.idxmin()]
    lines = ["| task | metric | raw, best of grid | " + " | ".join(f"{LABEL[e]}, depth ≤ 2" for e in FOURIER) + " |", "|---" * (3 + len(FOURIER)) + "|"]
    seen = set()
    for name in [n for n in MAIN if n in set(df.experiment)]:
        g = main_runs(df, name)
        for task in [t for t in EXPERIMENTS[name].tasks if t not in seen]:
            seen.add(task)
            t = g[g.task == task]
            ff = [cell(best(t[(t.encoding == e) & (t.depth <= 2)])) for e in FOURIER]
            lines.append(f"| {pretty(task)} | {t.metric.iloc[0]} | {cell(best(t[t.encoding == 'raw']))} | " + " | ".join(ff) + " |")
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    print(text, flush=True)


# ── Main ─────────────────────────────────────────────

FIGS = {"summary": fig_summary, "exchange_rate": fig_exchange_rate, "pareto": fig_pareto, "depth": fig_depth, "frequency": fig_frequency,
        "dimension": fig_dimension, "sawtooth": fig_sawtooth, "bandwidth": fig_bandwidth, "ablations": fig_ablations,
        "ntk": fig_ntk, "spectral_bias": fig_spectral_bias, "images": fig_images, "shapes": fig_shapes}


def main():
    """Draw the requested figures (default: all) into figures/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", choices=list(FIGS))
    names = parser.parse_args().figures or list(FIGS)  # argparse rejects a list default against choices
    set_style()
    df = load_results()
    for name in names:
        FIGS[name](df)
        print(f"[INFO] figure {name}", flush=True)


if __name__ == "__main__":
    main()
