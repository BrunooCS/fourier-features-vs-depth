"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Models: Fourier-family input encodings and MLP bodies (MLX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An encoding maps v in R^d to an embedding of width ~2m that feeds a ReLU MLP.
`siren` has no encoding: sine activations run through the whole body instead.

  raw        v
  pe         [cos, sin](2pi f_j v_i),   f_j log-spaced in [1, sigma], fixed      Tancik et al. 2020 ("positional")
  rff        [cos, sin](2pi B v),       B ~ N(0, sigma^2) of shape [d, m], fixed  Tancik et al. 2020 ("Gaussian")
  lff        as rff, B trainable                                                  learned Fourier features
  periodic   [cos, sin](2pi c_ij v_i),  c ~ N(0, sigma^2) of shape [d, m//d]      Gorishniy et al. 2022
  siren      sin(w0 (W h + b)) layers, w0 = 30                                    Sitzmann et al. 2020

Capacity accounting: `count_params` counts trainable tensors only (fixed Fourier matrices are
buffers); `flops_per_sample` counts forward matmul FLOPs of every 2D tensor (2 per MAC).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


# ── Encodings ────────────────────────────────────────

class Identity(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out_dim = d

    def __call__(self, x):
        return x


class DenseFourier(nn.Module):
    """Gaussian Fourier features: every frequency vector mixes all input axes."""

    def __init__(self, d, m, sigma, trainable):
        super().__init__()
        self.B = sigma * mx.random.normal((d, m))  # [d, m]
        self.out_dim = 2 * m
        if not trainable:
            self.freeze(keys=["B"])

    def __call__(self, x):
        z = 2 * math.pi * (x @ self.B)  # [N, d] -> [N, m]
        return mx.concatenate([mx.cos(z), mx.sin(z)], axis=-1)  # [N, 2m]


class AxisFourier(nn.Module):
    """Axis-aligned Fourier features: each input axis gets its own k frequencies."""

    def __init__(self, freqs, trainable):
        super().__init__()
        self.freqs = freqs  # [d, k]
        self.out_dim = 2 * freqs.size
        if not trainable:
            self.freeze(keys=["freqs"])

    def __call__(self, x):
        z = 2 * math.pi * (x[:, :, None] * self.freqs).reshape(x.shape[0], -1)  # [N, d, 1] * [d, k] -> [N, d*k]
        return mx.concatenate([mx.cos(z), mx.sin(z)], axis=-1)  # [N, 2dk]


def positional_freqs(d, m, sigma):
    return mx.array(np.tile(np.geomspace(1.0, sigma, m // d), (d, 1)).astype(np.float32))  # [d, m//d]


ENCODINGS = {
    "raw": lambda d, m, sigma: Identity(d),
    "pe": lambda d, m, sigma: AxisFourier(positional_freqs(d, m, sigma), trainable=False),
    "rff": lambda d, m, sigma: DenseFourier(d, m, sigma, trainable=False),
    "lff": lambda d, m, sigma: DenseFourier(d, m, sigma, trainable=True),
    "periodic": lambda d, m, sigma: AxisFourier(sigma * mx.random.normal((d, m // d)), trainable=True),
}
BANDWIDTH = ("pe", "rff", "lff", "periodic")  # encodings whose sigma is tuned per task


# ── Model ────────────────────────────────────────────

class MLP(nn.Module):
    """Encoding -> ReLU projection -> (depth - 1) hidden layers -> linear head.

    residual=True: pre-norm blocks h + relu(W LN(h)) and a final LN, trainable at any depth.
    residual=False: plain ReLU stack; raw baselines train both bodies and keep the better one.
    """

    def __init__(self, encoding, out_dim, depth, width, residual):
        super().__init__()
        self.encoding, self.residual = encoding, residual
        self.inp = nn.Linear(encoding.out_dim, width)
        self.hidden = [nn.Linear(width, width) for _ in range(depth - 1)]
        self.norms = [nn.LayerNorm(width) for _ in range(depth)] if residual else []
        self.head = nn.Linear(width, out_dim)
        for layer in [self.inp, *self.hidden]:
            layer.weight = nn.init.he_normal()(layer.weight)  # ReLU gain keeps plain deep stacks from vanishing

    def __call__(self, x):
        h = nn.relu(self.inp(self.encoding(x)))  # [N, d] -> [N, emb] -> [N, W]
        if self.residual:
            for layer, norm in zip(self.hidden, self.norms):
                h = h + nn.relu(layer(norm(h)))
            return self.head(self.norms[-1](h))  # [N, out]
        for layer in self.hidden:
            h = nn.relu(layer(h))
        return self.head(h)  # [N, out]


class Siren(nn.Module):
    """Sine MLP with the paper's initialization: first layer U(-1/n, 1/n), later U(+-sqrt(6/n)/w0)."""

    def __init__(self, d, out_dim, depth, width, w0=30.0):
        super().__init__()
        self.w0 = w0
        self.layers = [nn.Linear(d if i == 0 else width, width) for i in range(depth)]
        self.head = nn.Linear(width, out_dim)
        for i, layer in enumerate([*self.layers, self.head]):
            n = layer.weight.shape[1]
            bound = 1 / n if i == 0 else math.sqrt(6 / n) / w0
            layer.weight = mx.random.uniform(-bound, bound, layer.weight.shape)

    def __call__(self, x):
        h = x  # [N, d]
        for layer in self.layers:
            h = mx.sin(self.w0 * layer(h))  # [N, W]
        return self.head(h)  # [N, out]


def build_model(encoding, d, out_dim, depth, width, m, sigma, residual):
    if encoding == "siren":
        return Siren(d, out_dim, depth, width)
    return MLP(ENCODINGS[encoding](d, m, sigma), out_dim, depth, width, residual)


# ── Capacity accounting ──────────────────────────────

def count_params(model):
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def flops_per_sample(model):
    return sum(2 * v.size for _, v in tree_flatten(model.parameters()) if v.ndim == 2)
