#!/usr/bin/env python3
"""Create static blind-test PNGs from Takehi et al.'s released weights.

Requirements: Python 3, numpy, matplotlib; internet for the first run.
Usage: python make_weight_space_figures.py --out figures
       python make_weight_space_figures.py --cache data --out figures

This script never runs a neural-network forward pass.
All figures use numerical weights exactly as supplied to Astra, rounded to 3 decimals.
"""
from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCE_COMMIT = '4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
SOURCE = f'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{SOURCE_COMMIT}'
# Deliberately arbitrary presentation order: no game/shape names in PNGs.
GAMES = [
    ('A', 'bdf99279', 'Pac-Man'),
    ('B', 'a9f535cc', 'Breakout'),
    ('C', '0bbcc3ea', 'Freeway'),
    ('D', 'dfc6cb65', 'Pong'),
]
SHAPES = [
    ('A', 'a6a9dfb7', 'square'),
    ('B', '246a5b58', 'triangle'),
    ('C', 'd88aba69', 'circle'),
]
CHANNELS = ['agent', 'moving objects', 'reward targets', 'obstacles', 'previous frame']
DISPLAY_CHANNELS = (0, 1, 2, 4)


def download(repo_path: str, cache: Path) -> str:
    file = cache / repo_path
    if not file.is_file():
        file.parent.mkdir(parents=True, exist_ok=True)
        url = SOURCE + '/' + repo_path
        print('Fetching', url, flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'weight-space-figure-reproduction/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=50) as response:
                data = response.read()
        except Exception as e:
            raise RuntimeError(f'Could not fetch {url}. Check network access or use --cache with downloaded files. {e}') from e
        file.write_bytes(data)
    return file.read_text(encoding='utf-8')


def matrix_from_text(txt: str, name: str, rows: int, cols: int) -> np.ndarray:
    lines = txt.splitlines()
    candidates = [i for i, line in enumerate(lines) if re.match(r'^' + re.escape(name) + r'\s+\(', line)]
    if len(candidates) != 1:
        raise ValueError(f'Expected one {name} matrix header, found {len(candidates)}')
    idx = candidates[0] + 1
    matrix = []
    for line in lines[idx:]:
        text = line.strip()
        parts = text.split()
        try:
            vals = [float(s) for s in parts]
        except ValueError:
            break
        if len(vals) != cols:
            raise ValueError(f'Unexpected row length in {name}, got {len(vals)}, expected {cols}: {text[:60]}')
        matrix.append(vals)
        if len(matrix) == rows:
            break
    if len(matrix) != rows:
        raise ValueError(f'Expected {rows} rows for {name}, found {len(matrix)}')
    return np.array(matrix, dtype=np.float64)


def game_weight_maps(w1: np.ndarray) -> np.ndarray:
    """Spatial mean-|W1| maps indexed by input channel, not neural layer.

    W1[i,j] connects input feature i to first-hidden-layer neuron j.
    M[k,r,c] = mean_j |W1[(r*10+c)*5+k,j]|, over all 32 neurons.
    """
    return np.abs(w1.reshape(10, 10, 5, 32)).mean(axis=-1).transpose(2, 0, 1)


def game_channel_scales(maps: np.ndarray) -> np.ndarray:
    """An independent 99th-percentile scale for each input channel / model."""
    return np.maximum(np.quantile(maps, .99, axis=(1, 2)), 1e-12)


def style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_game(ax, m: np.ndarray, vmin=0, vmax=None):
    im = ax.imshow(m, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    style_ax(ax)
    return im


def save_game_figures(game_data, out):
    for label, maps in game_data.items():
        scales = game_channel_scales(maps)
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.8), layout='constrained')
        for channel, ax in zip(DISPLAY_CHANNELS, axes.flat):
            im = draw_game(ax, np.minimum(maps[channel] / scales[channel], 1.0), vmax=1)
            ax.set_title(f'{CHANNELS[channel]}\ninput channel {channel}', fontsize=14, pad=10)
            ax.set_xlabel(f'raw scale: 0–{scales[channel]:.3f}', fontsize=11)
        fig.suptitle(f'Network {label} — which game?', fontsize=21, weight='bold')
        fig.colorbar(im, ax=axes, location='bottom', fraction=.035, pad=.03,
                     label='mean |W1| over 32 first-hidden-layer neurons\nscale separately per input channel / model; 99th percentile = 1')
        fig.savefig(out / f'game_{label}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)


def diffusion_weight_map(w3: np.ndarray) -> np.ndarray:
    """Mean absolute final-layer W3 weight per pixel, over 64 incoming weights."""
    return np.abs(w3).mean(axis=0).reshape(12, 12)


def save_diffusion_figure(shape_data, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout='constrained')
    for ax, (label, weights) in zip(axes, shape_data.items()):
        values = diffusion_weight_map(weights)
        normalized = values / max(float(values.max()), 1e-12)
        im = ax.imshow(normalized, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        style_ax(ax)
        ax.set_title(f'Network {label}', fontsize=17, pad=10)
    fig.suptitle('What does each network generate?', fontsize=21, weight='bold')
    fig.colorbar(im, ax=axes, location='bottom', fraction=.04, pad=.04,
                 label='mean |final-layer W3 weight| over 64 incoming connections; scaled per model')
    fig.savefig(out / 'diffusion_simple.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_raw_weight_intro(cache, out):
    raw = download('exp1_which_game/networks/a9f535cc_actor.txt', cache)
    w1 = matrix_from_text(raw, 'W1', 500, 32)
    fig = plt.figure(figsize=(11, 6.2), facecolor='white')
    fig.text(.055, .92, 'What Astra received', fontsize=24, weight='bold', color='#18212b')
    fig.text(.055, .858, '17,253 numerical parameters + architecture + input / action meanings',
             fontsize=12, color='#465566')
    fig.text(.055, .755, '500 inputs  →  32 tanh units  →  32 tanh units  →  5 action logits',
             fontsize=15, weight='bold', color='#18212b')
    fig.text(.055, .683, 'h1 = tanh(x @ W1 + b1)    h2 = tanh(h1 @ W2 + b2)    logits = h2 @ W3 + b3',
             fontsize=10.5, family='monospace', color='#465566')
    fig.text(.055, .588, 'W1: first 6 input rows, first 8 of 32 columns', fontsize=12, weight='bold')
    excerpt = '\n'.join('  '.join(f'{v: .3f}' for v in row[:8]) + '  …' for row in w1[:6])
    fig.text(.055, .53, excerpt + '\n  ...', va='top', fontsize=13.5,
             family='monospace', linespacing=1.65, color='#18212b')
    fig.text(.055, .115, 'Actual released weights from Network B (a9f535cc), rounded to 3 decimals.',
             fontsize=10.5, color='#465566')
    fig.text(.055, .065, 'Only the excerpt is shortened here. Astra received every matrix and bias as text.',
             fontsize=10.5, color='#465566')
    fig.savefig(out / 'raw_weights.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('figures'))
    parser.add_argument('--cache', type=Path, default=Path('data'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.weight'] = 'normal'
    save_raw_weight_intro(args.cache, args.out)

    game_data = {}
    for label, model_id, _ in GAMES:
        raw = download(f'exp1_which_game/networks/{model_id}_actor.txt', args.cache)
        game_data[label] = game_weight_maps(matrix_from_text(raw, 'W1', 500, 32))
    save_game_figures(game_data, args.out)

    shape_data = {}
    for label, model_id, _ in SHAPES:
        raw = download(f'exp2_diffusion/networks/{model_id}_weights.txt', args.cache)
        shape_data[label] = matrix_from_text(raw, 'W3', 64, 144)
    save_diffusion_figure(shape_data, args.out)

    print('Saved raw-weight figure, four game figures and one diffusion figure to', args.out)



if __name__ == '__main__':
    main()
