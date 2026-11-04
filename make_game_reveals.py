#!/usr/bin/env python3
"""Fetch public gameplay and weights; render one sequential four-game GIF.

Usage: python make_game_reveals.py --cache data --out figures
No model is run. Gameplay is an overlay, not a reconstruction from weights.
"""
from __future__ import annotations

import argparse
import bisect
from pathlib import Path
import urllib.request

import matplotlib
from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from make_weight_space_figures import (
    CHANNELS, DISPLAY_CHANNELS, GAMES, download, game_channel_scales,
    game_weight_maps, matrix_from_text,
)

GAMEPLAY_SOURCE = 'https://rikiyat.github.io/blog/weight-space-programming/assets/which_game/'
FILENAMES = {'Pac-Man': 'pacman', 'Breakout': 'breakout', 'Freeway': 'freeway', 'Pong': 'pong'}
WIDTH, HEIGHT, CELL = 720, 720, 240
POSITIONS = [(70, 102), (410, 102), (70, 412), (410, 412)]
GAMEPLAY_MS, FADE_MS, HEATMAP_MS, FRAME_MS = 3000, 400, 3000, 100
SERIES_FILENAME = 'all_games_reveal.gif'
FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'


def centered(draw, text, x, y, size=18, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    draw.text((x, y), text, font=ImageFont.truetype(str(FONT_DIR / name), size),
              fill='#1b2632', anchor='mt')


def read_recording(name, cache):
    path = cache / 'gameplay' / (FILENAMES[name] + '.gif')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = GAMEPLAY_SOURCE + path.name
        print('Fetching', url, flush=True)
        with urllib.request.urlopen(url, timeout=50) as response:
            path.write_bytes(response.read())
    frames, ends, elapsed = [], [], 0
    with Image.open(path) as im:
        if im.size != (CELL, CELL):
            raise ValueError('Gameplay must be 240x240 to align with the 10x10 grid.')
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert('RGB').copy())
            elapsed += frame.info.get('duration', FRAME_MS)
            ends.append(elapsed)
    return frames, ends


def game_segment(label, model_id, name, cache):
    raw = download(f'exp1_which_game/networks/{model_id}_actor.txt', cache)
    maps = game_weight_maps(matrix_from_text(raw, 'W1', 500, 32))
    scales = game_channel_scales(maps)
    heatmaps = [Image.fromarray((colormaps['viridis'](
        np.minimum(maps[k] / scales[k], 1))[..., :3] * 255).astype('uint8')).resize(
            (CELL, CELL), Image.Resampling.NEAREST) for k in DISPLAY_CHANNELS]
    recording, frame_ends = read_recording(name, cache)
    frames, durations = [], []
    # One frame represents the entire static hold, saving memory.
    for ms in range(0, GAMEPLAY_MS + FADE_MS + FRAME_MS, FRAME_MS):
        if ms < GAMEPLAY_MS:
            alpha, phase = 1.0, 'Original gameplay'
        elif ms < GAMEPLAY_MS + FADE_MS:
            alpha, phase = 1 - (ms - GAMEPLAY_MS) / FADE_MS, ''
        else:
            alpha, phase = 0.0, 'Weight maps'
        # Start after the source recording's opening policy annotation.
        source_ms = 3500 + ms
        source = recording[min(bisect.bisect_right(frame_ends, source_ms), len(recording) - 1)]
        page = Image.new('RGB', (WIDTH, HEIGHT), 'white')
        draw = ImageDraw.Draw(page)
        centered(draw, f'Network {label}: {name}', WIDTH // 2, 12, 24, True)
        centered(draw, phase, WIDTH // 2, 48, 15)
        for k, heatmap, (x, y) in zip(DISPLAY_CHANNELS, heatmaps, POSITIONS):
            centered(draw, f'{CHANNELS[k]} (input channel {k})', x + CELL // 2, y - 26, 14, True)
            page.paste(Image.blend(heatmap, source, alpha), (x, y))
            centered(draw, f'weight scale 0–{scales[k]:.3f}', x + CELL // 2, y + CELL + 7, 11)
        frames.append(page)
        durations.append(HEATMAP_MS if alpha == 0 else FRAME_MS)
    return frames, durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('figures'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, durations = [], []
    montage = Image.new('RGB', (240 * 4, 240 * 4), 'white')
    for row, game in enumerate(GAMES):
        segment, timing = game_segment(*game, args.cache)
        for col, ms in enumerate([0, 1500, 3200, 3400]):
            montage.paste(segment[ms // FRAME_MS].resize((240, 240)), (col * 240, row * 240))
        frames.extend(segment)
        durations.extend(timing)
    # One palette across the sequence keeps the fade colors stable.
    palette = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    output = args.out / SERIES_FILENAME
    indexed[0].save(output, save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, optimize=True, disposal=1)
    print(f'Saved {output} ({sum(durations) / 1000:.1f} seconds)')


if __name__ == '__main__':
    main()
