# Weight-space figures

Fetch the public data and reproduce the raw-weight excerpt, four game heatmaps,
diffusion heatmaps and sequential gameplay reveal GIF. Python 3.10+.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python make_weight_space_figures.py
python make_game_reveals.py
```

The scripts download and cache inputs in `data/` and write six PNGs and one GIF
to `figures/`. Both accept `--cache PATH --out PATH`. Cached data and generated
figures are excluded from Git. The GIF script also works on its own.

- Games: mean absolute first-layer W1 weights to 32 hidden neurons, reshaped by
  input channel into 10×10 maps. Show channels 0, 1, 2 and 4; omit obstacles.
  Scale each channel within each model from zero to its 99th percentile, clipping
  above it. The same fixed scales are used in the static figures and GIF.
- Diffusion: mean absolute final-layer W3 weight per output pixel across its 64
  incoming connections; reshape to 12×12 and scale by each model's maximum.
- GIF: Pac-Man → Breakout → Freeway → Pong. Each game: 3 seconds of recorded
  gameplay, 0.4-second fade, 3 seconds of weight maps, then a hard cut.

No model is evaluated or trained. The maps average weights and omit biases.

Data: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab,
[released weights and logs](https://github.com/RikiyaT/lm_understand-logs),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), pinned to commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Gameplay recordings come from the
[study page](https://rikiyat.github.io/blog/weight-space-programming/).
