# bryLIC

This fork builds on the lovely [rLIC](https://github.com/neutrinoceros/rlic) project by @neutrinoceros.
Kudos to him.

## What is this?

`bryLIC` is a rust implementation of LIC with various idosyncratic features
useful for working on domains with inner boundaries. This repo currently
contains various adjustments to avoid aliasing caused by boundary effects and
further algorithmic adjustments near the boundary for dials letting you construct
aesthetically appealing "halos" around the boundaries.

This repo is heavily vibecoded and currently completely unstable so approach with caution.

## Features

### Edge Gain Effects

bryLIC supports two types of edge gain for creating "halo" effects:

- **Mask-based edge gain** (`edge_gain_strength`, `edge_gain_power`): Applied near internal mask boundaries
- **Domain edge gain** (`domain_edge_gain_strength`, `domain_edge_gain_power`): Applied near external domain boundaries (with closed boundary conditions)

Both can be used independently or together to create different visual effects. Edge gain
amplifies the LIC effect near boundaries by boosting pixel values based on how much the
convolution kernel was truncated by the boundary.

### Parallel Tiling

Supports multi-threaded processing via configurable tile decomposition for improved
performance on large images.

## Scripts

Utility scripts live under `scripts/`. Install the minimal extras with
`pip install -r scripts/requirements.txt`, then invoke them directly
(e.g. `python scripts/run_compare.py`).
