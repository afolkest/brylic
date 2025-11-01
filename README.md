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

## Scripts

Utility scripts live under `scripts/`. Install the minimal extras with
`pip install -r scripts/requirements.txt`, then invoke them directly
(e.g. `python scripts/run_compare.py`).
