# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2025-12-29

Initial release. Fork of [rLIC](https://github.com/neutrinoceros/rlic) with boundary-aware extensions:

- Mask support for inner boundaries (streamlines stop at masked pixels)
- Edge gain parameters (`edge_gain_strength`, `edge_gain_power`) for aesthetic halos near boundaries
- Tiled parallel execution (`tile_shape`, `num_threads`) for improved performance on large images (could probably be more easily achieved calling multiple threads with python.)
