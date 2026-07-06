# py_geo_fd

Python package for virtual deployment of flow diverters, following:
[Virtual flow diverter deployment and embedding for hemodynamic simulations](https://doi.org/10.1016/j.compbiomed.2024.109023)

## What It Does

Given:
- a vessel surface mesh (triangles), and
- a vessel centerline,

the package computes a deployed stent geometry and exports:
- centerline diagnostics,
- a stent envelope with porosity and wire-angle fields,
- wire-centerline curves,
- volumetric tetrahedral stent meshes.

## Installation

### Standard

```bash
pip install -e .
```

### Development (recommended)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
```

## Quickstart

The demo files live in demo.

```bash
cd demo
python3 demo.py
```

Equivalent Python usage:

```python
from py_geo_fd import Stent

stent = Stent("stent_config.json")
stent.save_centerline("centerline.vtp")
stent.save_envelope("stent_envelope")
stent.save_stent_to_vtp("stent.vtp")
stent.save_stent("stent")
```

## Required Inputs

### 1) Surface Mesh

- Any mesh format readable by meshio.
- Must contain triangle cells for wall distance intersection.
- Non-triangle elements are ignored by the current workflow.

### 2) Centerline

Supported centerline formats:
- .json (3D Slicer markups)
- .vtp
- .vtk
- .vtu

Centerlines can be generated with VMTK and curated in 3D Slicer.

## Configuration File

A complete example is provided in demo/stent_config.json.

Top-level structure:

- centerline_params
- stent_params
- wall_params
- n_cpu

Important controls:

- centerline_params.x_smoothing: smoothing applied to centerline geometry.
- centerline_params.dx_smoothing: smoothing applied to curvature/radius-related derivatives.
- centerline_params.s_start: deployment start position along centerline arc length.
- stent_params.numeric_params.n_segments: axial discretization of wire centerlines.
- stent_params.numeric_params.wire_resolution: circumferential discretization for tetra meshing.
- wall_params.n_axial_segments and wall_params.n_radial_segments: sampling resolution used to fit local wall radius.
- wall_params.smoothing: number of cylindrical smoothing passes on sampled radii.
- wall_params.margin: shrink/expand radius offset before deployment.

## Outputs

### save_centerline(path)

Writes .vtp centerline diagnostics, including local frames and radius fields.

### save_envelope(path, N=100, Nw=100, dim=2)

Writes .vtu envelope with fields:
- alpha (wire angle in degrees)
- porosity
- r (local adapted radius)

With dim=3, writes a volumetric envelope of stent thickness.

### save_stent_to_vtp(path)

Writes wire centerlines as polyline .vtp, useful for embedding workflows.

### save_stent(path)

Writes tetrahedral stent meshes:
- path_n.vtu: one winding direction
- path_p.vtu: opposite winding direction
- path.vtu: full stent

## Performance Notes

Primary runtime drivers:
- wall_params.n_axial_segments
- wall_params.n_radial_segments
- stent_params.numeric_params.n_segments
- stent_params.numeric_params.wire_resolution
- n_cpu

If runtime is high, first reduce n_axial_segments and n_radial_segments,
then n_segments.

## Troubleshooting

- If loading fails, verify centerline/surface file paths in the JSON config.
- If deployment starts at an unexpected location, adjust s_start and centerline orientation.
- If geometry looks noisy, increase centerline smoothing and/or wall smoothing.
- If wall intersection misses regions, ensure the surface is watertight and triangulated.

## Development

Run tests:

```bash
.venv/bin/pytest -q
```

Run lint:

```bash
.venv/bin/ruff check src tests
```

Run a simple construction benchmark:

```bash
cd demo
../.venv/bin/python benchmark.py --runs 3
```

## Funding

European Union (ERC, CURE, 101045042). The views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<p align="center">
  <img src="https://cure-erc.github.io/imgs/CURE_LOGO.png" alt="CURE Logo" width="325">
  <img src="https://cure-erc.github.io/imgs/LOGO-ERC.jpg" alt="Logo ERC" width="175">
</p>
