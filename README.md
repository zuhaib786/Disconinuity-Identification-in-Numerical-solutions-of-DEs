# tci — Troubled Cell Indicators for Discontinuous Galerkin Methods

Code for the thesis **"On Graph Neural Networks as Troubled Cell Indicators"**
(Zuhaib Ul Zamann, IIT Delhi, 2023), restructured as an installable,
config-driven experiment harness.

A *troubled-cell indicator* (TCI) decides in which mesh cells a DG solution
is losing regularity so that a limiter can act there — and only there. This
package implements classical indicators (minmod/TVB, KXRCF), the polynomial
annihilation (PA) detector, a fixed-stencil MLP baseline (in the style of
Ray & Hesthaven, JCP 2018), and a graph-neural-network detector that treats
the mesh as a graph (one node per cell, edges between face-adjacent cells)
and therefore works on meshes of any size without retraining.

Solvers: 1D nodal DG (Hesthaven & Warburton) for linear advection, inviscid
Burgers, and the compressible Euler equations (local Lax-Friedrichs flux,
SSP-RK3, per-stage limiting), plus an exact Riemann solver (Toro) for
shock-tube references. Step 3 adds fixed-`P1` scalar-advection DG on affine
triangles, including structured and seeded unstructured Delaunay meshes.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # numpy-only core: solver, classical + PA indicators
pip install -e '.[ml]'    # + torch, torch_geometric for the GNN indicator
pip install -e '.[dev]'   # + pytest
```

## Quickstart

Train the learned detectors (≈1 min each on CPU) and compare all indicators
across the benchmark problems. In every benchmark the indicator decides
where the minmod limiter acts; the learned indicators are trained on
**advection data only**, so the Burgers/Euler rows measure cross-PDE
generalization:

```bash
python -m tci.train configs/gnn1d.yaml --out runs/gnn1d
python -m tci.train configs/mlp1d.yaml --out runs/mlp1d
python scripts/run_benchmarks.py --problems all \
    --indicators minmod kxrcf pa gnn mlp \
    --gnn-model runs/gnn1d/model.pt --mlp-model runs/mlp1d/model.pt
```

GPU training is selected automatically when CUDA is available. The saved
checkpoint is moved back to CPU so it remains portable. Confirm the log says
`training device: cuda`; set `train.device: cuda` to fail explicitly instead
of falling back to CPU. On a Colab T4, increase `train.batch_size` to 64 or
128 for the 2D GNN configs. Mesh generation and NumPy DG benchmarks remain
CPU workloads.

Representative results (L1 error of cell means vs the exact/fine-grid
reference; "% flagged" is the mean fraction of cells limited per step):

**box** — advection of a box profile to T = 0.5 (thesis secs. 4.8/5.9):

| indicator | L1 error | % flagged | | **sod** — Sod tube, T = 0.2 | L1 error | % flagged |
|-----------|---------:|----------:|-|------------------------------|---------:|----------:|
| minmod    | 0.026    | 3.3       | | minmod                       | 0.0026   | 6.9       |
| kxrcf     | 0.043    | 48.3      | | kxrcf                        | 0.0040   | 2.1       |
| pa        | 0.023    | 0.0       | | pa                           | *diverged* | —       |
| gnn       | 0.029    | **0.6**   | | gnn                          | 0.0031   | **0.12**  |
| mlp       | 0.028    | 0.7       | | mlp                          | 0.0033   | 0.11      |

**burgers** — sine to shock, T = 0.3: all indicators comparable (L1
0.0060-0.0067); gnn/mlp/pa flag under 1% of cells vs minmod's 2.6%.

**shu_osher** — shock/entropy-wave interaction, T = 1.8: minmod L1 0.077
(over-limits the acoustic waves), kxrcf 0.044, pa 0.033, mlp 0.025. The
previous GNN row is not scientifically valid: the reported checkpoint used
`N=1` features while this benchmark defaults to `N=2`, and the old harness
misclassified that model-shape error as solver divergence.

Corrected threshold sweep using the exact-advection checkpoint and `N=1` for
both problems (full results: `runs/gnn1d/threshold-sweep.json`):

| threshold | Sod L1 | Sod TV | Sod % flagged | Shu-Osher L1 | Shu-Osher TV | Shu-Osher % flagged |
|----------:|-------:|-------:|--------------:|--------------:|--------------:|---------------------:|
| 0.02 | 0.00378 | 0.929 | 0.773 | 0.0494 | 12.83 | 1.973 |
| 0.05 | 0.00332 | 0.918 | 0.213 | 0.0440 | 14.24 | 1.486 |
| 0.10 | 0.00307 | 0.914 | 0.124 | 0.0423 | 14.82 | 1.179 |
| 0.20 | 0.00296 | 0.923 | 0.0346 | 0.0394 | 15.66 | 0.979 |
| 0.30 | **0.00290** | 0.924 | **0.0257** | **0.0372** | 16.25 | **0.863** |

No run diverged. Higher thresholds limit fewer cells and reduce L1 error on
both problems, but Shu-Osher TV rises substantially. Threshold selection is
therefore an accuracy/oscillation tradeoff, not a label-F1-only decision.
These `N=1` Shu-Osher rows must not be directly compared with the earlier
`N=2` baseline rows.

Findings so far, reproducing and extending the thesis (secs. 4.8, 5.9):

- The GNN localizes limiting to ~5-50x fewer cells than minmod at
  comparable solution error on box, Burgers, and Sod.
- KXRCF over-flags smeared fronts (48% on box); PA, an exact-data method,
  goes quiet on evolved numerical data and lets Sod blow up.
- At compatible `N=1`, the exact-advection GNN remains stable on Shu-Osher at
  every tested threshold. Numerical-advection and random-Euler training modes
  are available for the next data-distribution ablation.

Step 2.5 experiment commands:

```bash
# Train on evolved, oscillatory advection states or random Euler shock tubes.
python -m tci.train configs/gnn1d-numerical.yaml --out runs/gnn1d-numerical
python -m tci.train configs/gnn1d-euler.yaml --out runs/gnn1d-euler

# Sweep one checkpoint at its compatible N on Sod and Shu-Osher.
python scripts/sweep_gnn_thresholds.py \
    --model runs/gnn1d-numerical/model.pt \
    --thresholds 0.02 0.05 0.1 0.2 0.3 \
    --output runs/gnn1d-numerical/threshold-sweep.json

# Learned flags OR the KXRCF safety-net flags at every RK stage.
python scripts/run_benchmarks.py --problems box sod \
    --indicators gnn gnn-kxrcf --gnn-model runs/gnn1d-numerical/model.pt
```

## Library use

```python
import numpy as np
from tci import DG1D, BurgersDG1D, EulerDG1D, MinmodIndicator
from tci.solvers.euler import sod_initial

solver = DG1D(0.0, 1.0, K=100, N=1)                  # 100 elements, P1
u0 = solver.project(lambda x: np.where((x >= .4) & (x <= .6), 1.0, 0.0))
u, history = solver.advect(u0, a=1.0, final_time=0.5,
                           indicator=MinmodIndicator(), record_flags=True)

euler = EulerDG1D(0.0, 1.0, K=200, N=1)              # Sod shock tube
U = euler.solve(sod_initial, 0.2, indicator=MinmodIndicator())

from tci import AdvectionDG2D, random_delaunay_mesh

mesh = random_delaunay_mesh(n_interior=100, seed=0)
dg2 = AdvectionDG2D(mesh, velocity=(1.0, 0.25))
u2 = dg2.solve(lambda x, y: np.sin(2*np.pi*x) * np.cos(2*np.pi*y), 0.1)
```

Every indicator implements `flag(solver, u) -> bool[K]` (for Euler, `u` is
the density field); plug in your own by subclassing
`tci.indicators.base.Indicator`.

## Layout

```
tci/
├── solvers/        # nodal DG: advection, Burgers, Euler + exact Riemann
├── mesh.py         # validated triangular topology, geometry, cell graphs
├── indicators/     # minmod/TVB, KXRCF, polynomial annihilation, GNN, MLP
├── limiters.py     # minmod MUSCL limiter applied to flagged cells
├── data/           # piecewise-Fourier/Euler data, cell graphs, stencils
├── models.py       # GAT/GCN/SAGE + MLP detectors (architecture ablations)
├── train.py        # YAML-config-driven training (model.type: gnn | mlp)
└── evaluate.py     # solver-in-the-loop benchmarks (box/burgers/sod/shu_osher)
configs/            # one YAML per experiment/ablation
scripts/            # CLI entry points
tests/              # pytest suite
legacy/             # original thesis scripts and notebooks (unmaintained)
```

Ablations (architecture, data mode exact vs numerical, mesh-length
randomization, threshold, polynomial order) are new YAML files under
`configs/`, not new code.

The current 2D milestone supports scalar advection with inflow/outflow or
periodic traces on arbitrary affine triangular meshes. Exact line/circle-cut
data use one graph node per triangle, shared-face edges, and 10 node features:
three `P1` values plus seven dimensionless geometry features.

The first 2D GAT checkpoint (`runs/gnn2d-exact/model.pt`) was trained on
2,000 variable Delaunay meshes with approximately 130-330 cells. At threshold
0.1 it reached validation precision 0.405, recall 0.676, and F1 0.506. The
same checkpoint was then evaluated without resizing or retraining:

| mean cells | positive % | flagged % | precision | recall | F1 |
|-----------:|-----------:|----------:|----------:|-------:|---:|
| 70  | 16.76 | 45.08 | 0.311 | 0.835 | 0.453 |
| 630 | 6.46  | 7.14  | 0.565 | 0.625 | 0.593 |

Full results are in `runs/gnn2d-exact/mesh-generalization.json`. This supports
the variable-mesh graph claim, but the small-mesh over-flagging shows that
calibration does not yet transfer uniformly across mesh scales.

The first solver-in-the-loop test rotates a slotted disk for one revolution
on 288-cell meshes. L1 compares final cell means with the initial projected
DG field; TV is the sum of jumps over graph edges:

| mesh | indicator | L1 | TV | undershoot | % flagged | runtime (s) |
|------|-----------|---:|---:|-----------:|----------:|------------:|
| structured | unlimited | 0.0387 | 16.10 | 0.750 | 0.0 | 2.1 |
| structured | minmod2d | 0.0515 | 3.84 | 0 | 84.2 | 41.6 |
| structured | gnn2d | **0.0451** | 6.23 | 0.042 | **4.35** | **23.2** |
| unstructured | unlimited | 0.0233 | 12.06 | 0.318 | 0.0 | 2.6 |
| unstructured | minmod2d | **0.0489** | 3.87 | 0 | 82.9 | 49.0 |
| unstructured | gnn2d | 0.0492 | 4.78 | 0.012 | **15.2** | **30.1** |

The GNN matches or improves minmod L1 while limiting far fewer cells and
running faster, but leaves small negative undershoots. Results are stored in
`runs/gnn2d-exact/rotation-*-n12.json`. Unstructured rotation uses a jittered
grid followed by Delaunay triangulation; unconstrained random points produced
near-degenerate cells and impractical explicit CFL steps.

Additional 2D baselines expose different tradeoffs. On the structured mesh,
the fixed-feature MLP has L1 0.0400 with only 0.21% flagged, but undershoot
0.135; KXRCF has L1 0.0312, undershoot 0.416, and flags 37.9%. On the
unstructured mesh, MLP has L1 0.0343 / undershoot 0.0557 / 0.59% flagged;
KXRCF has L1 0.0254 / undershoot 0.0902 / 40.9% flagged. Thus label F1 alone
does not rank downstream robustness: the GNN is more monotone than MLP/KXRCF
while remaining much more selective than minmod.

A mixed exact/numerical GNN was also trained on 1,000 bounded trajectories.
It is a negative ablation: label F1 fell to 0.242; it slightly reduced the
unstructured undershoot (0.012 to 0.005) but doubled flags, and substantially
worsened structured undershoot. The exact-data checkpoint remains primary.

The fixed-P1 2D Euler solver now includes local Lax-Friedrichs fluxes,
conserved-component limiting, all-cell density/pressure positivity scaling,
SSP-RK stage rejection, periodic/transmissive/reflective boundaries, and
wall-clock limits. Bounded positive smoke runs are available for four-quadrant
Riemann, double Mach reflection, and a nonconvex forward-facing step under
`runs/euler2d-*-smoke.json`. Full-resolution shock runs are intentionally left
to the cluster harness because their local estimates exceed the runtime
policy.

`scripts/run_rotation2d.py` estimates completion time before solving and
enforces `--max-seconds` both before and during each run. Numerical 2D
training, threshold/grid, multi-checkpoint evaluation, Euler, and plotting
scripts all write incrementally or enforce hard deadlines. The completed
comparison figure is `images/rotation2d-metrics.png`.

## References

See [REFERENCES.md](REFERENCES.md). The DG solver follows Hesthaven &
Warburton, *Nodal Discontinuous Galerkin Methods* (2008).

## License

[MIT](LICENSE)
