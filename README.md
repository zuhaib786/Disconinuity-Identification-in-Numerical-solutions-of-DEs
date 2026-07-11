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
shock-tube references.

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
(over-limits the acoustic waves), kxrcf 0.044, pa 0.033, mlp 0.025;
**the advection-trained gnn diverges** — the strong-shock regime is outside
its training distribution (see Findings).

Findings so far, reproducing and extending the thesis (secs. 4.8, 5.9):

- The GNN localizes limiting to ~5-50x fewer cells than minmod at
  comparable solution error on box, Burgers, and Sod.
- KXRCF over-flags smeared fronts (48% on box); PA, an exact-data method,
  goes quiet on evolved numerical data and lets Sod blow up.
- Cross-PDE generalization from advection training has a limit: Shu-Osher
  breaks the GNN. Retraining on numerical/Euler data is the next ablation
  axis; a solve that blows up raises RuntimeError (never hangs) and is
  reported as DIVERGED in the tables.

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
```

Every indicator implements `flag(solver, u) -> bool[K]` (for Euler, `u` is
the density field); plug in your own by subclassing
`tci.indicators.base.Indicator`.

## Layout

```
tci/
├── solvers/        # nodal DG: advection, Burgers, Euler + exact Riemann
├── indicators/     # minmod/TVB, KXRCF, polynomial annihilation, GNN, MLP
├── limiters.py     # minmod MUSCL limiter applied to flagged cells
├── data/           # randomized piecewise-Fourier data, cell graphs, stencils
├── models.py       # GAT/GCN/SAGE + MLP detectors (architecture ablations)
├── train.py        # YAML-config-driven training (model.type: gnn | mlp)
└── evaluate.py     # solver-in-the-loop benchmarks (box/burgers/sod/shu_osher)
configs/            # one YAML per experiment/ablation
scripts/            # CLI entry points
tests/              # pytest suite (35 tests)
legacy/             # original thesis scripts and notebooks (unmaintained)
```

Ablations (architecture, data mode exact vs numerical, mesh-length
randomization, threshold, polynomial order) are new YAML files under
`configs/`, not new code.

## References

See [REFERENCES.md](REFERENCES.md). The DG solver follows Hesthaven &
Warburton, *Nodal Discontinuous Galerkin Methods* (2008).

## License

[MIT](LICENSE)
