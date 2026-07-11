# tci — Troubled Cell Indicators for Discontinuous Galerkin Methods

Code for the thesis **"On Graph Neural Networks as Troubled Cell Indicators"**
(Zuhaib Ul Zamann, IIT Delhi, 2023), restructured as an installable,
config-driven experiment harness.

A *troubled-cell indicator* (TCI) decides in which mesh cells a DG solution
is losing regularity so that a limiter can act there — and only there. This
package implements classical indicators (minmod/TVB, KXRCF), the polynomial
annihilation (PA) detector, and a graph-neural-network detector that treats
the mesh as a graph (one node per cell, edges between face-adjacent cells)
and therefore works on meshes of any size without retraining.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # numpy-only core: solver, classical + PA indicators
pip install -e '.[ml]'    # + torch, torch_geometric for the GNN indicator
pip install -e '.[dev]'   # + pytest
```

## Quickstart

Train the GNN detector (≈2 min on CPU) and compare all indicators on the
thesis benchmark — a box profile advected with speed 1 to T = 0.5 under
periodic BCs, minmod-limited in whichever cells each indicator flags:

```bash
python -m tci.train configs/gnn1d.yaml --out runs/gnn1d
python scripts/run_advection.py --indicators minmod kxrcf pa gnn \
    --model runs/gnn1d/model.pt --plot
```

Representative output (K = 100 elements, N = 1, exact TV of the profile is 2):

| indicator | L2 error | total variation | % cells flagged/step |
|-----------|---------:|----------------:|---------------------:|
| minmod    | 0.090    | 2.000           | 3.2                  |
| kxrcf     | 0.111    | 2.029           | 48.2                 |
| pa        | 0.083    | 2.412           | 0.0                  |
| gnn       | 0.095    | 2.095           | **0.6**              |

The GNN localizes limiting to ~5× fewer cells than minmod at comparable
solution error; KXRCF over-flags the smeared fronts; PA (an exact-data
method) goes quiet on evolved numerical data — reproducing the thesis
findings (sections 4.8, 5.9).

## Library use

```python
import numpy as np
from tci import DG1D, MinmodIndicator

solver = DG1D(0.0, 1.0, K=100, N=1)                  # 100 elements, P1
u0 = solver.project(lambda x: np.where((x >= .4) & (x <= .6), 1.0, 0.0))
u, history = solver.advect(u0, a=1.0, final_time=0.5,
                           indicator=MinmodIndicator(), record_flags=True)
```

Every indicator implements `flag(solver, u) -> bool[K]`; plug in your own by
subclassing `tci.indicators.base.Indicator`.

## Layout

```
tci/
├── solvers/        # nodal DG building blocks + 1D advection solver
├── indicators/     # minmod/TVB, KXRCF, polynomial annihilation, GNN
├── limiters.py     # minmod MUSCL limiter applied to flagged cells
├── data/           # randomized piecewise-Fourier data + cell-graph datasets
├── models.py       # GAT/GCN/SAGE detectors (architecture ablations)
├── train.py        # YAML-config-driven training
└── evaluate.py     # solver-in-the-loop benchmark + metrics
configs/            # one YAML per experiment/ablation
scripts/            # CLI entry points
tests/              # pytest suite (solver convergence, indicator sanity, GNN)
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
