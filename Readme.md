# Mesh-transferable troubled-cell indicators

This repository is being developed from the thesis **On Graph Neural Networks as
Troubled Cell Indicators** into a reproducible research package for learned and classical
troubled-cell indicators in discontinuous Galerkin (DG) solvers.

The central research question is whether a local graph-based indicator can transfer to
unseen mesh resolutions, nonuniform or unstructured meshes, polynomial orders, and
conservation laws while retaining solver accuracy and stability.

## Current status

The original implementations remain in `1D/`, `2D/`, `DG-1D/`, and
`ColabNotebooks/` for provenance. New reproducible code lives in `src/tci/`.

The new experiment core currently provides:

- deterministic piecewise-Fourier data with unambiguous cell labels;
- framework-independent line-graph and DG node-feature construction;
- a single tested implementation of binary classification metrics;
- command-line data generation and prediction evaluation;
- unit tests and continuous integration.

The metric module intentionally centralizes false-positive and false-negative definitions.
An original notebook evaluation path interchanged these counts, so thesis metrics should
be treated as provisional until regenerated through this package.

## Installation and smoke test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Generate a small deterministic dataset:

```bash
tci generate \
  --output artifacts/synthetic-smoke \
  --num-samples 10 \
  --n-cells 100 \
  --seed 2026
```

Evaluate saved positive-class scores:

```bash
tci evaluate \
  --labels labels.npy \
  --scores scores.npy \
  --threshold 0.5
```

Thresholds must be chosen on validation data and then frozen for test and
out-of-distribution evaluation.

## Planned experiment ladder

1. Reproduce the original 1D linear-advection CNN and GAT results with corrected metrics.
2. Add parameter-matched MLP, CNN, GCN, GraphSAGE, and GAT baselines.
3. Test transfer across resolution, nonuniformity, DG order, flux, and CFL number.
4. Add nonlinear Burgers and 1D Euler benchmarks.
5. Couple every indicator to the same limiter and measure solution error, conservation,
   total variation, overshoot, limited-cell fraction, stability, and runtime.
6. Extend the successful representation to 2D unstructured triangular meshes.

Every result intended for a paper must use a locked test manifest, at least five training
seeds, and uncertainty intervals. Classification accuracy alone is not sufficient: the
primary evidence is the quality and cost of the resulting DG solution.

## License

See [LICENSE](LICENSE).
