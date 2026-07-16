# Feature Design for Graph-Based Troubled-Cell Indicators

This is the post-Phase-4 feature roadmap. It distinguishes implemented
representations and measured evidence from proposed experiments. A proposed
benefit is a hypothesis until a complete controlled artifact is linked here.

The row-level review is
`runs/feature-review/gnn-features-audit.json`. It covers 31 suggestions with
implementation status, Phase 2--3 overlap, supporting or contrary evidence,
dependencies, priority, disposition, and a scheduled experiment ID where
appropriate.

## Frozen evidence and schema registry

The controlled sources of record are:

- `runs/paper/phase3-training-summary.json`: 20 five-seed training rows.
- `runs/paper/phase3-calibration-rows.json`: 400 calibration evaluations.
- `runs/paper/phase3-heldout-rows.json`: 120 held-out evaluations.
- `runs/paper/phase3-controlled-table.json`: controlled aggregates and
  permutation stress results.
- `runs/paper/phase4-primary-summary.json`: primary selection and limitations.
- `runs/paper/phase4-controlled-tables.json`: paper-ready controlled tables.

The implementations live in `tci/data/graphs.py` and are loaded from checkpoint
metadata by `tci/indicators/learned.py`.

| Schema | Node width | Edge width | Status |
|---|---:|---:|---|
| `ordered-global-v1` | 10 | none | Historical ordered/global control |
| `invariant-node-v2` | 8 | none | Phase 4 primary five-checkpoint family |
| `invariant-edge-v2` | 8 | 6 | Completed Phase 3 ablation; not primary |
| `invariant-local-v2` | 8 | 6 | Completed negative safety ablation |
| `invariant-extrema-v3` | 10 | none | Completed Phase 5 partial/negative ablation |

All five schemas use the same Phase 2 data and split IDs, GAT capacity,
optimizer, and training seeds 0--4. The selected downstream threshold is
`0.02` for every representation.

## What Phase 3 established

| Representation | F1 mean ± sample SD | PR-AUC | Held-out flags | Worst held-out undershoot | Interpretation |
|---|---:|---:|---:|---:|---|
| `ordered-global-v1` | 0.3716 ± 0.1186 | 0.3806 | 99.85% | 0 | Safe by near-global limiting; vertex-order dependent |
| `invariant-node-v2` | 0.5414 ± 0.0524 | 0.6835 | 35.57% | 0.00822 | Large invariance/selectivity gain; Phase 4 primary |
| `invariant-edge-v2` | 0.5427 ± 0.0583 | 0.6953 | 34.83% | 0.00848 | Only marginal gain over invariant nodes |
| `invariant-local-v2` | 0.7252 ± 0.0185 | 0.8531 | 9.75% | 0.22574 | Strong classifier, unacceptable solver safety |

The invariant schemas preserve node features, edge features when present, and
logits exactly under independent local triangle vertex permutations. The
ordered schema does not. This converts vertex-permutation sensitivity from a
hypothesis into a verified defect and validates invariant node construction.

The primary still narrowly fails the predeclared calibration safety rule: its
maximum calibration undershoot is `0.01447`, above the `0.01` limit. Its
held-out maximum is below the limit, but it is not universally
safety-qualified.

## Adopted primary representation

For a scalar `P1` cell `K`, `invariant-node-v2` uses:

```text
[(mean_K - global_median)/global_range,
 nodal_std/global_range,
 nodal_range/global_range,
 h_K*|grad u_K|/global_range,
 log(area_K/median(one-ring areas)),
 radius_ratio_quality,
 boundary_face_fraction,
 interior_neighbor_fraction]
```

The following choices are adopted for compatible future scalar controls:

- permutation-invariant mean/statistical features;
- dimensionless physical gradient magnitude;
- local relative area and bounded shape quality; and
- boundary/interior-neighbor fractions.

These features were evaluated as a bundle, so the individual contribution of
each geometry or gradient component is not separately identified.

## Completed features not adopted

### Directed interface bundle

`invariant-edge-v2` adds directed mean jumps, mean and maximum face-trace
jumps, normal-gradient jumps, relative face length, and relative centroid
distance. The edge-aware GAT interface is implemented and reusable, but the
bundle is not part of the primary representation: its predictive and flag-rate
changes were marginal while mean held-out runtime rose from `15.57` s to
`20.18` s.

This does not show that all edge context is unhelpful. Flow direction was
deliberately excluded from Phase 3, so a directed `a dot n` experiment remains
an open, separately scheduled hypothesis.

### Pure one-ring robust normalization

`invariant-local-v2` uses

```text
s_K = max(MAD(one-ring cell means),
          max_J |mean_J - mean_K|,
          1e-8 * max(1, median_J |mean_J|)).
```

This exact design is rejected as a primary normalization. It improved offline
classification and reduced excessive limiting, but its worst calibration and
held-out undershoots were `0.20618` and `0.22574`. The unsuccessful result must
remain visible. A bounded global/local hybrid is a new experiment, not a
reinterpretation of this result.

## Phase 5 controlled experiment order

Every experiment changes one factor at a time. Unless explicitly stated, use
the Phase 2 data/split IDs, GAT capacity, optimizer, five seeds, and Phase 3
downstream metrics.

### Completed: `P5-SAFETY-OR-001` — primary GNN OR KXRCF

This experiment directly tested the primary model's small calibration safety
gap without retraining or changing its features.

- Methods: frozen `invariant-node-v2` GNN, KXRCF, and their boolean union.
- GNN checkpoints: all five Phase 4 primary checkpoints.
- Thresholds: frozen GNN `0.02`; existing KXRCF `1.0`; no tuning.
- Calibration: structured and Delaunay `n={10,14}`.
- Held out: structured and Delaunay `n={8,12,16}`.
- Metrics: L1/L2, TV, bounds violations, mass error, flags, and runtime.
- Result: all 60 new rows completed. The union reduced worst calibration
  undershoot from `0.01447` to `0.00849` with zero overshoot and passed both
  held-out L1/L2 guards.
- Rejection: mean held-out flagging rose from `35.57%` to `62.71%`, exceeding
  the predeclared `60%` selectivity limit.
- Artifact: `runs/feature-review/p5-safety-or-summary.json` with raw calibration
  and held-out rows in the adjacent JSON files.

The hard union is therefore a useful safety/selectivity tradeoff and a negative
overall acceptance result. It is not incorporated into the primary detector.

### Completed: `P5-LOCAL-EXTREMA-001` — explicit neighbor-envelope features

Add bounded upper/lower neighbor-extremum ratios to `invariant-node-v2`. This
targets false negatives directly without replacing global normalization. Keep
the architecture and labels fixed. Acceptance requires all calibration rows to
have undershoot/overshoot at most `0.01`, mean held-out flagging at most `40%`,
mean held-out L1/L2 no more than 10% above the Phase 4 primary, and mean offline
F1 and PR-AUC above the primary values. Threshold selection uses the unchanged
Phase 3 candidate set and safety rule before these held-out checks.

All 5 training, 100 calibration, and 30 held-out rows completed. Mean F1 rose
from `0.5414` to `0.5818`, PR-AUC from `0.6835` to `0.7284`, and held-out
flagging fell from `35.57%` to `32.99%`. Held-out L1/L2 and bounds passed their
guards, and permutation stress was exact. The worst calibration undershoot was
`0.01418`, so the experiment failed the `0.01` safety criterion and is not
adopted. Source: `runs/feature-review/p5-local-extrema-summary.json`.

### Completed: `P5-LABEL-SAFETY-001` — one-cell training-label halo

This experiment expanded only the training labels by exactly one
face-adjacency hop; validation and checkpoint selection continued to use the
original geometric labels. Acceptance required all calibration rows to have
undershoot/overshoot at most `0.01`, mean held-out flagging at most `50%`, mean
held-out L1/L2 no more than 10% above the Phase 4 primary, mean validation
recall above `0.7638`, and PR-AUC at least 90% of the primary value. Use the
unchanged five-threshold selection protocol and exact permutation stress test.

The halo increased the training positive fraction from `10.47%` to `19.57%`.
Mean original-label validation recall rose from `0.7638` to `0.8830`, but
PR-AUC fell from `0.6835` to `0.5360`, below the required `0.6151`. All 100
calibration rows completed, yet the least-violating threshold (`0.02`) had
maximum undershoot `0.01113`. Of 30 held-out attempts, 25 completed and 5 hit
the unchanged runtime limit; successful rows averaged `54.46%` flagged cells,
above the `50%` limit. Their mean L1/L2 remained within the error guards, and
permutation stress was exact. The halo therefore failed four gates and is not
adopted. Source: `runs/feature-review/p5-label-halo-summary.json`.

### 1. `P5-LABEL-SAFETY-002..003` — remaining label/objective controls

Next test one soft distance kernel and then a predeclared false-negative cost.
Keep the primary feature schema fixed and report original binary-label metrics
as a secondary evaluation. These are label/objective experiments and must not
be attributed to feature design.

### 2. `P5-SENSOR-FUSION-001` — continuous classical-sensor input

Add one continuous KXRCF magnitude to the primary raw features. Compare with
both the raw primary and the hard OR safety net. This tests learned sensor
fusion; it must not be described as learning solely from DG coefficients.

### 3. `P5-HYBRID-SCALE-001` — bounded global/local scale

Test a predeclared cap that prevents the local scale from becoming arbitrarily
different from the global scale. Run a calibration-only pilot, freeze the cap,
then perform the same five-seed held-out protocol. Pure local scaling remains a
failed control.

### 4. `P5-FLOW-CFL-001..002` — direction, then local CFL

First add only normal advection speed and an inflow/outflow sign to the tested
edge schema. Include reversed flow and rotated fields in held-out tests. Add
local CFL only in a second experiment after the directional effect is known.
Raw gradient components or face normals require a flow-aligned local frame or
an equivariant design; otherwise they would reintroduce rotation dependence.

## Deferred feature families

### Boundary context

Physical boundary-condition edges are deferred to `P5-BOUNDARY-001`. The
current graph contains interior face edges, and periodic rotation cannot
isolate reflective/inflow/outflow boundary semantics. The future test requires
explicit boundary graph edges and a multi-boundary benchmark suite.

### Higher-order modal features

For degree above one, normalized modal energy by degree and a Persson-type
highest-mode ratio remain appropriate hypotheses (`P5-HIGHER-ORDER-001`). They
are deferred because the controlled two-dimensional study is fixed `P1`.

### Euler-aware features

A scalar-density detector is not an Euler indicator. The staged roadmap is:

1. `P5-EULER-PRIMITIVE-001`: locally normalized density, pressure, velocity,
   sound speed, Mach, entropy, gradients, and positivity margins.
2. `P5-EULER-CHAR-001`: face-normal characteristic jumps, compression, and a
   normalized Rankine-Hugoniot residual.
3. `P5-EULER-POSITIVITY-001`: local spectral radius, positivity activation,
   and LLF dissipation, one family at a time.

These require Euler stage-state data, nondimensionalization, and a frozen
multi-problem split. Until then, the existing Euler results establish solver
feasibility only.

### Residual and history features

`P5-RESIDUAL-001` may add one normalized semi-discrete residual after a
numerical-stage dataset exists. Previous-stage changes, previous probabilities,
limiter coefficients, and consecutive-flag counts are deferred to
`P5-HISTORY-001..002`. Stateful experiments must define reset/restart semantics
and verify restart equivalence and hysteresis.

### Downstream-aware training

A differentiable loss for oscillation and excessive limiting is deferred to
`P5-DOWNSTREAM-LOSS-001`. It is substantially more complex than halo labels or
cost-sensitive weighting and should be attempted only after those clean
controls fail.

### Architecture sweeps

GAT depth and matched-capacity GCN/SAGE sweeps (`P5-ARCH-001`) remain low
priority. Phase 3 intentionally held architecture fixed, and no evidence yet
shows that network capacity is the limiting factor.

## Reproducibility rules

- Never combine feature, label, architecture, and training-loss changes in one
  run and attribute the result to a single factor.
- Predeclare data/split IDs, seeds, thresholds, runtime bounds, and
  acceptance/failure criteria.
- Report offline classification and solver-in-the-loop metrics together.
- Retain failed constraints and negative ablations in machine-readable output.
- Do not update the frozen Phase 4 primary silently; a successful Phase 5
  experiment is an extension until a new evidence-freeze phase explicitly
  promotes it.
