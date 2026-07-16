#!/usr/bin/env python
"""Freeze the `data-v3` label constants on a held-out calibration batch (plan 6.1).

The batch is disjoint from every training data seed and is measured once,
before any `data-v3` model is trained.  The predeclared selection rule lives in
`tci.phase6.select_label_constants`.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tci.data.curves2d import CURVE_TYPES, sample_curve
from tci.data.generate2d_v3 import (
    ROTATION_CENTER,
    resolve_spec,
    sample_mesh,
    smooth_extremum_field,
)
from tci.data.labels2d import duffy_quadrature, label_cells, uniform_refine
from tci.phase6 import (
    ALPHA_CANDIDATES,
    CORE_AREA_FRACTION,
    GAMMA_CANDIDATES,
    select_label_constants,
)


def discontinuity_batch(spec, count, rng):
    """Sharp, O(1)-amplitude reference discontinuities over the curve family."""
    barycentric, area_weights = duffy_quadrature(spec["label"]["quadrature_order"])
    batch = []
    for index in range(count):
        mesh, _ = sample_mesh(rng, spec)
        curve = sample_curve(rng, CURVE_TYPES[index % len(CURVE_TYPES)])
        amplitude = float(rng.choice([-1.0, 1.0]))
        background_slope = rng.normal(0.0, 0.3, size=2)

        def field(x, y, curve=curve, amplitude=amplitude, slope=background_slope):
            smooth = slope[0] * x + slope[1] * y
            return smooth + amplitude * (curve.distance(x, y) < 0.0)

        fraction = curve.inside_fraction(mesh, barycentric, area_weights)
        core = (fraction >= CORE_AREA_FRACTION[0]) & (fraction <= CORE_AREA_FRACTION[1])
        batch.append({"mesh": mesh, "field": field, "core": core, "refinement": uniform_refine(mesh)})
    return batch


def smooth_batch(spec, count, rng):
    batch = []
    for _ in range(count):
        mesh, _ = sample_mesh(rng, spec)
        field, _ = smooth_extremum_field(rng, spec, float(np.sqrt(np.mean(mesh.areas))))
        batch.append({"mesh": mesh, "field": field, "refinement": uniform_refine(mesh)})
    return batch


def resolution_awareness(spec, alpha, gamma):
    """One steep layer resolved on a fine mesh must not be labelled troubled."""
    from tci.mesh import rectangular_mesh

    rows = []
    for resolution in (8, 16, 32, 64):
        mesh = rectangular_mesh(nx=resolution, ny=resolution)
        h = float(np.sqrt(np.mean(mesh.areas)))
        width = 1.0 / 16.0  # fixed physical layer width: resolved only when h << width
        field = lambda x, y: 0.5 * (1.0 + np.tanh((x + 0.5 * y - 0.7) / width))
        labels, _ = label_cells(
            mesh, field, alpha=alpha, gamma=gamma, order=spec["label"]["quadrature_order"]
        )
        rows.append(
            {
                "resolution": resolution,
                "h_over_layer_width": h / width,
                "positive_fraction": float(np.mean(labels)),
                "positive_cells": int(np.sum(labels)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/paper/phase6-label-calibration.json"))
    parser.add_argument("--discontinuity-samples", type=int, default=60)
    parser.add_argument("--smooth-samples", type=int, default=40)
    parser.add_argument("--calibration-seed", type=int, default=987654321)
    args = parser.parse_args()

    spec = resolve_spec({"ladder": "v3-b"})
    rng = np.random.default_rng(args.calibration_seed)
    print("building the held-out calibration batch", flush=True)
    discontinuities = discontinuity_batch(spec, args.discontinuity_samples, rng)
    smooth = smooth_batch(spec, args.smooth_samples, rng)

    candidates = []
    for gamma in GAMMA_CANDIDATES:
        for alpha in ALPHA_CANDIDATES:
            core_hits = core_total = 0
            smooth_hits = smooth_total = 0
            for case in discontinuities:
                labels, _ = label_cells(
                    case["mesh"],
                    case["field"],
                    alpha=alpha,
                    gamma=gamma,
                    order=spec["label"]["quadrature_order"],
                    refinement=case["refinement"],
                )
                core_hits += int(np.sum(labels & case["core"]))
                core_total += int(np.sum(case["core"]))
            for case in smooth:
                labels, _ = label_cells(
                    case["mesh"],
                    case["field"],
                    alpha=alpha,
                    gamma=gamma,
                    order=spec["label"]["quadrature_order"],
                    refinement=case["refinement"],
                )
                smooth_hits += int(np.sum(labels))
                smooth_total += int(labels.size)
            row = {
                "alpha": alpha,
                "gamma": gamma,
                "core_cells": core_total,
                "core_recall": core_hits / core_total if core_total else 0.0,
                "smooth_cells": smooth_total,
                "smooth_false_positive_rate": smooth_hits / smooth_total if smooth_total else 0.0,
            }
            candidates.append(row)
            print(
                f"alpha={alpha:<5} gamma={gamma:<4} core recall {row['core_recall']:.3f} "
                f"smooth FP {row['smooth_false_positive_rate']:.5f}",
                flush=True,
            )

    selection = select_label_constants(candidates)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "phase 6 data-v3 label-constant calibration",
        "batch": {
            "calibration_seed": args.calibration_seed,
            "discontinuity_samples": args.discontinuity_samples,
            "smooth_samples": args.smooth_samples,
            "curve_family": list(CURVE_TYPES),
            "mesh_spec": spec["meshes"],
            "note": "held out from every training data seed; measured before any data-v3 training",
        },
        "proposed_constants": {"alpha": 0.5, "gamma": 3.0, "source": "plan 6.1 starting values"},
        "selection": selection,
        "resolution_awareness": resolution_awareness(
            spec, selection["alpha"], selection["gamma"]
        ),
        "rotation_center": ROTATION_CENTER.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"selection": {k: selection[k] for k in ("alpha", "gamma", "constraint")}}, indent=2))


if __name__ == "__main__":
    main()
