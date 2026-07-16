#!/usr/bin/env python
"""Freeze the primary comparison and disposition for P5-LOCAL-EXTREMA-001."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tci.extrema_evaluation import assess_extrema
from tci.safety_evaluation import aggregate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def by_representation(rows, representation):
    matches = [row for row in rows if row["representation"] == representation]
    if len(matches) != 1:
        raise ValueError(f"expected one summary for {representation}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extrema-controlled",
        type=Path,
        default=Path("runs/feature-review/p5-local-extrema-controlled.json"),
    )
    parser.add_argument(
        "--extrema-heldout",
        type=Path,
        default=Path("runs/feature-review/p5-local-extrema-heldout.json"),
    )
    parser.add_argument(
        "--phase3-controlled", type=Path, default=Path("runs/paper/phase3-controlled-table.json")
    )
    parser.add_argument(
        "--phase3-heldout", type=Path, default=Path("runs/paper/phase3-heldout-rows.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/feature-review/p5-local-extrema-summary.json")
    )
    args = parser.parse_args()

    paths = (args.extrema_controlled, args.extrema_heldout, args.phase3_controlled, args.phase3_heldout)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    extrema_controlled = json.loads(args.extrema_controlled.read_text())
    extrema_rows = json.loads(args.extrema_heldout.read_text())["rows"]
    primary_controlled = json.loads(args.phase3_controlled.read_text())
    primary_rows = [
        row
        for row in json.loads(args.phase3_heldout.read_text())["rows"]
        if row["representation"] == "invariant-node-v2"
    ]
    extrema_offline = by_representation(
        extrema_controlled["offline_summary"], "invariant-extrema-v3"
    )
    primary_offline = by_representation(
        primary_controlled["offline_summary"], "invariant-node-v2"
    )
    selections = extrema_controlled["threshold_selection"]
    if len(selections) != 1 or selections[0]["representation"] != "invariant-extrema-v3":
        raise ValueError("extrema controlled table must contain one threshold selection")
    extrema_heldout = aggregate(extrema_rows)
    primary_heldout = aggregate(primary_rows)
    acceptance = assess_extrema(
        extrema_offline,
        selections[0],
        extrema_heldout,
        primary_offline,
        primary_heldout,
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_id": "P5-LOCAL-EXTREMA-001",
        "scientific_change": "two bounded lower/upper neighbor-envelope violation features added to invariant-node-v2",
        "data_id": extrema_controlled["data_id"],
        "split_id": extrema_controlled["split_id"],
        "sources": {
            str(path): {"sha256": sha256(path)} for path in paths
        },
        "offline": {"primary": primary_offline, "extrema": extrema_offline},
        "threshold_selection": selections[0],
        "heldout": {"primary": primary_heldout, "extrema": extrema_heldout},
        "permutation_stress": extrema_controlled["permutation_stress"],
        "acceptance": acceptance,
        "disposition": (
            "adopt as a Phase 5 extension; Phase 4 remains frozen"
            if acceptance["passes"]
            else "do not adopt; retain as a partial/negative controlled result"
        ),
        "gate": extrema_controlled["gate"],
    }
    atomic_json(args.output, payload)
    print(json.dumps({"acceptance": acceptance, "disposition": payload["disposition"]}, indent=2))


if __name__ == "__main__":
    main()
