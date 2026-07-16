#!/usr/bin/env python
"""Freeze the primary comparison and disposition for P5-LABEL-SAFETY-001."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tci.halo_evaluation import assess_label_halo
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
        "--halo-training",
        type=Path,
        default=Path("runs/feature-review/p5-label-halo-training-summary.json"),
    )
    parser.add_argument(
        "--halo-controlled",
        type=Path,
        default=Path("runs/feature-review/p5-label-halo-controlled.json"),
    )
    parser.add_argument(
        "--halo-heldout",
        type=Path,
        default=Path("runs/feature-review/p5-label-halo-heldout.json"),
    )
    parser.add_argument(
        "--phase3-controlled", type=Path, default=Path("runs/paper/phase3-controlled-table.json")
    )
    parser.add_argument(
        "--phase3-heldout", type=Path, default=Path("runs/paper/phase3-heldout-rows.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/feature-review/p5-label-halo-summary.json")
    )
    args = parser.parse_args()

    paths = (
        args.halo_training,
        args.halo_controlled,
        args.halo_heldout,
        args.phase3_controlled,
        args.phase3_heldout,
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    training = json.loads(args.halo_training.read_text())
    controlled = json.loads(args.halo_controlled.read_text())
    halo_rows = json.loads(args.halo_heldout.read_text())["rows"]
    primary_controlled = json.loads(args.phase3_controlled.read_text())
    primary_rows = [
        row
        for row in json.loads(args.phase3_heldout.read_text())["rows"]
        if row["representation"] == "invariant-node-v2"
    ]
    if controlled["data_id"] != training["data_id"] or controlled["split_id"] != training["split_id"]:
        raise ValueError("halo training and controlled tables use different data/split IDs")
    policies = [row["label_policy"] for row in training["rows"]]
    if len(policies) != 5 or any(
        policy["training_hops"] != 1 or policy["validation_hops"] != 0 for policy in policies
    ):
        raise ValueError("expected five one-hop training / zero-hop validation label policies")

    halo_offline = by_representation(controlled["offline_summary"], "invariant-node-v2")
    primary_offline = by_representation(
        primary_controlled["offline_summary"], "invariant-node-v2"
    )
    selections = controlled["threshold_selection"]
    if len(selections) != 1 or selections[0]["representation"] != "invariant-node-v2":
        raise ValueError("halo controlled table must contain one threshold selection")
    halo_heldout = aggregate(halo_rows)
    primary_heldout = aggregate(primary_rows)
    acceptance = assess_label_halo(
        halo_offline,
        selections[0],
        halo_heldout,
        primary_offline,
        primary_heldout,
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_id": "P5-LABEL-SAFETY-001",
        "scientific_change": "expand training labels by exactly one face-adjacency hop while retaining original validation labels",
        "data_id": controlled["data_id"],
        "split_id": controlled["split_id"],
        "sources": {str(path): {"sha256": sha256(path)} for path in paths},
        "label_policy": policies[0],
        "offline": {"primary": primary_offline, "label_halo": halo_offline},
        "threshold_selection": selections[0],
        "heldout": {"primary": primary_heldout, "label_halo": halo_heldout},
        "permutation_stress": controlled["permutation_stress"],
        "acceptance": acceptance,
        "disposition": (
            "adopt as a Phase 5 label-policy extension; Phase 4 remains frozen"
            if acceptance["passes"]
            else "do not adopt; retain as a negative controlled label-policy result"
        ),
        "gate": controlled["gate"],
    }
    atomic_json(args.output, payload)
    print(json.dumps({"acceptance": acceptance, "disposition": payload["disposition"]}, indent=2))


if __name__ == "__main__":
    main()
