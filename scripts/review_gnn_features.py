#!/usr/bin/env python
"""Write the Phase 5 machine-readable audit of the GNN feature roadmap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tci.feature_review import SUGGESTIONS, validate_suggestions


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase3", type=Path, default=Path("runs/paper/phase3-controlled-table.json")
    )
    parser.add_argument(
        "--phase4", type=Path, default=Path("runs/paper/phase4-primary-summary.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/feature-review/gnn-features-audit.json")
    )
    args = parser.parse_args()

    for path in (args.phase3, args.phase4):
        if not path.exists():
            raise FileNotFoundError(path)
    validate_suggestions()
    dispositions = Counter(row["disposition"] for row in SUGGESTIONS)
    experiments = sorted(
        {
            row["scheduled_experiment_id"]
            for row in SUGGESTIONS
            if row["scheduled_experiment_id"]
        }
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Post-Phase-3 audit of every suggestion in GNN_FEATURES.md plus the explicit Phase 5 safety/architecture extensions.",
        "evidence": {
            "phase3": {"path": str(args.phase3), "sha256": sha256(args.phase3)},
            "phase4": {"path": str(args.phase4), "sha256": sha256(args.phase4)},
        },
        "review_policy": {
            "one_factor_at_a_time": True,
            "fixed_data_id": "be8eb9faab9ba9647e27d04c8eee4d4af079dd7c3d9f9451bb36da559662bf56",
            "fixed_split_id": "cf1b1a002250b2c7c41b0e0cdb6d3285b28bc9eb446b38e8a49f3ae96f82ed94",
            "training_seeds": [0, 1, 2, 3, 4],
            "primary_representation": "invariant-node-v2",
            "primary_threshold": 0.02,
        },
        "summary": {
            "suggestion_count": len(SUGGESTIONS),
            "dispositions": dict(sorted(dispositions.items())),
            "scheduled_experiment_ids": experiments,
            "first_experiment": "P5-LABEL-SAFETY-002",
        },
        "suggestions": SUGGESTIONS,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
