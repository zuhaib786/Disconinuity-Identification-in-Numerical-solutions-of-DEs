#!/usr/bin/env python
"""Run a pre-estimated, wall-clock-bounded triangular 2D Euler benchmark."""

import argparse
import json
from pathlib import Path

from tci.evaluate_euler2d import (
    double_mach_setup,
    estimate_euler_setup,
    forward_step_setup,
    four_quadrant_setup,
    run_euler_setup,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=["riemann", "double_mach", "step"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=0.05)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.problem == "riemann":
        setup = four_quadrant_setup(args.n, args.final_time or 0.3)
    elif args.problem == "double_mach":
        setup = double_mach_setup(4 * args.n, args.n, args.final_time or 0.2)
    else:
        setup = forward_step_setup(3 * args.n, args.n, args.final_time or 4.0)
    estimate = estimate_euler_setup(setup, args.cfl)
    print(json.dumps({"estimate": estimate}, indent=2), flush=True)
    if estimate["estimated_runtime_s"] > args.max_seconds:
        raise TimeoutError("estimated Euler runtime exceeds --max-seconds")
    result = run_euler_setup(setup, args.cfl, args.max_seconds)
    text = json.dumps({"estimate": estimate, "result": result}, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
