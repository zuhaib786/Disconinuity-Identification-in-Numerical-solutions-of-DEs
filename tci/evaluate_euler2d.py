"""Bounded 2D Euler benchmark setups on triangular meshes."""

import time
from pathlib import Path

import numpy as np

from tci.indicators.classical2d import MinmodIndicator2D
from tci.mesh import forward_step_mesh, rectangular_mesh
from tci.solvers.euler2d import (
    EulerDG2D,
    EulerTimeoutError,
    conserved_to_primitive_2d,
    primitive_to_conserved_2d,
    reflective_boundary_2d,
)


def _constant_primitive(rho, u, v, p):
    return primitive_to_conserved_2d(
        np.asarray(rho), np.asarray(u), np.asarray(v), np.asarray(p)
    )


def four_quadrant_setup(n=20, final_time=0.3):
    solver = EulerDG2D(rectangular_mesh(nx=n, ny=n))
    centroids = solver.mesh.centroids
    states = {
        "ne": _constant_primitive(1.5, 0.0, 0.0, 1.5),
        "nw": _constant_primitive(0.5323, 1.206, 0.0, 0.3),
        "sw": _constant_primitive(0.138, 1.206, 1.206, 0.029),
        "se": _constant_primitive(0.5323, 0.0, 1.206, 0.3),
    }
    U = np.empty((3, solver.K, 4))
    for cell, (x, y) in enumerate(centroids):
        key = ("n" if y >= 0.5 else "s") + ("e" if x >= 0.5 else "w")
        U[:, cell] = states[key]
    return solver, U, final_time


def double_mach_setup(nx=48, ny=12, final_time=0.2):
    mesh = rectangular_mesh((0.0, 4.0), (0.0, 1.0), nx, ny)
    pre = _constant_primitive(1.4, 0.0, 0.0, 1.0)
    post = _constant_primitive(
        8.0, 8.25 * np.cos(np.pi / 6), -8.25 * np.sin(np.pi / 6), 116.5
    )

    def boundary(x, y, current_time, U_minus, normal):
        U_plus = U_minus.copy()
        left = np.all(np.isclose(x, 0.0), axis=1)
        top = np.all(np.isclose(y, 1.0), axis=1)
        bottom = np.all(np.isclose(y, 0.0), axis=1)
        U_plus[left] = post
        if np.any(top):
            shock = 1.0 / 6.0 + (1.0 + 20.0 * current_time) / np.sqrt(3.0)
            U_plus[top] = np.where((x[top] < shock)[..., None], post, pre)
        if np.any(bottom):
            reflected = reflective_boundary_2d(
                x[bottom], y[bottom], current_time, U_minus[bottom], normal[bottom]
            )
            U_plus[bottom] = np.where(
                (x[bottom] < 1.0 / 6.0)[..., None], post, reflected
            )
        return U_plus

    solver = EulerDG2D(mesh, boundary_state=boundary)
    nodes = solver.nodes
    shock = 1.0 / 6.0 + nodes[:, :, 1] / np.sqrt(3.0)
    U = np.where((nodes[:, :, 0] < shock)[..., None], post, pre)
    return solver, U, final_time


def forward_step_setup(nx=30, ny=10, final_time=4.0):
    mesh = forward_step_mesh(nx, ny)
    inflow = _constant_primitive(1.4, 3.0, 0.0, 1.0)

    def boundary(x, y, current_time, U_minus, normal):
        U_plus = U_minus.copy()
        mean_x, mean_y = np.mean(x, axis=1), np.mean(y, axis=1)
        left = np.isclose(mean_x, 0.0)
        right = np.isclose(mean_x, 3.0)
        wall = ~left & ~right
        U_plus[left] = inflow
        if np.any(wall):
            U_plus[wall] = reflective_boundary_2d(
                x[wall], y[wall], current_time, U_minus[wall], normal[wall]
            )
        return U_plus

    solver = EulerDG2D(mesh, boundary_state=boundary)
    U = np.broadcast_to(inflow, (3, solver.K, 4)).copy()
    return solver, U, final_time


def estimate_euler_setup(setup, cfl=0.05):
    solver, U, final_time = setup
    dt = solver.stable_dt(U, cfl)
    steps = int(np.ceil(final_time / dt))
    indicator = MinmodIndicator2D()
    started = time.perf_counter()
    U1 = U + dt * solver.rhs(U, 0.0)
    U1 = solver._postprocess(U1, indicator)
    U2 = 0.75 * U + 0.25 * (U1 + dt * solver.rhs(U1, dt))
    U2 = solver._postprocess(U2, indicator)
    candidate = U / 3.0 + 2.0 / 3.0 * (
        U2 + dt * solver.rhs(U2, 0.5 * dt)
    )
    solver._postprocess(candidate, indicator)
    seconds_per_step = time.perf_counter() - started
    return {
        "cells": solver.K,
        "steps": steps,
        "stable_dt": dt,
        "seconds_per_initial_step": seconds_per_step,
        "estimated_runtime_s": 1.5 * steps * seconds_per_step,
    }


def run_euler_setup(setup, cfl=0.05, max_seconds=300.0, checkpoint_path=None):
    solver, U0, final_time = setup
    initial = solver.integral(U0)
    start_time = 0.0
    prior_runtime = 0.0
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None and checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            U0 = checkpoint["U"]
            start_time = float(checkpoint["time"])
            saved_final_time = float(checkpoint["final_time"])
            if "runtime_s" in checkpoint:
                prior_runtime = float(checkpoint["runtime_s"])
        if U0.shape != (3, solver.K, 4) or not np.isclose(saved_final_time, final_time):
            raise ValueError("Euler checkpoint is incompatible with this benchmark")
        print(
            f"resuming Euler checkpoint at t={start_time:.6g}/{final_time:.6g}",
            flush=True,
        )

    initial_dt = solver.stable_dt(U0, cfl)
    estimated_remaining_steps = max(
        1, int(np.ceil((final_time - start_time) / initial_dt))
    )
    progress_interval = max(1, estimated_remaining_steps // 100)
    started = time.perf_counter()

    def report(current_time, U, steps):
        rho, _, _, pressure = conserved_to_primitive_2d(U)
        elapsed = time.perf_counter() - started
        fraction = current_time / final_time if final_time else 1.0
        run_fraction = (
            (current_time - start_time) / (final_time - start_time)
            if final_time > start_time
            else 1.0
        )
        eta = (
            elapsed * (1.0 - run_fraction) / run_fraction
            if run_fraction > 0
            else float("inf")
        )
        print(
            f"Euler progress: {100*fraction:6.2f}%  "
            f"t={current_time:.6g}/{final_time:.6g}  steps={steps}  "
            f"elapsed={elapsed:.1f}s  eta={eta:.1f}s  "
            f"rho_min={np.min(rho):.3e}  p_min={np.min(pressure):.3e}",
            flush=True,
        )

    try:
        U = solver.solve(
            U0,
            final_time,
            indicator=MinmodIndicator2D(),
            cfl=cfl,
            max_seconds=max_seconds,
            start_time=start_time,
            progress_callback=report,
            progress_interval=progress_interval,
        )
    except EulerTimeoutError as exc:
        if checkpoint_path is None:
            raise
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint_path,
            U=exc.U,
            time=exc.current_time,
            final_time=exc.final_time,
            runtime_s=prior_runtime + time.perf_counter() - started,
        )
        rho, _, _, pressure = conserved_to_primitive_2d(exc.U)
        return {
            "status": "timeout",
            "cells": solver.K,
            "completed_time": exc.current_time,
            "final_time": final_time,
            "accepted_steps_this_run": exc.steps,
            "runtime_s": prior_runtime + time.perf_counter() - started,
            "min_density": float(np.min(rho)),
            "min_pressure": float(np.min(pressure)),
            "checkpoint": str(checkpoint_path),
        }
    runtime = prior_runtime + time.perf_counter() - started
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    rho, _, _, pressure = conserved_to_primitive_2d(U)
    return {
        "status": "ok",
        "cells": solver.K,
        "runtime_s": runtime,
        "min_density": float(np.min(rho)),
        "min_pressure": float(np.min(pressure)),
        "max_density": float(np.max(rho)),
        "integral_change": np.abs(solver.integral(U) - initial).tolist(),
        "U": U,
    }
