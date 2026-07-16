"""Solver-in-the-loop benchmarks for triangular 2D advection."""

import time

import numpy as np

from tci.mesh import perturbed_delaunay_mesh, rectangular_mesh
from tci.solvers.dg2d import AdvectionDG2D


def rotational_velocity(x, y, time=0.0, center=(0.5, 0.5), omega=2 * np.pi):
    return np.stack(
        [-omega * (y - center[1]), omega * (x - center[0])], axis=-1
    )


def slotted_disk(x, y, center=(0.5, 0.75), radius=0.15, slot_width=0.05):
    disk = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    slot = (np.abs(x - center[0]) < slot_width / 2.0) & (y >= center[1])
    return (disk & ~slot).astype(float)


def graph_total_variation(values, mesh):
    edges = mesh.graph_edges
    return float(np.sum(np.abs(values[edges[:, 0]] - values[edges[:, 1]])))


def rotation_solver(n=16, mesh_type="structured", seed=0):
    if mesh_type == "structured":
        mesh = rectangular_mesh(nx=n, ny=n)
    elif mesh_type == "delaunay":
        mesh = perturbed_delaunay_mesh(nx=n, ny=n, jitter=0.2, seed=seed)
    else:
        raise ValueError("mesh_type must be 'structured' or 'delaunay'")
    return AdvectionDG2D(
        mesh,
        velocity=rotational_velocity,
        boundary_func=lambda x, y, time: np.zeros_like(x),
    )


def estimate_rotation(indicator=None, n=16, mesh_type="structured", seed=0, cfl=0.15):
    solver = rotation_solver(n, mesh_type, seed)
    u = solver.project(slotted_disk)
    steps = int(np.ceil(1.0 / solver.stable_dt(cfl)))
    repeats = 3
    started = time.perf_counter()
    for _ in range(repeats):
        for _ in range(3):
            trial = solver.rhs(u)
            if indicator is not None:
                flags = indicator.flag(solver, trial)
                from tci.limiters2d import limit_p1

                limit_p1(solver, trial, flags)
    seconds_per_step = (time.perf_counter() - started) / repeats
    return {
        "cells": solver.K,
        "steps": steps,
        "stable_dt": solver.stable_dt(cfl),
        "max_skewness": float(np.max(solver.mesh.skewness)),
        "estimated_runtime_s": steps * seconds_per_step,
    }


def run_slotted_rotation(
    indicator=None,
    n=16,
    mesh_type="structured",
    seed=0,
    cfl=0.15,
    max_seconds=None,
):
    solver = rotation_solver(n, mesh_type, seed)
    mesh = solver.mesh
    u0 = solver.project(slotted_disk)
    initial_means = solver.cell_means(u0)
    initial_mass = solver.integral(u0)
    started = time.perf_counter()
    u, history = solver.solve(
        u0,
        1.0,
        cfl=cfl,
        indicator=indicator,
        record_flags=True,
        max_seconds=max_seconds,
    )
    runtime = time.perf_counter() - started
    means = solver.cell_means(u)
    l1 = float(
        np.sum(mesh.areas * np.abs(means - initial_means)) / np.sum(mesh.areas)
    )
    l2 = float(
        np.sqrt(
            np.sum(mesh.areas * (means - initial_means) ** 2)
            / np.sum(mesh.areas)
        )
    )
    metrics = {
        "cells": mesh.K,
        "l1_error": l1,
        "l2_error": l2,
        "initial_total_variation": graph_total_variation(initial_means, mesh),
        "total_variation": graph_total_variation(means, mesh),
        "undershoot": max(0.0, -float(np.min(u))),
        "overshoot": max(0.0, float(np.max(u)) - 1.0),
        "mass_error": abs(solver.integral(u) - initial_mass),
        "flagged_pct": 100.0 * float(np.mean([flags.mean() for _, flags in history]))
        if history
        else 0.0,
        "runtime_s": runtime,
    }
    return metrics, {
        "solver": solver,
        "u": u,
        "history": history,
        "v_exact": initial_means,
    }
