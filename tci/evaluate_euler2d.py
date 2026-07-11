"""Bounded 2D Euler benchmark setups on triangular meshes."""

import time

import numpy as np

from tci.indicators.classical2d import MinmodIndicator2D
from tci.mesh import forward_step_mesh, rectangular_mesh
from tci.solvers.euler2d import (
    EulerDG2D,
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
    solver.rhs(U)
    indicator.flag(solver, U[:, :, 0])
    seconds_per_stage = time.perf_counter() - started
    return {
        "cells": solver.K,
        "steps": steps,
        "stable_dt": dt,
        "estimated_runtime_s": 3.0 * steps * seconds_per_stage,
    }


def run_euler_setup(setup, cfl=0.05, max_seconds=300.0):
    solver, U0, final_time = setup
    initial = solver.integral(U0)
    started = time.perf_counter()
    U = solver.solve(
        U0,
        final_time,
        indicator=MinmodIndicator2D(),
        cfl=cfl,
        max_seconds=max_seconds,
    )
    runtime = time.perf_counter() - started
    rho, _, _, pressure = conserved_to_primitive_2d(U)
    return {
        "cells": solver.K,
        "runtime_s": runtime,
        "min_density": float(np.min(rho)),
        "min_pressure": float(np.min(pressure)),
        "max_density": float(np.max(rho)),
        "integral_change": np.abs(solver.integral(U) - initial).tolist(),
    }
