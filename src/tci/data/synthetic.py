"""Deterministic synthetic functions with exact troubled-cell labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticExample:
    vertices: np.ndarray
    values: np.ndarray
    labels: np.ndarray
    jump_locations: np.ndarray


def _validate_domain(domain: tuple[float, float]) -> tuple[float, float]:
    left, right = map(float, domain)
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        raise ValueError("domain must be a finite pair (left, right) with left < right")
    return left, right


def generate_piecewise_fourier(
    *,
    n_cells: int = 100,
    max_jumps: int = 5,
    n_modes: int = 15,
    domain: tuple[float, float] = (-1.0, 1.0),
    rng: np.random.Generator | None = None,
) -> SyntheticExample:
    """Generate one piecewise Fourier function on a uniform one-dimensional mesh.

    Jump cells are sampled without replacement. The jump is placed strictly inside
    its cell, removing the vertex-label ambiguity present in the original generator.
    """

    if n_cells < 2:
        raise ValueError("n_cells must be at least 2")
    if max_jumps < 0 or max_jumps > n_cells:
        raise ValueError("max_jumps must lie between 0 and n_cells")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")

    generator = np.random.default_rng() if rng is None else rng
    left, right = _validate_domain(domain)
    vertices = np.linspace(left, right, n_cells + 1, dtype=np.float64)

    n_jumps = int(generator.integers(0, max_jumps + 1))
    jump_cells = np.sort(
        generator.choice(n_cells, size=n_jumps, replace=False).astype(int)
    )
    widths = np.diff(vertices)
    offsets = generator.uniform(0.2, 0.8, size=n_jumps)
    jump_locations = vertices[jump_cells] + offsets * widths[jump_cells]

    # One independent Fourier expansion for each smooth segment. Including a
    # constant coefficient makes jumps in the generated values overwhelmingly likely.
    coefficients = generator.normal(size=(n_jumps + 1, n_modes + 1, 2))
    segment = np.searchsorted(jump_locations, vertices, side="right")
    scaled_x = 2.0 * np.pi * (vertices - left) / (right - left)
    values = np.empty_like(vertices)
    modes = np.arange(1, n_modes + 1, dtype=np.float64)

    for index, (x_value, segment_index) in enumerate(zip(scaled_x, segment)):
        coeff = coefficients[segment_index]
        values[index] = coeff[0, 0] + np.sum(
            coeff[1:, 0] * np.cos(modes * x_value)
            + coeff[1:, 1] * np.sin(modes * x_value)
        )

    labels = np.zeros(n_cells, dtype=np.uint8)
    labels[jump_cells] = 1
    return SyntheticExample(vertices, values, labels, jump_locations)

