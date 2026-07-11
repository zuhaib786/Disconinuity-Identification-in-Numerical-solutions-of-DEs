"""Exact cut-curve training data on variable triangular meshes."""

from dataclasses import dataclass, field
import time

import numpy as np

from tci.mesh import (
    TriangleMesh,
    perturbed_delaunay_mesh,
    random_delaunay_mesh,
    rectangular_mesh,
)


@dataclass
class Sample2D:
    mesh: TriangleMesh
    u: np.ndarray
    labels: np.ndarray
    curve: str
    parameters: dict = field(default_factory=dict)
    source: str = "exact"
    time: float = 0.0
    trajectory_id: int = -1


def cells_cut_by_line(mesh, point, normal):
    """Cells intersected by the infinite line ``normal . (x-point) = 0``."""
    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if point.shape != (2,) or normal.shape != (2,) or np.linalg.norm(normal) == 0:
        raise ValueError("point and nonzero normal must have shape (2,)")
    values = (mesh.points[mesh.cells] - point) @ normal
    scale = max(float(np.max(np.abs(values))), 1.0)
    tolerance = 64.0 * np.finfo(float).eps * scale
    return (np.min(values, axis=1) <= tolerance) & (
        np.max(values, axis=1) >= -tolerance
    )


def cells_cut_by_circle(mesh, center, radius):
    """Cells intersected by a circle, including circles contained in a cell."""
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center must be a finite vector of shape (2,)")
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be positive and finite")

    vertices = mesh.points[mesh.cells]
    edges = np.roll(vertices, -1, axis=1) - vertices
    to_center = center - vertices
    cross = edges[:, :, 0] * to_center[:, :, 1] - edges[:, :, 1] * to_center[:, :, 0]
    tolerance = 64.0 * np.finfo(float).eps * max(float(radius), 1.0)
    inside = np.all(cross >= -tolerance, axis=1)

    edge_length_sq = np.sum(edges * edges, axis=2)
    projection = np.clip(
        np.sum(to_center * edges, axis=2) / edge_length_sq, 0.0, 1.0
    )
    closest = vertices + projection[:, :, None] * edges
    edge_distance = np.linalg.norm(closest - center, axis=2)
    min_distance = np.where(inside, 0.0, np.min(edge_distance, axis=1))
    max_distance = np.max(np.linalg.norm(vertices - center, axis=2), axis=1)
    return (min_distance <= radius + tolerance) & (
        max_distance >= radius - tolerance
    )


def _polynomial_values(x, y, coefficients, domain):
    (xmin, xmax), (ymin, ymax) = domain
    xn = 2.0 * (x - 0.5 * (xmin + xmax)) / (xmax - xmin)
    yn = 2.0 * (y - 0.5 * (ymin + ymax)) / (ymax - ymin)
    basis = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn])
    return np.tensordot(coefficients, basis, axes=(0, 0))


def generate_exact_2d_samples(
    n_samples,
    domain=((0.0, 1.0), (0.0, 1.0)),
    mesh_type="delaunay",
    n_interior_range=(50, 150),
    boundary_divisions=(8, 8),
    structured_range=(6, 12),
    curves=("line", "circle"),
    coefficient_sigma=1.0,
    seed=0,
):
    """Generate piecewise-quadratic fields separated by random cut curves."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if mesh_type not in ("delaunay", "structured"):
        raise ValueError("mesh_type must be 'delaunay' or 'structured'")
    if not curves or any(curve not in ("line", "circle") for curve in curves):
        raise ValueError("curves must contain 'line' and/or 'circle'")
    (xmin, xmax), (ymin, ymax) = domain
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("domain limits must be strictly increasing")

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        if mesh_type == "delaunay":
            n_interior = int(rng.integers(n_interior_range[0], n_interior_range[1] + 1))
            mesh = random_delaunay_mesh(
                (xmin, xmax),
                (ymin, ymax),
                n_interior=n_interior,
                boundary_divisions=boundary_divisions,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
        else:
            nx = int(rng.integers(structured_range[0], structured_range[1] + 1))
            ny = int(rng.integers(structured_range[0], structured_range[1] + 1))
            mesh = rectangular_mesh((xmin, xmax), (ymin, ymax), nx, ny)

        curve = str(rng.choice(curves))
        nodes = mesh.points[mesh.cells].transpose(1, 0, 2)
        if curve == "line":
            angle = float(rng.uniform(0.0, np.pi))
            normal = np.array([np.cos(angle), np.sin(angle)])
            point = rng.uniform((xmin, ymin), (xmax, ymax))
            phase = (nodes - point) @ normal
            labels = cells_cut_by_line(mesh, point, normal)
            parameters = {"point": point.tolist(), "normal": normal.tolist()}
        else:
            width = min(xmax - xmin, ymax - ymin)
            center = rng.uniform(
                (xmin + 0.2 * width, ymin + 0.2 * width),
                (xmax - 0.2 * width, ymax - 0.2 * width),
            )
            radius = float(rng.uniform(0.1 * width, 0.3 * width))
            phase = np.sum((nodes - center) ** 2, axis=2) - radius**2
            labels = cells_cut_by_circle(mesh, center, radius)
            parameters = {"center": center.tolist(), "radius": radius}

        left_coefficients = rng.normal(0.0, coefficient_sigma, size=6)
        right_coefficients = rng.normal(0.0, coefficient_sigma, size=6)
        left = _polynomial_values(
            nodes[:, :, 0], nodes[:, :, 1], left_coefficients, domain
        )
        right = _polynomial_values(
            nodes[:, :, 0], nodes[:, :, 1], right_coefficients, domain
        )
        u = np.where(phase < 0.0, left, right)
        samples.append(
            Sample2D(mesh=mesh, u=u, labels=labels, curve=curve, parameters=parameters)
        )
    return samples


def _rotate(values, center, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return center + (np.asarray(values) - center) @ rotation.T


def generate_numerical_2d_samples(
    n_samples,
    mesh_range=(8, 12),
    curves=("line", "circle"),
    time_range=(0.02, 0.2),
    limited_fraction=0.0,
    cfl=0.15,
    max_steps=1500,
    max_seconds_per_sample=5.0,
    max_generation_seconds=600.0,
    seed=0,
):
    """Evolved rotational-advection states with exact rotated-curve labels."""
    from tci.evaluate2d import rotational_velocity
    from tci.indicators.classical2d import MinmodIndicator2D
    from tci.solvers.dg2d import AdvectionDG2D

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not 0.0 <= limited_fraction <= 1.0:
        raise ValueError("limited_fraction must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + max_generation_seconds
    center = np.array([0.5, 0.5])
    samples = []
    attempts = 0
    while len(samples) < n_samples:
        attempts += 1
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"2D numerical generation produced {len(samples)}/{n_samples} samples "
                f"within {max_generation_seconds:.1f}s"
            )
        if attempts > 10 * n_samples:
            raise RuntimeError("too many rejected 2D numerical trajectories")
        n = int(rng.integers(mesh_range[0], mesh_range[1] + 1))
        if rng.random() < 0.75:
            mesh = perturbed_delaunay_mesh(
                nx=n,
                ny=n,
                jitter=float(rng.uniform(0.05, 0.2)),
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
        else:
            mesh = rectangular_mesh(nx=n, ny=n)
        solver = AdvectionDG2D(
            mesh,
            velocity=rotational_velocity,
            boundary_func=None,
        )
        final_time = float(rng.uniform(*time_range))
        estimated_steps = int(np.ceil(final_time / solver.stable_dt(cfl)))
        if estimated_steps > max_steps:
            continue

        curve = str(rng.choice(curves))
        if curve == "line":
            angle = float(rng.uniform(0.0, np.pi))
            normal0 = np.array([np.cos(angle), np.sin(angle)])
            point0 = rng.uniform((0.25, 0.25), (0.75, 0.75))
            curve_parameters = {"point": point0, "normal": normal0}
        else:
            radius = float(rng.uniform(0.08, 0.18))
            orbit = float(rng.uniform(0.0, 0.25 - radius))
            angle = float(rng.uniform(0.0, 2 * np.pi))
            circle0 = center + orbit * np.array([np.cos(angle), np.sin(angle)])
            curve_parameters = {"center": circle0, "radius": radius}

        left_coefficients = rng.normal(size=6)
        right_coefficients = rng.normal(size=6)

        def initial(x, y):
            xy = np.stack([x, y], axis=-1)
            if curve == "line":
                phase = (xy - curve_parameters["point"]) @ curve_parameters["normal"]
            else:
                phase = (
                    np.sum((xy - curve_parameters["center"]) ** 2, axis=-1)
                    - curve_parameters["radius"] ** 2
                )
            left = _polynomial_values(x, y, left_coefficients, ((0.0, 1.0), (0.0, 1.0)))
            right = _polynomial_values(x, y, right_coefficients, ((0.0, 1.0), (0.0, 1.0)))
            return np.where(phase < 0.0, left, right)

        def boundary(x, y, current_time):
            xy0 = _rotate(np.stack([x, y], axis=-1), center, -2 * np.pi * current_time)
            return initial(xy0[..., 0], xy0[..., 1])

        solver.boundary_func = boundary
        use_limiter = rng.random() < limited_fraction
        indicator = MinmodIndicator2D() if use_limiter else None
        try:
            u = solver.solve(
                initial,
                final_time,
                cfl=cfl,
                indicator=indicator,
                max_seconds=max_seconds_per_sample,
            )
        except TimeoutError:
            continue
        assert isinstance(u, np.ndarray)

        rotation_angle = 2 * np.pi * final_time
        if curve == "line":
            point = _rotate(curve_parameters["point"], center, rotation_angle)
            normal = _rotate(center + curve_parameters["normal"], center, rotation_angle) - center
            labels = cells_cut_by_line(mesh, point, normal)
            parameters = {"point": point.tolist(), "normal": normal.tolist()}
        else:
            circle_center = _rotate(curve_parameters["center"], center, rotation_angle)
            labels = cells_cut_by_circle(mesh, circle_center, curve_parameters["radius"])
            parameters = {
                "center": circle_center.tolist(),
                "radius": curve_parameters["radius"],
            }
        parameters.update(
            {
                "estimated_steps": estimated_steps,
                "left_coefficients": left_coefficients.tolist(),
                "right_coefficients": right_coefficients.tolist(),
            }
        )
        samples.append(
            Sample2D(
                mesh=mesh,
                u=u,
                labels=labels,
                curve=curve,
                parameters=parameters,
                source="limited" if use_limiter else "unlimited",
                time=final_time,
                trajectory_id=len(samples),
            )
        )
    return samples


def generate_mixed_2d_samples(n_samples, exact_fraction=0.25, seed=0, **kwargs):
    """Deterministic mixture of static exact and evolved numerical samples."""
    n_exact = int(round(n_samples * exact_fraction))
    exact = generate_exact_2d_samples(n_exact, seed=seed)
    numerical = generate_numerical_2d_samples(
        n_samples - n_exact, seed=seed + 1, **kwargs
    )
    samples = exact + numerical
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    return samples
