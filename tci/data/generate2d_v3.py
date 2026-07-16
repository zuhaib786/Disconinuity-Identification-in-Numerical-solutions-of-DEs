"""`data-v3`: distribution-matched 2D training data (plan phase 6).

The generator addresses the three drift sources of `tci/data/generate2d.py`
relative to solver-in-the-loop deployment:

* geometry drift -- an expanded signed-distance curve family with corners,
  endpoints, near-tangencies, and several interacting fronts per sample;
* value drift -- a steep-layer continuum and evolved DG states, including
  post-limiter states;
* label drift -- every label comes from `tci.data.labels2d`, i.e. from the
  reference field, so it is resolution- and amplitude-aware.

The composition is frozen in the config.  ``ladder`` selects the controlled
one-factor-at-a-time steps of plan 6.3:

``v3-a``
    the new label rule on the *old* line/circle exact geometry and meshes;
``v3-b``
    ``v3-a`` plus components 1--3 (expanded curves, amplitude continuum, steep
    layers, smooth negatives) and the widened mesh diversity;
``v3-c``
    ``v3-b`` plus component 4 (evolved solver states) -- the full schema.
"""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tci.data.curves2d import CURVE_TYPES, LEGACY_CURVE_TYPES, rotate_curve, sample_curve
from tci.data.generate2d import Sample2D, _polynomial_values
from tci.data.labels2d import ALPHA, GAMMA, QUADRATURE_ORDER, duffy_quadrature, label_cells, uniform_refine
from tci.mesh import delaunay_mesh, perturbed_delaunay_mesh, random_delaunay_mesh, rectangular_mesh

SCHEMA_VERSION = "data-v3"
COMPONENTS = ("exact_curves", "steep_layer", "smooth", "evolved")
MESH_FAMILIES = ("structured", "delaunay_random", "delaunay_perturbed", "delaunay_graded")
ROTATION_CENTER = np.array([0.5, 0.5])

DEFAULT_SPEC = {
    "schema_version": SCHEMA_VERSION,
    "ladder": "v3-c",
    "field_style": "additive_jump",
    "n_samples": 2000,
    "seed": 0,
    "domain": [[0.0, 1.0], [0.0, 1.0]],
    "label": {"alpha": ALPHA, "gamma": GAMMA, "quadrature_order": QUADRATURE_ORDER},
    "mixture": {
        "exact_curves": 0.30,
        "steep_layer": 0.25,
        "smooth": 0.15,
        "evolved": 0.30,
    },
    "curves": list(CURVE_TYPES),
    "curves_per_sample": [1, 3],
    "amplitude_range": [1e-3, 3.0],
    "layer_width_range": [0.0625, 4.0],
    "coefficient_sigma": 1.0,
    "meshes": {
        "structured": 0.3,
        "delaunay_random": 0.3,
        "delaunay_perturbed": 0.25,
        "delaunay_graded": 0.15,
    },
    "n_interior_range": [30, 600],
    "structured_range": [6, 24],
    "boundary_divisions": [8, 8],
    "jitter_range": [0.05, 0.2],
    "grading_strength": 2.0,
    "evolved": {
        "mesh_range": [6, 16],
        "structured_fraction": 0.25,
        "time_range": [0.02, 0.2],
        "snapshot_range": [2, 4],
        "limited_fraction": 0.5,
        "steep_fraction": 0.45,
        "cfl": 0.15,
        "max_seconds_per_trajectory": 60.0,
    },
    "max_generation_seconds": 7200.0,
}

# The controlled ladder.  Each step adds exactly one experimental factor.
LADDER_OVERRIDES = {
    "v3-a": {
        "mixture": {"exact_curves": 1.0, "steep_layer": 0.0, "smooth": 0.0, "evolved": 0.0},
        "curves": list(LEGACY_CURVE_TYPES),
        "curves_per_sample": [1, 1],
        "field_style": "legacy_piecewise_quadratic",
        "meshes": {
            "structured": 0.0,
            "delaunay_random": 1.0,
            "delaunay_perturbed": 0.0,
            "delaunay_graded": 0.0,
        },
        "n_interior_range": [50, 150],
    },
    "v3-b": {
        "mixture": {
            "exact_curves": 0.43,
            "steep_layer": 0.36,
            "smooth": 0.21,
            "evolved": 0.0,
        },
        "field_style": "additive_jump",
    },
    "v3-c": {"field_style": "additive_jump"},
}


def resolve_spec(spec):
    """Merge a user spec onto the defaults and the ladder step, then validate."""
    spec = dict(spec or {})
    ladder = str(spec.get("ladder", DEFAULT_SPEC["ladder"]))
    if ladder not in LADDER_OVERRIDES:
        raise ValueError(f"unknown ladder step {ladder!r}; choose from {sorted(LADDER_OVERRIDES)}")

    resolved = copy.deepcopy(DEFAULT_SPEC)
    for override in (LADDER_OVERRIDES[ladder], spec):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(resolved.get(key), dict):
                resolved[key] = {**resolved[key], **value}
            else:
                resolved[key] = value
    resolved["ladder"] = ladder

    if int(resolved["n_samples"]) < 1:
        raise ValueError("n_samples must be positive")
    for name, fractions in (("mixture", resolved["mixture"]), ("meshes", resolved["meshes"])):
        unknown = set(fractions) - set(COMPONENTS if name == "mixture" else MESH_FAMILIES)
        if unknown:
            raise ValueError(f"unknown {name} keys {sorted(unknown)}")
        if any(value < 0.0 for value in fractions.values()):
            raise ValueError(f"{name} fractions must be nonnegative")
        if abs(sum(fractions.values()) - 1.0) > 1e-9:
            raise ValueError(f"{name} fractions must sum to 1, got {sum(fractions.values())}")
    unknown_curves = set(resolved["curves"]) - set(CURVE_TYPES)
    if unknown_curves:
        raise ValueError(f"unknown curve kinds {sorted(unknown_curves)}")
    if resolved["field_style"] not in ("legacy_piecewise_quadratic", "additive_jump"):
        raise ValueError(f"unknown field_style {resolved['field_style']!r}")
    return resolved


def spec_id(spec):
    """Content hash of a resolved spec; the on-disk dataset cache key."""
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _counts(n_samples, fractions, keys):
    """Deterministic integer counts that sum exactly to ``n_samples``."""
    exact = np.array([n_samples * fractions.get(key, 0.0) for key in keys])
    counts = np.floor(exact).astype(int)
    remainder = n_samples - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact - counts))
        counts[order[:remainder]] += 1
    return dict(zip(keys, counts.tolist()))


def _graded_points(rng, domain, n_interior, strength):
    """Interior points whose density concentrates toward a random attractor."""
    (xmin, xmax), (ymin, ymax) = domain
    margin = 0.02 * min(xmax - xmin, ymax - ymin)
    attractor = rng.uniform((xmin, ymin), (xmax, ymax))
    candidates = rng.uniform(
        (xmin + margin, ymin + margin), (xmax - margin, ymax - margin), size=(8 * n_interior, 2)
    )
    distance = np.linalg.norm(candidates - attractor, axis=1)
    scale = 0.1 * min(xmax - xmin, ymax - ymin)
    weights = 1.0 / (scale + distance) ** strength
    chosen = rng.choice(len(candidates), size=n_interior, replace=False, p=weights / weights.sum())
    return candidates[chosen], attractor


def sample_mesh(rng, spec, evolved=False):
    """Draw one mesh and its provenance record."""
    (xlim, ylim) = spec["domain"]
    if evolved:
        low, high = spec["evolved"]["mesh_range"]
        n = int(rng.integers(low, high + 1))
        if rng.random() < spec["evolved"]["structured_fraction"]:
            return rectangular_mesh(xlim, ylim, nx=n, ny=n), {"family": "structured", "n": n}
        jitter = float(rng.uniform(*spec["jitter_range"]))
        mesh = perturbed_delaunay_mesh(
            xlim, ylim, nx=n, ny=n, jitter=jitter, seed=int(rng.integers(0, 2**31 - 1))
        )
        return mesh, {"family": "delaunay_perturbed", "n": n, "jitter": jitter}

    families = [name for name in MESH_FAMILIES if spec["meshes"].get(name, 0.0) > 0.0]
    weights = np.array([spec["meshes"][name] for name in families])
    family = str(rng.choice(families, p=weights / weights.sum()))
    boundary = tuple(spec["boundary_divisions"])
    if family == "structured":
        low, high = spec["structured_range"]
        n = int(rng.integers(low, high + 1))
        return rectangular_mesh(xlim, ylim, nx=n, ny=n), {"family": family, "n": n}

    low, high = spec["n_interior_range"]
    n_interior = int(rng.integers(low, high + 1))
    if family == "delaunay_perturbed":
        n = max(2, int(round(np.sqrt(n_interior))))
        jitter = float(rng.uniform(*spec["jitter_range"]))
        mesh = perturbed_delaunay_mesh(
            xlim, ylim, nx=n, ny=n, jitter=jitter, seed=int(rng.integers(0, 2**31 - 1))
        )
        return mesh, {"family": family, "n": n, "jitter": jitter}
    if family == "delaunay_random":
        mesh = random_delaunay_mesh(
            xlim,
            ylim,
            n_interior=n_interior,
            boundary_divisions=boundary,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        return mesh, {"family": family, "n_interior": n_interior}

    interior, attractor = _graded_points(rng, spec["domain"], n_interior, spec["grading_strength"])
    x = np.linspace(xlim[0], xlim[1], boundary[0] + 1)
    y = np.linspace(ylim[0], ylim[1], boundary[1] + 1)
    edge = np.concatenate(
        [
            np.column_stack([x, np.full_like(x, ylim[0])]),
            np.column_stack([x, np.full_like(x, ylim[1])]),
            np.column_stack([np.full(len(y) - 2, xlim[0]), y[1:-1]]),
            np.column_stack([np.full(len(y) - 2, xlim[1]), y[1:-1]]),
        ]
    )
    mesh = delaunay_mesh(np.concatenate([edge, interior]))
    return mesh, {
        "family": family,
        "n_interior": n_interior,
        "attractor": attractor.tolist(),
        "grading_strength": spec["grading_strength"],
    }


def _quadratic(rng, spec):
    coefficients = rng.normal(0.0, spec["coefficient_sigma"], size=6)
    domain = tuple(tuple(limits) for limits in spec["domain"])

    def field(x, y):
        return _polynomial_values(x, y, coefficients, domain)

    return field, coefficients


def smooth_extremum_field(rng, spec, h):
    """Component 3: a smooth field with genuine extrema and no interface."""
    quadratic, coefficients = _quadratic(rng, spec)
    (xmin, xmax), (ymin, ymax) = spec["domain"]
    width = min(xmax - xmin, ymax - ymin)
    count = int(rng.integers(1, 4))
    centers = rng.uniform((xmin, ymin), (xmax, ymax), size=(count, 2))
    # Aim for resolved bumps -- an unresolvable spike is a discontinuity in
    # disguise -- but never invert the range on a mesh so coarse that 2h already
    # exceeds the upper bound.  The label rule, not the component, has the final
    # say on such borderline cases.
    widest = 0.25 * width
    narrowest = min(max(2.0 * h, 0.05 * width), 0.8 * widest)
    sigmas = rng.uniform(narrowest, widest, size=count)
    heights = rng.normal(0.0, 1.0, size=count)
    wave = rng.integers(1, 3, size=2)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wave_amplitude = float(rng.uniform(0.0, 0.5))

    def field(x, y):
        values = quadratic(x, y)
        for center, sigma, height in zip(centers, sigmas, heights):
            squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
            values = values + height * np.exp(-0.5 * squared / sigma**2)
        return values + wave_amplitude * np.sin(
            2.0 * np.pi * wave[0] * x + phase
        ) * np.cos(2.0 * np.pi * wave[1] * y)

    parameters = {
        "coefficients": coefficients.tolist(),
        "bump_centers": centers.tolist(),
        "bump_sigmas": sigmas.tolist(),
        "bump_heights": heights.tolist(),
        "wave": wave.tolist(),
        "wave_amplitude": wave_amplitude,
    }
    return field, parameters


def curve_field(rng, spec, mesh, steep):
    """Components 1, 2, and the evolved initial conditions."""
    (xmin, xmax), (ymin, ymax) = spec["domain"]
    domain = ((xmin, xmax), (ymin, ymax))
    low, high = spec["curves_per_sample"]
    count = int(rng.integers(low, high + 1))
    curves = [sample_curve(rng, str(rng.choice(spec["curves"])), domain) for _ in range(count)]
    h = float(np.sqrt(np.mean(mesh.areas)))

    if spec["field_style"] == "legacy_piecewise_quadratic":
        if len(curves) != 1:
            raise ValueError("the legacy field style uses exactly one curve")
        left, left_coefficients = _quadratic(rng, spec)
        right, right_coefficients = _quadratic(rng, spec)
        curve = curves[0]

        def piecewise(x, y):
            return np.where(curve.distance(x, y) < 0.0, left(x, y), right(x, y))

        parameters = {
            "curves": [curve.as_json()],
            "left_coefficients": left_coefficients.tolist(),
            "right_coefficients": right_coefficients.tolist(),
            "field_style": "legacy_piecewise_quadratic",
        }
        return piecewise, curves, parameters

    background, coefficients = _quadratic(rng, spec)
    nodes = mesh.points[mesh.cells]
    variation = float(np.ptp(background(nodes[:, :, 0], nodes[:, :, 1])))
    variation = max(variation, 1e-6)

    log_low, log_high = np.log(spec["amplitude_range"])
    amplitudes = np.exp(rng.uniform(log_low, log_high, size=count)) * variation
    amplitudes *= rng.choice([-1.0, 1.0], size=count)
    if steep:
        width_low, width_high = np.log(spec["layer_width_range"])
        widths = h * np.exp(rng.uniform(width_low, width_high, size=count))
    else:
        widths = np.zeros(count)

    def field(x, y):
        values = background(x, y)
        for curve, amplitude, width in zip(curves, amplitudes, widths):
            distance = curve.distance(x, y)
            if width > 0.0:
                profile = 0.5 * (1.0 + np.tanh(-distance / width))
            else:
                profile = (distance < 0.0).astype(float)
            values = values + amplitude * profile
        return values

    parameters = {
        "curves": [curve.as_json() for curve in curves],
        "coefficients": coefficients.tolist(),
        "amplitudes": amplitudes.tolist(),
        "layer_widths": widths.tolist(),
        "smooth_variation": variation,
        "field_style": "additive_jump",
    }
    return field, curves, parameters


def geometric_cut_labels(mesh, distances, order=QUADRATURE_ORDER):
    """Scale-free auxiliary diagnostic: does any curve cross the cell?

    This reproduces the intent of the legacy ``cells_cut_by_line/circle``
    labels for the whole curve family, so the old and new label definitions
    stay comparable offline.  It is never a training target.
    """
    barycentric, _ = duffy_quadrature(order)
    vertices = mesh.points[mesh.cells]
    corners = np.eye(3)
    cut = np.zeros(mesh.K, dtype=bool)
    for distance in distances:
        signs = []
        for point in np.concatenate([corners, barycentric]):
            xy = np.einsum("i,kid->kd", point, vertices)
            signs.append(np.asarray(distance(xy[:, 0], xy[:, 1])) < 0.0)
        stacked = np.stack(signs)
        cut |= np.any(stacked, axis=0) & np.any(~stacked, axis=0)
    return cut


def _rotated_reference(field, angle):
    """The exact rotational-advection solution: pull back to the initial time."""
    cosine, sine = np.cos(-angle), np.sin(-angle)

    def reference(x, y):
        dx = x - ROTATION_CENTER[0]
        dy = y - ROTATION_CENTER[1]
        return field(
            ROTATION_CENTER[0] + cosine * dx - sine * dy,
            ROTATION_CENTER[1] + sine * dx + cosine * dy,
        )

    return reference


def _label(mesh, field, spec, refinement):
    label_spec = spec["label"]
    labels, _ = label_cells(
        mesh,
        field,
        alpha=label_spec["alpha"],
        gamma=label_spec["gamma"],
        order=label_spec["quadrature_order"],
        refinement=refinement,
    )
    return labels


def _static_sample(rng, spec, component):
    mesh, mesh_record = sample_mesh(rng, spec)
    h = float(np.sqrt(np.mean(mesh.areas)))
    if component == "smooth":
        field, parameters = smooth_extremum_field(rng, spec, h)
        curves = []
    else:
        field, curves, parameters = curve_field(rng, spec, mesh, steep=component == "steep_layer")
    nodes = mesh.points[mesh.cells].transpose(1, 0, 2)
    u = np.asarray(field(nodes[:, :, 0], nodes[:, :, 1]), dtype=float)
    labels = _label(mesh, field, spec, None)
    aux = geometric_cut_labels(mesh, [curve.distance for curve in curves]) if curves else np.zeros(mesh.K, bool)
    parameters["mesh"] = mesh_record
    return Sample2D(
        mesh=mesh,
        u=u,
        labels=labels,
        curve=component,
        parameters=parameters,
        source=component,
        aux_labels=aux,
    )


def smooth_negative_samples(count, seed, spec=None):
    """A held-out batch of component-3 fields: smooth, with extrema, no interface.

    Every cell is a true negative, so any flag a detector raises here is a
    smooth-extremum false positive.
    """
    resolved = resolve_spec({"ladder": "v3-b", **(spec or {})})
    rng = np.random.default_rng(int(seed))
    return [_static_sample(rng, resolved, "smooth") for _ in range(int(count))]


def _evolved_trajectory(rng, spec, trajectory_id, deadline):
    """Component 4: evolved rotational-advection states with reference labels."""
    from tci.evaluate2d import rotational_velocity
    from tci.indicators.classical2d import MinmodIndicator2D
    from tci.solvers.dg2d import AdvectionDG2D

    evolved = spec["evolved"]
    mesh, mesh_record = sample_mesh(rng, spec, evolved=True)
    steep = rng.random() < evolved["steep_fraction"]
    field, curves, parameters = curve_field(rng, spec, mesh, steep=steep)
    limited = rng.random() < evolved["limited_fraction"]

    count = int(rng.integers(evolved["snapshot_range"][0], evolved["snapshot_range"][1] + 1))
    times = np.sort(rng.uniform(evolved["time_range"][0], evolved["time_range"][1], size=count))

    def boundary(x, y, current_time):
        return _rotated_reference(field, 2.0 * np.pi * current_time)(x, y)

    solver = AdvectionDG2D(mesh, velocity=rotational_velocity, boundary_func=boundary)
    indicator = MinmodIndicator2D() if limited else None
    budget = min(
        evolved["max_seconds_per_trajectory"], max(1.0, deadline - time.monotonic())
    )
    started = time.monotonic()

    refinement = uniform_refine(mesh)
    samples = []
    u = solver.project(field)
    previous = 0.0
    for snapshot in times:
        u = solver.solve(
            u,
            float(snapshot - previous),
            cfl=evolved["cfl"],
            indicator=indicator,
            max_seconds=max(0.5, budget - (time.monotonic() - started)),
        )
        assert isinstance(u, np.ndarray)
        previous = float(snapshot)
        angle = 2.0 * np.pi * previous
        reference = _rotated_reference(field, angle)
        labels = _label(mesh, reference, spec, refinement)
        aux = geometric_cut_labels(
            mesh, [rotate_curve(curve, ROTATION_CENTER, angle).distance for curve in curves]
        )
        snapshot_parameters = dict(parameters)
        snapshot_parameters["mesh"] = mesh_record
        snapshot_parameters["limited"] = limited
        snapshot_parameters["steep"] = steep
        samples.append(
            Sample2D(
                mesh=mesh,
                u=u.copy(),
                labels=labels,
                curve="evolved",
                parameters=snapshot_parameters,
                source="limited" if limited else "unlimited",
                time=previous,
                trajectory_id=trajectory_id,
                aux_labels=aux,
            )
        )
    return samples


def generate_data_v3(spec, progress=None):
    """Generate the frozen `data-v3` mixture for one resolved spec."""
    spec = resolve_spec(spec)
    rng = np.random.default_rng(int(spec["seed"]))
    n_samples = int(spec["n_samples"])
    counts = _counts(n_samples, spec["mixture"], COMPONENTS)
    deadline = time.monotonic() + float(spec["max_generation_seconds"])

    samples = []
    for component in ("exact_curves", "steep_layer", "smooth"):
        for _ in range(counts[component]):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"data-v3 generation exceeded {spec['max_generation_seconds']}s")
            samples.append(_static_sample(rng, spec, component))
            if progress and len(samples) % 100 == 0:
                progress(f"generated {len(samples)}/{n_samples} samples")

    trajectory_id = 0
    attempts = 0
    while len(samples) < n_samples:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"data-v3 generation produced {len(samples)}/{n_samples} samples "
                f"within {spec['max_generation_seconds']}s"
            )
        attempts += 1
        if attempts > 10 * max(1, counts["evolved"]):
            raise RuntimeError("too many rejected data-v3 trajectories")
        try:
            trajectory = _evolved_trajectory(rng, spec, trajectory_id, deadline)
        except TimeoutError:
            continue
        # Keep whole trajectories: the split groups snapshots by trajectory_id.
        if len(samples) + len(trajectory) > n_samples:
            trajectory = trajectory[: n_samples - len(samples)]
        samples.extend(trajectory)
        trajectory_id += 1
        if progress:
            progress(f"generated {len(samples)}/{n_samples} samples")

    order = rng.permutation(len(samples))
    return [samples[int(index)] for index in order]


def _pack(samples):
    points, cells, u, labels, aux = [], [], [], [], []
    metadata = []
    for sample in samples:
        points.append(sample.mesh.points)
        cells.append(sample.mesh.cells)
        u.append(np.asarray(sample.u, dtype=float).T)
        labels.append(np.asarray(sample.labels, dtype=bool))
        aux.append(np.asarray(sample.aux_labels, dtype=bool))
        metadata.append(
            {
                "curve": sample.curve,
                "parameters": sample.parameters,
                "source": sample.source,
                "time": sample.time,
                "trajectory_id": sample.trajectory_id,
            }
        )
    return {
        "points": np.concatenate(points),
        "point_counts": np.array([len(item) for item in points], dtype=np.int64),
        "cells": np.concatenate(cells),
        "cell_counts": np.array([len(item) for item in cells], dtype=np.int64),
        "u": np.concatenate(u),
        "labels": np.concatenate(labels),
        "aux_labels": np.concatenate(aux),
        "metadata": np.array(json.dumps(metadata)),
    }


def _unpack(archive):
    from tci.mesh import TriangleMesh

    # Indexing an NpzFile decompresses the whole array, so read each one once.
    stored = {
        key: archive[key]
        for key in ("points", "point_counts", "cells", "cell_counts", "u", "labels", "aux_labels")
    }
    metadata = json.loads(str(archive["metadata"]))
    point_offsets = np.concatenate([[0], np.cumsum(stored["point_counts"])])
    cell_offsets = np.concatenate([[0], np.cumsum(stored["cell_counts"])])
    samples = []
    for index, record in enumerate(metadata):
        points = stored["points"][point_offsets[index] : point_offsets[index + 1]]
        start, stop = cell_offsets[index], cell_offsets[index + 1]
        samples.append(
            Sample2D(
                mesh=TriangleMesh(points, stored["cells"][start:stop]),
                u=stored["u"][start:stop].T,
                labels=stored["labels"][start:stop],
                curve=record["curve"],
                parameters=record["parameters"],
                source=record["source"],
                time=record["time"],
                trajectory_id=record["trajectory_id"],
                aux_labels=stored["aux_labels"][start:stop],
            )
        )
    return samples


def load_or_generate(spec, cache_dir="runs/data-v3", progress=print):
    """Generate the dataset once per spec and cache it on disk.

    The five training seeds of a ladder step share one dataset, so generation
    -- which advects hundreds of DG trajectories -- must not repeat per run.
    """
    spec = resolve_spec(spec)
    identifier = spec_id(spec)
    cache = Path(cache_dir) / f"{spec['ladder']}-{identifier[:16]}.npz"
    if cache.exists():
        if progress:
            progress(f"data-v3: loading cached dataset {cache}")
        with np.load(cache, allow_pickle=False) as archive:
            samples = _unpack(archive)
        if len(samples) != int(spec["n_samples"]):
            raise ValueError(f"cached dataset {cache} has {len(samples)} samples")
        return samples

    if progress:
        progress(f"data-v3: generating {spec['n_samples']} samples for {spec['ladder']}")
    started = time.monotonic()
    samples = generate_data_v3(spec, progress=progress)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **_pack(samples))
    temporary.replace(cache)
    (cache.with_suffix(".json")).write_text(json.dumps({"spec": spec, "spec_id": identifier}, indent=2))
    if progress:
        progress(f"data-v3: generated and cached {cache} in {time.monotonic() - started:.1f}s")
    return samples
