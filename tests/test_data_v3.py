"""Phase 6 `data-v3`: label rule, curve family, generator, and split protocol."""

import numpy as np
import pytest

from tci.data.curves2d import CURVE_TYPES, Circle, Line, rotate_curve, sample_curve
from tci.data.generate2d_v3 import (
    generate_data_v3,
    load_or_generate,
    resolve_spec,
    spec_id,
)
from tci.data.graphs import TriangleFeatureBuilder, one_ring_robust_scale
from tci.data.labels2d import (
    face_mean_jumps,
    label_cells,
    project_p1,
    uniform_refine,
)
from tci.mesh import perturbed_delaunay_mesh, rectangular_mesh
from tci.phase6 import assess_data_v3, select_label_constants
from tci.train import dataset_id, deterministic_split, split_groups


def smooth(x, y):
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def step(x, y):
    return np.where(x + 0.6 * y < 0.7, 0.0, 1.0)


# --- P1 reference projection ------------------------------------------------


def test_projection_reproduces_linear_fields_exactly():
    mesh = perturbed_delaunay_mesh(nx=6, ny=6, jitter=0.2, seed=3)
    nodes = mesh.points[mesh.cells].transpose(1, 0, 2)
    projected = project_p1(mesh, lambda x, y: 2.0 * x - 3.0 * y + 1.0)
    expected = 2.0 * nodes[:, :, 0] - 3.0 * nodes[:, :, 1] + 1.0
    assert np.allclose(projected, expected, atol=1e-12)
    # A continuous P1 field has no face-mean jumps at all.
    assert np.max(face_mean_jumps(mesh, projected)) < 1e-12


def test_smooth_face_jumps_decay_at_second_order():
    """The O(h^2) decay of a smooth field is what the gamma criterion detects."""
    jumps = [
        np.mean(face_mean_jumps(mesh, project_p1(mesh, smooth)))
        for mesh in (rectangular_mesh(nx=16, ny=16), rectangular_mesh(nx=32, ny=32))
    ]
    assert 3.5 <= jumps[0] / jumps[1] <= 4.5


def test_uniform_refine_quadruples_cells_and_maps_children():
    mesh = rectangular_mesh(nx=3, ny=3)
    refined, children = uniform_refine(mesh)
    assert refined.K == 4 * mesh.K
    assert children.shape == (mesh.K, 4)
    assert sorted(children.ravel().tolist()) == list(range(refined.K))
    assert np.isclose(np.sum(refined.areas), np.sum(mesh.areas))
    # Every child lies inside its parent.
    for parent, kids in enumerate(children):
        assert np.allclose(
            np.sum(refined.areas[kids]), mesh.areas[parent], rtol=1e-12
        )


# --- the label rule ---------------------------------------------------------


def test_smooth_fields_carry_no_positive_labels():
    for mesh in (rectangular_mesh(nx=8, ny=8), perturbed_delaunay_mesh(nx=8, ny=8, jitter=0.2, seed=1)):
        labels, _ = label_cells(mesh, smooth)
        assert not np.any(labels)


def test_discontinuity_labels_are_localized_at_the_front():
    mesh = rectangular_mesh(nx=12, ny=12)
    labels, _ = label_cells(mesh, step)
    assert np.any(labels)
    assert np.mean(labels) < 0.25
    centroids = mesh.centroids[labels]
    distance = np.abs(centroids[:, 0] + 0.6 * centroids[:, 1] - 0.7) / np.hypot(1.0, 0.6)
    assert np.max(distance) <= np.sqrt(np.mean(mesh.areas))


def test_labels_are_amplitude_aware():
    mesh = rectangular_mesh(nx=12, ny=12)
    background = lambda x, y: 0.5 * x + 0.3 * y  # noqa: E731

    def vanishing(x, y):
        return background(x, y) + 1e-6 * (x + 0.6 * y >= 0.7)

    def strong(x, y):
        return background(x, y) + 1.0 * (x + 0.6 * y >= 0.7)

    assert not np.any(label_cells(mesh, vanishing)[0])
    assert np.any(label_cells(mesh, strong)[0])


def test_labels_are_resolution_aware():
    """One steep layer of fixed physical width: troubled only while unresolved."""
    width = 1.0 / 16.0
    layer = lambda x, y: 0.5 * (1.0 + np.tanh((x + 0.5 * y - 0.7) / width))  # noqa: E731
    coarse, _ = label_cells(rectangular_mesh(nx=8, ny=8), layer)
    fine, _ = label_cells(rectangular_mesh(nx=32, ny=32), layer)
    assert np.any(coarse)
    assert not np.any(fine)


def test_label_scale_matches_the_phase3_feature_scale():
    """The label scale s_K is the same one-ring robust scale the features use."""
    mesh = perturbed_delaunay_mesh(nx=6, ny=6, jitter=0.2, seed=5)
    u = project_p1(mesh, step)
    builder = TriangleFeatureBuilder(mesh, "invariant-local-v2")
    scales, centers = builder._scales(u, np.mean(u, axis=0))
    shared, shared_centers = one_ring_robust_scale(np.mean(u, axis=0), mesh.neighbors)
    assert np.allclose(scales, shared)
    assert np.allclose(centers, shared_centers)


# --- the curve family -------------------------------------------------------


@pytest.mark.parametrize("kind", CURVE_TYPES)
def test_signed_distance_is_normalized_near_the_interface(kind):
    curve = sample_curve(np.random.default_rng(11), kind)
    grid = np.linspace(0.005, 0.995, 200)
    x, y = np.meshgrid(grid, grid)
    distance = curve.distance(x, y)
    gradient = np.hypot(*np.gradient(distance, grid[1] - grid[0]))
    near = np.abs(distance) < 0.02
    assert np.any(distance < 0.0) and np.any(distance > 0.0)
    assert 0.9 <= float(np.median(gradient[near])) <= 1.1


def test_rotating_a_curve_rotates_its_zero_level_set():
    center = np.array([0.5, 0.5])
    angle = 0.7
    for curve in (Line((0.4, 0.3), (0.6, 0.8)), Circle((0.4, 0.6), 0.2)):
        rotated = rotate_curve(curve, center, angle)
        points = np.random.default_rng(2).uniform(0.0, 1.0, size=(50, 2))
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        turned = center + (points - center) @ rotation.T
        assert np.allclose(
            rotated.distance(turned[:, 0], turned[:, 1]),
            curve.distance(points[:, 0], points[:, 1]),
            atol=1e-12,
        )


# --- the generator ----------------------------------------------------------


def test_resolve_spec_rejects_inconsistent_compositions():
    with pytest.raises(ValueError, match="ladder"):
        resolve_spec({"ladder": "v4"})
    with pytest.raises(ValueError, match="sum to 1"):
        resolve_spec({"mixture": {"exact_curves": 0.5, "steep_layer": 0.1, "smooth": 0.1, "evolved": 0.1}})
    with pytest.raises(ValueError, match="curve kinds"):
        resolve_spec({"curves": ["spiral"]})


def test_ladder_steps_change_one_factor_at_a_time():
    a = resolve_spec({"ladder": "v3-a"})
    b = resolve_spec({"ladder": "v3-b"})
    c = resolve_spec({"ladder": "v3-c"})
    assert a["field_style"] == "legacy_piecewise_quadratic"
    assert a["curves"] == ["line", "circle"]
    assert a["mixture"]["evolved"] == 0.0 and b["mixture"]["evolved"] == 0.0
    assert c["mixture"]["evolved"] > 0.0
    # v3-C differs from v3-B only by adding the evolved component.
    assert {key: value for key, value in b.items() if key != "mixture"} == {
        key: value for key, value in c.items() if key not in ("mixture", "ladder")
    } | {"ladder": "v3-b"}


def test_generated_samples_carry_reference_labels_and_diagnostics():
    samples = generate_data_v3({"ladder": "v3-b", "n_samples": 6, "seed": 3})
    assert len(samples) == 6
    for sample in samples:
        assert sample.labels.shape == (sample.mesh.K,)
        assert sample.aux_labels.shape == (sample.mesh.K,)
        assert sample.labels.dtype == bool
    # Smooth-component samples are the pure negatives of the mixture.
    smooth_samples = [s for s in samples if s.curve == "smooth"]
    assert smooth_samples
    assert all(np.mean(s.labels) < 0.02 for s in smooth_samples)


def test_evolved_trajectories_are_grouped_and_partly_limited():
    spec = {
        "ladder": "v3-c",
        "n_samples": 12,
        "seed": 1,
        "mixture": {"exact_curves": 0.0, "steep_layer": 0.0, "smooth": 0.0, "evolved": 1.0},
        "evolved": {"mesh_range": [6, 7], "time_range": [0.02, 0.05]},
    }
    samples = generate_data_v3(spec)
    assert len(samples) == 12
    assert all(sample.trajectory_id >= 0 for sample in samples)
    assert len({sample.trajectory_id for sample in samples}) < 12
    assert all(sample.time > 0.0 for sample in samples)
    assert {sample.source for sample in samples} <= {"limited", "unlimited"}


def test_dataset_cache_round_trips_identically(tmp_path):
    spec = {"ladder": "v3-b", "n_samples": 5, "seed": 4}
    first = load_or_generate(spec, cache_dir=tmp_path, progress=None)
    second = load_or_generate(spec, cache_dir=tmp_path, progress=None)
    assert dataset_id(first) == dataset_id(second)
    for a, b in zip(first, second):
        assert np.array_equal(a.mesh.cells, b.mesh.cells)
        assert np.allclose(a.u, b.u)
        assert np.array_equal(a.labels, b.labels)
        assert np.array_equal(a.aux_labels, b.aux_labels)


def test_spec_id_tracks_the_label_constants():
    base = resolve_spec({"ladder": "v3-b"})
    tuned = resolve_spec({"ladder": "v3-b", "label": {"alpha": 0.3, "gamma": 2.0}})
    assert spec_id(base) != spec_id(tuned)


# --- the split protocol -----------------------------------------------------


class _Sample:
    def __init__(self, trajectory_id):
        self.trajectory_id = trajectory_id


def test_grouped_split_keeps_whole_trajectories_together():
    samples = [_Sample(index // 3) for index in range(30)]
    train, validation, _ = deterministic_split(30, 0.2, 0, "data", split_groups(samples))
    validation_set = set(validation.tolist())
    assert not validation_set & set(train.tolist())
    for trajectory in range(10):
        members = {index for index in range(30) if index // 3 == trajectory}
        assert not (members & validation_set) or members <= validation_set


def test_legacy_split_is_unchanged_without_groups():
    """The frozen Phase 2--4 split IDs must survive the grouped-split change."""
    assert deterministic_split(50, 0.2, 0, "data")[2] == deterministic_split(
        50, 0.2, 0, "data", None
    )[2]


# --- the predeclared rules --------------------------------------------------


def test_label_constants_maximize_recall_under_the_false_positive_ceiling():
    candidates = [
        {"alpha": 0.1, "gamma": 2.0, "core_recall": 0.95, "smooth_false_positive_rate": 0.02},
        {"alpha": 0.15, "gamma": 2.0, "core_recall": 0.89, "smooth_false_positive_rate": 0.0005},
        {"alpha": 0.5, "gamma": 3.0, "core_recall": 0.16, "smooth_false_positive_rate": 0.0002},
    ]
    selection = select_label_constants(candidates)
    assert selection["constraint"] == "satisfied"
    assert (selection["alpha"], selection["gamma"]) == (0.15, 2.0)


def test_label_calibration_reports_failure_instead_of_relaxing_the_ceiling():
    candidates = [
        {"alpha": 0.1, "gamma": 4.0, "core_recall": 0.99, "smooth_false_positive_rate": 0.06},
        {"alpha": 0.5, "gamma": 2.0, "core_recall": 0.20, "smooth_false_positive_rate": 0.004},
    ]
    selection = select_label_constants(candidates)
    assert selection["constraint"] == "failed"
    assert selection["alpha"] == 0.5  # the safest candidate, not the best recall


def _downstream(flag_pct_by_resolution, undershoot, l2):
    groups = [
        {
            "mesh": mesh,
            "resolution": resolution,
            "metrics": {"flagged_pct": {"mean": flagged}},
        }
        for mesh in ("structured", "delaunay")
        for resolution, flagged in flag_pct_by_resolution.items()
    ]
    overall = {
        "metrics": {
            "undershoot": {"maximum": undershoot, "mean": undershoot},
            "flagged_pct": {"mean": float(np.mean(list(flag_pct_by_resolution.values())))},
            "l2_error": {"mean": l2},
        }
    }
    return {"groups": groups, "overall": overall}


def test_acceptance_requires_a_gain_at_bounded_error_cost():
    baseline = _downstream({8: 60.0, 12: 40.0, 16: 20.0}, undershoot=0.01, l2=0.10)
    candidate = _downstream({8: 30.0, 12: 25.0, 16: 20.0}, undershoot=0.005, l2=0.105)
    convergence = {"fits": [{"dimension": "2d-seed0", "l2_slope": 1.9}]}
    extrema = {"candidate_false_positive_rate": 0.01, "baseline_false_positive_rate": 0.02}
    assessment = assess_data_v3(candidate, baseline, convergence, extrema)
    assert assessment["decision"] == "adopt"
    assert all(assessment["criteria"].values())


def test_acceptance_rejects_a_selectivity_gain_bought_with_error():
    baseline = _downstream({8: 60.0, 12: 40.0, 16: 20.0}, undershoot=0.01, l2=0.10)
    candidate = _downstream({8: 10.0, 12: 8.0, 16: 6.0}, undershoot=0.002, l2=0.13)
    convergence = {"fits": [{"dimension": "2d-seed0", "l2_slope": 1.9}]}
    extrema = {"candidate_false_positive_rate": 0.01, "baseline_false_positive_rate": 0.02}
    assessment = assess_data_v3(candidate, baseline, convergence, extrema)
    assert assessment["decision"] == "reject"
    assert not assessment["criteria"]["1_selectivity_or_safety_gain_at_bounded_l2_cost"]


def test_acceptance_rejects_a_broken_smooth_convergence_rate():
    baseline = _downstream({8: 60.0, 12: 40.0, 16: 20.0}, undershoot=0.01, l2=0.10)
    candidate = _downstream({8: 30.0, 12: 25.0, 16: 20.0}, undershoot=0.005, l2=0.10)
    convergence = {"fits": [{"dimension": "2d-seed0", "l2_slope": 1.9}, {"dimension": "2d-seed1", "l2_slope": 1.2}]}
    extrema = {"candidate_false_positive_rate": 0.01, "baseline_false_positive_rate": 0.02}
    assessment = assess_data_v3(candidate, baseline, convergence, extrema)
    assert assessment["decision"] == "reject"
    assert not assessment["criteria"]["3_smooth_convergence_preserved"]
