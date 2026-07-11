import numpy as np

from tci.data.generate import (
    generate_euler_riemann_samples,
    generate_exact_samples,
    generate_numerical_samples,
    random_piecewise_fourier,
)


def test_random_piecewise_fourier_jumps():
    rng = np.random.default_rng(3)
    f, locs = random_piecewise_fourier(rng, max_disc=5)
    x = np.linspace(0, 1, 1001)
    vals = f(x)
    assert vals.shape == x.shape
    assert np.all((locs > 0) & (locs < 1))
    # Function is finite and (for pieces) bounded.
    assert np.all(np.isfinite(vals))


def test_generate_exact_samples_shapes_and_labels():
    samples = generate_exact_samples(5, N=1, k_range=(30, 60), seed=1)
    for s in samples:
        Np, K = s.u.shape
        assert Np == 2
        assert 30 <= K <= 60
        assert s.labels.shape == (K,)
        assert s.labels.sum() == len(np.unique(s.disc_locs)) or s.labels.sum() <= len(
            s.disc_locs
        )


def test_generate_numerical_samples_runs():
    samples = generate_numerical_samples(
        2, N=1, k_range=(30, 40), seed=2, t_range=(0.05, 0.1)
    )
    for s in samples:
        assert np.all(np.isfinite(s.u))
        assert s.labels.dtype == bool


def test_variable_mesh_lengths():
    samples = generate_exact_samples(10, k_range=(30, 120), seed=4)
    lengths = {s.u.shape[1] for s in samples}
    assert len(lengths) > 3


def test_generate_euler_riemann_samples_runs_and_is_reproducible():
    first = generate_euler_riemann_samples(
        2, N=1, k_range=(20, 24), seed=7, t_range=(0.01, 0.02)
    )
    second = generate_euler_riemann_samples(
        2, N=1, k_range=(20, 24), seed=7, t_range=(0.01, 0.02)
    )
    for a, b in zip(first, second):
        assert a.u.shape[0] == 2
        assert a.labels.shape == (a.u.shape[1],)
        assert a.labels.dtype == bool
        assert a.labels.any()
        assert np.all(np.isfinite(a.u)) and np.min(a.u) > 0
        assert np.array_equal(a.u, b.u)
        assert np.array_equal(a.labels, b.labels)
