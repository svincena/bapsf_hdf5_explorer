import numpy as np
import pytest
from lapd_explorer.spatial import average_spatial
from lapd_explorer.model import Dataset, preprocess, quantity


@pytest.mark.parametrize("method", ["box", "gaussian", "disk", "median"])
@pytest.mark.parametrize("mode", ["reflect", "nearest", "wrap"])
def test_constants_and_independent_noncontiguous_planes(method, mode):
    # Spatial axes can straddle case/shot axes, and either can be singleton.
    a = np.broadcast_to(np.arange(24.).reshape(2, 1, 3, 1, 4), (2, 5, 3, 7, 4))
    original = a.copy()
    result = average_spatial(a, method, spatial_axes=(3, 1), mode=mode)
    np.testing.assert_allclose(result, a, atol=1e-13)
    np.testing.assert_array_equal(a, original)
    assert not np.shares_memory(result, a)
    singleton = average_spatial(np.ones((1, 2, 3)), method, mode=mode)
    np.testing.assert_allclose(singleton, 1)
    noisy = np.random.default_rng(42).normal(size=(2, 5, 3, 7, 4))
    result = average_spatial(noisy, method, spatial_axes=(3, 1), mode=mode,
                             window_size=(5, 3), sigma=(.7, 1.3))
    for case in range(2):
        for shot in range(3):
            for time in range(4):
                expected = average_spatial(noisy[case, :, shot, :, time].T, method,
                                           mode=mode, window_size=(5, 3), sigma=(.7, 1.3)).T
                np.testing.assert_allclose(result[case, :, shot, :, time], expected)


def test_box_disk_and_gaussian_impulse_weights():
    a = np.zeros((15, 15))
    a[7, 7] = 1
    box = average_spatial(a, "box", window_size=(3, 5))
    expected = np.zeros_like(a)
    expected[6:9, 5:10] = 1/15
    np.testing.assert_allclose(box, expected, atol=1e-16)
    disk = average_spatial(a, "disk", radius=1)
    expected[:] = 0
    expected[7, 6:9] = 1/5
    expected[6:9, 7] = 1/5
    np.testing.assert_allclose(disk, expected)
    gaussian = average_spatial(a, "gaussian", sigma=(.5, 1))
    y = np.exp(-.5*(np.arange(-2, 3)/.5)**2)
    x = np.exp(-.5*np.arange(-4, 5)**2)
    expected[:] = 0
    expected[5:10, 3:12] = np.outer(y/y.sum(), x/x.sum())
    np.testing.assert_allclose(gaussian, expected, atol=1e-16)


def test_median_removes_spike_and_boundary_modes_are_distinct():
    a = np.ones((7, 7))
    a[3, 3] = 1000
    np.testing.assert_array_equal(average_spatial(a, "median"), 1)
    a = np.broadcast_to(np.arange(4.), (3, 4))
    expected = {"reflect": 4/5, "nearest": 3/5, "wrap": 8/5}
    for mode, first in expected.items():
        out = average_spatial(a, window_size=(1, 5), mode=mode)
        assert out[0, 0] == pytest.approx(first)


@pytest.mark.parametrize("method", ["box", "gaussian", "disk", "median"])
def test_missing_support_renormalization_and_no_cross_time_leak(method):
    a = np.ones((15, 15, 3))
    a[7, 7, 0] = np.inf
    a[..., 2] = np.nan
    original = a.copy()
    propagated = average_spatial(a, method, sigma=.5)
    expected = np.zeros((15, 15), dtype=bool)
    if method == "gaussian":
        expected[5:10, 5:10] = True
    elif method == "disk":
        expected[7, 6:9] = True
        expected[6:9, 7] = True
    else:
        expected[6:9, 6:9] = True
    np.testing.assert_array_equal(np.isnan(propagated[..., 0]), expected)
    np.testing.assert_allclose(propagated[..., 1], 1)
    assert np.isnan(propagated[..., 2]).all()
    omitted = average_spatial(a, method, sigma=.5, nan_policy="omit")
    np.testing.assert_allclose(omitted[..., :2], 1)
    assert np.isnan(omitted[..., 2]).all()
    np.testing.assert_array_equal(a, original)


def test_omit_averages_only_available_values():
    a = np.array([[1., np.nan, 5.], [7., 9., np.inf], [13., 15., 17.]])
    assert average_spatial(a, "box", nan_policy="omit")[1, 1] == pytest.approx(67/7)
    assert average_spatial(a, "disk", nan_policy="omit")[1, 1] == pytest.approx(31/3)
    assert average_spatial(a, "median", nan_policy="omit")[1, 1] == 9


@pytest.mark.parametrize("kwargs,match", [
    ({"spatial_axes": (0, 0)}, "distinct"),
    ({"spatial_axes": (0, 3)}, "valid spatial"),
    ({"spatial_axes": (0,)}, "valid spatial"),
    ({"window_size": (2, 3)}, "odd"),
    ({"window_size": 2.5}, "integer"),
    ({"window_size": (3,)}, "pair"),
    ({"method": "gaussian", "sigma": (1, 0)}, "positive"),
    ({"method": "gaussian", "sigma": np.nan}, "positive"),
    ({"method": "disk", "radius": -1}, "integer"),
    ({"mode": "unknown"}, "Boundary"),
    ({"nan_policy": "interp"}, "policy"),
    ({"method": "unknown"}, "Unknown spatial"),
])
def test_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        average_spatial(np.ones((3, 4, 5)), **kwargs)


@pytest.mark.parametrize("a", [np.ones(5), np.ones((2, 3), complex), np.empty((0, 3))])
def test_invalid_arrays(a):
    with pytest.raises(ValueError):
        average_spatial(a)


def test_preprocessing_order_components_and_saved_history(tmp_path):
    from lapd_explorer.io import save_dataset, load_dataset
    from lapd_explorer.smoothing import smooth_time_series
    from scipy.integrate import cumulative_trapezoid
    a = np.random.default_rng(7).normal(size=(2, 3, 5, 4, 8))
    dims = ("case", "z", "shot", "x", "time")
    coords = {dim: np.arange(size, dtype=float) for dim, size in zip(dims, a.shape)}
    data = Dataset({"A": a, "B": -a}, dims, coords)
    settings = dict(method="median", window_size=(3, 3))
    result = preprocess(data, integrate=True, smoothing=dict(window_size=3), gain=2,
                        average=True, spatial_averaging=settings)
    integrated = cumulative_trapezoid(a, coords["time"], axis=-1, initial=0)
    temporal = smooth_time_series(integrated, window_size=3)*2
    expected = average_spatial(temporal.mean(axis=2), spatial_axes=(1, 2), **settings)
    np.testing.assert_allclose(result.channels["A"], expected)
    np.testing.assert_allclose(result.channels["B"], -expected)
    np.testing.assert_allclose(quantity(result, ["A", "B"], "Magnitude"), np.sqrt(2)*abs(expected[0]))
    assert result.dims == ("case", "z", "x", "time")
    assert "Average 5 shots" in result.history[-2]
    assert "Spatial averaging: median" in result.history[-1]
    assert "grid points along z, x" in result.history[-1]
    assert result.units == "V·s"
    path = tmp_path / "spatial.h5"
    save_dataset(path, result)
    restored = load_dataset(path)
    assert restored.history == result.history
    np.testing.assert_array_equal(restored.channels["A"], expected)
    np.testing.assert_array_equal(data.channels["A"], a)
    np.testing.assert_array_equal(preprocess(data).channels["A"], a)


@pytest.mark.parametrize("dims,shape", [(("time",), (5,)), (("x", "time"), (3, 5))])
def test_preprocessing_rejects_nonplanes(dims, shape):
    data = Dataset({"A": np.ones(shape)}, dims, {d: np.arange(n) for d, n in zip(dims, shape)})
    with pytest.raises(ValueError, match="2D data plane"):
        preprocess(data, spatial_averaging={})
