import numpy as np
import pytest
from lapd_explorer.smoothing import smooth_time_series
from lapd_explorer.model import Dataset, preprocess


@pytest.mark.parametrize("method,settings", [("moving", {"window_size": 5}),
    ("savgol", {"window_size": 5, "polyorder": 2}), ("gaussian", {"sigma": 1}),
    ("butterworth", {"fs": 100, "cutoff": 10}),
    ("butterworth", {"fs": 100, "cutoff": 10, "butter_type": "highpass"}),
    ("butterworth", {"fs": 100, "cutoff": 10, "cutoff_upper": 30, "butter_type": "bandpass"})])
@pytest.mark.parametrize("shape,axis", [((80,), 0), ((3, 80), -1),
    ((2, 80, 3), 1), ((2, 3, 2, 4, 80), -1)])
def test_arbitrary_axes_match_independent_traces(method, settings, shape, axis):
    a = np.random.default_rng(42).normal(size=shape)
    original = a.copy()
    result = smooth_time_series(a, method, time_axis=axis, **settings)
    expected = np.apply_along_axis(lambda row: smooth_time_series(row, method, **settings), axis, a)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(a, original)
    assert result.shape == a.shape


def test_moving_missing_values_have_local_support():
    a = np.ones(20)
    a[7] = np.nan
    result = smooth_time_series(a, window_size=3)
    assert np.array_equal(np.flatnonzero(np.isnan(result)), [6, 7, 8])
    np.testing.assert_allclose(result[np.isfinite(result)], 1)


@pytest.mark.parametrize("method", ["moving", "savgol", "gaussian", "butterworth"])
def test_interpolation_all_missing_and_noncontiguous_axis(method):
    a = np.tile(np.arange(40.), (2, 3, 1)).transpose(1, 2, 0)
    a[0, 10:12, 0] = np.nan
    a[1, :, 1] = np.nan
    original = a.copy()
    out = smooth_time_series(a, method, time_axis=1, nan_policy="interp", fs=100, cutoff=10)
    assert np.isnan(out[1, :, 1]).all()
    assert np.isfinite(out[0, :, 0]).all()
    np.testing.assert_array_equal(a, original)


def test_interp_uses_actual_time_and_extends_endpoints():
    a = [np.nan, 2, np.nan, 10, np.nan]
    result = smooth_time_series(a, window_size=1, nan_policy="interp", time=[0, 1, 2, 5, 8])
    np.testing.assert_allclose(result, [2, 2, 4, 10, 10])


def test_butterworth_suppresses_high_frequency_without_phase_shift():
    t = np.arange(2000)/1000
    low = np.sin(2*np.pi*10*t)
    a = low + np.sin(2*np.pi*200*t)
    out = smooth_time_series(a, "butterworth", time=t, cutoff=40)
    np.testing.assert_allclose(out[200:-200], low[200:-200], atol=3e-5)


@pytest.mark.parametrize("kind,cutoff,upper,expected", [
    ("lowpass", 20, None, [1, 0, 0]),
    ("highpass", 100, None, [0, 0, 1]),
    ("bandpass", 20, 100, [0, 1, 0]),
])
def test_butterworth_passbands(kind, cutoff, upper, expected):
    t = np.arange(8000)/1000
    waves = np.sin(2*np.pi*np.array([5, 50, 250])[:, None]*t)
    out = smooth_time_series(waves.sum(axis=0), "butterworth", time=t,
                             butter_type=kind, cutoff=cutoff, cutoff_upper=upper)
    # Project away from edge transients to check gain and phase in each band.
    amplitudes = 2*np.mean(waves[:, 1000:-1000]*out[1000:-1000], axis=1)
    quadrature = 2*np.mean(np.cos(2*np.pi*np.array([5, 50, 250])[:, None]*t[1000:-1000])
                           * out[1000:-1000], axis=1)
    # Fourth-order roll-off leaves a small finite stop-band response.
    np.testing.assert_allclose(amplitudes, expected, atol=.005)
    np.testing.assert_allclose(quadrature, 0, atol=1e-6)


@pytest.mark.parametrize("kind,lower,upper,match", [
    ("unknown", 10, 20, "type"),
    ("bandpass", 10, None, "lower < upper"),
    ("bandpass", 10, 10, "lower < upper"),
    ("bandpass", 20, 10, "lower < upper"),
    ("bandpass", 10, np.nan, "lower < upper"),
    ("bandpass", 10, np.inf, "lower < upper"),
    ("bandpass", 10, 50, "Nyquist"),
    ("highpass", 50, None, "Nyquist"),
    ("bandpass", 0, 20, "Cutoff"),
])
def test_butterworth_invalid_bands(kind, lower, upper, match):
    with pytest.raises(ValueError, match=match):
        smooth_time_series(np.ones(100), "butterworth", fs=100,
                           butter_type=kind, cutoff=lower, cutoff_upper=upper)


def test_bandpass_padding_and_history(tmp_path):
    from lapd_explorer.io import save_dataset, load_dataset
    settings = dict(method="butterworth", butter_type="bandpass", cutoff=10, cutoff_upper=30)
    with pytest.raises(ValueError, match="at least 28"):
        smooth_time_series(np.ones(20), fs=100, **settings)
    data = Dataset({"A": np.ones(100)}, ("time",), {"time": np.arange(100)/100})
    result = preprocess(data, smoothing=settings)
    assert "butter_type=bandpass" in result.history[-1]
    assert "cutoff=10 Hz" in result.history[-1]
    assert "cutoff_upper=30 Hz" in result.history[-1]
    path = tmp_path / "bandpass.h5"
    save_dataset(path, result)
    assert load_dataset(path).history == result.history


@pytest.mark.parametrize("a,kwargs,match", [
    ([1+2j, 3j], {}, "real"), ([1, 2], {"time_axis": 2}, "axis"),
    ([1], {}, "two"), (np.ones(20), {"window_size": 2.5}, "integer"),
    (np.ones(20), {"method": "savgol", "window_size": 4}, "odd"),
    (np.ones(20), {"method": "savgol", "polyorder": -1}, "Polynomial"),
    (np.ones(20), {"method": "gaussian", "sigma": np.inf}, "sigma"),
    (np.ones(20), {"method": "butterworth", "dt": 0, "cutoff": 1}, "positive"),
    (np.ones(20), {"method": "butterworth", "dt": 1, "fs": 1, "cutoff": .1}, "exactly"),
    (np.ones(20), {"method": "butterworth", "fs": 10, "cutoff": 5}, "Nyquist"),
    (np.ones(5), {"method": "butterworth", "fs": 10, "cutoff": 1}, "at least"),
    (np.ones(20), {"method": "butterworth", "time": np.arange(20.)**2, "cutoff": .1}, "uniform"),
])
def test_invalid_inputs(a, kwargs, match):
    with pytest.raises(ValueError, match=match):
        smooth_time_series(a, **kwargs)


def test_preprocessing_order_metadata_and_export(tmp_path):
    from scipy.integrate import cumulative_trapezoid
    from lapd_explorer.io import save_dataset, load_dataset
    t = np.arange(30.) / 100
    a = np.random.default_rng(7).normal(size=(2, 3, 30))
    data = Dataset({"A": a, "B": 2*a}, ("case", "shot", "time"),
                   {"case": np.arange(2), "shot": np.arange(3), "time": t})
    settings = dict(method="moving", window_size=5)
    result = preprocess(data, integrate=True, smoothing=settings, gain=2, average=True)
    integrated = cumulative_trapezoid(a, t, axis=-1, initial=0)
    expected = smooth_time_series(integrated, **settings).mean(axis=1)*2
    np.testing.assert_allclose(result.channels["A"], expected)
    np.testing.assert_allclose(result.channels["B"], expected*2)
    assert result.dims == ("case", "time") and result.units == "V·s"
    assert "integration" in result.history[0] and "smoothing" in result.history[1]
    path = tmp_path / "smooth.h5"
    save_dataset(path, result)
    restored = load_dataset(path)
    assert restored.history == result.history
    np.testing.assert_array_equal(restored.channels["A"], expected)
    np.testing.assert_array_equal(preprocess(data).channels["A"], a)


def test_savgol_missing_edge_does_not_contaminate_other_traces():
    a = np.ones((2, 25))
    a[0, 4] = np.nan
    out = smooth_time_series(a, "savgol", window_size=5, polyorder=2)
    assert np.isnan(out[0, :7]).all()
    np.testing.assert_allclose(out[0, 7:], 1)
    np.testing.assert_allclose(out[1], 1)
