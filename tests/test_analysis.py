import numpy as np
import pytest

from bapsf_explorer.analysis import filter_signal, subtract_baseline
from bapsf_explorer.data import record_array, acquisition_reshape


def test_baseline_is_per_trace_and_does_not_mutate_original():
    raw = record_array([[4., 6., 9.], [10., 12., 15.]], [0, 1, 2], [1, 2], attrs={"units": "V"})
    corrected = subtract_baseline(raw, 0, 1)
    np.testing.assert_allclose(corrected, [[-1, 1, 4], [-1, 1, 4]])
    assert raw[0, 0] == 4
    assert corrected.attrs["units"] == "V"
    with pytest.raises(ValueError, match="no samples"):
        subtract_baseline(raw, 4, 5)


def test_filter_rejects_high_frequency_and_preserves_low_frequency_phase():
    t = np.arange(10000) / 10000
    low = np.sin(2 * np.pi * 50 * t)
    high = np.sin(2 * np.pi * 2000 * t)
    raw = record_array((low + high)[None, :], t, [1])
    result = filter_signal(raw, "lowpass", 200)
    np.testing.assert_allclose(result.values[0, 500:-500], low[500:-500], atol=.001)
    assert np.max(np.abs(raw.values[0] - low)) > .9


def test_filter_preserves_empty_shots_but_rejects_partial_gaps():
    raw = record_array(np.stack([np.ones(100), np.full(100, np.nan)]), np.arange(100)*.001, [1, 2])
    result = filter_signal(raw, "lowpass", 100)
    assert np.isnan(result[1]).all()
    raw.values[0, 4] = np.nan
    with pytest.raises(ValueError, match="partial missing"):
        filter_signal(raw, "lowpass", 100)


@pytest.mark.parametrize("kind,cutoff", [("lowpass", 500), ("lowpass", -1), ("bandpass", [200, 100]), ("highpass", [1, 2])])
def test_invalid_cutoffs(kind, cutoff):
    raw = record_array(np.ones((1, 100)), np.arange(100)*.001, [1])
    with pytest.raises(ValueError):
        filter_signal(raw, kind, cutoff)


def test_nonuniform_time_and_short_windows_fail():
    raw = record_array(np.ones((1, 5)), [0, .001, .002, .004, .005], [1])
    with pytest.raises(ValueError, match="uniformly"):
        filter_signal(raw, "lowpass", 100)
    raw = record_array(np.ones((1, 5)), np.arange(5)*.001, [1])
    with pytest.raises(ValueError, match="too short"):
        filter_signal(raw, "lowpass", 100)


def test_filter_uses_named_time_axis():
    raw = record_array(np.ones((6, 100)), np.arange(100)*.001, np.arange(6))
    grid = acquisition_reshape(raw, {"y": 2, "shot": 3}).transpose("time", "shot", "y")
    result = filter_signal(grid, "highpass", 10)
    assert result.dims == grid.dims
    np.testing.assert_allclose(result, 0, atol=1e-12)
