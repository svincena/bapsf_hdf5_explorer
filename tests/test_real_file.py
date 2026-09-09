"""Opt-in integration tests: BAPSF_TEST_FILE=/path/to/file.hdf5 pytest."""
import os

import numpy as np
import pytest
from bapsflib import lapd

from bapsf_explorer.reader import inspect_file, read_channel
from bapsf_explorer.data import motion_reshape


@pytest.fixture(scope="module")
def catalog():
    path = os.environ.get("BAPSF_TEST_FILE")
    if not path:
        pytest.skip("Set BAPSF_TEST_FILE for read-only real-file tests")
    return inspect_file(path)


def test_sample_channels_time_scaling_and_motion(catalog):
    assert [ch.label for ch in catalog.channels] == ["Isat", "Isweep", "Vsweep"]
    ch = catalog.channels[0]
    raw = read_channel(catalog.path, ch, rows=slice(0, ch.records), samples=slice(100, 2148), motion=catalog.motions[0])
    grid = motion_reshape(raw)
    assert grid.sizes == {"x": 91, "shot": 5, "time": 2048}
    np.testing.assert_allclose(grid.x, np.linspace(-22.5, 22.5, 91))
    assert grid.x.attrs["units"] == "cm"
    assert grid.time.values[0] == pytest.approx(2e-6)
    np.testing.assert_allclose(np.diff(grid.time), 20e-9)
    np.testing.assert_array_equal(grid.shot_id.values.ravel(), np.arange(1, 456))
    assert grid.attrs["missing_cells"] == 0
    with lapd.File(catalog.path, mode="r") as file:
        expected = file.read_data(ch.board, ch.channel, index=slice(0, 5), time_slice=slice(100, 2148), **ch.read_kwargs)
    np.testing.assert_array_equal(raw.values[:5], expected["signal"])


def test_all_channels_full_time_trace(catalog):
    for ch in catalog.channels:
        raw = read_channel(catalog.path, ch, rows=slice(0, 1))
        assert raw.sizes == {"record": 1, "time": 139264}
        assert np.isfinite(raw.values).all()
        assert raw.attrs["units"] == "V"
        assert raw.time.values[-1] == pytest.approx((139264 - 1) * 20e-9)
