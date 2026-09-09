import numpy as np
import pytest
import xarray as xr

from bapsf_explorer.data import record_array, acquisition_reshape, motion_reshape


def test_motion_uses_positions_not_acquisition_order_and_preserves_ids():
    raw = record_array(np.arange(12).reshape(6, 2), [0., 1.], [11, 12, 13, 14, 15, 16],
                       positions={"x": [1, 0, 1, 0, 1, 0], "measured_x": [1.01, .01, 1.02, .02, 1.03, .03]})
    grid = motion_reshape(raw)
    assert grid.dims == ("x", "shot", "time")
    np.testing.assert_array_equal(grid.sel(x=0).shot_id, [12, 14, 16])
    np.testing.assert_array_equal(grid.sel(x=1).isel(shot=1), [4, 5])
    np.testing.assert_allclose(grid.measured_x.sel(x=0), [.01, .02, .03])


def test_missing_grid_and_unequal_repeats_do_not_fabricate_measurements():
    raw = record_array(np.ones((4, 2)), [0, 1], [1, 2, 3, 4],
                       positions={"x": [0, 0, 1, 0], "y": [0, 0, 0, 1]})
    grid = motion_reshape(raw)
    assert grid.sizes == {"y": 2, "x": 2, "shot": 2, "time": 2}
    assert grid.attrs["missing_cells"] == 4
    assert np.isnan(grid.sel(x=1, y=1)).all()
    assert (grid.sel(x=1, y=1).shot_id == -1).all()
    assert grid.mean("shot").sel(x=1, y=0).values.tolist() == [1, 1]


def test_cases_are_explicit_and_not_confused_with_repeats():
    raw = record_array(np.arange(8).reshape(4, 2), [0, 1], [1, 2, 3, 4], positions={"x": [0, 1, 0, 1]})
    grid = motion_reshape(raw, cases=["a", "a", "b", "b"])
    assert grid.dims == ("x", "case", "shot", "time")
    assert grid.sel(x=1, case="b").shot_id.item() == 4


def test_acquisition_order_and_arbitrary_transpose():
    raw = record_array(np.arange(48).reshape(24, 2), [0, 1], np.arange(24) + 50)
    grid = acquisition_reshape(raw, {"y": 2, "x": 3, "case": 2, "shot": 2})
    assert grid.sel(y=1, x=2, case=1, shot=1).shot_id.item() == 73
    np.testing.assert_array_equal(grid.transpose("shot", "case", "x", "y", "time").sel(y=1, x=2, case=1, shot=1), [46, 47])


@pytest.mark.parametrize("dims", [{"x": 3}, {"x": 0}, {"time": 4}, {"x": -4}])
def test_bad_shapes_fail(dims):
    raw = record_array(np.zeros((4, 2)), [0, 1], np.arange(4))
    with pytest.raises(ValueError):
        acquisition_reshape(raw, dims)


def test_memory_guard_before_grid_allocation():
    raw = record_array(np.ones((3, 2)), [0, 1], [1, 2, 3], positions={"x": [0, 1, 2]})
    with pytest.raises(ValueError, match="Motion grid needs"):
        motion_reshape(raw, max_bytes=1)


def test_export_roundtrip(tmp_path):
    raw = record_array(np.arange(12).reshape(6, 2), [0, .1], np.arange(6),
                       positions={"x": [0, 0, 0, 1, 1, 1]}, attrs={"units": "V"})
    grid = motion_reshape(raw)
    path = tmp_path / "analysis.nc"
    grid.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataarray(path, engine="h5netcdf") as reopened:
        xr.testing.assert_identical(grid, reopened.load())
