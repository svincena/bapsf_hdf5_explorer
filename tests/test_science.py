import numpy as np
import pytest
from lapd_explorer.model import Dataset, preprocess, quantity, spectrum
from lapd_explorer.io import (
    Records, infer_shots_from_positions, load_dataset, map_manual, map_motion,
    read_raw, save_dataset,
)


def point(values, t=None):
    a = np.asarray(values, float)
    return Dataset({"A": a}, ("time",), {"time": np.arange(a.size) if t is None else t})


def test_detrend_actual_nonuniform_time_and_integration():
    t = np.array([0, .1, .3, .7, 1.1])
    data = point(3*t+7, t)
    out = preprocess(data, baseline="Linear detrend")
    np.testing.assert_allclose(out.channels["A"], 0, atol=1e-14)
    integrated = preprocess(point(2*t+1, t), integrate=True)
    np.testing.assert_allclose(integrated.channels["A"], t*t+t)
    assert integrated.units == "V·s"
    np.testing.assert_array_equal(data.channels["A"], 3*t+7)


def test_average_only_named_shots_with_shot_axis_not_last():
    a = np.arange(2*3*4*5.).reshape(2, 3, 4, 5)
    data = Dataset({"A": a}, ("case", "shot", "x", "time"),
                   {"case": np.arange(2), "shot": np.arange(3), "x": np.arange(4), "time": np.arange(5)})
    out = preprocess(data, average=True)
    assert out.dims == ("case", "x", "time")
    np.testing.assert_allclose(out.channels["A"], a.mean(axis=1))
    assert "shot" not in out.coords


def test_no_synthetic_shot_axis_and_magnitude():
    data = Dataset({"A": np.full((3, 4), 3.), "B": np.full((3, 4), 4.)},
                   ("x", "time"), {"x": np.arange(3), "time": np.arange(4)})
    assert preprocess(data, average=True).dims == data.dims
    np.testing.assert_allclose(quantity(data, ["A", "B"], "Magnitude"), 5)


def test_psd_peak_and_nonuniform_rejection():
    t = np.arange(2048)/2048
    f, p = spectrum(np.sin(2*np.pi*128*t), t)
    assert f[p.argmax()] == 128
    with pytest.raises(ValueError, match="uniform"):
        spectrum(np.ones(5), np.array([0, 1, 2, 4, 6]))


def records():
    # Scrambled spatial order; repeated records remain in acquisition order.
    xyz = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]*4)
    a = np.arange(16*5.).reshape(16, 5)
    return Records({"A": a}, np.arange(5)*.1, np.arange(100, 116), xyz, "test", {})


def test_motion_shot_alignment_and_cases():
    rec = records()
    data = map_motion(rec, cases=2, repeats=2)
    assert data.dims == ("y", "x", "case", "shot", "time")
    np.testing.assert_array_equal(data.channels["A"][0, 0, 1, 0], rec.channels["A"][9])
    assert data.shot_numbers[0, 0, 1, 0] == 109
    alternative = map_motion(rec, cases=2, repeats=2, repeat_order="shot,case")
    assert alternative.shot_numbers[0, 0, 1, 0] == 105


def test_infer_shots_from_point_line_and_plane():
    point = np.zeros((7, 3))
    assert infer_shots_from_positions(point) == (7, 1, ())

    line = np.repeat(np.column_stack((np.arange(4), np.zeros(4), np.zeros(4))), 3, axis=0)
    assert infer_shots_from_positions(line) == (3, 4, (4,))

    yy, xx = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
    plane = np.repeat(np.column_stack((xx.ravel(), yy.ravel(), np.zeros(6))), 5, axis=0)
    assert infer_shots_from_positions(plane) == (5, 6, (3, 2))


def test_infer_shots_rejects_unbalanced_positions():
    xyz = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="equal numbers"):
        infer_shots_from_positions(xyz)


def test_motion_missing_and_nonplanar_rejected():
    rec = records()
    with pytest.raises(ValueError, match="requires"):
        map_motion(rec)
    rec.xyz[0] = [0, 0, 0]
    with pytest.raises(ValueError, match="Unequal"):
        map_motion(rec, 2, 2)
    rec.xyz[0, 2] = 1
    with pytest.raises(ValueError, match="3 spatial"):
        map_motion(rec, 2, 2)


def test_manual_axis_order_and_roundtrip(tmp_path):
    rec = records()
    d = map_manual(rec, [("x", 4, -1, 1), ("case", 2, 0, 1), ("shot", 2, 0, 1)])
    assert d.shot_numbers[2, 1, 0] == 110
    path = tmp_path/"processed.h5"
    save_dataset(path, d)
    out = load_dataset(path)
    assert out.dims == d.dims
    assert out.history == d.history
    np.testing.assert_array_equal(out.channels["A"], d.channels["A"])
    np.testing.assert_array_equal(out.shot_numbers, d.shot_numbers)


def test_point_roundtrip_and_raw_line(tmp_path):
    import h5py
    path = tmp_path/"point.h5"
    d = map_manual(Records({"A": np.ones((1, 5))}, np.arange(5), np.array([99]), None, "test", {}), [])
    save_dataset(path, d)
    assert load_dataset(path).shot_numbers == 99
    raw = tmp_path/"raw.h5"
    with h5py.File(raw, "w") as f:
        f["signal"] = np.arange(20).reshape(4, 5)
    line = read_raw(raw, {"A": "signal"}, [("z", 4, 0, 3)], 1e-6)
    assert line.dims == ("z", "time")
    assert line.shot_numbers is None
    np.testing.assert_allclose(line.coords["time"], np.arange(5)*1e-6)


@pytest.mark.parametrize("dims,shape", [(('x','y','z','time'), (2,2,2,3)), (('x','x','time'), (2,2,3))])
def test_dimension_validation(dims, shape):
    with pytest.raises(ValueError):
        Dataset({"A": np.ones(shape)}, dims, {d: np.arange(n) for d, n in zip(dims, shape)})
