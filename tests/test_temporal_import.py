"""Disk selections, reduced sampling rates, bounded reads, and preview selectors."""
import h5py
import numpy as np
import pytest
from bapsflib import lapd
from bapsflib._hdf.maps.tests import FauxHDFBuilder
from lapd_explorer import io, spectral
from lapd_explorer.temporal import sample_slice, preview_record


def test_inclusive_time_limits_snap_to_original_samples():
    dt, t0 = 1e-8, -2e-6
    expected = slice(17, 201, 7)
    assert sample_slice(1024, dt, t0, sample_limits=(17, 200), decimation=7) == expected
    assert sample_slice(1024, dt, t0, time_limits=(t0+17*dt, t0+200*dt), decimation=7) == expected
    assert sample_slice(1024, dt, t0, time_limits=(t0+17.2*dt, t0+200.8*dt)).start == 18


@pytest.mark.parametrize("options", [dict(decimation=0), dict(decimation=True), dict(decimation=2.5),
    dict(sample_limits=(-1, 10)), dict(sample_limits=(10, 9)), dict(sample_limits=(0, 100)),
    dict(sample_limits=(1, 1)), dict(time_limits=(np.nan, 1)), dict(sample_limits=(0, 10), time_limits=(0, 10))])
def test_invalid_temporal_selection_rejected(options):
    with pytest.raises(ValueError):
        sample_slice(100, 1., **options)


def test_raw_hyperslabs_and_effective_sampling_rate(tmp_path, monkeypatch):
    path = tmp_path/"raw.h5"
    original = np.arange(2*3*4*257).reshape(2, 3, 4, 257)
    with h5py.File(path, "w") as f:
        f["A"], f["B"] = original, original + 100
    getitem = h5py.Dataset.__getitem__
    reads = []
    def read(dataset, key, *args, **kwargs):
        if dataset.name in ("/A", "/B"):
            reads.append(key)
            assert key[-1] == slice(10, 201, 6)
        return getitem(dataset, key, *args, **kwargs)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    data = io.read_raw(path, {"A": "A", "B": "B"}, [("x", 2, 0, 1), ("y", 3, 0, 2), ("shot", 4, 0, 3)],
                       1e-6, t0=-1e-3, sample_limits=(10, 200), decimation=6)
    np.testing.assert_array_equal(data.channels["A"], original[..., 10:201:6])
    np.testing.assert_array_equal(data.channels["B"], (original+100)[..., 10:201:6])
    np.testing.assert_allclose(data.coords["time"], -1e-3 + np.arange(10, 201, 6)*1e-6)
    assert spectral.sampling_rate(data.coords["time"]) == pytest.approx(1/(6e-6))
    assert reads
    temporal = data.metadata["temporal selection"]["A"]
    assert temporal["effective sampling rate (Hz)"] == pytest.approx(1/(6e-6))
    assert temporal["anti-alias filtering"] is False


@pytest.mark.parametrize("digitizer", ["SIS 3301", "SIS crate"])
def test_lapd_thinning_batches_and_calibration_match_direct_read(tmp_path, monkeypatch, digitizer):
    path = tmp_path/"acquisition.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module(digitizer, {"sn_size": 513, "nt": 64})
    spec = io.inspect_file(path)["channels"][0]
    opts = {k: spec[k] for k in ("digitizer", "config_name", "adc") if spec.get(k)}
    with lapd.File(path, mode="r") as f:
        reference = f.read_data(spec["board"], spec["channel"], **opts, index=slice(1, None))
        dt = reference.dt.to_value("s")
    original = lapd.File.read_data
    reads = []
    def read(f, *args, **kwargs):
        reads.append(kwargs)
        assert len(kwargs["index"]) <= 256
        assert kwargs["time_slice"] == slice(3, 60, 4)
        return original(f, *args, **kwargs)
    monkeypatch.setattr(lapd.File, "read_data", read)
    data = io.read_lapd(path, {"A": spec}, start=1, t0=-1.,
                        time_limits=(-1.+3*dt, -1.+59*dt), decimation=4)
    np.testing.assert_array_equal(data.channels["A"], reference["signal"][:, 3:60:4])
    np.testing.assert_array_equal(data.shots, reference["shotnum"])
    np.testing.assert_allclose(data.time, -1.+np.arange(3, 60, 4)*dt)
    assert len(reads) == 2


def test_lecroy_without_sample_interval_requires_explicit_raw_interval(tmp_path, monkeypatch):
    path = tmp_path/"scope.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module("LeCroy_scope", {"sn_size": 8, "nt": 64})
    info = io.inspect_file(path)
    spec = info["channels"][0]
    dataset_path = next(name for name, shape in info["datasets"] if shape == (8, 64))
    with h5py.File(path, "r") as f:
        expected = f[dataset_path][:, 3:60:4]
    monkeypatch.setattr(lapd.File, "read_data", lambda *args, **kwargs: pytest.fail(
        "Missing time metadata must be rejected before reading signals"))
    with pytest.raises(ValueError, match="sample interval is missing"):
        io.import_metadata(path, spec=spec)
    with pytest.raises(ValueError, match="sample interval is missing"):
        io.read_lapd(path, {"A": spec}, sample_limits=(3, 59), decimation=4)
    data = io.read_raw(path, {"A": dataset_path}, [("shot", 8, 0, 7)], 1e-6,
                       sample_limits=(3, 59), decimation=4)
    np.testing.assert_array_equal(data.channels["A"], expected)
    np.testing.assert_allclose(data.coords["time"], np.arange(3, 60, 4)*1e-6)


def test_preview_reads_only_one_bounded_trace_and_refines_zoom(tmp_path):
    path = tmp_path/"raw.h5"
    original = np.arange(4*100001).reshape(4, 100001)
    with h5py.File(path, "w") as f:
        f["signal"] = original
    metadata = io.import_metadata(path, dataset_path="signal", dt=1e-6)
    index, row = preview_record(metadata)
    assert index == row == 2
    time, values = io.preview_trace(path, metadata, index, 0, 100000, dataset_path="signal")
    assert len(time) <= 20000
    np.testing.assert_array_equal(values, original[2, ::6])
    time, values = io.preview_trace(path, metadata, index, 1000, 1100, dataset_path="signal")
    assert len(time) == 101
    np.testing.assert_array_equal(values, original[2, 1000:1101])


def test_record_selection_by_index_global_shot_and_nearest_coordinate():
    metadata = dict(indices=np.arange(10, 16), shots=np.arange(100, 106),
                    xyz=np.array([[0., 0., 1.]]*3 + [[2., 0., 1.]]*3))
    assert preview_record(metadata) == (13, 3)
    assert preview_record(metadata, mode="index", index=11) == (11, 1)
    assert preview_record(metadata, mode="shot", shot=104) == (14, 4)
    assert preview_record(metadata, mode="position", position=(1.9, None, None), repeat=2) == (15, 5)
    with pytest.raises(ValueError, match="No matching record"):
        preview_record(metadata, mode="shot", shot=99)
    with pytest.raises(ValueError, match="contains 3 records"):
        preview_record(metadata, mode="position", position=(0., 0., 1.), repeat=3)
