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
                       1e-6, t0=-1e-3, sample_limits=(10, 200), decimation=6, downsampling="simple")
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
                        time_limits=(-1.+3*dt, -1.+59*dt), decimation=4, downsampling="simple")
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
        io.read_lapd(path, {"A": spec}, sample_limits=(3, 59), decimation=4, downsampling="simple")
    data = io.read_raw(path, {"A": dataset_path}, [("shot", 8, 0, 7)], 1e-6,
                       sample_limits=(3, 59), decimation=4, downsampling="simple")
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


@pytest.mark.parametrize("factor", [2, 3, 7, 41])
def test_chunked_fir_matches_whole_trace_without_seams(factor):
    from scipy.signal import resample_poly
    from lapd_explorer.temporal import resample_into, retained_samples
    values = np.random.default_rng(5).normal(size=(3, 4011))
    selection = sample_slice(4011, 1., sample_limits=(17, 3995), decimation=factor)
    output = np.empty((3, retained_samples(selection)))
    reads, progress = [], []
    def read(part):
        reads.append(part)
        return values[:, part]
    resample_into(read, output, selection, chunk_samples=113, progress=progress.append)
    reference = resample_poly(values[:, 17:3996], 1, factor, axis=-1)
    np.testing.assert_allclose(output, reference, atol=1e-12, rtol=1e-12)
    assert len(reads) > 1
    assert all(part.step == 1 and part.stop-part.start <= 113+20*factor for part in reads)
    assert progress == sorted(progress) and progress[-1] == output.shape[-1]


def test_fir_suppresses_alias_and_preserves_passband():
    from lapd_explorer.temporal import resample_into, retained_samples
    n, factor = 32768, 8
    time = np.arange(n)
    values = np.array([np.sin(2*np.pi*.01*time), np.sin(2*np.pi*.2*time)])
    selection = sample_slice(n, 1., decimation=factor)
    filtered = np.empty((2, retained_samples(selection)))
    resample_into(lambda part: values[:, part], filtered, selection, chunk_samples=1024)
    interior = filtered[:, 20:-20]
    assert np.sqrt(np.mean(interior[0]**2)) == pytest.approx(2**-.5, rel=.005)
    assert np.sqrt(np.mean(interior[1]**2)) < .001
    assert np.sqrt(np.mean(values[1, ::factor]**2)) > .7


@pytest.mark.parametrize("method", ["polyphase", "simple", "average"])
def test_raw_resampling_methods_coordinates_metadata_and_progress(tmp_path, method):
    from scipy.signal import resample_poly
    path = tmp_path/"methods.h5"
    values = np.random.default_rng(3).normal(size=(2, 3, 101))
    with h5py.File(path, "w") as f:
        f["A"] = values
    progress = []
    result = io.read_raw(path, {"A": "A"}, [("x", 2, 0, 1), ("shot", 3, 0, 2)],
                         .01, t0=-2., sample_limits=(3, 97), decimation=4, downsampling=method,
                         progress=lambda *args: progress.append(args))
    crop = values[..., 3:98]
    if method == "polyphase":
        expected = resample_poly(crop, 1, 4, axis=-1)
    elif method == "simple":
        expected = crop[..., ::4]
    else:
        expected = crop[..., :92].reshape(2, 3, 23, 4).mean(axis=-1)
    np.testing.assert_allclose(result.channels["A"], expected, atol=1e-12)
    offset = 1.5 if method == "average" else 0
    np.testing.assert_allclose(result.coords["time"], -2.+(3+offset+4*np.arange(expected.shape[-1]))*.01)
    meta = result.metadata["temporal selection"]["A"]
    assert meta["downsampling method"] == method
    assert meta["anti-alias filtering"] == (method != "simple")
    if method == "average":
        assert meta["discarded trailing samples"] == 3
    assert method in result.history[-1]
    assert progress[-1] == (1, 1, "Mapping")
    reads = [done/total for done, total, stage in progress if stage != "Mapping"]
    assert reads == sorted(reads) and reads[-1] == 1


@pytest.mark.parametrize("method", ["polyphase", "average"])
@pytest.mark.parametrize("digitizer", ["SIS 3301", "SIS crate"])
def test_filtered_lapd_matches_calibrated_reference(tmp_path, digitizer, method):
    from scipy.signal import resample_poly
    path = tmp_path/"filtered.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module(digitizer, {"sn_size": 12, "nt": 257})
    spec = io.inspect_file(path)["channels"][0]
    opts = {k: spec[k] for k in ("digitizer", "config_name", "adc") if spec.get(k)}
    with lapd.File(path, mode="r") as f:
        reference = f.read_data(spec["board"], spec["channel"], **opts)
    crop = reference["signal"][:, 3:254]
    expected = resample_poly(crop, 1, 4, axis=-1) if method == "polyphase" else crop[:, :248].reshape(12, 62, 4).mean(axis=-1)
    result = io.read_lapd(path, {"A": spec}, sample_limits=(3, 253), decimation=4, downsampling=method)
    np.testing.assert_allclose(result.channels["A"], expected, atol=1e-6, rtol=2e-6)
    assert result.channels["A"].dtype == np.float32


def test_large_average_blocks_stay_bounded_and_require_complete_blocks():
    from lapd_explorer.temporal import resample_into
    values = np.arange(101., dtype=float)[None, :]
    selection = sample_slice(101, 1., decimation=41, downsampling="average")
    output = np.empty((1, 2))
    reads = []
    def read(part):
        reads.append(part)
        return values[:, part]
    resample_into(read, output, selection, "average", chunk_samples=13)
    np.testing.assert_array_equal(output, [[20., 61.]])
    assert max(part.stop-part.start for part in reads) <= 13
    with pytest.raises(ValueError, match="at least two"):
        sample_slice(81, 1., decimation=41, downsampling="average")
    with pytest.raises(ValueError, match="Unknown"):
        sample_slice(100, 1., downsampling="bad")


def test_parallel_fir_matches_serial_and_reads_from_caller_thread(monkeypatch):
    import threading
    from lapd_explorer import temporal
    values = np.random.default_rng(7).normal(size=(8, 40003)).astype(np.float32)
    selection = sample_slice(values.shape[-1], 1., decimation=5)
    serial = np.empty((8, temporal.retained_samples(selection)), dtype=np.float32)
    parallel = np.empty_like(serial)
    caller = threading.get_ident()
    filter_threads = set()
    original = temporal.resample_poly
    def filter(*args, **kwargs):
        filter_threads.add(threading.get_ident())
        return original(*args, **kwargs)
    def read(part):
        assert threading.get_ident() == caller
        return values[:, part]
    monkeypatch.setattr(temporal, "resample_poly", filter)
    monkeypatch.setattr(temporal.os, "cpu_count", lambda: 1)
    temporal.resample_into(read, serial, selection, chunk_samples=2048)
    assert filter_threads == {caller}
    filter_threads.clear()
    monkeypatch.setattr(temporal.os, "cpu_count", lambda: 4)
    temporal.resample_into(read, parallel, selection, chunk_samples=2048)
    assert caller not in filter_threads and 1 <= len(filter_threads) <= 4
    np.testing.assert_array_equal(parallel, serial)
