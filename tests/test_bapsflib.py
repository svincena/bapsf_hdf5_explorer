"""Exercise the actual installed bapsflib against its synthetic HDF5 builder."""
import numpy as np
import h5py
import pytest
from bapsflib import lapd
from bapsflib._hdf.maps.tests import FauxHDFBuilder
from lapd_explorer.io import channel_name, guess_shots_per_case, infer_shots_from_positions, inspect_file, read_lapd


def forbid_signal_reads(monkeypatch):
    original = h5py.Dataset.__getitem__
    def metadata_only(dataset, key, *args, **kwargs):
        if dataset.ndim == 2 and dataset.dtype.kind in "fiu":
            pytest.fail(f"Shot estimation read signal samples from {dataset.name}")
        return original(dataset, key, *args, **kwargs)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", metadata_only)
    monkeypatch.setattr(lapd.File, "read_data", lambda *args, **kwargs: pytest.fail(
        "Shot estimation must not call the signal reader"))


@pytest.mark.parametrize("adc,channel,field", [
    ("SIS 3302", 8, "Data type 8"),
    ("SIS 3305", 1, "FPGA 1 Data type 1"),
    ("SIS 3305", 5, "FPGA 2 Data type 1"),
])
def test_sis_descriptions_match_board_configuration_and_fpga(tmp_path, monkeypatch, adc, channel, field):
    path = tmp_path/"descriptions.h5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module("SIS crate", {"n_configs": 2, "sn_size": 4, "nt": 16})
        knobs = f.modules["SIS crate"].knobs
        enabled = knobs.active_brdch
        enabled[adc][1, channel-1] = True
        knobs.active_brdch = enabled
        knobs.active_config = ("config01", "config02")
        root = f["Raw data + config/SIS crate"]
        group = f"SIS crate {adc.split()[-1]} configurations"
        root[f"config01/{group}[1]"].attrs[field] = "wrong configuration"
        root[f"config02/{group}[0]"].attrs[field] = "wrong board"
        attrs = root[f"config02/{group}[1]"].attrs
        attrs[field] = np.bytes_("  Bxhf_mov_p25  ")
    forbid_signal_reads(monkeypatch)
    def selected():
        info = inspect_file(path)
        assert not info["error"]
        return next(s for s in info["channels"] if s["adc"] == adc and
                    s["config_name"] == "config02" and s["board"] == 2 and s["channel"] == channel)
    spec = selected()
    assert spec["data_type"] == "Bxhf_mov_p25"
    assert spec["data_type_field"] == field
    assert channel_name(spec, 1) == "C1 · Bxhf_mov_p25"
    for value, expected in (("", f"B2 Ch{channel}"), (" \t\n ", f"B2 Ch{channel}"),
                            ("By_μprobe", "By_μprobe"), (np.array([b"array-label"]), "array-label")):
        with h5py.File(path, "r+") as f:
            f[f"Raw data + config/SIS crate/config02/{group}[1]"].attrs[field] = value
        spec = selected()
        assert channel_name(spec) == expected
    with h5py.File(path, "r+") as f:
        del f[f"Raw data + config/SIS crate/config02/{group}[1]"].attrs[field]
    assert channel_name(selected()) == f"B2 Ch{channel}"


@pytest.mark.parametrize("digitizer", ["SIS 3301", "SIS crate", "LeCroy_scope"])
def test_shot_guess_uses_dataset_shape_without_signals(tmp_path, monkeypatch, digitizer):
    path = tmp_path/"counts.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module(digitizer, {"sn_size": 8, "nt": 16})
    spec = inspect_file(path)["channels"][0]
    forbid_signal_reads(monkeypatch)
    for start, stop, count in [(0, None, 8), (1, 5, 4), (3, 100, 5)]:
        guess = guess_shots_per_case(path, spec, start=start, stop=stop)
        assert guess == {"shots_per_case": count, "spatial_points": 1,
                         "spatial_shape": (), "records": count,
                         "position_field": "none (point assumed)"}
    with pytest.raises(ValueError, match="No records"):
        guess_shots_per_case(path, spec, start=8)


@pytest.mark.parametrize("position_source", ["target", "measured"])
@pytest.mark.parametrize("start,stop,motion_records", [(0, None, 8), (1, 7, 8), (0, None, 4)])
def test_motion_shot_guess_matches_import_without_signals(
        tmp_path, monkeypatch, position_source, start, stop, motion_records):
    path = tmp_path/"motion.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module("SIS 3301", {"sn_size": 8, "nt": 16})
        f.add_module("bmotion", {"sn_size": motion_records})
        positions = f["Raw data + config/bmotion/bmotion_positions"]
        rows = positions[:]
        rows["a0"] = np.repeat([0., 1.], motion_records // 2)
        rows["a1"] = 0.
        positions[:] = rows
    info = inspect_file(path)
    spec, control = info["channels"][0], info["controls"][0]
    reference = read_lapd(path, {"A": spec}, control, start, stop, position_source=position_source)
    if position_source == "target" and start == 1:
        # This slice cuts across the target grid and must still be rejected.
        forbid_signal_reads(monkeypatch)
        with pytest.raises(ValueError, match="complete rectangular grid"):
            guess_shots_per_case(path, spec, control, start, stop, position_source)
        return
    shots, points, shape = infer_shots_from_positions(reference.xyz)
    forbid_signal_reads(monkeypatch)
    guess = guess_shots_per_case(path, spec, control, start, stop, position_source)
    assert guess == {"shots_per_case": shots, "spatial_points": points,
                     "spatial_shape": shape, "records": len(reference.shots),
                     "position_field": "xyz_target" if position_source == "target" else "xyz"}


def test_real_bapsflib_discovery_and_voltage_read(tmp_path):
    path = tmp_path/"acquisition.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module("SIS 3301", {"n_configs": 1, "sn_size": 8, "nt": 16})
    info = inspect_file(path)
    assert info["channels"], info["error"]
    rec = read_lapd(path, {"A": info["channels"][0]}, start=1, stop=5)
    assert rec.channels["A"].shape == (4, 16)
    assert rec.time[1] > 0
    assert np.all(np.isfinite(rec.channels["A"]))
    assert len(np.unique(rec.shots)) == 4


def test_user_sample_when_available():
    import os
    import pytest
    from lapd_explorer.io import map_motion
    path = os.environ.get("LAPD_SAMPLE_FILE")
    if not path:
        pytest.skip("Set LAPD_SAMPLE_FILE to the supplied September 2026 x-line acquisition")
    info = inspect_file(path)
    assert {(s['board'], s['channel']) for s in info['channels']} == {(3,6),(3,7),(3,8)}
    control = next(c for c in info['controls'] if '<Hermes>' in c[1])
    rec = read_lapd(path, {n:s for n,s in zip(['Bx','By','Bz'],info['channels'])}, control)
    data = map_motion(rec, cases=1, repeats=5)
    assert data.dims == ('x','shot','time')
    assert data.shape == (91,5,6144)
    np.testing.assert_allclose(data.coords['x'][[0,-1]], [-20,20])
    np.testing.assert_allclose(np.diff(data.coords['time']), 1e-8)
    np.testing.assert_array_equal(np.sort(data.shot_numbers.ravel()), np.arange(1,456))
    assert rec.metadata['Bx']['position field used'] == 'xyz_target'
