"""Exercise the actual installed bapsflib against its synthetic HDF5 builder."""
import numpy as np
from bapsflib._hdf.maps.tests import FauxHDFBuilder
from lapd_explorer.io import inspect_file, read_lapd


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
