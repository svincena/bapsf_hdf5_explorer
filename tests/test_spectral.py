"""Scientific conventions, geometry adapters, caching, and Qt workflow tests."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lapd-matplotlib")
from dataclasses import replace
import time
import numpy as np
import pytest
from scipy import signal
from lapd_explorer.model import Dataset
from lapd_explorer import spectral as sp


def acquisition(dims=(), shape=(), phase=.6):
    t = np.arange(1024)/1024
    spatial_phase = np.arange(int(np.prod(shape))).reshape(shape)*.2
    a = 2*np.cos(2*np.pi*64*t + spatial_phase[..., None])
    b = 3*np.cos(2*np.pi*64*t + spatial_phase[..., None] + phase)
    data = Dataset({"A": a, "B": b}, dims+("time",),
                   dict(zip(dims, [np.arange(n) for n in shape]), time=t),
                   shot_numbers=np.arange(int(np.prod(shape))).reshape(shape))
    settings = sp.Settings(interval=(t[0], t[-1]), nperseg=128, nfft=256, overlap=64, max_lag=20)
    return data, settings


def test_welch_cross_sign_coherence_and_coherent_amplitude():
    data, s = acquisition()
    result = sp.process(data, ("A", "B"), s)
    kw = dict(fs=1024, window="hann", nperseg=128, nfft=256, noverlap=64)
    f, power = signal.welch(data.channels["A"], **kw)
    _, cross = signal.csd(data.channels["A"], data.channels["B"], **kw)
    np.testing.assert_array_equal(result.frequency, f)
    np.testing.assert_allclose(result.values("Auto-power A"), power)
    np.testing.assert_allclose(result.values("Cross-power"), cross)
    k = result.frequency_index(64.1)
    assert result.frequency[k] == 64
    assert result.values("Cross-phase")[k] == pytest.approx(.6)
    assert result.values("Magnitude-squared coherence")[k] == pytest.approx(1.)
    assert result.arrays["Phasor A"][k] == pytest.approx(2.)
    assert result.arrays["Phasor B"][k] == pytest.approx(3*np.exp(.6j))
    assert result.units("Auto-power A") == "(V)²/Hz"
    assert result.units("Complex coherency") == "dimensionless"
    assert result.units("Cross-phase", degrees=True) == "deg"
    np.testing.assert_allclose(result.displayed("Cross-power", "Real"), cross.real)
    assert result.displayed("Cross-power", "Phase", True)[k] == pytest.approx(np.rad2deg(.6))


@pytest.mark.parametrize("nfft", [1024, 1025])
def test_density_integrates_to_variance_and_dc_nyquist(nfft):
    data, s = acquisition()
    s = replace(s, nperseg=1024, nfft=nfft, overlap=0, window="boxcar", detrend="none")
    rng = np.random.default_rng(5)
    data.channels["A"] = rng.normal(size=1024)
    r = sp.process(data, ("A",), s)
    assert sum(r.values("Auto-power A"))*1024/nfft == pytest.approx(np.mean(data.channels["A"]**2))
    data.channels["A"] = np.full(1024, 3.)
    r = sp.process(data, ("A",), s)
    assert r.arrays["Phasor A"][0] == pytest.approx(3.)
    if nfft == 1024:
        data.channels["A"] = (-1.)**np.arange(1024)*2
        r = sp.process(data, ("A",), s)
        assert r.arrays["Phasor A"][-1] == pytest.approx(2.)


def test_lag_covariance_is_separate_from_frequency():
    data, s = acquisition()
    a = np.zeros(1024)
    a[200:205] = 1
    b = np.roll(a, 7)
    data.channels.update(A=a, B=b)
    r = sp.process(data, ("A", "B"), s)
    assert r.lags[np.argmax(r.values("Cross-covariance"))] == pytest.approx(7/1024)
    zero = s.max_lag
    assert r.values("Covariance A")[zero] == pytest.approx(np.var(a))
    assert r.values("Cross-covariance")[zero] == pytest.approx(np.mean((a-a.mean())*(b-b.mean())))
    assert r.coordinate("Cross-covariance") is r.lags
    assert r.units("Cross-covariance") == "(V)²"
    assert r.frame("Cross-covariance", zero).channels["Cross-covariance"][0] == pytest.approx(r.values("Cross-covariance")[zero])


@pytest.mark.parametrize("dims,shape", [((), ()), (("shot",), (2,)), (("z", "case", "shot"), (3, 2, 2)),
                                       (("y", "x", "case", "shot"), (2, 3, 2, 2))])
def test_geometry_bad_traces_shots_and_point_processing(dims, shape):
    data, s = acquisition(dims, shape)
    calls = []
    r = sp.process(data, ("A", "B"), s, lambda n, total: calls.append((n, total)))
    assert r.dims == dims+("frequency",)
    assert r.coords["frequency"] is r.frequency
    frame = r.frame("Auto-power A", 16)
    assert frame.dims == data.dims and frame.shape == shape+(1,)
    assert frame.shot_numbers is data.shot_numbers
    assert calls[-1] == (int(np.prod(shape)), int(np.prod(shape)))
    for d in dims:
        np.testing.assert_array_equal(frame.coords[d], data.coords[d])
    point = sp.process(data, ("A", "B"), s, index=(0,)*len(shape))
    np.testing.assert_allclose(point.values("Auto-power A"), r.values("Auto-power A")[(0,)*len(shape)], atol=1e-28)
    index = (0,)*len(shape)
    data.channels["A"][index+(slice(0, 4),)] = np.nan
    failed = sp.process(data, ("A", "B"), s)
    assert len(failed.failures) == 1
    assert failed.failures[0][0] == index
    assert np.isnan(failed.values("Cross-power")[index]).all()
    if "shot" in dims:
        averaged = sp.process(data, ("A", "B"), replace(s, average_shots=True))
        assert "shot" not in averaged.source.dims
        assert averaged.source.shot_numbers is None
    with pytest.raises(ValueError, match="canceled"):
        sp.process(data, ("A", "B"), s, canceled=lambda: True)


def test_averaging_waveforms_is_not_averaging_power():
    data, s = acquisition(("shot",), (2,))
    data.channels["A"][1] = -data.channels["A"][0]
    individual = sp.process(data, ("A",), s)
    s.average_shots = True
    averaged = sp.process(data, ("A",), s)
    assert individual.values("Auto-power A").max() > 0
    assert averaged.values("Auto-power A").max() == 0
    point = sp.process(data, ("A",), s, index=(1,))
    np.testing.assert_array_equal(point.values("Auto-power A"), averaged.values("Auto-power A"))


def test_median_and_incoherent_signals():
    data, s = acquisition()
    rng = np.random.default_rng(7)
    data.channels = {n: rng.normal(size=1024) for n in data.channels}
    r = sp.process(data, ("A", "B"), s)
    assert np.mean(r.values("Magnitude-squared coherence")) < .2
    s.average = "median"
    r = sp.process(data, ("A", "B"), s)
    _, expected = signal.csd(data.channels["A"], data.channels["B"], fs=1024,
                             nperseg=128, nfft=256, noverlap=64, average="median", window="hann")
    np.testing.assert_allclose(r.values("Cross-power"), expected)


def test_estimator_failure_isolated_and_input_selection_validation(monkeypatch):
    data, s = acquisition(("shot",), (3,))
    original = sp._estimates
    def fail_one(traces, fs, settings):
        if any(np.isclose(row[0], data.channels["A"][1, 0]) for row in traces[0]):
            raise ValueError("bad location")
        return original(traces, fs, settings)
    monkeypatch.setattr(sp, "_estimates", fail_one)
    r = sp.process(data, ("A", "B"), s)
    assert r.failures == [((1,), "bad location")]
    assert np.isfinite(r.values("Auto-power A")[[0, 2]]).all()
    for names in ((), ("A", "A"), ("A", "B", "C"), ("missing",)):
        with pytest.raises(ValueError, match="Select"):
            sp.process(data, names, s)


def test_animation_preserves_polarization_and_spatial_phase():
    data, s = acquisition(("y", "x"), (2, 3), phase=np.pi/2)
    r = sp.process(data, ("A", "B"), s)
    k = r.frequency_index(64)
    phases = np.arange(6).reshape(2, 3)*.2
    for phase in (0., np.pi/2, np.pi):
        frame = r.frame("Cross-power", k, phase=phase, vector=True)
        np.testing.assert_allclose(frame.channels["A"][..., 0], 2*np.cos(phases+phase), atol=1e-12)
        np.testing.assert_allclose(frame.channels["B"][..., 0], 3*np.cos(phases+phase+np.pi/2), atol=1e-12)
        for amplitude in "AB":
            scalar = r.frame("Cross-power", k, phase=phase, amplitude=amplitude)
            np.testing.assert_allclose(scalar.channels["Cross-power"][..., 0],
                                       np.sqrt(r.values(f"Auto-power {amplitude}")[..., k])*np.cos(phase-np.pi/2), atol=1e-12)
    np.testing.assert_array_equal(r.frequency_indices(60, 72, 2), [15, 17])
    with pytest.raises(ValueError, match="no calculated bins"):
        r.frequency_indices(61, 62)
    with pytest.raises(ValueError, match="outside"):
        r.frequency_index(600)


@pytest.mark.parametrize("changes,match", [({"overlap": 128}, "Overlap"), ({"nperseg": 2000}, "Segment"),
    ({"nfft": 32}, "FFT"), ({"interval": (1., 2.)}, "Interval"), ({"max_lag": 1024}, "lag"),
    ({"average": "bad"}, "averaging"), ({"window": "bad"}, "window"), ({"nfft": 128.5}, "integer")])
def test_invalid_estimator_settings(changes, match):
    data, s = acquisition()
    with pytest.raises(ValueError, match=match):
        sp.process(data, ("A",), replace(s, **changes))


def test_single_channel_zero_power_and_invalid_timebase():
    data, s = acquisition()
    r = sp.process(data, ("A",), s)
    assert r.quantities == sp.SINGLE
    with pytest.raises(ValueError, match="two selected"):
        r.values("Cross-power")
    data.channels = {n: np.zeros_like(a) for n, a in data.channels.items()}
    r = sp.process(data, ("A", "B"), s)
    assert np.isnan(r.values("Complex coherency")).all()
    assert np.isnan(r.values("Cross-phase")).all()
    assert np.isnan(r.displayed("Cross-power", "Phase")).all()
    data.coords["time"][10] += 1e-5
    with pytest.raises(ValueError, match="uniformly"):
        sp.process(data, ("A",), s)


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets as W
    return W.QApplication.instance() or W.QApplication([])


@pytest.mark.parametrize("dims,shape", [((), ()), (("z", "shot"), (3, 2)), (("y", "x", "shot"), (2, 3, 2))])
def test_gui_geometry_cached_views_intervals_and_animation(app, tmp_path, monkeypatch, dims, shape):
    from lapd_explorer.spectral_gui import SpectralDialog
    data, s = acquisition(dims, shape)
    dialog = SpectralDialog(data, ("A", "B"), len(dims) == 3, s, {}, "Light")
    dialog.show()
    app.processEvents()
    result = sp.process(data, ("A", "B"), s)
    dialog.batch_ready(result)
    def no_fft(*args, **kwargs):
        pytest.fail("Display changes must not recalculate spectra")
    monkeypatch.setattr(sp.signal, "welch", no_fft)
    for q in result.quantities:
        dialog.quantity.setCurrentText(q)
        dialog.draw_spectrum()
        dialog.result_canvas.draw()
    dialog.quantity.setCurrentText("Cross-power")
    dialog.bin_slider.setValue(result.frequency_index(64))
    dialog.representation.setCurrentText("Phase")
    dialog.degrees.setChecked(True)
    dialog.interpretation.setCurrentText("Phase projection")
    dialog.amplitude.setCurrentText("A")
    dialog.animation_mode.setCurrentIndex(1)
    dialog.step_frame()
    first_axis = dialog.result_ax
    dialog.step_frame()
    assert dialog.result_ax is first_axis
    assert "phase = 10" in dialog.frame_label.text()
    dialog.bin_slider.setValue(result.frequency_index(80))
    dialog.step_frame()
    assert "f = 80" in dialog.frame_label.text()
    if len(dims) == 3:
        dialog.interpretation.setCurrentText("Vector components")
        dialog.step_frame()
        dialog.step_frame()
        assert len(dialog.result_ax.collections) >= 2
    dialog.animation_mode.setCurrentIndex(0)
    dialog.frequency_start.setValue(60)
    dialog.frequency_stop.setValue(68)
    dialog.frequency_step.setValue(2)
    dialog.loop.setChecked(False)
    dialog.step_frame()
    dialog.step_frame()
    assert "f = 68" in dialog.frame_label.text()
    dialog.advance()
    assert not dialog.timer.isActive()
    assert dialog.batch is result
    settings = replace(s)
    dialog.reset_view()
    assert s == settings
    dialog.update_appearance("Dark")
    dialog.control_tabs.setCurrentIndex(2)
    dialog.grab().save(str(tmp_path / f"spectral-{len(dims)}.png"))
    dialog.editor.first.setValue(20)
    assert dialog.batch is None
    assert s.interval[0] == data.coords["time"][20]
    np.testing.assert_allclose(dialog.trace_view.selectors[0].extents, np.array(s.interval)*1000)
    dialog.reject()


def test_main_selection_worker_persistence_and_preprocessing(app, monkeypatch):
    from lapd_explorer.app import MainWindow
    from lapd_explorer.spectral_gui import SpectralDialog
    from lapd_explorer.model import preprocess
    w = MainWindow()
    data, s = acquisition(("x", "shot"), (3, 2))
    w.set_data(data)
    w.spectral_settings = s
    w.slice_boxes[0].setValue(2)
    w.shot.setValue(1)
    monkeypatch.setattr(SpectralDialog, "exec", lambda self: 0)
    w.open_spectral()
    dialog = w.spectral_dialog
    assert dialog.index() == (2, 1)
    assert dialog.names == ("A", "B") and not dialog.vector
    def wait():
        deadline = time.monotonic()+15
        while not dialog.action_buttons[0].isEnabled() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert dialog.action_buttons[0].isEnabled()
    dialog.process_point()
    wait()
    assert dialog.point is not None
    dialog.process_all()
    wait()
    batch = dialog.batch
    assert batch is not None
    dialog.reject()
    w.open_spectral()
    assert w.spectral_dialog is dialog and dialog.batch is batch
    dialog.reject()
    processed = preprocess(data, average=True, gain=2)
    w.processed(processed)
    w.components[1].setCurrentText("None")
    w.open_spectral()
    new = w.spectral_dialog
    assert new is not dialog and new.data is processed
    assert new.names == ("A",)
    assert not new.quantity.model().item(2).isEnabled()
    assert not new.average_shots.isEnabled()
    new.reject()
    w.close()
