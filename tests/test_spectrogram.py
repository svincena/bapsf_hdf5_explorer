"""Spectrogram normalization, cross-phase, bounded FFTs, and trace-navigation workflow."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lapd-matplotlib")
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import fft, signal
from lapd_explorer.model import Dataset
from lapd_explorer import spectrogram as sg, spectral


def acquisition(dims=(), shape=(), samples=2048):
    t = -.25+np.arange(samples)/1024
    offset = np.arange(int(np.prod(shape))).reshape(shape)*.15
    a = 2*np.cos(2*np.pi*64*t+offset[..., None])
    b = 3*np.cos(2*np.pi*64*t+offset[..., None]+.6)
    data = Dataset({"A": a, "B": b}, dims+("time",), dict(zip(dims, [np.arange(n)*.5 for n in shape]), time=t),
                   shot_numbers=(100+np.arange(int(np.prod(shape)))).reshape(shape))
    settings = sg.Settings(nperseg=128, nfft=256, overlap=64)
    return data, settings


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
@pytest.mark.parametrize("sides", ["onesided", "twosided"])
@pytest.mark.parametrize("nfft", [256, 257])
def test_matches_scipy_periodograms_and_mean_csd(scaling, sides, nfft):
    data, s = acquisition()
    rng = np.random.default_rng(14)
    data.channels = {n: rng.normal(size=data.shape) for n in data.channels}
    s = replace(s, nfft=nfft, scaling=scaling, sides=sides)
    result = sg.process(data, ("A", "B"), s)
    kwargs = dict(fs=1024, nperseg=s.nperseg, nfft=nfft, noverlap=s.overlap, window=s.window,
                  detrend=s.detrend, scaling=scaling, return_onesided=sides == "onesided")
    f, t, power = signal.spectrogram(data.channels["A"], **kwargs)
    _, cross = signal.csd(data.channels["A"], data.channels["B"], **kwargs)
    if sides == "twosided":
        f, power, cross = fft.fftshift(f), fft.fftshift(power, axes=0), fft.fftshift(cross)
    np.testing.assert_array_equal(result.frequency, f)
    np.testing.assert_allclose(result.time, t+data.coords["time"][0])
    np.testing.assert_allclose(result.arrays["Auto-power A"], power, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.arrays["Cross-power"].mean(axis=-1), cross, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("window", sg.WINDOWS)
def test_all_window_options_and_odd_centers_match_short_time_fft(window):
    data, s = acquisition(samples=517)
    param = sg.PARAMETERS.get(window, (None, 0))[1]
    s = replace(s, nperseg=65, nfft=129, overlap=31, window=window, window_parameter=param)
    result = sg.process(data, ("A", "B"), s)
    stft = signal.ShortTimeFFT(sg.window_values(s), s.nperseg-s.overlap, fs=1024, mfft=s.nfft,
                              fft_mode="onesided", scale_to="psd", phase_shift=None)
    a = stft.stft_detrend(data.channels["A"], s.detrend, p0=0, p1=len(result.time), k_offset=s.nperseg//2)
    b = stft.stft_detrend(data.channels["B"], s.detrend, p0=0, p1=len(result.time), k_offset=s.nperseg//2)
    factors = np.r_[1., np.full(s.nfft//2, 2.)][:, None]
    np.testing.assert_allclose(result.arrays["Auto-power A"], abs(a)**2*factors, atol=1e-12)
    np.testing.assert_allclose(result.arrays["Cross-power"], np.conj(a)*b*factors, atol=1e-12)
    np.testing.assert_allclose(result.time, data.coords["time"][0]+(32+np.arange(len(result.time))*34)/1024)


@pytest.mark.parametrize("boundary", ["zeros", "edge", "even", "odd"])
@pytest.mark.parametrize("samples,overlap", [(4099, 61), (66, 0)])
def test_global_boundary_padding_matches_reference_even_across_chunks(boundary, samples, overlap):
    data, s = acquisition(samples=samples)
    data.channels["A"] = np.arange(samples, dtype=float)**2
    s = replace(s, nperseg=64, nfft=64, overlap=overlap, boundary=boundary, pad_end=True, detrend="none")
    result = sg.process(data, ("A",), s)
    # ShortTimeFFT may omit a requested fully exterior final hop; compare the
    # independent padded-window FFT, including exactly the requested centers.
    starts = sg.layout(samples, s)
    left, right = max(0, -starts[0]), max(0, starts[-1]+s.nperseg-samples)
    kw = dict(mode="constant") if boundary == "zeros" else dict(mode="edge") if boundary == "edge" else dict(mode="reflect", reflect_type=boundary)
    padded = np.pad(data.channels["A"], (left, right), **kw)
    windows = np.stack([padded[start+left:start+left+s.nperseg] for start in starts])
    z = fft.rfft(windows*sg.window_values(s), axis=-1)
    expected = (abs(z)**2/(1024*np.sum(sg.window_values(s)**2))).T
    expected[1:-1] *= 2
    np.testing.assert_allclose(result.arrays["Auto-power A"], expected, rtol=1e-12, atol=1e-6)


def test_trace_index_interval_phase_dc_and_nyquist():
    data, s = acquisition(("y", "x", "case", "shot"), (2, 3, 2, 4))
    index = (1, 2, 1, 3)
    s = replace(s, interval=(data.coords["time"][100], data.coords["time"][800]), window="boxcar",
                nperseg=128, nfft=128, overlap=0, detrend="none")
    result = sg.process(data, ("A", "B"), s, index)
    k = np.argmin(abs(result.frequency-64))
    np.testing.assert_allclose(result.displayed("Cross-power", "Phase")[k], .6, atol=1e-12)
    np.testing.assert_allclose(result.arrays["Auto-power A"][k], .25, atol=1e-12)
    assert result.index == index and result.global_shot == int(data.shot_numbers[index])
    assert result.location == {"y": .5, "x": 1., "case": .5, "shot": 1.5}
    assert result.time[0] == pytest.approx(data.coords["time"][164])
    data.channels["A"][index] = 3.
    result = sg.process(data, ("A",), s, index)
    np.testing.assert_allclose(result.arrays["Auto-power A"][0], 9*128/1024)
    data.channels["A"][index] = (-1.)**np.arange(data.shape[-1])
    result = sg.process(data, ("A",), s, index)
    np.testing.assert_allclose(result.arrays["Auto-power A"][-1], 128/1024)
    np.testing.assert_allclose(result.arrays["Auto-power A"].sum(axis=0)*8, 1.)


def test_chirp_ridge_moves_in_time():
    data, s = acquisition()
    t = np.arange(data.shape[-1])/1024
    data.channels["A"] = signal.chirp(t, f0=30, t1=t[-1], f1=250)
    result = sg.process(data, ("A",), replace(s, nperseg=256, nfft=512, overlap=224, detrend="none"))
    ridge = result.frequency[np.argmax(result.arrays["Auto-power A"], axis=0)]
    expected = 30+220*(result.time-data.coords["time"][0])/t[-1]
    np.testing.assert_allclose(ridge, expected, atol=3.)


@pytest.mark.parametrize("changes", [dict(nperseg=3), dict(nfft=32), dict(overlap=128), dict(window="invalid"),
    dict(window="tukey", window_parameter=2), dict(window="kaiser", window_parameter=-1), dict(fft_workers=0),
    dict(overlap=1.5), dict(sides="bad"), dict(boundary="bad"), dict(scaling="bad"), dict(detrend="bad")])
def test_invalid_options_rejected(changes):
    data, s = acquisition()
    with pytest.raises(ValueError):
        sg.process(data, ("A",), replace(s, **changes))


def test_bad_trace_nonfinite_memory_budget_cancellation_and_fft_workers(monkeypatch):
    data, s = acquisition(("shot",), (2,), samples=65536)
    with pytest.raises(ValueError, match="index"):
        sg.process(data, ("A",), s, (2,))
    with pytest.raises(ValueError, match="distinct"):
        sg.process(data, ("A", "A"), s, (0,))
    with pytest.raises(ValueError, match="MiB"):
        sg.process(data, ("A",), s, (0,), max_output_bytes=100)
    data.channels["A"][0, 30] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        sg.process(data, ("A",), s, (0,))
    seen, progress = [], []
    original = sg.fft.rfft
    def transform(values, *args, **kwargs):
        seen.append((values.size, fft.get_workers()))
        return original(values, *args, **kwargs)
    monkeypatch.setattr(sg.fft, "rfft", transform)
    with fft.set_workers(3):
        with pytest.raises(ValueError, match="canceled"):
            sg.process(data, ("A",), replace(s, fft_workers=2), (1,),
                       progress=lambda n, total: progress.append(n), canceled=lambda: any(n>0 for n in progress))
        assert fft.get_workers() == 3
        serial = sg.process(data, ("A",), replace(s, fft_workers=1), (1,))
        parallel = sg.process(data, ("A",), replace(s, fft_workers=2), (1,))
        assert fft.get_workers() == 3
    assert seen and max(size for size, _ in seen) <= 1_000_000
    assert {workers for _, workers in seen} == {1, 2}
    np.testing.assert_allclose(serial.arrays["Auto-power A"], parallel.arrays["Auto-power A"], atol=1e-12)


def test_display_units_db_reference_and_lossless_export(tmp_path):
    data, s = acquisition()
    result = sg.process(data, ("A", "B"), s)
    values, units = sg.color_values(result, "Auto-power A", scale="dB", reference=2, dynamic_range=60)
    k = np.argmin(abs(result.frequency-64))
    np.testing.assert_allclose(values[k], 10*np.log10(result.arrays["Auto-power A"][k]/2))
    assert "dB re 2 (V)²/Hz" == units
    assert values.max()-values.min() == pytest.approx(60)
    assert result.value_units("Cross-power", "Phase", True) == "deg"
    with pytest.raises(ValueError, match="nonnegative"):
        sg.color_values(result, "Cross-power", "Real", scale="dB")
    path = tmp_path/"result.npz"
    sg.save(path, result)
    with np.load(path, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["Cross_power"], result.arrays["Cross-power"])
        np.testing.assert_array_equal(f["time"], result.time)
        metadata = json.loads(str(f["metadata"]))
        assert metadata["global_shot"] == 100
        assert metadata["settings"]["scaling"] == "density"
        assert metadata["names"] == ["A", "B"]


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_trace_picker_coordinates_indices_shots_cycles_and_interactive_map(app):
    from lapd_explorer.spectral_gui import SpectralDialog
    data, s = acquisition(("y", "x", "case", "shot"), (2, 3, 2, 4))
    dialog = SpectralDialog(data, ("A", "B"), False, spectral.Settings(), {}, "Light")
    dialog.show()
    app.processEvents()
    picker = dialog.trace_picker
    picker.coordinates["y"].setValue(.45)
    picker.coordinates["x"].setValue(.8)
    picker.choose_coordinates()
    assert dialog.index() == (1, 2, 0, 0)
    picker.global_shot.setText("107")
    picker.choose_global_shot()
    assert dialog.index() == (0, 0, 1, 3)
    picker.cycle_axis.setCurrentText("Shot")
    picker.cycle(1)
    assert dialog.index() == (0, 0, 1, 0)
    picker.cycle_axis.setCurrentText("Location")
    picker.cycle(1)
    assert dialog.index() == (0, 1, 1, 0)
    picker.cycle_axis.setCurrentText("Case")
    picker.cycle(1)
    assert dialog.index() == (0, 1, 0, 0)
    picker.flat_index.setValue(47)
    assert dialog.index() == (1, 2, 1, 3)
    picker.wrap.setChecked(False)
    picker.cycle_axis.setCurrentText("Trace (all indices)")
    picker.cycle(1)
    assert picker.flat_index.value() == 47
    dialog.location_clicked(SimpleNamespace(inaxes=dialog.location_ax, xdata=.05, ydata=.49))
    assert dialog.index() == (1, 0, 1, 3)
    np.testing.assert_allclose(dialog.trace_view.lines[0].get_ydata(), data.channels["A"][1, 0, 1, 3])
    dialog.reject()


def test_gui_worker_cached_displays_live_cycling_and_controls(app, tmp_path, monkeypatch):
    from lapd_explorer.spectral_gui import SpectralDialog
    data, s = acquisition(("x", "case", "shot"), (2, 2, 2))
    dialog = SpectralDialog(data, ("A", "B"), False, spectral.Settings(), {}, "Light")
    dialog.show()
    app.processEvents()
    controls = dialog.spectrogram_controls
    controls.nperseg.setValue(128)
    controls.overlap.setValue(64)
    assert controls.duration.value() == 125
    controls.overlap_percent.setValue(75)
    controls.percent_changed()
    assert controls.overlap.value() == 96
    controls.duration.setValue(250)
    controls.duration_changed()
    assert controls.nperseg.value() == 256
    controls.window.setCurrentText("kaiser")
    assert not controls.window_parameter.isHidden() and controls.window_parameter.value() == 8.6
    observed = []
    main_thread = threading.get_ident()
    original = sg.process
    def process(*args, **kwargs):
        observed.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(sg, "process", process)
    def wait():
        deadline = time.monotonic()+10
        while (dialog.worker and dialog.worker.isRunning() or dialog.spectrogram_timer.isActive() or not dialog.spectrogram_button.isEnabled()) and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.005)
        app.processEvents()
        assert dialog.spectrogram_result is not None, dialog.message.text()
    dialog.process_spectrogram()
    wait()
    first = dialog.spectrogram_result
    assert observed and main_thread not in observed
    assert dialog.tabs.currentIndex() == dialog.spectrogram_tab
    original_limits = controls.last_axis.get_xlim()
    controls.last_axis.set_xlim(.1, .2)
    dialog.reset_view()
    np.testing.assert_allclose(controls.last_axis.get_xlim(), original_limits)
    assert dialog.spectrogram_result is first
    for kind in ("Phase", "Real", "Imaginary", "Magnitude"):
        controls.representation.setCurrentText(kind)
        assert dialog.spectrogram_result is first
    controls.cmap.setCurrentText("twilight_r")
    assert controls.last_image.cmap.name == "twilight_r"
    assert dialog.spectrogram_result is first
    assert len(observed) == 1
    controls.quantity.setCurrentText("Auto-power A")
    controls.color_scale.setCurrentText("Logarithmic")
    controls.log_frequency.setChecked(True)
    dialog.spectrogram_canvas.draw()
    assert controls.last_axis.get_yscale() == "log"
    controls.last_axis.set_xlim(.2, .4)
    controls.last_axis.set_ylim(30., 200.)
    dialog.trace_picker.cycle_axis.setCurrentText("Shot")
    dialog.trace_picker.next.click()
    wait()
    assert dialog.spectrogram_result.index == (0, 0, 1)
    assert len(observed) == 2
    np.testing.assert_allclose(controls.last_axis.get_xlim(), (.2, .4))
    np.testing.assert_allclose(controls.last_axis.get_ylim(), (30., 200.))
    controls.auto_update.setChecked(False)
    dialog.trace_picker.next.click()
    app.processEvents()
    assert dialog.spectrogram_result is None and not dialog.spectrogram_timer.isActive()
    dialog.process_spectrogram()
    wait()
    dialog.update_appearance("Dark")
    dialog.grab().save(str(tmp_path/"spectrogram-dark.png"))
    assert controls.last_axis.get_facecolor() != (1., 1., 1., 1.)
    dialog.reject()


def test_db_extreme_reference_stays_finite():
    data, s = acquisition()
    data.channels["A"] *= 1e100
    result = sg.process(data, ("A",), s)
    values, _ = sg.color_values(result, "Auto-power A", scale="dB", reference=1e-300)
    assert np.isfinite(values).all() and values.max() > 4000


@pytest.mark.parametrize("dims,shape", [((), ()), (("z", "shot"), (1, 2)), (("y", "x", "shot"), (1, 2, 2))])
def test_single_channel_and_single_window_gui_geometries(app, dims, shape):
    from lapd_explorer.spectral_gui import SpectralDialog
    data, s = acquisition(dims, shape, samples=128)
    data.shot_numbers = None
    dialog = SpectralDialog(data, ("A",), False, spectral.Settings(nperseg=128, nfft=128, overlap=64, max_lag=20), {}, "Light")
    dialog.show()
    app.processEvents()
    controls = dialog.spectrogram_controls
    settings = controls.settings(dialog.editor.interval())
    result = sg.process(data, ("A",), settings, dialog.index())
    dialog.spectrogram_ready(result)
    assert len(result.time) == 1
    assert controls.quantity.count() == 1
    assert not controls.representation.isEnabled()
    assert controls.last_image.get_array().shape == result.arrays["Auto-power A"].shape
    if dims:
        assert hasattr(dialog, "location_ax")
    dialog.reject()


def test_zero_power_phase_is_masked_and_map_stays_open_when_cycling(app):
    from lapd_explorer.spectral_gui import SpectralDialog
    data, s = acquisition(("x", "shot"), (2, 2))
    data.channels = {n: np.zeros_like(a) for n, a in data.channels.items()}
    dialog = SpectralDialog(data, ("A", "B"), False, spectral.Settings(), {}, "Light")
    dialog.show()
    app.processEvents()
    result = sg.process(data, ("A", "B"), dialog.spectrogram_controls.settings(dialog.editor.interval()), dialog.index())
    dialog.spectrogram_ready(result)
    dialog.spectrogram_controls.representation.setCurrentText("Phase")
    assert np.ma.getmaskarray(dialog.spectrogram_controls.last_image.get_array()).all()
    dialog.spectrogram_controls.representation.setCurrentText("Magnitude")
    dialog.spectrogram_controls.color_scale.setCurrentText("Logarithmic")
    dialog.spectrogram_canvas.draw()
    dialog.tabs.setCurrentIndex(4)
    dialog.control_tabs.setCurrentIndex(3)
    dialog._spectrogram_requested = True
    dialog.trace_picker.cycle_axis.setCurrentText("Location")
    dialog.trace_picker.next.click()
    deadline = time.monotonic()+10
    while (dialog.spectrogram_timer.isActive() or dialog.worker and dialog.worker.isRunning() or not dialog.spectrogram_button.isEnabled()) and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert dialog.spectrogram_result.index == (1, 0)
    assert dialog.tabs.currentIndex() == 4 and dialog.control_tabs.currentIndex() == 3
    assert dialog.trace_picker.next.isVisible()
    dialog.reject()
