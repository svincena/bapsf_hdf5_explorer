"""Physical pipeline regressions, using analytic signals rather than FFT replicas."""
from dataclasses import replace
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from lapd_explorer.model import Dataset, preprocess
from lapd_explorer import spectral as sp, spectrogram as sg


def paired_shots():
    t = np.arange(2048)/1024
    wave = np.cos(2*np.pi*64*t)
    zero = np.zeros_like(wave)
    data = Dataset({"A": np.stack((wave, zero)), "B": np.stack((zero, wave))},
                   ("shot", "time"), {"shot": np.arange(2), "time": t},
                   shot_numbers=np.array([101, 102]))
    settings = sp.Settings(interval=(t[0], t[-1]), nperseg=256, nfft=256,
                           overlap=128, fft_workers=1)
    return data, settings


def test_ensemble_has_no_cross_shot_terms_and_preserves_fluctuation_power():
    data, settings = paired_shots()
    result = sp.process(data, ("A", "B"), replace(settings, average_shots=True))
    k = result.frequency_index(64)
    assert result.values("Auto-power A")[k] > 0
    assert result.values("Auto-power B")[k] > 0
    assert result.values("Cross-power")[k] == 0
    assert result.values("Magnitude-squared coherence")[k] == 0
    assert result.valid_shots == 2
    data.channels["A"][1] = -data.channels["A"][0]
    individual = sp.process(data, ("A",), settings)
    ensemble = sp.process(data, ("A",), replace(settings, average_shots=True))
    np.testing.assert_allclose(ensemble.values("Auto-power A"), individual.values("Auto-power A").mean(axis=0))
    assert ensemble.values("Auto-power A")[k] > 0


def test_ensemble_uses_joint_valid_support_in_any_axis_order_and_point_mode():
    data, settings = paired_shots()
    data = Dataset({n: np.broadcast_to(a[:, None, None, :], (2, 3, 2, 2048)).copy()
                    for n, a in data.channels.items()}, ("shot", "x", "case", "time"),
                   dict(data.coords, x=np.arange(3), case=np.arange(2)))
    data.channels["A"][0, 1, 0, 0] = np.nan
    s = replace(settings, average_shots=True)
    result = sp.process(data, ("A", "B"), s)
    point = sp.process(data, ("A", "B"), s, index=(0, 1, 0))
    assert result.source.dims == ("x", "case", "time")
    assert result.valid_shots[1, 0] == point.valid_shots == 1
    assert result.valid_shots[0, 0] == 2
    assert result.failures[0][0] == (0, 1, 0)
    for name in result.arrays:
        np.testing.assert_allclose(point.arrays[name], result.arrays[name][1, 0], atol=1e-14, equal_nan=True)


@pytest.mark.parametrize("kind", ["shots", "positions", "legacy"])
def test_mixed_waveforms_rejected_and_spatial_shot_provenance_removed(kind, tmp_path):
    from lapd_explorer import io
    data, settings = paired_shots()
    if kind == "shots":
        mixed = preprocess(data, average=True)
    elif kind == "positions":
        data = Dataset({n: a[None, ...] for n, a in data.channels.items()},
                       ("y", "x", "time"), {"y": np.array([0.]), "x": np.arange(2), "time": data.coords["time"]},
                       shot_numbers=np.array([[101, 102]]))
        mixed = preprocess(data, spatial_averaging={"method": "box", "window_size": (1, 3)})
    else:
        mixed = replace(data, history=("Spatial averaging: box",))
    if kind != "legacy":
        assert mixed.shot_numbers is None
        assert kind in mixed.metadata["waveform_mixing"]
        path = tmp_path / "mixed.h5"
        io.save_dataset(path, mixed)
        mixed = io.load_dataset(path)
    with pytest.raises(ValueError, match="unmixed"):
        sp.process(mixed, ("A", "B"), settings)
    with pytest.raises(ValueError, match="unmixed"):
        sg.process(mixed, ("A", "B"), sg.Settings(), (0,)*(len(mixed.dims)-1))


def test_single_estimate_and_median_coherence_masked_but_ensemble_supported():
    data, settings = paired_shots()
    rng = np.random.default_rng(23)
    data.channels = {n: rng.normal(size=(2, 2048)) for n in data.channels}
    single = replace(settings, nperseg=2048, nfft=2048, overlap=0)
    result = sp.process(data, ("A", "B"), single)
    assert np.isnan(result.values("Magnitude-squared coherence")).all()
    assert np.isnan(result.values("Cross-phase")).all()
    assert np.isnan(result.displayed("Cross-power", "Phase")).all()
    ensemble = sp.process(data, ("A", "B"), replace(single, average_shots=True))
    assert np.isfinite(ensemble.values("Magnitude-squared coherence")).all()
    median = sp.process(data, ("A", "B"), replace(settings, average="median"))
    assert np.isfinite(median.values("Cross-power")).all()
    assert np.isnan(median.values("Complex coherency")).all()


def test_effective_averages_account_for_overlap_and_phase_noise_is_masked():
    data, settings = paired_shots()
    rng = np.random.default_rng(75)
    data.channels = {n: rng.normal(size=(2, 2048)) for n in data.channels}
    result = sp.process(data, ("A", "B"), settings)
    count, effective, _ = result.support
    assert 1 < effective < count
    assert np.isnan(result.values("Cross-phase")).mean() > .85
    count, effective, _ = sp.segment_support(2048, replace(settings, overlap=0))
    assert effective == count


def test_component_projections_preserve_quadrature_and_require_physical_reference():
    data, settings = paired_shots()
    t = data.coords["time"]
    data = Dataset({"A": 2*np.cos(2*np.pi*64*t), "B": 3*np.cos(2*np.pi*64*t+np.pi/2)},
                   ("time",), {"time": t})
    unconfirmed = sp.process(data, ("A", "B"), settings)
    k = unconfirmed.frequency_index(64)
    with pytest.raises(ValueError, match="common phase reference"):
        unconfirmed.frame("Cross-power", k, phase=0., amplitude="A")
    with pytest.raises(ValueError, match="common phase reference"):
        unconfirmed.frame("Cross-power", k, phase=0., vector=True)
    result = sp.process(data, ("A", "B"), replace(settings, phase_reference_confirmed=True))
    for phase in (0., .5, np.pi/2):
        a = result.frame("Cross-power", k, phase=phase, amplitude="A")
        b = result.frame("Cross-power", k, phase=phase, amplitude="B")
        assert a.channels["Cross-power"][0] == pytest.approx(2*np.cos(phase), abs=1e-12)
        assert b.channels["Cross-power"][0] == pytest.approx(3*np.cos(phase+np.pi/2), abs=1e-12)
        assert a.units == b.units == "V"


def test_adaptive_starting_window_resolves_documented_sample_tone():
    fs = 100e6
    t = np.arange(6144)/fs
    data = Dataset({"A": np.cos(2*np.pi*100e3*t)}, ("time",), {"time": t})
    n = sp.default_segment_samples(len(t))
    settings = sp.Settings(interval=(t[0], t[-1]), nperseg=n, nfft=n, overlap=n//2, fft_workers=1)
    result = sp.process(data, ("A",), settings)
    power = result.values("Auto-power A")
    assert result.frequency[np.argmax(power)] == pytest.approx(100e3, abs=fs/n)
    assert power.sum()*fs/n == pytest.approx(.5, rel=.01)


def test_db_conversion_retains_weak_peak_despite_strong_out_of_band_tone():
    t = np.arange(2048)/1024
    a = 1000*np.cos(2*np.pi*16*t)+.1*np.cos(2*np.pi*200*t)
    data = Dataset({"A": a}, ("time",), {"time": t})
    result = sg.process(data, ("A",), sg.Settings(nperseg=1024, nfft=1024, overlap=512,
                                                window="boxcar", detrend="none", fft_workers=1))
    shown, _ = sg.color_values(result, "Auto-power A", scale="dB", dynamic_range=60)
    assert shown[200, 0] == pytest.approx(-23.0102999566)
    assert np.ptp(shown[180:221]) > 60


def test_quick_vector_power_sums_components_without_nonlinear_norm():
    from matplotlib.figure import Figure
    from lapd_explorer.plotting import draw_psd
    t = np.arange(2048)/1024
    a, b = 2*np.cos(2*np.pi*64*t), 2*np.sin(2*np.pi*64*t)
    data = Dataset({"A": a, "B": b}, ("time",), {"time": t})
    ax = Figure().subplots()
    draw_psd(ax, np.hypot(a, b), data, components=[a, b], label="Sum of component PSDs")
    frequency, power = ax.lines[0].get_data()
    assert frequency[np.argmax(power)] == pytest.approx(.064)
    assert power.sum()*(frequency[1]-frequency[0])*1000 == pytest.approx(4.)
    assert ax.get_title() == "Sum of component PSDs"


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_ensemble_gui_preview_and_spectrogram_keep_the_selected_shot(app):
    from lapd_explorer.spectral_gui import SpectralDialog
    data, settings = paired_shots()
    dialog = SpectralDialog(data, ("A", "B"), False, settings, {}, "Light")
    dialog.average_shots.setChecked(True)
    np.testing.assert_array_equal(dialog.trace_view.lines[0].get_ydata(), data.channels["A"][0])
    dialog.trace_picker.cycle_axis.setCurrentText("Shot")
    dialog.trace_picker.next.click()
    np.testing.assert_array_equal(dialog.trace_view.lines[0].get_ydata(), data.channels["A"][1])
    spectrogram = sg.process(data, ("A", "B"), dialog.spectrogram_controls.settings(dialog.editor.interval()), dialog.index())
    assert np.all(spectrogram.arrays["Auto-power A"] == 0)
    assert spectrogram.global_shot == 102
    ensemble = sp.process(data, ("A", "B"), dialog.collect(), index=dialog.index())
    assert ensemble.values("Auto-power A").max() > 0
    assert np.all(ensemble.values("Cross-power") == 0)
    assert not dialog.amplitude.model().item(1).isEnabled()
    assert not dialog.interpretation.model().item(2).isEnabled()
    dialog.reject()


def test_gui_db_limits_use_visible_band_and_export_keeps_acquisition_metadata(app, tmp_path):
    import json
    from matplotlib.figure import Figure
    from lapd_explorer.spectrogram_gui import Controls
    t = np.arange(2048)/1024
    data = Dataset({"A": 1000*np.cos(2*np.pi*16*t)+.1*np.cos(2*np.pi*200*t)},
                   ("time",), {"time": t}, metadata={"A": {"temporal selection": {"downsampling method": "polyphase"}}})
    result = sg.process(data, ("A",), sg.Settings(nperseg=1024, nfft=1024, overlap=512,
                                                window="boxcar", detrend="none", fft_workers=1))
    controls = Controls(1024, len(t), ("A",))
    controls.frequency_min.setValue(180)
    controls.frequency_max.setValue(220)
    controls.dynamic_range.setValue(60)
    controls.render(result, Figure(), "Light")
    assert controls.last_image.norm.vmax == pytest.approx(-23.0102999566)
    assert controls.last_image.norm.vmin == pytest.approx(-83.0102999566)
    path = tmp_path / "spectrogram.npz"
    sg.save(path, result)
    with np.load(path, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata"]))["source_metadata"] == data.metadata
    controls.close()
