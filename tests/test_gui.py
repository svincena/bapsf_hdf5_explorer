import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lapd-matplotlib")
import numpy as np
import pytest
from PySide6 import QtWidgets as W
from lapd_explorer.app import MainWindow, STYLE, export_mp4, atomic_save
from lapd_explorer.model import Dataset
from lapd_explorer.widgets import ImportDialog


@pytest.fixture(scope="module")
def app():
    a = W.QApplication.instance() or W.QApplication([])
    a.setStyle("Fusion")
    a.setStyleSheet(STYLE)
    yield a


def test_gui_linked_slices_processing_point_line_and_movie(app, tmp_path):
    w = MainWindow()
    w.show()
    w.draw()
    app.processEvents()
    w.mode.setCurrentText("Vector")
    w.draw()
    assert w._values.shape == (25, 31, 160)
    w.cmap.setCurrentText("plasma")
    w.arrow_cmap.setCurrentText("cividis")
    w.draw()
    assert w.main_ax.collections[0].cmap.name == "plasma"
    assert w.main_ax.collections[1].cmap.name == "cividis"
    w.frame.setValue(1)
    w.draw()
    expected = np.hypot(w.data.selected("Bx")[...,1], w.data.selected("By")[...,1])
    np.testing.assert_allclose(w.main_ax.collections[1].get_array(), expected.ravel())
    from types import SimpleNamespace
    w.clicked(SimpleNamespace(inaxes=w.main_ax, xdata=10, ydata=-10))
    assert w.slice_boxes[1].value() == 21
    assert w.slice_boxes[0].value() == 6
    w.frame.setValue(12)
    w.draw()
    assert w.slider.value() == 12
    w.canvas.print_png(str(tmp_path/"plane.png"))
    w.grab().save(str(tmp_path/"window.png"))
    t = np.arange(16)*1e-6
    for dims, a in [(("x", "time"), np.ones((3, 16))), (("time",), np.ones(16))]:
        coords = {"time": t}
        if len(dims) == 2:
            coords["x"] = np.arange(3)
        w.set_data(Dataset({"A": a}, dims, coords))
        w.draw()
        app.processEvents()
    movie = tmp_path/"movie.mp4"
    export_mp4(movie, w.data, w.options(), [0, 1], 5)
    assert movie.stat().st_size > 1000
    w.close()


def test_atomic_save_preserves_existing_destination_on_failure(tmp_path):
    path = tmp_path/"result.h5"
    path.write_bytes(b"original")
    def fail(tmp):
        from pathlib import Path
        Path(tmp).write_bytes(b"partial")
        raise ValueError("canceled")
    with pytest.raises(ValueError):
        atomic_save(path, fail)
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".lapd-*"))


def test_import_dialog_applies_shot_guess_and_keeps_fields_editable(app, monkeypatch):
    info = {
        "channels": [{"digitizer": "D", "config_name": "cfg", "adc": "A",
                      "board": 1, "channel": 2}],
        "controls": [("bmotion", "scan")],
        "datasets": [],
    }
    monkeypatch.setattr(
        "lapd_explorer.widgets.io.guess_shots_per_case",
        lambda *args: {"shots_per_case": 5, "spatial_points": 2601,
                       "spatial_shape": (51, 51), "records": 13005,
                       "position_field": "xyz_target"},
    )
    dialog = ImportDialog("fake.h5", info)
    dialog.start_guess()
    while dialog.guess_worker.isRunning():
        app.processEvents()
    app.processEvents()
    assert dialog.cases.value() == 1
    assert dialog.repeats.value() == 5
    assert dialog.cases.isEnabled() and dialog.repeats.isEnabled()
    assert "2601 spatial points" in dialog.message.text()
    dialog.cases.setValue(3)
    dialog.repeats.setValue(4)
    assert (dialog.cases.value(), dialog.repeats.value()) == (3, 4)
    dialog.guess_timer.stop()
    dialog.reject()


def test_smoothing_dialog_and_processing(app, monkeypatch, tmp_path):
    from lapd_explorer.widgets import SmoothingDialog
    from lapd_explorer.smoothing import DEFAULT_SMOOTHING, smooth_time_series
    dialog = SmoothingDialog(dict(DEFAULT_SMOOTHING), np.arange(40.)/10000)
    dialog.show()
    app.processEvents()
    assert dialog.fields["window_size"].isVisible()
    assert not dialog.fields["cutoff"].isVisible()
    dialog.method.setCurrentText("Butterworth")
    assert dialog.fields["cutoff"].isVisible()
    assert not dialog.fields["window_size"].isVisible()
    dialog.fields["cutoff"].setText("5000")
    dialog.accept()
    assert "Nyquist" in dialog.error_label.text()
    assert dialog.result() != W.QDialog.Accepted
    dialog.method.setCurrentText("Savitzky–Golay")
    dialog.fields["window_size"].setValue(4)
    dialog.accept()
    assert "odd" in dialog.error_label.text()
    dialog.fields["window_size"].setValue(7)
    dialog.accept()
    assert dialog.result() == W.QDialog.Accepted
    w = MainWindow()
    a = np.arange(40.)**2
    data = Dataset({"A": a}, ("time",), {"time": np.arange(40.)})
    w.set_data(data)
    assert not w.smooth.isChecked() and not w.smooth_button.isEnabled()
    w.smooth.setChecked(True)
    assert w.smooth_button.isEnabled()
    w.smoothing_settings.update(method="moving", window_size=3)
    monkeypatch.setattr(w, "launch", lambda fn, done, message: done(fn()))
    w.apply_processing()
    expected = smooth_time_series(a, window_size=3)
    np.testing.assert_allclose(w.data.channels["A"], expected)
    w.apply_processing()
    np.testing.assert_allclose(w.data.channels["A"], expected)
    w.show()
    app.processEvents()
    w.grab().save(str(tmp_path / "smoothing-window.png"))
    w.reset_processing()
    assert not w.smooth.isChecked() and w.data is w.raw
    assert w.smoothing_settings == DEFAULT_SMOOTHING
    w.smooth.setChecked(True)
    w.set_data(data)
    assert not w.smooth.isChecked()
    w.close()


def test_butterworth_type_controls_and_settings(app, monkeypatch):
    from lapd_explorer.widgets import SmoothingDialog
    from lapd_explorer.smoothing import DEFAULT_SMOOTHING
    t = np.arange(100.)/10000
    settings = dict(DEFAULT_SMOOTHING, method="butterworth")
    dialog = SmoothingDialog(settings, t)
    dialog.show()
    app.processEvents()
    assert dialog.fields["butter_type"].isVisible()
    assert not dialog.fields["cutoff_upper"].isVisible()
    dialog.fields["butter_type"].setCurrentText("High-pass")
    assert dialog.settings()["butter_type"] == "highpass"
    assert "cutoff_upper" not in dialog.settings()
    dialog.fields["butter_type"].setCurrentText("Band-pass")
    assert dialog.fields["cutoff_upper"].isVisible()
    assert dialog.form.labelForField(dialog.fields["cutoff"]).text() == "Lower cutoff (Hz)"
    dialog.fields["cutoff_upper"].setText("500")
    dialog.accept()
    assert "lower < upper" in dialog.error_label.text()
    dialog.fields["cutoff_upper"].setText("2e3")
    dialog.accept()
    assert dialog.result() == W.QDialog.Accepted
    settings.update(dialog.settings())
    restored = SmoothingDialog(settings, t)
    assert restored.settings() == dialog.settings()
    restored.method.setCurrentText("Gaussian")
    assert restored.fields["butter_type"].isHidden()
    assert restored.fields["cutoff_upper"].isHidden()
    restored.reject()
    w = MainWindow()
    w.set_data(Dataset({"A": np.ones(100)}, ("time",), {"time": t}))
    w.smooth.setChecked(True)
    w.smoothing_settings = settings
    monkeypatch.setattr(w, "launch", lambda fn, done, message: done(fn()))
    w.apply_processing()
    np.testing.assert_allclose(w.data.channels["A"], 0, atol=1e-12)
    assert "butter_type=bandpass" in w.data.history[-1]
    w.reset_processing()
    assert w.smoothing_settings["butter_type"] == "lowpass"
    w.close()
