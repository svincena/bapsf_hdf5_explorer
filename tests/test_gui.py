import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lapd-matplotlib")
import numpy as np
import pytest
from PySide6 import QtGui as G, QtWidgets as W
from lapd_explorer.app import MainWindow, STYLE, export_mp4, atomic_save
from lapd_explorer.model import Dataset
from lapd_explorer.widgets import ImportDialog, ScientificDoubleSpinBox


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
    assert w.facility_logo.variant == "Color"
    assert w.facility_logo.height() == 56
    w.appearance.setCurrentText("Dark")
    assert w.facility_logo.variant == "White"
    w.appearance.setCurrentText("Light")
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


def test_scientific_double_spin_box_accepts_exponents(app):
    box = ScientificDoubleSpinBox()
    box.setDecimals(9)
    box.setRange(-1e12, 1e12)
    acceptable, _, _ = box.validate("1.4e5", 5)
    intermediate, _, _ = box.validate("1.4e", 4)
    assert acceptable == G.QValidator.Acceptable
    assert intermediate == G.QValidator.Intermediate
    box.lineEdit().setText("1.4e5")
    box.interpretText()
    assert box.value() == 1.4e5


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
    assert dialog.facility_logo.variant == "Black"
    assert dialog.facility_logo.height() == 46
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


def test_plane_slice_axis_modes_and_navigation(app, tmp_path):
    from lapd_explorer import plotting
    w = MainWindow()
    values = np.arange(24.).reshape(2, 3, 4)
    values[1, 2, 3] = 1000  # Outside both initially selected slices.
    data = Dataset({"A": values}, ("y", "x", "time"),
                   {"y": np.arange(2), "x": np.arange(3), "time": np.arange(4)})
    w.set_data(data)
    for box in w.slice_boxes:
        box.setValue(0)
    w.show()
    w.draw()
    app.processEvents()
    horizontal, vertical = w.fig._lapd_slice_axes
    np.testing.assert_allclose(horizontal.get_ylim(), (0, 11))
    np.testing.assert_allclose(vertical.get_ylim(), (0, 15))
    assert "x slice" in w.slice_axis_controls[0].title()
    assert "y slice" in w.slice_axis_controls[1].title()
    w.frame.setValue(3)
    w.draw()
    assert w.fig._lapd_slice_axes[0] is horizontal
    np.testing.assert_allclose(horizontal.get_ylim(), (0, 11))
    np.testing.assert_allclose(vertical.get_ylim(), (0, 15))
    w.slice_boxes[0].setValue(1)
    w.draw()
    np.testing.assert_allclose(w.fig._lapd_slice_axes[0].get_ylim(), (12, 1000))

    control = w.slice_axis_controls[0]
    control.mode.setCurrentText("Manual")
    control.minimum.setText("-2e1")
    control.maximum.setText("30")
    control.accept_limits()
    w.draw()
    np.testing.assert_allclose(w.fig._lapd_slice_axes[0].get_ylim(), (-20, 30))
    np.testing.assert_allclose(w.fig._lapd_slice_axes[1].get_ylim(), (0, 15))
    control.minimum.setText("nan")
    control.accept_limits()
    w.frame.setValue(0)
    w.draw()
    assert "Previous limits remain active" in control.message.text()
    np.testing.assert_allclose(w.fig._lapd_slice_axes[0].get_ylim(), (-20, 30))
    control.minimum.setText("40")
    control.accept_limits()
    assert control.settings()["limits"] == (-20, 30)

    control.mode.setCurrentText("Interactive")
    w.draw()
    horizontal, vertical = w.fig._lapd_slice_axes
    assert horizontal.get_navigate() and not vertical.get_navigate()
    horizontal.set_ylim(-5, 5)
    horizontal.set_xlim(.25, 1.75)
    w.toolbar.push_current()
    w.toolbar.back()
    np.testing.assert_allclose(horizontal.get_ylim(), (-20, 30))
    w.toolbar.forward()
    np.testing.assert_allclose(horizontal.get_ylim(), (-5, 5))
    w.frame.setValue(2)
    w.draw()
    assert w.fig._lapd_slice_axes[0] is horizontal
    np.testing.assert_allclose(horizontal.get_ylim(), (-5, 5))
    w.cmap.setCurrentText("plasma")
    w.draw()
    horizontal, vertical = w.fig._lapd_slice_axes
    np.testing.assert_allclose(horizontal.get_ylim(), (-5, 5))
    np.testing.assert_allclose(horizontal.get_xlim(), (.25, 1.75))

    # The standalone movie renderer gets the same interactive/manual bounds.
    other = w.slice_axis_controls[1]
    other.mode.setCurrentText("Manual")
    other.minimum.setText("-10")
    other.maximum.setText("20")
    other.accept_limits()
    w.draw()
    horizontal, vertical = w.fig._lapd_slice_axes
    exported = plotting.figure()
    plotting.render(exported, data, w.options())
    exported._lapd_update_frame(3)
    np.testing.assert_allclose(exported._lapd_slice_axes[0].get_ylim(), (-5, 5))
    np.testing.assert_allclose(exported._lapd_slice_axes[1].get_ylim(), (-10, 20))
    exported.clear()
    w.toolbar.home()
    np.testing.assert_allclose(horizontal.get_ylim(), (12, 1000))
    np.testing.assert_allclose(vertical.get_ylim(), (-10, 20))
    w.canvas.draw()
    w.grab().save(str(tmp_path / "slice-axis-controls.png"))
    w.set_data(Dataset({"A": np.ones(4)}, ("time",), {"time": np.arange(4)}))
    w.draw()
    assert w.slice_axis_panel.isHidden()
    assert not w.fig._lapd_slice_axes
    w.close()


def test_slice_limits_constant_and_missing():
    from lapd_explorer.plotting import finite_limits
    assert finite_limits(np.array([np.nan, np.inf])) == (0, 1)
    low, high = finite_limits(np.full((3, 4), 2.))
    assert low < 2 < high
    assert finite_limits(np.array([np.nan, -7, 4, np.inf])) == (-7, 4)


def test_appearance_switch_preserves_data_and_slice_limits(app, tmp_path):
    from matplotlib.colors import to_rgba
    from PySide6 import QtGui as G
    from lapd_explorer.appearance import colors
    from lapd_explorer.widgets import SmoothingDialog
    w = MainWindow()
    assert w.appearance.currentText() == "Light"
    w.show()
    w.mode.setCurrentText("Vector")
    w.draw()
    data = w.data
    control = w.slice_axis_controls[0]
    control.mode.setCurrentText("Manual")
    control.minimum.setText("-2")
    control.maximum.setText("2")
    control.accept_limits()
    for appearance in ("Dark", "Light"):
        w.appearance.setCurrentText(appearance)
        w.draw()
        app.processEvents()
        w.canvas.draw()
        theme = colors(appearance)
        assert w.data is data
        np.testing.assert_allclose(w.fig.get_facecolor(), to_rgba(theme["bg"]))
        np.testing.assert_allclose(w.main_ax.get_facecolor(), to_rgba(theme["panel"]))
        assert w.toolbar.palette().color(G.QPalette.Window).name() == theme["bg"]
        assert w.main_ax.xaxis.label.get_color() == theme["muted"]
        assert w.main_ax.yaxis.get_offset_text().get_color() == theme["muted"]
        np.testing.assert_allclose(w.fig._lapd_slice_axes[0].get_ylim(), (-2, 2))
        assert w.options()["appearance"] == appearance
        w.grab().save(str(tmp_path / f"plane-{appearance}.png"))
        dialog = SmoothingDialog(w.smoothing_settings, w.data.coords["time"], w)
        dialog.show()
        app.processEvents()
        assert dialog.palette().color(G.QPalette.WindowText).name() == theme["fg"]
        dialog.grab().save(str(tmp_path / f"dialog-{appearance}.png"))
        dialog.reject()
    w.close()


@pytest.mark.parametrize("appearance", ["Light", "Dark"])
def test_appearance_rendering_and_exports(app, tmp_path, appearance, monkeypatch):
    from matplotlib.colors import to_rgba
    from lapd_explorer.appearance import colors
    from lapd_explorer import plotting
    w = MainWindow()
    w.appearance.setCurrentText(appearance)
    t = np.arange(64)*1e-6
    wave = np.sin(np.arange(64)*.3)*1e-7
    for spatial in (True, False):
        data = (Dataset({"A": np.tile(wave, (4, 1))}, ("z", "time"),
                        {"z": np.arange(4), "time": t}) if spatial else
                Dataset({"A": wave, "B": wave*2}, ("time",), {"time": t}))
        w.set_data(data)
        w.show()
        w.draw()
        w.canvas.draw()
        w.grab().save(str(tmp_path / f"{'line' if spatial else 'point'}-{appearance}.png"))
        for ax in w.fig.axes:
            assert ax.xaxis.get_offset_text().get_color() == colors(appearance)["muted"]
        # Standalone renderer must not depend on the live QApplication palette.
        opts = w.options()
        fig = plotting.figure()
        plotting.render(fig, data, opts)
        fig._lapd_update_frame(1)
        np.testing.assert_allclose(fig.get_facecolor(), to_rgba(colors(appearance)["bg"]))
        fig.clear()
    saved_colors = []
    def capture_frame(writer, **kwargs):
        saved_colors.append(kwargs["facecolor"])
    monkeypatch.setattr("matplotlib.animation.FFMpegWriter.grab_frame", capture_frame)
    export_mp4(tmp_path / f"theme-{appearance}.mp4", data, opts, [0, 1], 5)
    for color in saved_colors:
        np.testing.assert_allclose(color, to_rgba(colors(appearance)["bg"]))
    assert len(saved_colors) == 2
    w.close()


def test_theme_text_and_trace_contrast():
    from matplotlib.colors import to_rgb
    from lapd_explorer.appearance import THEMES
    def luminance(color):
        rgb = np.array(to_rgb(color))
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        return linear @ np.array([.2126, .7152, .0722])
    def contrast(a, b):
        lo, hi = sorted((luminance(a), luminance(b)))
        return (hi+.05)/(lo+.05)
    for theme in THEMES.values():
        for foreground, background in [("fg", "bg"), ("fg", "panel"), ("muted", "bg"),
                                       ("muted", "panel"), ("fg", "selection"), ("on_accent", "accent")]:
            assert contrast(theme[foreground], theme[background]) >= 4.5
        for curve in ("accent", "secondary", "third", "cursor"):
            assert contrast(theme[curve], theme["panel"]) >= 3
