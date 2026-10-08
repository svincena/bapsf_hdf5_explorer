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


def test_colormap_catalog_previews_and_reversed_colors(app):
    from matplotlib import colormaps
    from PySide6 import QtCore as C
    from lapd_explorer.colormaps import COLORMAPS
    from lapd_explorer.widgets import colormap_combo
    selector = colormap_combo()
    assert selector.currentText() == "viridis"
    assert len(set(COLORMAPS)) == len(COLORMAPS)
    for index, name in enumerate(COLORMAPS):
        assert not selector.itemIcon(index).isNull()
        assert selector.itemData(index, C.Qt.ToolTipRole)
        if not name.endswith("_r"):
            np.testing.assert_allclose(colormaps[name](np.array([0., 1.])),
                                       colormaps[name + "_r"](np.array([1., 0.])))
    left = selector.itemIcon(0).pixmap(64, 14).toImage()
    reversed_left = selector.itemIcon(1).pixmap(64, 14).toImage()
    assert left.pixelColor(0, 7) != reversed_left.pixelColor(0, 7)


def test_vector_arrow_controls_redraw_cache_reset_and_export_options(app, tmp_path):
    from lapd_explorer.vector_style import DEFAULT_ARROW_STYLE
    w = MainWindow()
    w.show()
    assert not w.arrow_style.isEnabled()
    w.mode.setCurrentText("Vector")
    w.draw()
    cached = w._values
    style = w.arrow_style
    style.toggle.click()
    style.fields["width"].setValue(2.5)
    style.fields["length"].setValue(1.5)
    style.stride.setValue(3)
    style.pivot.setCurrentText("Tail")
    w.draw()
    assert w._values is cached
    arrows = w.main_ax.collections[1]
    assert arrows.width == pytest.approx(2.5/72) and arrows.pivot == "tail"
    assert arrows.N == 9*11
    exported = w.options()["arrow_style"]
    assert exported["width"] == 2.5 and exported["length"] == 1.5 and exported["stride"] == 3
    app.processEvents()
    w.grab().save(str(tmp_path/"vector-arrow-controls.png"))
    style.reset()
    assert style.settings() == DEFAULT_ARROW_STYLE
    w.mode.setCurrentText("Scalar")
    assert not style.isEnabled()
    w.close()


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
    w.cmap.setCurrentText("YlGnBu_r")
    w.arrow_cmap.setCurrentText("twilight_shifted")
    w.draw()
    assert w.main_ax.collections[0].cmap.name == "YlGnBu_r"
    assert w.main_ax.collections[1].cmap.name == "twilight_shifted"
    assert w.options()["cmap"] == "YlGnBu_r"
    assert w.options()["arrow_cmap"] == "twilight_shifted"
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


def test_file_dialog_locations_are_independent_and_persist(app, monkeypatch, tmp_path):
    from pathlib import Path
    from PySide6 import QtCore as C
    from lapd_explorer.widgets import Worker
    settings_path = str(tmp_path / "preferences.ini")
    settings = C.QSettings(settings_path, C.QSettings.IniFormat)
    w = MainWindow(settings=settings)
    data = Dataset({"A": np.arange(8.)}, ("time",), {"time": np.arange(8.)})
    w.set_data(data)
    operations = [("open_data", w.open_file, ""),
                  ("save_data", w.save_data, "lapd-processed.h5"),
                  ("save_image", w.save_image, "lapd-frame.png"),
                  ("save_movie", w.export_movie, "lapd-animation.mp4")]
    selected = ""
    shown = []
    def choose(parent, title, initial, filters):
        shown.append(initial)
        return selected, ""
    monkeypatch.setattr(W.QFileDialog, "getOpenFileName", choose)
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", choose)
    # Canceling a dialog must not set preferences or inherit another category.
    for category, operation, filename in operations:
        operation()
        assert shown[-1] == str(Path.home() / filename)
        assert not settings.contains(f"file_dialogs/{category}")

    def launch(fn, done, message):
        w.jobs.append(Worker(fn, w))  # Movie dialog connects to the job's signals.
        done(fn())
    monkeypatch.setattr(w, "launch", launch)
    monkeypatch.setattr("lapd_explorer.app.io.inspect_file", lambda path: {"portable": True})
    monkeypatch.setattr("lapd_explorer.app.io.load_dataset", lambda path: data)
    monkeypatch.setattr(w, "draw", lambda: None)
    monkeypatch.setattr(w.fig, "savefig", lambda path, **kwargs: Path(path).write_bytes(b"image"))
    monkeypatch.setattr("lapd_explorer.app.export_mp4", lambda path, *args: Path(path).write_bytes(b"movie"))
    for category, operation, filename in operations:
        directory = tmp_path / category
        directory.mkdir()
        selected = str(directory / (filename or "input.h5"))
        operation()
        assert settings.value(f"file_dialogs/{category}") == str(directory)
        assert w.file_dialog_path(category, filename) == str(directory / filename)
    w.redraw_timer.stop()
    w.close()

    # A new settings object/window simulates restarting the application.
    restored = MainWindow(settings=C.QSettings(settings_path, C.QSettings.IniFormat))
    for category, _, filename in operations:
        assert restored.file_dialog_path(category, filename) == str(tmp_path / category / filename)
    restored.settings.setValue("file_dialogs/save_image", str(tmp_path / "deleted-directory"))
    assert restored.file_dialog_path("save_image", "lapd-frame.png") == str(Path.home() / "lapd-frame.png")
    assert restored.file_dialog_path("save_data") == str(tmp_path / "save_data")
    restored.redraw_timer.stop()
    restored.close()


def test_failed_io_and_canceled_import_preserve_locations(app, monkeypatch, tmp_path):
    from PySide6 import QtCore as C
    settings = C.QSettings(str(tmp_path / "preferences.ini"), C.QSettings.IniFormat)
    w = MainWindow(settings=settings)
    previous = tmp_path / "previous"
    previous.mkdir()
    for category in ("open_data", "save_data", "save_image", "save_movie"):
        w.remember_file_location(category, previous / "old.h5")
    selected = str(tmp_path / "new.h5")
    monkeypatch.setattr(W.QFileDialog, "getOpenFileName", lambda *args: (selected, ""))
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", lambda *args: (selected, ""))
    errors = []
    monkeypatch.setattr(w, "error", errors.append)
    def fail(*args, **kwargs):
        raise OSError("Test I/O failure")
    def launch(fn, done, message):
        from lapd_explorer.widgets import Worker
        w.jobs.append(Worker(fn, w))
        try:
            result = fn()
        except OSError as exc:
            errors.append(str(exc))
        else:
            done(result)
    monkeypatch.setattr(w, "launch", launch)
    monkeypatch.setattr("lapd_explorer.app.io.inspect_file", fail)
    monkeypatch.setattr("lapd_explorer.app.io.save_dataset", fail)
    monkeypatch.setattr(w, "draw", lambda: None)
    monkeypatch.setattr(w.fig, "savefig", fail)
    monkeypatch.setattr("lapd_explorer.app.export_mp4", fail)
    for operation in (w.open_file, w.save_data, w.save_image, w.export_movie):
        operation()
    assert len(errors) == 4
    monkeypatch.setattr("lapd_explorer.app.io.inspect_file", lambda path: {"portable": False})
    class CanceledImport:
        def __init__(self, *args):
            pass
        def exec(self):
            return W.QDialog.Rejected
    monkeypatch.setattr("lapd_explorer.app.ImportDialog", CanceledImport)
    w.open_file()
    for category in ("open_data", "save_data", "save_image", "save_movie"):
        assert w.file_dialog_path(category) == str(previous)
    w.redraw_timer.stop()
    w.close()
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
    box.setDecimals(12)
    for value in (0., 100., 1e-9, 1.234567890123, 1000.000000000001):
        box.setValue(value)
        assert float(box.text()) == value
        box.interpretText()
        assert box.value() == value
    box.setValue(100.)
    assert box.text() == "100"


def test_sis_descriptions_in_import_channel_controls_and_plot_legends(app, tmp_path, monkeypatch):
    import time
    from PySide6 import QtCore as C
    from bapsflib._hdf.maps.tests import FauxHDFBuilder
    from lapd_explorer import io
    from lapd_explorer.import_time_gui import ImportTimeDialog
    path = tmp_path/"named-channels.h5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module("SIS crate", {"sn_size": 4, "nt": 16})
        knobs = f.modules["SIS crate"].knobs
        enabled = knobs.active_brdch
        enabled["SIS 3302"][0, :3] = True
        knobs.active_brdch = enabled
        attrs = f["Raw data + config/SIS crate/config01/SIS crate 3302 configurations[0]"].attrs
        attrs["Data type 1"] = np.bytes_(" Bxhf_mov_p25 ")
        attrs["Data type 2"] = np.bytes_(" \t ")
        attrs["Data type 3"] = "Bxhf_mov_p25"
    dialog = ImportDialog(path, io.inspect_file(path))
    dialog.show()
    dialog.channels.clearSelection()
    for i in range(dialog.channels.count()):
        item = dialog.channels.item(i)
        spec = item.data(C.Qt.UserRole)
        if spec["adc"] == "SIS 3302":
            item.setSelected(True)
            assert f"B1 Ch{spec['channel']}" in item.text()
            if spec["channel"] != 2:
                assert "Bxhf_mov_p25" in item.text()
    dialog.add_axis()
    dialog.axes.cellWidget(0, 0).setCurrentText("shot")
    dialog.axes.item(0, 1).setText("4")
    dialog.axes.item(0, 3).setText("3")
    previewed = []
    def inspect_preview(preview):
        assert "B1 Ch1" in preview.channel.itemText(0)
        assert "Bxhf_mov_p25" in preview.channel.itemText(0)
        preview.worker.wait()
        app.processEvents()
        assert preview.axis.get_title() == "Bxhf_mov_p25"
        preview.reject()
        previewed.append(True)
        return W.QDialog.Rejected
    monkeypatch.setattr(ImportTimeDialog, "exec", inspect_preview)
    dialog.preview_time()
    deadline = time.monotonic()+10
    while not previewed and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert previewed, dialog.message.text()
    dialog.begin()
    deadline = time.monotonic()+10
    while dialog.dataset is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert dialog.dataset is not None, dialog.message.text()
    names = ["C1 · Bxhf_mov_p25", "C2 · B1 Ch2", "C3 · Bxhf_mov_p25"]
    assert list(dialog.dataset.channels) == names
    assert dialog.dataset.metadata[names[0]]["data type"] == "Bxhf_mov_p25"
    saved = tmp_path/"named-processed.h5"
    io.save_dataset(saved, dialog.dataset)
    assert list(io.load_dataset(saved).channels) == names
    w = MainWindow()
    w.set_data(dialog.dataset)
    w.show()
    app.processEvents()
    assert [w.components[0].itemText(i) for i in range(w.components[0].count())] == names
    for name in names:
        w.components[0].setCurrentText(name)
        w.draw()
        assert w.main_ax.get_legend().get_texts()[0].get_text() == name
    w.grab().save(str(tmp_path/"sis-named-channel-plots.png"))
    w.close()
    dialog.reject()


def test_import_dialog_applies_shot_guess_and_keeps_fields_editable(app, monkeypatch):
    info = {
        "channels": [{"digitizer": "D", "config_name": "cfg", "adc": "A",
                      "board": 1, "channel": 2}],
        "controls": [("bmotion", "scan")],
        "datasets": [],
    }
    monkeypatch.setattr(
        "lapd_explorer.widgets.io.guess_shots_per_case",
        lambda *args, **kwargs: {"shots_per_case": 5, "spatial_points": 2601,
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


def test_temporal_preview_zoom_manual_limits_and_record_selection(app, tmp_path):
    import time
    import h5py
    from lapd_explorer import io
    from lapd_explorer.import_time_gui import ImportTimeDialog
    path = tmp_path/"preview.h5"
    t = np.arange(100001)*.001
    values = np.array([np.sin(t)+i for i in range(3)])
    with h5py.File(path, "w") as f:
        f["signal"] = values
    metadata = io.import_metadata(path, dataset_path="signal", dt=.001)
    dialog = ImportTimeDialog(path, [("Signal", None, "signal")], dict(dt=.001), metadata, 0., {})
    dialog.show()
    def wait():
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            app.processEvents()
            if not dialog.timer.isActive() and not dialog._pending and dialog.worker and not dialog.worker.isRunning():
                app.processEvents()
                return
            time.sleep(.005)
        pytest.fail("Preview did not finish")
    wait()
    assert "record index 1" in dialog.record_label.text()
    assert len(dialog.line.get_xdata()) <= 20000
    assert not dialog.mode.model().item(1).isEnabled()
    assert not dialog.mode.model().item(2).isEnabled()
    dialog.axis.set_xlim(10., 20.)
    wait()
    assert dialog.sample_limits() == (10000, 20000)
    assert len(dialog.line.get_xdata()) == 10001
    dialog.start.setValue(12.345)
    dialog.last.setValue(20001)
    wait()
    assert dialog.sample_limits() == (12345, 20001)
    assert dialog.stop.value() == pytest.approx(20.001)
    dialog.index.setValue(2)
    dialog.reload_button.click()
    wait()
    assert "record index 2" in dialog.record_label.text()
    np.testing.assert_allclose(dialog.line.get_ydata(), values[2, 12345:20002])
    dialog.grab().save(str(tmp_path/"temporal-preview.png"))
    dialog.accept()
    assert dialog.result() == W.QDialog.Accepted


def test_import_dialog_reads_typed_temporal_limits_and_thins_on_disk(app, tmp_path):
    import time
    import h5py
    from lapd_explorer import io
    path = tmp_path/"raw.h5"
    values = np.arange(120).reshape(3, 40)
    with h5py.File(path, "w") as f:
        f["signal"] = values
    dialog = ImportDialog(path, io.inspect_file(path))
    dialog.show()
    dialog.raw_paths.item(0).setSelected(True)
    dialog.add_axis()
    dialog.axes.cellWidget(0, 0).setCurrentText("shot")
    dialog.axes.item(0, 1).setText("3")
    dialog.axes.item(0, 3).setText("2")
    dialog.dt.setText("0.001")
    dialog.t0.setText("-0.010")
    dialog.time_range_mode.setCurrentIndex(2)
    dialog.time_first.setText("-0.005")
    dialog.time_last.setText("0.021")
    dialog.decimation.setValue(3)
    dialog.downsampling.setCurrentText("Simple decimation")
    dialog.begin()
    deadline = time.monotonic()+10
    while dialog.dataset is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert dialog.dataset is not None, dialog.message.text()
    np.testing.assert_array_equal(dialog.dataset.channels[next(iter(dialog.dataset.channels))], values[:, 5:32:3])
    np.testing.assert_allclose(dialog.dataset.coords["time"], -.01+np.arange(5, 32, 3)*.001)
    assert dialog.dataset.shape == (3, 9)
    dialog.reject()


def test_import_preview_global_shot_coordinate_and_applied_limits(app, tmp_path, monkeypatch):
    import time
    from bapsflib._hdf.maps.tests import FauxHDFBuilder
    from lapd_explorer import io
    from lapd_explorer.import_time_gui import ImportTimeDialog
    path = tmp_path/"acquisition.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module("SIS 3301", {"sn_size": 8, "nt": 128})
        f.add_module("bmotion", {"sn_size": 8})
    importer = ImportDialog(path, io.inspect_file(path))
    importer.decimation.setValue(4)
    importer.show()
    previewed = []
    def choose(dialog):
        dialog.show()
        def wait():
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                app.processEvents()
                if not dialog.timer.isActive() and not dialog._pending and dialog.worker and not dialog.worker.isRunning():
                    app.processEvents()
                    return
                time.sleep(.005)
            pytest.fail("Preview did not finish")
        wait()
        assert "record index 4" in dialog.record_label.text()
        dialog.mode.setCurrentIndex(1)
        shot = dialog.metadata["shots"][1]
        dialog.shot.setText(str(shot))
        dialog.reload_button.click()
        wait()
        assert f"global shot {shot}" in dialog.record_label.text()
        dialog.mode.setCurrentIndex(2)
        for box, value in zip(dialog.coordinates, dialog.metadata["xyz"][6]):
            box.setText(str(value+.001))
        dialog.reload_button.click()
        wait()
        assert "record index 6" in dialog.record_label.text()
        dialog.first.setValue(17)
        dialog.last.setValue(87)
        wait()
        dialog.grab().save(str(tmp_path/"motion-temporal-preview.png"))
        dialog.accept()
        previewed.append(dialog.sample_limits())
        return dialog.result()
    monkeypatch.setattr(ImportTimeDialog, "exec", choose)
    importer.preview_time()
    deadline = time.monotonic()+15
    while not previewed and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert previewed == [(17, 87)], importer.message.text()
    assert importer.temporal_options() == dict(sample_limits=(17, 87), decimation=4, downsampling="polyphase")
    assert "Imported rate:" in importer.time_summary.text()
    importer.grab().save(str(tmp_path/"import-temporal-controls.png"))
    if importer.guess_worker:
        importer.guess_worker.wait()
    importer.guess_timer.stop()
    app.processEvents()
    importer.reject()


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


def test_spatial_averaging_dialog_processing_render_and_reset(app, monkeypatch, tmp_path):
    from lapd_explorer.widgets import SpatialAveragingDialog
    from lapd_explorer.spatial import DEFAULT_SPATIAL_AVERAGING, average_spatial
    from lapd_explorer import plotting
    dialog = SpatialAveragingDialog(DEFAULT_SPATIAL_AVERAGING, ("z", "x"))
    dialog.show()
    app.processEvents()
    assert dialog.windows[0].isVisible() and dialog.sigmas[0].isHidden()
    assert "z window" in dialog.form.labelForField(dialog.windows[0]).text()
    dialog.windows[0].setValue(4)
    dialog.accept()
    assert "odd" in dialog.error_label.text()
    dialog.method.setCurrentText("Gaussian average")
    assert dialog.windows[0].isHidden() and dialog.sigmas[0].isVisible()
    dialog.sigmas[0].setText("nan")
    dialog.accept()
    assert "positive" in dialog.error_label.text()
    dialog.sigmas[0].setText(".8")
    dialog.sigmas[1].setText("1.5")
    assert dialog.settings()["sigma"] == (.8, 1.5)
    dialog.method.setCurrentText("Disk average")
    assert dialog.radius.isVisible() and dialog.sigmas[0].isHidden()
    dialog.radius.setValue(2)
    dialog.accept()
    assert dialog.result() == W.QDialog.Accepted
    restored = SpatialAveragingDialog(dialog.settings(), ("z", "x"))
    assert restored.settings() == dialog.settings()
    restored.reject()

    w = MainWindow()
    a = np.arange(5*7*4., dtype=float).reshape(5, 7, 4)**2
    data = Dataset({"A": a, "B": a*2}, ("z", "x", "time"),
                   {"z": np.arange(5), "x": np.arange(7), "time": np.arange(4)})
    w.set_data(data)
    assert w.spatial_average.isEnabled() and not w.spatial_average.isChecked()
    assert not w.spatial_button.isEnabled()
    w.spatial_average.setChecked(True)
    assert w.spatial_button.isEnabled()
    w.spatial_settings.update(dialog.settings())
    monkeypatch.setattr(w, "launch", lambda fn, done, message: done(fn()))
    w.apply_processing()
    expected = average_spatial(a, **dialog.settings())
    np.testing.assert_allclose(w.data.channels["A"], expected)
    w.apply_processing()
    np.testing.assert_allclose(w.data.channels["A"], expected)
    assert "Spatial averaging: disk" in w.history.text()
    w.mode.setCurrentText("Vector")
    w.show()
    w.draw()
    app.processEvents()
    np.testing.assert_allclose(w._values, expected*np.sqrt(5))
    for frame in (0, 3):
        w.frame.setValue(frame)
        w.draw()
        np.testing.assert_allclose(w.main_ax.collections[0].get_array(), expected[..., frame]*np.sqrt(5))
        np.testing.assert_allclose(w.main_ax.collections[1].U, expected[..., frame].ravel())
        horizontal, vertical = w.fig._lapd_slice_axes
        np.testing.assert_allclose(horizontal.lines[0].get_ydata(), w._values[w.slice_boxes[0].value(), :, frame])
        np.testing.assert_allclose(vertical.lines[0].get_ydata(), w._values[:, w.slice_boxes[1].value(), frame])
    # The standalone export renderer receives the same processed channels.
    fig = plotting.figure()
    ax = plotting.render(fig, w.data, w.options())
    fig._lapd_update_frame(1)
    np.testing.assert_allclose(ax.collections[0].get_array(), expected[..., 1]*np.sqrt(5))
    np.testing.assert_allclose(ax.collections[1].U, expected[..., 1].ravel())
    fig.clear()
    w.grab().save(str(tmp_path / "spatial-window.png"))
    preview = SpatialAveragingDialog(w.spatial_settings, data.spatial_dims, w)
    preview.show()
    app.processEvents()
    preview.grab().save(str(tmp_path / "spatial-dialog.png"))
    preview.reject()
    w.reset_processing()
    assert w.data is w.raw and not w.spatial_average.isChecked()
    assert w.spatial_settings == DEFAULT_SPATIAL_AVERAGING
    for dims, shape in [(("x", "time"), (3, 4)), (("time",), (4,))]:
        w.spatial_average.setChecked(True)
        w.set_data(Dataset({"A": np.ones(shape)}, dims,
                           {d: np.arange(n) for d, n in zip(dims, shape)}))
        assert not w.spatial_average.isEnabled() and not w.spatial_average.isChecked()
        assert not w.spatial_button.isEnabled()
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


def test_manual_colormap_range_playback_and_export(app):
    from lapd_explorer import plotting
    w = MainWindow()
    values = np.arange(24.).reshape(2, 3, 4)
    data = Dataset({"A": values}, ("y", "x", "time"),
                   {"y": np.arange(2), "x": np.arange(3), "time": np.arange(4)})
    w.set_data(data)
    w.draw()
    control = w.color_range
    assert control.mode.currentText() == "Automatic"
    assert w.options()["color_limits"] is None
    np.testing.assert_allclose(w.main_ax.collections[0].get_clim(), (0, 23))
    w.lock.setChecked(False)
    w.draw()
    np.testing.assert_allclose(w.main_ax.collections[0].get_clim(), (0, 20))
    control.mode.setCurrentText("Manual")
    control.minimum.setText("-2e1")
    control.maximum.setText("1e1")
    control.accept_limits()
    w.draw()
    mesh = w.main_ax.collections[0]
    for frame in (1, 3, 0):
        w.frame.setValue(frame)
        w.draw()
        assert w.main_ax.collections[0] is mesh
        np.testing.assert_allclose(mesh.get_clim(), (-20, 10))
        np.testing.assert_allclose(mesh.get_array(), values[..., frame])
    for lower, upper in [("10", "10"), ("20", "10"), ("nan", "10"), ("-20", "inf"), ("bad", "10")]:
        control.minimum.setText(lower)
        control.maximum.setText(upper)
        control.accept_limits()
        assert w.options()["color_limits"] == (-20, 10)
    exported = plotting.figure()
    plotting.render(exported, data, w.options())
    exported._lapd_update_frame(3)
    np.testing.assert_allclose(exported.axes[0].collections[0].get_clim(), (-20, 10))
    exported.clear()
    control.mode.setCurrentText("Automatic")
    w.frame.setValue(3)
    w.draw()
    np.testing.assert_allclose(w.main_ax.collections[0].get_clim(), (3, 23))
    w.set_data(Dataset({"A": np.ones(4)}, ("time",), {"time": np.arange(4)}))
    assert control.isHidden()
    assert w.options()["color_limits"] is None
    w.close()


def test_import_downsampling_choices_summary_and_progress(app, tmp_path):
    from lapd_explorer import io
    import h5py
    path = tmp_path/"choices.h5"
    with h5py.File(path, "w") as f:
        f["A"] = np.arange(101)[None, :]
    dialog = ImportDialog(path, io.inspect_file(path))
    dialog.show()
    app.processEvents()
    assert dialog.downsampling.isHidden()
    dialog.decimation.setValue(4)
    assert not dialog.downsampling.isHidden()
    assert dialog.downsampling.currentText() == "Polyphase FIR resampling"
    assert dialog.temporal_options()["downsampling"] == "polyphase"
    assert "FIR" in dialog.alias_note.text()
    dialog.temporal_metadata = dict(samples=101, dt=.01)
    dialog.t0.setText("0")
    dialog.downsampling.setCurrentText("Block averaging")
    assert "25 / 101 samples" in dialog.time_summary.text()
    assert "block-center" in dialog.alias_note.text()
    dialog.downsampling.setCurrentText("Simple decimation")
    assert "26 / 101 samples" in dialog.time_summary.text()
    assert "can alias" in dialog.alias_note.text()
    dialog.decimation.setValue(1)
    assert dialog.downsampling.isHidden() and dialog.alias_note.isHidden()
    assert dialog.temporal_options() == dict(decimation=1)
    dialog.update_import_progress(750, 1000, "Reading / resampling")
    assert dialog.import_progress.value() == 750
    assert "750 / 1,000 shots (acquisitions)" in dialog.progress_detail.text()
    assert "250 remaining" in dialog.progress_detail.text()
    from lapd_explorer.import_progress import ShotProgress
    dialog.update_import_progress(1, 2, ShotProgress("Reading / resampling", 0, 1, channel=2, channels=2))
    assert dialog.import_progress.value() == 500
    assert "channel 2/2: 0 / 1 shots (acquisitions)" in dialog.progress_detail.text()
    assert "1 remaining" in dialog.progress_detail.text()
    assert "samples" not in dialog.progress_detail.text()
    dialog.update_import_progress(7, 12, ShotProgress("Mapping", 0, 6, phase=2))
    assert "Stage 2/2" in dialog.import_progress.format()
    assert "0 / 6 shots (acquisitions)" in dialog.progress_detail.text()
    dialog.failed("Read failed")
    assert dialog.import_progress.isHidden() and dialog.load.isEnabled()
    dialog.reject()


def test_status_bar_counts_imported_acquisitions_once_and_retains_count_after_averaging(app, tmp_path):
    import h5py
    from lapd_explorer import io
    from lapd_explorer.model import preprocess
    path = tmp_path/"acquisitions.h5"
    with h5py.File(path, "w") as f:
        f["A"] = np.ones((2, 3, 2, 4, 17))
        f["B"] = np.ones((2, 3, 2, 4, 17))
    axes = [("y", 2, 0, 1), ("x", 3, 0, 2), ("case", 2, 0, 1), ("shot", 4, 0, 3)]
    imported = io.read_raw(path, {"A": "A", "B": "B"}, axes, .001)
    w = MainWindow()
    w.set_data(imported)
    w.draw()
    assert "48 imported shots (acquisitions)" in w.statusBar().currentMessage()
    assert "samples" not in w.statusBar().currentMessage()
    w.processed(preprocess(imported, average=True))
    w.draw()
    assert "shot" not in w.data.dims
    assert "48 imported shots (acquisitions)" in w.statusBar().currentMessage()
    w.set_data(Dataset({"A": np.ones(17)}, ("time",), {"time": np.arange(17.)}))
    w.draw()
    assert "1 imported shot (acquisition)" in w.statusBar().currentMessage()
    w.close()
