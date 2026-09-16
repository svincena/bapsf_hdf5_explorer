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
