import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from bapsf_explorer.window import ExplorerWindow
from bapsf_explorer.data import demo_data


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_task(app, window):
    deadline = time.monotonic() + 15
    while window.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert window.worker is None


def test_demo_selection_animation_analysis_and_reset(app):
    window = ExplorerWindow()
    errors = []
    window.error = errors.append
    window.load_demo()
    wait_task(app, window)
    assert not errors
    assert window.data.dims == ("y", "x", "case", "shot", "time")
    window.selectors["case"][0].setValue(1)
    expected = window.data.isel(y=0, x=0, case=1).mean("shot").values
    np.testing.assert_allclose(window.trace.yData, expected)
    window.shot_mode.setCurrentIndex(0)
    window.selectors["shot"][0].setValue(2)
    expected = window.data.isel(y=0, x=0, case=1, shot=2).values
    np.testing.assert_array_equal(window.trace.yData, expected)
    window.frame.setValue(50)
    window.next_frame()
    assert window.frame.value() > 50
    window.vertical.setCurrentText("None")
    assert window.profile.isVisible()
    window.baseline.setChecked(True)
    window.filter_kind.setCurrentText("Lowpass")
    window.cutoff.setText("10000")
    window.apply_analysis()
    wait_task(app, window)
    assert not errors
    assert window.data is not window.base
    assert "Butterworth" in window.data.attrs["history"]
    window.reset_analysis()
    assert window.data is window.base
    window.close()


def test_invalid_shape_retains_previous_data(app):
    window = ExplorerWindow()
    window.demo_loaded(demo_data())
    before = window.data
    errors = []
    window.error = errors.append
    window.layout_mode.setCurrentText("Acquisition order")
    window.dimensions.setText("x=999")
    window.reshape_data()
    wait_task(app, window)
    assert errors and window.data is before
    window.close()


def test_raw_record_mean_and_trace_autorange(app):
    from bapsf_explorer.data import record_array
    raw = record_array(np.array([[1., 2., 3.], [100., 200., 300.]]), [0, 1, 2], [1, 2])
    window = ExplorerWindow()
    window.raw_loaded(raw)
    window.selectors["record"][0].setValue(1)
    assert window.wave.viewRange()[1][1] >= 300
    window.shot_mode.setCurrentIndex(1)
    np.testing.assert_allclose(window.trace.yData, [50.5, 101., 151.5])
    window.close()
