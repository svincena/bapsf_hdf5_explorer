"""Conversion, fit integration, batch isolation and GUI workflow regression tests."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import copy
import numpy as np
import pytest
from lapd_explorer.model import Dataset
from lapd_explorer import langmuir as lp


def acquisition(shape=()):
    voltage = np.linspace(-30, 20, 1500)
    current = .01 * np.exp(np.minimum((voltage-5)/3, 0)) - .0003
    time = np.arange(voltage.size)*1e-6
    dims = {0: (), 1: ("shot",), 2: ("x", "shot"), 3: ("y", "x", "shot")}[len(shape)]
    coords = dict(zip(dims, [np.arange(n) for n in shape]), time=time)
    data = Dataset({"V_sweep": np.broadcast_to(voltage/10, shape + voltage.shape).copy(),
                    "I_sweep": np.broadcast_to(-current*50/2 + .03, shape + current.shape).copy()},
                   dims + ("time",), coords)
    settings = lp.Settings(voltage_channel="V_sweep", current_channel="I_sweep",
                           voltage_factor=10, current_factor=2, resistance=50, area_mm2=2,
                           invert=True, offset_mode="Constant", offset_volts=.03,
                           interval=(time[0], time[-1]))
    return data, settings, voltage, current


def test_physical_conversion_and_real_core():
    data, settings, v, i = acquisition()
    actual_v, actual_i = lp.traces(data, settings, ())
    np.testing.assert_allclose(actual_v, v)
    np.testing.assert_allclose(actual_i, i, atol=1e-18)
    values, notes = lp.analyze(data, settings, ())
    assert values["T_e"] == pytest.approx(3, rel=.05)
    assert values["V_p"] == pytest.approx(5, abs=1)
    assert values["V_f"] == pytest.approx(5 + 3*np.log(.03), abs=.1)
    assert values["n_e"] > 0
    settings.area_mm2 *= 2
    assert lp.analyze(data, settings, ())[0]["n_e"] == pytest.approx(values["n_e"]/2)


def test_offset_per_trace_and_validation():
    data, settings, _, _ = acquisition((2,))
    data.channels["I_sweep"][:, -20:] = np.array([.1, .2])[:, None]
    settings.offset_mode = "Recorded interval"
    settings.offset_interval = tuple(data.coords["time"][[1480, -1]])
    for index in [(0,), (1,)]:
        assert np.mean(lp.traces(data, settings, index)[1][-20:]) == pytest.approx(0, abs=1e-16)
    for attr, value, match in [("resistance", 0, "Resistance"), ("area_mm2", np.nan, "area"),
                                ("voltage_channel", "missing", "Select both"),
                                ("offset_interval", (-1., 2.), "within")]:
        broken = copy.deepcopy(settings)
        setattr(broken, attr, value)
        with pytest.raises(ValueError, match=match):
            lp.validate(data, broken)
    settings.offset_mode = "None"
    with pytest.raises(ValueError, match="Calibrate"):
        lp.validate(data, settings)
    settings.zero_calibrated = True
    lp.validate(data, settings)
    settings.interval = (0., 1e-6)
    with pytest.raises(ValueError, match="7 samples"):
        lp.validate(data, settings)


def test_batch_preserves_geometry_failure_and_settings():
    data, settings, _, _ = acquisition((2, 3, 2))
    data.channels["I_sweep"][1, 2, 1] = np.nan
    progress = []
    batch = lp.process_all(data, settings, lambda n, total: progress.append((n, total)))
    assert progress[-1] == (12, 12)
    assert len(batch.failures) == 1
    assert batch.failures[0][0] == (1, 2, 1)
    for name in lp.QUANTITIES:
        derived = batch.dataset(data, name)
        assert derived.dims == data.dims
        assert derived.shape == (2, 3, 2, 1)
        assert np.isnan(derived.channels[name][1, 2, 1, 0])
        assert np.isfinite(derived.channels[name][0, 0, 0, 0])
    settings.area_mm2 = 500
    assert batch.settings.area_mm2 == 2
    with pytest.raises(ValueError, match="canceled"):
        lp.process_all(data, settings, canceled=lambda: True)


def test_rejected_core_estimates_are_not_cached(monkeypatch):
    data, settings, _, _ = acquisition()
    def rejected(*args, **kwargs):
        return dict(ok=False, warnings=["rejected fit"], te_eV=3, n_e_m3=1e18, vp_V=5, vf_V=-5)
    monkeypatch.setattr(lp.core, "analyze_iv_trace", rejected)
    result = lp.process_all(data, settings)
    assert result.failures == [((), "rejected fit")]
    assert all(np.isnan(a) for a in result.values.values())


def test_dialog_intervals_offset_persistence_and_cached_results(tmp_path):
    from PySide6 import QtWidgets as W
    from lapd_explorer.langmuir_gui import LangmuirDialog, OffsetDialog
    app = W.QApplication.instance() or W.QApplication([])
    for shape in [(), (2,), (3, 2), (2, 3, 2)]:
        data, settings, _, _ = acquisition(shape)
        dialog = LangmuirDialog(data, settings, {}, "Light")
        dialog.show()
        app.processEvents()
        dialog.trace_view.selected.emit((.0001, .0013))
        assert dialog.editor.first.value() == 100
        assert settings.interval == dialog.editor.interval()
        dialog.editor.first.setValue(200)
        assert settings.interval[0] == data.coords["time"][200]
        np.testing.assert_allclose(dialog.trace_view.selectors[1].extents, np.array(settings.interval)*1000)
        saved = copy.deepcopy(settings)
        dialog.reset_view()
        saved.interval = tuple(data.coords["time"][[0, -1]])
        assert settings == saved
        offset = OffsetDialog(data, settings, dialog.index(), "Light", dialog)
        offset.mode.setCurrentText("Constant")
        offset.constant.setText("0.03")
        offset.accept()
        assert offset.result() == W.QDialog.Accepted
        assert offset.settings.offset_volts == .03
        dialog.editor.set_interval(tuple(data.coords["time"][[0, -1]]))
        batch = lp.process_all(data, settings)
        dialog.batch_ready(batch)
        for name in lp.QUANTITIES:
            dialog.quantity.setCurrentText(name)
            assert dialog.batch is batch
            dialog.result_canvas.draw()
        dialog.grab().save(str(tmp_path / f"langmuir-{len(shape)}.png"))
        dialog.reject()
        dialog.show()
        assert settings.offset_volts == .03
        dialog.reject()


def test_main_integration_and_background_execution(monkeypatch):
    import time
    from PySide6 import QtWidgets as W
    from lapd_explorer.app import MainWindow
    from lapd_explorer.langmuir_gui import LangmuirDialog
    app = W.QApplication.instance() or W.QApplication([])
    w = MainWindow()
    data, settings, _, _ = acquisition((3, 2))
    w.set_data(data)
    w.langmuir_settings = settings
    w.slice_boxes[0].setValue(2)
    w.shot.setValue(1)
    monkeypatch.setattr(LangmuirDialog, "exec", lambda self: 0)
    w.open_langmuir()
    dialog = w.langmuir_dialog
    assert dialog.index() == (2, 1)
    dialog.process_shot()
    deadline = time.monotonic() + 15
    while not dialog.action_buttons[0].isEnabled() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert "T_e =" in dialog.results.toPlainText()
    dialog.process_all()
    while not dialog.action_buttons[0].isEnabled() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert dialog.batch is not None
    batch = dialog.batch
    dialog.reject()
    w.appearance.setCurrentText("Dark")
    w.open_langmuir()
    assert w.langmuir_dialog is dialog
    assert dialog.batch is batch
    assert w.data is data and w.raw is data
    dialog.reject()
    w.close()


def test_navigation_and_typed_bounds_control_the_fitted_samples(monkeypatch):
    from PySide6 import QtWidgets as W
    from matplotlib.backend_bases import MouseEvent
    from lapd_explorer.langmuir_gui import LangmuirDialog, OffsetDialog
    app = W.QApplication.instance() or W.QApplication([])
    data, settings, voltage, current = acquisition()
    settings.interval = tuple(data.coords["time"][[100, 1350]])
    dialog = LangmuirDialog(data, settings, {}, "Light")
    dialog.show()
    app.processEvents()
    view, editor = dialog.trace_view, dialog.editor
    original_voltage_limits = view.axes[0].get_ylim()

    def check_bounds():
        interval = editor.interval()
        assert settings.interval == interval
        for axis, selector in zip(view.axes, view.selectors):
            np.testing.assert_allclose(axis.get_xlim(), np.array(interval)*1000)
            np.testing.assert_allclose(selector.extents, np.array(interval)*1000)
        assert editor.start.value() == pytest.approx(interval[0]*1000)
        assert editor.stop.value() == pytest.approx(interval[1]*1000)

    check_bounds()  # Opening must preserve an already configured interval.
    editor.start.setValue(.2)
    editor.last.setValue(1200)
    check_bounds()
    assert editor.first.value() == 200

    # Exercise actual toolbar zoom with canvas mouse events on the current plot.
    view.canvas.draw()
    view.toolbar.zoom()
    axis = view.axes[1]
    y = np.mean(axis.get_ylim())
    x0, y0 = axis.transData.transform((.3, y))
    x1, y1 = axis.transData.transform((1.1, y + np.ptp(axis.get_ylim())*.1))
    for event, x, yy in [("button_press_event", x0, y0), ("motion_notify_event", x1, y1),
                          ("button_release_event", x1, y1)]:
        MouseEvent(event, view.canvas, x, yy, button=1,
                   **({"buttons": {1}} if event == "motion_notify_event" else {}))._process()
    app.processEvents()
    view.toolbar.zoom()
    check_bounds()
    assert .29 < editor.start.value() < .31
    assert 1.09 < editor.stop.value() < 1.11
    zoomed = editor.interval()
    view.toolbar.back()
    app.processEvents()
    check_bounds()
    assert editor.interval() != zoomed
    view.toolbar.forward()
    app.processEvents()
    check_bounds()
    assert editor.interval() == zoomed

    view.canvas.draw()
    view.toolbar.pan()
    x0, y0 = axis.transData.transform((.5, np.mean(axis.get_ylim())))
    for event, x in [("button_press_event", x0), ("motion_notify_event", x0+35),
                      ("button_release_event", x0+35)]:
        MouseEvent(event, view.canvas, x, y0, button=1,
                   **({"buttons": {1}} if event == "motion_notify_event" else {}))._process()
    app.processEvents()
    view.toolbar.pan()
    check_bounds()
    assert editor.interval() != zoomed

    # Flush a pending navigation update before analysis, even without an event-loop turn.
    view.axes[0].set_xlim(.1002, 1.3508)
    snapshot = dialog.collect()
    check_bounds()
    assert (editor.first.value(), editor.last.value()) == (101, 1350)
    captured = []
    original = lp.core.analyze_iv_trace
    def spy(v, i, **kwargs):
        captured.append((v.copy(), i.copy()))
        return original(v, i, **kwargs)
    monkeypatch.setattr(lp.core, "analyze_iv_trace", spy)
    lp.analyze(data, snapshot, ())
    np.testing.assert_allclose(captured[0][0], voltage[101:1351])
    np.testing.assert_allclose(captured[0][1], current[101:1351], atol=1e-18)

    view.toolbar.home()
    app.processEvents()
    check_bounds()
    assert (editor.first.value(), editor.last.value()) == (0, 1499)
    np.testing.assert_allclose(view.axes[0].get_ylim(), original_voltage_limits)
    view.axes[1].set_xlim(-.5, 1.2)
    app.processEvents()
    check_bounds()
    assert (editor.first.value(), editor.last.value()) == (0, 1200)

    # Out-of-recording pans and sub-sample zooms stay synchronized and drawable.
    view.axes[0].set_xlim(2., 3.)
    app.processEvents()
    check_bounds()
    assert (editor.first.value(), editor.last.value()) == (1498, 1499)
    view.axes[0].set_xlim(.5001, .5002)
    app.processEvents()
    check_bounds()
    assert editor.last.value() > editor.first.value()

    offset = OffsetDialog(data, settings, (), "Light", dialog)
    offset.view.axes[1].set_xlim(1.2, 1.4)
    app.processEvents()
    assert (offset.editor.first.value(), offset.editor.last.value()) == (1200, 1400)
    offset.editor.first.setValue(1250)
    for ax in offset.view.axes:
        np.testing.assert_allclose(ax.get_xlim(), (1.25, 1.4))
    offset.reject()
    dialog.reject()
