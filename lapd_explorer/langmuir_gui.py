"""Langmuir workflow; numerical analysis lives in langmuir.py and the supplied core."""
import copy
import threading
import numpy as np
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.widgets import SpanSelector
from . import langmuir as analysis, plotting
from .appearance import colors
from .widgets import Worker, combo, spin


class IntervalEditor(W.QWidget):
    changed = C.Signal(object)

    def __init__(self, time, interval, parent=None):
        super().__init__(parent)
        self.time = time
        layout = W.QFormLayout(self)
        layout.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        self.start, self.stop = W.QDoubleSpinBox(), W.QDoubleSpinBox()
        self.first, self.last = spin(0, len(time)-1), spin(0, len(time)-1)
        for box in (self.start, self.stop):
            box.setDecimals(9)
            box.setRange(time[0]*1e3, time[-1]*1e3)
            box.setSingleStep((time[-1]-time[0])*1e3/max(len(time)-1, 1))
        layout.addRow("Start (ms, inclusive)", self.start)
        layout.addRow("End (ms, inclusive)", self.stop)
        layout.addRow("First sample (0-based)", self.first)
        layout.addRow("Last sample (inclusive)", self.last)
        for box in (self.start, self.stop, self.first, self.last):
            box.setKeyboardTracking(False)
        self.set_interval(interval or (time[0], time[-1]))
        self.start.valueChanged.connect(self.from_times)
        self.stop.valueChanged.connect(self.from_times)
        self.first.valueChanged.connect(self.from_indices)
        self.last.valueChanged.connect(self.from_indices)

    def interval(self):
        return (float(self.time[self.first.value()]), float(self.time[self.last.value()]))

    def set_interval(self, interval):
        # Snap graphical/numeric bounds to actual samples; indices are authoritative.
        tolerance = max(abs(self.time[0]), abs(self.time[-1]), np.ptp(self.time)) * np.finfo(float).eps * 8
        lo = int(np.clip(np.searchsorted(self.time, interval[0] - tolerance, side="left"), 0, len(self.time)-1))
        hi = int(np.clip(np.searchsorted(self.time, interval[1] + tolerance, side="right")-1, 0, len(self.time)-1))
        for box, value in zip((self.start, self.stop, self.first, self.last),
                              (self.time[lo]*1e3, self.time[hi]*1e3, lo, hi)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.changed.emit(self.interval())

    def from_times(self):
        # Roundoff in millisecond widgets must not shift an exact sample endpoint.
        values = [self.start.value()/1e3, self.stop.value()/1e3]
        indices = [int(np.argmin(abs(self.time-v))) for v in values]
        self.set_interval(tuple(self.time[i] for i in indices))

    def from_indices(self):
        self.set_interval(self.interval())


class TraceView(W.QWidget):
    selected = C.Signal(object)

    def __init__(self, time, appearance, parent=None):
        super().__init__(parent)
        self.time = time
        self.appearance = appearance
        self.fig = plotting.figure(appearance)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = W.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.axes = [self.fig.add_subplot(211), self.fig.add_subplot(212)]
        self.axes[1].sharex(self.axes[0])
        self.lines = []
        self._history_initialized = False
        self._pending_limits = None
        self._syncing_limits = False
        self.limit_timer = C.QTimer(self)
        self.limit_timer.setSingleShot(True)
        self.limit_timer.timeout.connect(self.flush_limits)
        theme = colors(appearance)
        for ax, label in zip(self.axes, ("Probe voltage (V)", "Probe current (A)")):
            plotting.style(ax, theme)
            ax.set_ylabel(label)
            self.lines.append(ax.plot([], [])[0])
        self.axes[1].set_xlabel("Time (ms)")
        self.selectors = [SpanSelector(ax, self.select, "horizontal", useblit=True,
                                      props=dict(facecolor=theme["accent"], alpha=.25),
                                      interactive=True, drag_from_anywhere=True) for ax in self.axes]
        self.set_interval((time[0], time[-1]))
        for ax in self.axes:
            ax.callbacks.connect("xlim_changed", self.limits_changed)

    def limits_changed(self, axis):
        if self._syncing_limits:
            return
        # Wait until Matplotlib finishes updating both shared axes/history. This
        # also coalesces pan motion events and avoids recursive limit callbacks.
        self._pending_limits = tuple(sorted(v/1e3 for v in axis.get_xlim()))
        self.limit_timer.start(0)

    def flush_limits(self):
        interval = self._pending_limits
        self._pending_limits = None
        self.limit_timer.stop()
        if interval is not None:
            self.select(interval[0]*1e3, interval[1]*1e3)

    def select(self, lo, hi):
        interval = np.clip(sorted((lo/1e3, hi/1e3)), self.time[0], self.time[-1])
        first = np.searchsorted(self.time, interval[0], side="left")
        last = np.searchsorted(self.time, interval[1], side="right") - 1
        if first >= last and len(self.time) > 1:
            # A pan past the recording or sub-sample zoom still needs a drawable
            # interval. Analysis separately checks its minimum sample count.
            first = int(np.clip(first, 0, len(self.time)-2))
            interval = self.time[[first, first+1]]
        self.selected.emit(tuple(interval))

    def set_interval(self, interval):
        self._pending_limits = None
        self.limit_timer.stop()
        limits = tuple(v*1e3 for v in interval)
        self._syncing_limits = True
        try:
            if limits[0] < limits[1]:
                # emit=False does not propagate to shared axes, so set both.
                for ax in self.axes:
                    ax.set_xlim(*limits, emit=False)
        finally:
            self._syncing_limits = False
        for selector in self.selectors:
            selector.extents = tuple(v*1e3 for v in interval)
            selector.set_visible(True)
        self.canvas.draw_idle()

    def traces(self, voltage, current):
        for ax, line, values in zip(self.axes, self.lines, (voltage, current)):
            line.set_data(self.time*1e3, values)
            ax.relim()
            ax.autoscale_view(scalex=False)
        if not self._history_initialized:
            # Seed Home after data have established sensible vertical limits.
            limits = self.axes[0].get_xlim()
            for ax in self.axes:
                ax.set_xlim(self.time[0]*1e3, self.time[-1]*1e3, emit=False)
            self.toolbar.push_current()
            for ax in self.axes:
                ax.set_xlim(*limits, emit=False)
            self._history_initialized = True
        self.canvas.draw_idle()

    def reset_view(self):
        interval = (self.time[0], self.time[-1])
        self.set_interval(interval)
        self.selected.emit(interval)
        for ax in self.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="y")
        self.toolbar.update()
        self.toolbar.push_current()
        self.canvas.draw_idle()


class OffsetDialog(W.QDialog):
    def __init__(self, data, settings, index, appearance, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Current zero / instrumental offset")
        self.resize(1000, 750)
        self.data, self.settings, self.index = data, copy.deepcopy(settings), index
        layout = W.QVBoxLayout(self)
        note = W.QLabel("Choose a region where the true current is zero. The mean is estimated separately for each shot.\nConstant offsets are digitizer volts, subtracted before attenuation, resistance and polarity conversion.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.mode = combo(["None", "Recorded interval", "Constant"])
        self.mode.setCurrentText(settings.offset_mode)
        self.constant = W.QLineEdit(str(settings.offset_volts))
        form = W.QFormLayout()
        form.addRow("Correction", self.mode)
        form.addRow("Constant (digitizer V)", self.constant)
        layout.addLayout(form)
        self.editor = IntervalEditor(data.coords["time"], settings.offset_interval)
        layout.addWidget(self.editor)
        self.view = TraceView(data.coords["time"], appearance)
        self.view.axes[0].set_ylabel("Raw V signal (V)")
        self.view.axes[1].set_ylabel("Raw I signal (V)")
        layout.addWidget(self.view, 1)
        self.view.traces(data.channels[settings.voltage_channel][index], data.channels[settings.current_channel][index])
        self.view.reset_view()
        self.view.selected.connect(self.editor.set_interval)
        self.editor.changed.connect(self.update_interval)
        self.message = W.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_interval(self.editor.interval())

    def update_interval(self, interval):
        self.view.set_interval(interval)
        try:
            selected = analysis.interval_slice(self.data.coords["time"], interval)
            mean = np.mean(self.data.channels[self.settings.current_channel][self.index][selected])
            self.message.setText(f"Representative mean: {mean:.6g} digitizer V")
        except ValueError as exc:
            self.message.setText(str(exc))

    def accept(self):
        self.view.flush_limits()
        try:
            self.settings.offset_mode = self.mode.currentText()
            self.settings.offset_volts = float(self.constant.text())
            self.settings.offset_interval = self.editor.interval()
            analysis.validate(self.data, self.settings, analysis=False)
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        super().accept()


class LangmuirDialog(W.QDialog):
    progress = C.Signal(int, int)

    def __init__(self, data, settings, selection, appearance, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Langmuir probe analysis")
        self.resize(1350, 900)
        self.data, self.settings, self.appearance = data, settings, appearance
        self.batch = None
        self.worker = None
        self.cancel_event = threading.Event()
        layout = W.QVBoxLayout(self)
        split = W.QSplitter()
        layout.addWidget(split, 1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(470)
        scroll.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff)
        panel = W.QWidget()
        self.form = W.QFormLayout(panel)
        self.form.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        scroll.setWidget(panel)
        split.addWidget(scroll)
        self.voltage = combo([""] + list(data.channels))
        self.current = combo([""] + list(data.channels))
        self.voltage.setCurrentText(settings.voltage_channel if settings.voltage_channel in data.channels else "")
        self.current.setCurrentText(settings.current_channel if settings.current_channel in data.channels else "")
        for widget, name in ((self.voltage, "V_sweep"), (self.current, "I_sweep")):
            if not widget.currentText() and name in data.channels:
                widget.setCurrentText(name)
        self.form.addRow("V_sweep channel", self.voltage)
        self.form.addRow("I_sweep channel", self.current)
        self.indices = {}
        for d, n in zip(data.dims[:-1], data.shape[:-1]):
            widget = spin(0, n-1, min(selection.get(d, 0), n-1))
            self.indices[d] = widget
            self.form.addRow(f"{d} index (0-based)", widget)
            widget.valueChanged.connect(self.selection_changed)
        self.location = W.QLabel()
        self.location.setWordWrap(True)
        self.form.addRow(self.location)
        self.fields = {}
        for key, label in [("voltage_factor", "V attenuation multiplier"), ("current_factor", "I attenuation multiplier"),
                           ("resistance", "Measurement resistor (Ω)"), ("area_mm2", "Collection area (mm²)")]:
            widget = W.QLineEdit(str(getattr(settings, key)))
            self.fields[key] = widget
            self.form.addRow(label, widget)
            widget.editingFinished.connect(self.settings_changed)
        self.invert = W.QCheckBox("Invert current (electron current positive)")
        self.invert.setChecked(settings.invert)
        self.form.addRow(self.invert)
        self.zero = W.QCheckBox("Current zero independently calibrated")
        self.zero.setChecked(settings.zero_calibrated)
        self.zero.setToolTip("Required when no offset correction is selected. Do not zero the ion-saturation branch.")
        self.form.addRow(self.zero)
        offset = W.QPushButton("Configure current offset…")
        offset.clicked.connect(self.configure_offset)
        self.form.addRow(offset)
        self.offset_label = W.QLabel()
        self.offset_label.setWordWrap(True)
        self.form.addRow(self.offset_label)
        self.editor = IntervalEditor(data.coords["time"], settings.interval)
        self.form.addRow(W.QLabel("SWEEP INTERVAL · drag across either trace"))
        self.form.addRow(self.editor)
        fit = W.QGroupBox("Physical / fitting settings")
        fitform = W.QFormLayout(fit)
        fitform.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        self.fit_fields = {}
        labels = {"voltage_bin_width": "Voltage bin width (V)", "vp_smoothing_width_V": "Vp smoothing width (V)",
                  "te_min_eV": "Minimum Te (eV)", "te_max_eV": "Maximum Te (eV)",
                  "te_min_r2": "Minimum fit R²", "te_margin_from_vp": "Fit margin below Vp (V)",
                  "te_current_floor_frac": "Electron current floor fraction", "te_min_points": "Minimum fit points",
                  "ion_min_snr": "Minimum ion-current SNR"}
        for key, label in labels.items():
            widget = W.QLineEdit(str(settings.fit[key]))
            self.fit_fields[key] = widget
            fitform.addRow(label, widget)
            widget.editingFinished.connect(self.settings_changed)
        for key, label, choices in [("vp_smoothing", "Vp smoothing", ["savgol", "moving", "none"]),
                                    ("ies_method", "Electron saturation", ["high_bias_median", "at_vp"])]:
            widget = combo(choices)
            widget.setCurrentText(settings.fit[key])
            self.fit_fields[key] = widget
            fitform.addRow(label, widget)
            widget.currentTextChanged.connect(self.settings_changed)
        self.ideal = W.QCheckBox("Enforce ideal planar-Maxwellian checks")
        self.ideal.setChecked(settings.fit["enforce_ideal_model_checks"])
        fitform.addRow(self.ideal)
        self.form.addRow(fit)
        self.tabs = W.QTabWidget()
        split.addWidget(self.tabs)
        self.trace_view = TraceView(data.coords["time"], appearance)
        self.tabs.addTab(self.trace_view, "Representative traces")
        result_panel = W.QWidget()
        result_layout = W.QVBoxLayout(result_panel)
        self.quantity = combo(list(analysis.QUANTITIES))
        result_layout.addWidget(self.quantity)
        self.result_fig = plotting.figure(appearance)
        self.result_canvas = FigureCanvasQTAgg(self.result_fig)
        self.result_toolbar = NavigationToolbar2QT(self.result_canvas, self)
        result_layout.addWidget(self.result_toolbar)
        result_layout.addWidget(self.result_canvas)
        self.tabs.addTab(result_panel, "Derived quantities")
        self.result_canvas.mpl_connect("button_press_event", self.result_clicked)
        self.results = W.QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setMaximumHeight(125)
        layout.addWidget(self.results)
        self.message = W.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = W.QHBoxLayout()
        self.action_buttons = []
        for label, callback in [("Process This Shot", self.process_shot), ("Process All Shots", self.process_all),
                                ("Reset View", self.reset_view), ("Exit to Main", self.reject)]:
            button = W.QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
            self.action_buttons.append(button)
        self.cancel_button = W.QPushButton("Cancel processing")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_event.set)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.progress.connect(lambda n, total: self.message.setText(f"Processed {n:,} / {total:,} traces…"))
        self.trace_view.selected.connect(self.editor.set_interval)
        self.editor.changed.connect(self.window_changed)
        for widget in (self.voltage, self.current):
            widget.currentTextChanged.connect(self.settings_changed)
        for widget in (self.invert, self.zero, self.ideal):
            widget.toggled.connect(self.settings_changed)
        self.quantity.currentTextChanged.connect(self.draw_results)
        self.settings.interval = self.editor.interval()
        self.selection_changed()
        self.trace_view.toolbar.push_current()

    def index(self):
        return tuple(w.value() for w in self.indices.values())

    def collect(self):
        self.trace_view.flush_limits()
        s = copy.deepcopy(self.settings)
        s.voltage_channel, s.current_channel = self.voltage.currentText(), self.current.currentText()
        for key, widget in self.fields.items():
            setattr(s, key, float(widget.text()))
        s.invert, s.zero_calibrated = self.invert.isChecked(), self.zero.isChecked()
        s.interval = self.editor.interval()
        for key, widget in self.fit_fields.items():
            s.fit[key] = widget.currentText() if isinstance(widget, W.QComboBox) else float(widget.text())
        s.fit["enforce_ideal_model_checks"] = self.ideal.isChecked()
        self.settings.__dict__.update(s.__dict__)
        return s

    def settings_changed(self, *args):
        self.batch = None
        self.result_fig.clear()
        self.result_canvas.draw_idle()
        self.results.clear()
        self.refresh_traces()

    def window_changed(self, interval):
        if interval == self.settings.interval:
            self.trace_view.set_interval(interval)
            return
        self.settings.interval = interval
        self.trace_view.set_interval(interval)
        self.settings_changed()

    def selection_changed(self, *args):
        if not hasattr(self, "trace_view"):
            return
        self.results.clear()
        self.refresh_traces()
        self.draw_results()

    def refresh_traces(self):
        try:
            s = self.collect()
            analysis.validate(self.data, s, analysis=False)
            self.trace_view.traces(*analysis.traces(self.data, s, self.index()))
            self.trace_view.set_interval(s.interval)
            self.message.setText("Zoom, pan, drag or edit the bounds to select the analyzed time interval.")
        except (ValueError, KeyError) as exc:
            empty = np.full(len(self.data.coords["time"]), np.nan)
            self.trace_view.traces(empty, empty)
            self.message.setText(str(exc))
        self.offset_label.setText(f"Offset: {self.settings.offset_mode}" +
                                  (f" ({self.settings.offset_volts:g} digitizer V)" if self.settings.offset_mode == "Constant" else ""))
        description = ", ".join(f"{d}={self.data.coords[d][w.value()]:g}" for d, w in self.indices.items())
        if self.data.shot_numbers is not None:
            description += f" · Global shot {self.data.shot_numbers[self.index()]}"
        self.location.setText(description or "Single point / trace")

    def configure_offset(self):
        try:
            s = self.collect()
            # Permit fixing a previously invalid offset region.
            preview = copy.deepcopy(s)
            preview.offset_mode = "None"
            analysis.validate(self.data, preview, analysis=False)
            dialog = OffsetDialog(self.data, s, self.index(), self.appearance, self)
            if dialog.exec() == W.QDialog.Accepted:
                for key in ("offset_mode", "offset_interval", "offset_volts"):
                    setattr(self.settings, key, getattr(dialog.settings, key))
                self.settings_changed()
        except ValueError as exc:
            self.message.setText(str(exc))

    def start_job(self, fn, done):
        self.cancel_event.clear()
        self.tabs.setEnabled(False)
        self.form.parentWidget().setEnabled(False)
        for button in self.action_buttons:
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = Worker(fn, self)
        self.worker.result.connect(done)
        self.worker.failed.connect(self.message.setText)
        self.worker.finished.connect(self.job_finished)
        self.worker.start()

    def job_finished(self):
        self.tabs.setEnabled(True)
        self.form.parentWidget().setEnabled(True)
        for button in self.action_buttons:
            button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def process_shot(self):
        try:
            s, index = self.collect(), self.index()
            analysis.validate(self.data, s)
            self.results.clear()
            self.message.setText("Analyzing representative trace…")
            self.start_job(lambda: analysis.analyze(self.data, s, index), self.shot_ready)
        except ValueError as exc:
            self.message.setText(str(exc))

    def shot_ready(self, result):
        values, notes = result
        self.results.setPlainText("    ".join(f"{name} = {value:.6g} {analysis.QUANTITIES[name][1]}" for name, value in values.items()) +
                                  ("\nModel notes: " + "; ".join(notes) if notes else ""))
        self.message.setText("Representative trace passed the enabled quality checks.")

    def process_all(self):
        try:
            s = self.collect()
            analysis.validate(self.data, s)
            self.message.setText("Processing all positions, cases and stored shots…")
            self.start_job(lambda: analysis.process_all(self.data, s, self.progress.emit, self.cancel_event.is_set), self.batch_ready)
        except ValueError as exc:
            self.message.setText(str(exc))

    def batch_ready(self, batch):
        self.batch = batch
        total = int(np.prod(self.data.shape[:-1]))
        self.message.setText(f"{total-len(batch.failures):,}/{total:,} traces accepted; {len(batch.failures):,} failed (NaN). Switch quantity without refitting.")
        self.results.setPlainText("\n".join(f"Index {index}: {reason}" for index, reason in batch.failures[:20]) or "All traces passed.")
        self.draw_results()
        self.tabs.setCurrentIndex(1)

    def draw_results(self, *args):
        if self.batch is None:
            return
        name = self.quantity.currentText()
        self.result_ax = plotting.render_derived(self.result_fig, self.batch.dataset(self.data, name), name,
                    case=self.indices["case"].value() if "case" in self.indices else 0,
                    shot=self.indices["shot"].value() if "shot" in self.indices else 0,
                    slices=[self.indices[d].value() for d in self.data.spatial_dims], appearance=self.appearance,
                    cmap=self.parent().cmap.currentText() if hasattr(self.parent(), "cmap") else "viridis")
        value = self.batch.values[name][self.index()]
        self.result_ax.set_title(self.result_ax.get_title() + f"\nSelected: {value:.6g} {analysis.QUANTITIES[name][1]}")
        self.result_toolbar.update()
        self.result_canvas.draw_idle()

    def result_clicked(self, event):
        if event.inaxes != getattr(self, "result_ax", None) or self.result_toolbar.mode or event.xdata is None:
            return
        dims = self.data.spatial_dims
        positions = [event.ydata, event.xdata] if len(dims) == 2 else [event.xdata]
        for d, value in zip(dims, positions):
            self.indices[d].setValue(int(np.argmin(abs(self.data.coords[d]-value))))

    def update_appearance(self, appearance):
        self.appearance = appearance
        self.trace_view.appearance = appearance
        theme = colors(appearance)
        self.trace_view.fig.set_facecolor(theme["bg"])
        for ax, line, selector in zip(self.trace_view.axes, self.trace_view.lines, self.trace_view.selectors):
            plotting.style(ax, theme)
            line.set_color(theme["accent"])
            selector.set_props(facecolor=theme["accent"], alpha=.25)
        self.trace_view.canvas.draw_idle()
        self.draw_results()

    def reset_view(self):
        self.trace_view.reset_view()
        self.tabs.setCurrentIndex(0)
        # Full-range view now also restores the full analysis interval.

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            self.message.setText("Cancel processing or wait for it to finish before exiting.")
            return
        try:
            self.collect()
        except ValueError:
            pass  # The persistent dialog retains unfinished numeric edits as well.
        super().reject()
