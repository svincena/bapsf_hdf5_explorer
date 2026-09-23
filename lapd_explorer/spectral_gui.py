"""Browser spectral workflow; estimators and result coordinates live in spectral.py."""
import copy
import threading
import numpy as np
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from . import spectral as analysis, plotting
from .appearance import colors
from .langmuir_gui import IntervalEditor, TraceView
from .widgets import Worker, combo, spin


def number(value=0., low=-1e15, high=1e15, decimals=6):
    box = W.QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


class SpectralDialog(W.QDialog):
    progress = C.Signal(int, int)

    def __init__(self, data, names, vector, settings, selection, appearance, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectral Analysis")
        self.resize(1380, 900)
        self.data, self.names, self.vector = data, tuple(names), vector
        self.settings, self.appearance = settings, appearance
        self.batch = self.point = self.worker = None
        self.point_index = None
        self.cancel_event = threading.Event()
        self._render_key = None
        self._animation_frames = None
        self._animation_position = 0
        self._phase = None
        self.timer = C.QTimer(self)
        self.timer.timeout.connect(self.advance)
        layout = W.QVBoxLayout(self)
        description = " · ".join(f"{c}: {n}" for c, n in zip("AB", names))
        note = W.QLabel(description + (" · Vector components" if vector else " · Scalar inputs") +
                        "\nUses the browser's currently processed data. " + (" → ".join(data.history) or "Original data") +
                        "\nS_AB = conj(A) × B; positive cross-phase means B leads A. "
                        "PSD is one-sided density; covariance has a separate lag axis.")
        note.setWordWrap(True)
        layout.addWidget(note)
        split = W.QSplitter()
        layout.addWidget(split, 1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        self.control_tabs = W.QTabWidget()
        self.controls = W.QWidget()
        form = W.QFormLayout(self.controls)
        form.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        scroll.setWidget(self.controls)
        self.control_tabs.setMinimumWidth(390)
        self.control_tabs.addTab(scroll, "Estimate")
        split.addWidget(self.control_tabs)
        self.indices = {}
        for d in data.dims[:-1]:
            widget = spin(0, len(data.coords[d])-1, min(selection.get(d, 0), len(data.coords[d])-1))
            self.indices[d] = widget
            form.addRow(f"{d} index", widget)
            widget.valueChanged.connect(self.selection_changed)
        self.location = W.QLabel()
        self.location.setWordWrap(True)
        form.addRow(self.location)
        self.editor = IntervalEditor(data.coords["time"], settings.interval)
        form.addRow(W.QLabel("Analysis interval"))
        form.addRow(self.editor)
        self.fs = analysis.sampling_rate(data.coords["time"])
        form.addRow(W.QLabel(f"Sampling: {self.fs:g} Hz · Δt = {1/self.fs:g} s"))
        self.nperseg = spin(4, 2**24, settings.nperseg)
        self.nfft = spin(4, 2**24, settings.nfft)
        self.overlap = spin(0, 2**24, settings.overlap)
        self.max_lag = spin(0, len(data.coords["time"])-1, settings.max_lag)
        self.window = combo(["hann", "hamming", "blackman", "boxcar"])
        self.detrend = combo(["constant", "linear", "none"])
        self.average = combo(["mean", "median"])
        self.average.setToolTip("Welch segment averaging, distinct from averaging shots. Median estimates can yield coherence > 1; values are not clipped.")
        self.average_shots = W.QCheckBox("Average stored shots before spectra")
        self.average_shots.setChecked(settings.average_shots and "shot" in data.dims)
        self.average_shots.setEnabled("shot" in data.dims)
        self.average_shots.setToolTip("Off: process stored shots independently. On: average waveforms at each position/case first. If the main browser already averaged shots, that average is the input.")
        for key, label in [("nperseg", "Segment samples"), ("nfft", "FFT samples"),
                           ("overlap", "Overlap samples"), ("window", "Window"),
                           ("detrend", "Detrend per segment"), ("average", "Segment averaging"),
                           ("max_lag", "Maximum covariance lag (samples)")]:
            widget = getattr(self, key)
            if isinstance(widget, W.QComboBox):
                widget.setCurrentText(getattr(settings, key))
            form.addRow(label, widget)
        form.addRow(self.average_shots)

        self.tabs = W.QTabWidget()
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([410, 970])
        self.trace_view = TraceView(data.coords["time"], appearance)
        for i, axis in enumerate(self.trace_view.axes):
            axis.set_ylabel(f"{names[i]} ({data.units})" if i < len(names) else "No second channel")
        self.trace_view.axes[1].set_visible(len(names) == 2)
        self.tabs.addTab(self.trace_view, "Input traces / interval")
        self.spectrum_fig, self.spectrum_canvas, self.spectrum_toolbar = self.plot_tab("Spectrum / covariance")
        self.result_fig, self.result_canvas, self.result_toolbar = self.plot_tab("Spatial field / animation")
        self.result_canvas.mpl_connect("button_press_event", self.result_clicked)

        display = W.QGroupBox("Cached result display")
        df = W.QFormLayout(display)
        df.setRowWrapPolicy(W.QFormLayout.WrapAllRows)
        self.quantity = combo(list(analysis.SINGLE + analysis.PAIRED))
        for i in range(len(analysis.SINGLE), self.quantity.count()):
            self.quantity.model().item(i).setEnabled(len(names) == 2)
        self.representation = combo(analysis.REPRESENTATIONS)
        self.degrees = W.QCheckBox("Phase in degrees")
        self.bin_slider = W.QSlider(C.Qt.Horizontal)
        self.bin_slider.setRange(0, 0)
        self.coordinate = number(0., 0.)
        self.coordinate_label = W.QLabel("Frequency (Hz), nearest bin")
        self.actual = W.QLabel("Process a point or the full dataset first.")
        self.actual.setWordWrap(True)
        self.frequency_start = number(0., 0.)
        self.frequency_stop = number(self.fs/2, 0.)
        self.frequency_step = spin(1, 2**24, 1)
        self.cmap = combo(plotting.COLORMAPS)
        self.cmap.setCurrentText(parent.cmap.currentText() if hasattr(parent, "cmap") else "viridis")
        self.lock = W.QCheckBox("Fixed scale during animation")
        self.lock.setChecked(True)
        df.addRow("Quantity", self.quantity)
        df.addRow("Complex representation", self.representation)
        df.addRow(self.degrees)
        df.addRow("Coordinate bin", self.bin_slider)
        df.addRow(self.coordinate_label, self.coordinate)
        df.addRow(self.actual)
        df.addRow("Display / sweep start (Hz)", self.frequency_start)
        df.addRow("Display / sweep end (Hz)", self.frequency_stop)
        df.addRow("Sweep every Nth bin", self.frequency_step)
        df.addRow("Color map", self.cmap)
        df.addRow(self.lock)
        display_scroll = W.QScrollArea()
        display_scroll.setWidgetResizable(True)
        display_scroll.setWidget(display)
        self.control_tabs.addTab(display_scroll, "Display")

        animation = W.QGroupBox("Animation")
        af = W.QFormLayout(animation)
        af.setRowWrapPolicy(W.QFormLayout.WrapAllRows)
        self.animation_mode = combo(["Frequency at fixed phase", "Phase at fixed frequency"])
        self.animation_mode.model().item(1).setEnabled(len(names) == 2)
        self.interpretation = combo(["Quantity representation", "Phase projection", "Vector components"])
        self.interpretation.model().item(1).setEnabled(len(names) == 2)
        self.interpretation.model().item(2).setEnabled(vector and len(data.spatial_dims) == 2)
        self.interpretation.setToolTip("Vector components use coherent peak phasors referenced to interval start, not Welch power. Phase projection uses the cross-phase convention above.")
        self.amplitude = combo(["Quantity", "A", "B"])
        self.amplitude.setToolTip("A/B: sqrt(PSD) in input units/√Hz, with phase from S_AB. Quantity: complex cross-power or coherency itself.")
        self.phase_steps = spin(2, 10000, 36)
        self.phase = number(0., -36000., 36000., 2)
        self.fps = spin(1, 60, 10)
        self.loop = W.QCheckBox("Loop")
        self.loop.setChecked(True)
        for label, widget in [("Advance", self.animation_mode), ("Field", self.interpretation),
                              ("Scalar amplitude", self.amplitude), ("Phase frames", self.phase_steps),
                              ("Fixed / starting phase (deg)", self.phase), ("Frames per second", self.fps)]:
            af.addRow(label, widget)
        af.addRow(self.loop)
        self.play = W.QPushButton("Play")
        self.play.clicked.connect(self.toggle_play)
        self.step = W.QPushButton("Step frame")
        self.step.clicked.connect(self.step_frame)
        row = W.QHBoxLayout()
        row.addWidget(self.play)
        row.addWidget(self.step)
        af.addRow(row)
        self.frame_label = W.QLabel("Frequency and phase are independent coordinates.")
        self.frame_label.setWordWrap(True)
        af.addRow(self.frame_label)
        animation_scroll = W.QScrollArea()
        animation_scroll.setWidgetResizable(True)
        animation_scroll.setWidget(animation)
        self.control_tabs.addTab(animation_scroll, "Animate")
        self.message = W.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = W.QHBoxLayout()
        self.action_buttons = []
        for label, callback in [("Process This Point/Shot", self.process_point), ("Process All", self.process_all),
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
        self.progress.connect(lambda n, total: self.message.setText(f"Processed {n:,} / {total:,} locations/shots…"))
        self.trace_view.selected.connect(self.editor.set_interval)
        self.editor.changed.connect(self.window_changed)
        for widget in (self.nperseg, self.nfft, self.overlap, self.max_lag):
            widget.valueChanged.connect(self.settings_changed)
        for widget in (self.window, self.detrend, self.average):
            widget.currentTextChanged.connect(self.settings_changed)
        self.average_shots.toggled.connect(self.settings_changed)
        self.quantity.currentTextChanged.connect(self.quantity_changed)
        for widget in (self.representation, self.cmap, self.interpretation, self.amplitude, self.animation_mode):
            widget.currentTextChanged.connect(self.display_changed)
        for widget in (self.degrees, self.lock):
            widget.toggled.connect(self.display_changed)
        for widget in (self.phase, self.phase_steps, self.frequency_step, self.fps, self.frequency_start, self.frequency_stop):
            widget.valueChanged.connect(self.display_changed)
        self.bin_slider.valueChanged.connect(self.coordinate_bin_changed)
        self.coordinate.editingFinished.connect(self.coordinate_changed)
        self.settings.interval = self.editor.interval()
        self.selection_changed()
        self.quantity_changed()

    def plot_tab(self, title):
        panel = W.QWidget()
        layout = W.QVBoxLayout(panel)
        fig = plotting.figure(self.appearance)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self.tabs.addTab(panel, title)
        return fig, canvas, toolbar

    def index(self):
        return tuple(w.value() for w in self.indices.values())

    def result_index(self, result):
        return tuple(self.indices[d].value() for d in result.source.dims[:-1])

    def collect(self):
        self.trace_view.flush_limits()
        s = analysis.Settings(interval=self.editor.interval(),
                              nperseg=self.nperseg.value(), nfft=self.nfft.value(), overlap=self.overlap.value(),
                              window=self.window.currentText(), detrend=self.detrend.currentText(),
                              average=self.average.currentText(), average_shots=self.average_shots.isChecked(),
                              max_lag=self.max_lag.value())
        self.settings.__dict__.update(vars(s))
        return copy.deepcopy(s)

    def settings_changed(self, *args):
        self.stop_play()
        self.collect()
        self.batch = self.point = None
        self._animation_frames = None
        self._phase = None
        self._render_key = None
        for fig, canvas in ((self.result_fig, self.result_canvas), (self.spectrum_fig, self.spectrum_canvas)):
            fig.clear()
            canvas.draw_idle()
        self.selection_changed()
        self.message.setText("Analysis settings changed. Process a point or all locations to refresh the spectra.")

    def window_changed(self, interval):
        self.trace_view.set_interval(interval)
        if interval != self.settings.interval:
            self.settings.interval = interval
            self.settings_changed()

    def selection_changed(self, *args):
        if not hasattr(self, "trace_view"):
            return
        self.stop_play()
        self._animation_frames = None
        self._phase = None
        index = self.index()
        key = tuple(slice(None) if d == "shot" and self.average_shots.isChecked() else i
                    for d, i in zip(self.data.dims[:-1], index))
        traces = [self.data.channels[n][key] for n in self.names]
        if self.average_shots.isChecked() and "shot" in self.data.dims:
            traces = [a.mean(axis=0) for a in traces]
        self.trace_view.traces(traces[0], traces[-1])
        self.trace_view.set_interval(self.editor.interval())
        if "shot" in self.indices:
            self.indices["shot"].setEnabled(not self.average_shots.isChecked())
        description = ", ".join(f"{d}={self.data.coords[d][w.value()]:g}" for d, w in self.indices.items()
                                if d != "shot" or not self.average_shots.isChecked())
        if self.average_shots.isChecked():
            description += " · averaged shots"
        elif self.data.shot_numbers is not None:
            description += f" · Global shot {self.data.shot_numbers[index]}"
        self.location.setText(description or "Single point / trace")
        self._render_key = None
        self.draw_spectrum()
        self.draw_results()

    def current_result(self):
        return self.batch or (self.point if self.point_index == self.index() else None)

    def start_job(self, all_points):
        try:
            s = self.collect()
            analysis.validate(self.data, self.names, s)
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self.stop_play()
        self.cancel_event.clear()
        self.control_tabs.setEnabled(False)
        self.tabs.setEnabled(False)
        for button in self.action_buttons:
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        index = None if all_points else self.index()
        self.message.setText("Processing all locations/shots…" if all_points else "Processing selected point/shot…")
        self.worker = Worker(lambda: analysis.process(self.data, self.names, s, self.progress.emit,
                                                     self.cancel_event.is_set, index=index), self)
        self.worker.result.connect(self.batch_ready if all_points else self.point_ready)
        self.worker.failed.connect(self.message.setText)
        self.worker.finished.connect(self.job_finished)
        self.worker.start()

    def process_point(self):
        self.start_job(False)

    def process_all(self):
        self.start_job(True)

    def job_finished(self):
        self.control_tabs.setEnabled(True)
        self.tabs.setEnabled(True)
        for button in self.action_buttons:
            button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def point_ready(self, result):
        self.point, self.point_index = result, self.index()
        self.results_ready(result)
        self.tabs.setCurrentIndex(1)

    def batch_ready(self, result):
        self.batch = result
        self.results_ready(result)
        self.tabs.setCurrentIndex(2 if result.source.spatial_dims else 1)

    def results_ready(self, result):
        total = int(np.prod(result.source.shape[:-1]))
        self.message.setText(f"{total-len(result.failures):,}/{total:,} accepted; {len(result.failures):,} failed (NaN). "
                             "Display changes use cached arrays." +
                             (" Failures: " + "; ".join(f"{i}: {reason}" for i, reason in result.failures[:4]) if result.failures else ""))
        for widget in (self.frequency_start, self.frequency_stop):
            widget.blockSignals(True)
            widget.setRange(0., result.frequency[-1])
            widget.blockSignals(False)
        self.quantity_changed()

    def quantity_changed(self, *args):
        result = self.current_result()
        q = self.quantity.currentText()
        covariance = "covariance" in q.lower()
        self.representation.setEnabled(q in analysis.COMPLEX)
        self.degrees.setEnabled(q == "Cross-phase" or q in analysis.COMPLEX)
        self.coordinate_label.setText("Lag (s), nearest sample" if covariance else "Frequency (Hz), nearest bin")
        if result is not None:
            coord = result.coordinate(q)
            self.bin_slider.blockSignals(True)
            self.bin_slider.setRange(0, len(coord)-1)
            if covariance:
                self.bin_slider.setValue(len(coord)//2)
            self.bin_slider.blockSignals(False)
            self.coordinate.setRange(coord[0], coord[-1])
        self.play.setEnabled(not covariance)
        self.step.setEnabled(not covariance)
        self.display_changed()

    def display_changed(self, *args):
        self.stop_play()
        self._animation_frames = None
        self._phase = None
        self._render_key = None
        self.amplitude.model().item(0).setEnabled(self.quantity.currentText() in analysis.COMPLEX)
        if (self.interpretation.currentText() == "Phase projection" and
                self.quantity.currentText() not in analysis.COMPLEX and self.amplitude.currentText() == "Quantity"):
            self.amplitude.setCurrentText("A")
        self.amplitude.setEnabled(self.interpretation.currentText() == "Phase projection")
        self.phase_steps.setEnabled(self.animation_mode.currentIndex() == 1)
        self.draw_spectrum()
        self.draw_results()

    def coordinate_changed(self):
        result = self.current_result()
        if result is not None:
            coord = result.coordinate(self.quantity.currentText())
            self.bin_slider.setValue(int(np.argmin(abs(coord-self.coordinate.value()))))
            self.coordinate_bin_changed()

    def coordinate_bin_changed(self, *args):
        self.stop_play()
        self._animation_frames = None
        self._phase = None
        self._render_key = None
        self.bin_changed()

    def bin_changed(self, *args):
        self.draw_results()
        result = self.current_result()
        if result is not None and hasattr(self, "spectrum_cursor"):
            coord = result.coordinate(self.quantity.currentText())[self.bin_slider.value()]
            self.spectrum_cursor.set_xdata([coord, coord])
            self.spectrum_canvas.draw_idle()

    def draw_spectrum(self):
        result = self.current_result()
        self.spectrum_fig.clear()
        if result is None:
            self.spectrum_canvas.draw_idle()
            return
        q = self.quantity.currentText()
        kind, degrees = self.representation.currentText(), self.degrees.isChecked()
        values = result.displayed(q, kind, degrees, self.result_index(result))
        coord = result.coordinate(q)
        ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_fig.set_facecolor(colors(self.appearance)["bg"])
        plotting.style(ax, colors(self.appearance))
        ax.plot(coord, values)
        self.spectrum_cursor = ax.axvline(coord[min(self.bin_slider.value(), len(coord)-1)], ls="--", lw=.9)
        covariance = "covariance" in q.lower()
        ax.set(xlabel=("Lag (s); positive = B follows A" if q == "Cross-covariance" else "Lag (s)") if covariance else "Frequency (Hz)",
               ylabel=result.units(q, kind, degrees), title=f"{q} · {', '.join(self.names)}\n{self.location.text()}")
        if not covariance and self.frequency_start.value() < self.frequency_stop.value():
            ax.set_xlim(self.frequency_start.value(), self.frequency_stop.value())
        self.spectrum_toolbar.update()
        self.spectrum_canvas.draw_idle()

    def frame(self, result, bin_index, phase=None):
        q = self.quantity.currentText()
        covariance = "covariance" in q.lower()
        interpretation = self.interpretation.currentText() if not covariance else "Quantity representation"
        vector = interpretation == "Vector components"
        projection = interpretation == "Phase projection" or vector
        if projection and phase is None:
            phase = np.deg2rad(self.phase.value())
        data = result.frame(q, bin_index, kind=self.representation.currentText(), degrees=self.degrees.isChecked(),
                            phase=phase if projection else None, amplitude=self.amplitude.currentText(), vector=vector)
        coord = result.coordinate(q)[bin_index]
        label = f"lag = {coord:g} s" if covariance else f"f = {coord:g} Hz"
        detail = f" · phase = {np.rad2deg(phase):.4g}°" if projection else ""
        if vector:
            label = "Coherent peak vector · " + label
        elif projection:
            label = f"{q if self.amplitude.currentText() == 'Quantity' else 'sqrt(PSD '+self.amplitude.currentText()+')'} × cos(phase − cross-phase) · " + label
        else:
            label = q + (f" ({self.representation.currentText()})" if q in analysis.COMPLEX else "") + " · " + label
        return data, f"{label}{detail}\n{', '.join(self.names)} · {self.location.text()}", vector

    def draw_results(self):
        result = self.current_result()
        if result is None:
            self.result_fig.clear()
            self.result_canvas.draw_idle()
            return
        try:
            bin_index = min(self.bin_slider.value(), len(result.coordinate(self.quantity.currentText()))-1)
            coord = result.coordinate(self.quantity.currentText())[bin_index]
            self.coordinate.blockSignals(True)
            self.coordinate.setValue(coord)
            self.coordinate.blockSignals(False)
            self.actual.setText(f"Actual {'lag (s)' if 'covariance' in self.quantity.currentText().lower() else 'frequency (Hz)'}: {coord:.9g} · bin {bin_index}")
            if self.batch is None:
                self.result_fig.clear()
                ax = self.result_fig.add_subplot(111)
                plotting.style(ax, colors(self.appearance))
                ax.text(.5, .5, "Process All to inspect spatial fields and animate.", ha="center", transform=ax.transAxes)
                self.result_canvas.draw_idle()
                return
            data, title, vector = self.frame(result, bin_index, self._phase)
            key = (id(result), self.quantity.currentText(), self.representation.currentText(),
                   self.degrees.isChecked(), self.cmap.currentText(), self.interpretation.currentText(),
                   self.amplitude.currentText(), self.index(), self.appearance, self.lock.isChecked())
            if self._render_key == key:
                self.result_fig._lapd_update_derived(data, title)
            else:
                self.result_ax = plotting.render_derived(
                    self.result_fig, data, next(iter(data.channels)),
                    case=self.indices["case"].value() if "case" in self.indices else 0,
                    shot=self.indices["shot"].value() if "shot" in data.dims else 0,
                    slices=[self.indices[d].value() for d in data.spatial_dims],
                    cmap=self.cmap.currentText(), appearance=self.appearance, title=title, vector=vector,
                    limits=getattr(self, "_animation_limits", None) if self._animation_frames is not None else None,
                    arrow_cmap=self.parent().arrow_cmap.currentText() if hasattr(self.parent(), "arrow_cmap") else "Solid white")
                self._render_key = key
                self.result_toolbar.update()
            self.result_canvas.draw_idle()
        except (ValueError, IndexError) as exc:
            self.message.setText(str(exc))

    def result_clicked(self, event):
        if event.inaxes != getattr(self, "result_ax", None) or self.result_toolbar.mode or event.xdata is None:
            return
        dims = self.data.spatial_dims
        positions = [event.ydata, event.xdata] if len(dims) == 2 else [event.xdata]
        for d, value in zip(dims, positions):
            self.indices[d].setValue(int(np.argmin(abs(self.data.coords[d]-value))))

    def prepare_animation(self):
        if self.batch is None:
            raise ValueError("Process All before animating spatial fields.")
        if "covariance" in self.quantity.currentText().lower():
            raise ValueError("Covariance uses lag, not frequency; choose a spectral quantity for animation.")
        if self.animation_mode.currentIndex() == 1:
            if self.interpretation.currentText() == "Quantity representation":
                raise ValueError("Phase animation needs Phase projection or Vector components.")
            self._animation_frames = [(self.bin_slider.value(), np.deg2rad(self.phase.value()) + 2*np.pi*i/self.phase_steps.value())
                                      for i in range(self.phase_steps.value())]
        else:
            indices = self.batch.frequency_indices(self.frequency_start.value(), self.frequency_stop.value(), self.frequency_step.value())
            self._animation_frames = [(int(i), np.deg2rad(self.phase.value())) for i in indices]
        # One frame at a time: no space × frequency × phase allocation.
        self._animation_limits = None
        if self.lock.isChecked():
            low, high = np.inf, -np.inf
            for i, phase in self._animation_frames:
                data, _, vector = self.frame(self.batch, i, phase)
                parts = [data.selected(n, self.indices['case'].value() if 'case' in self.indices else 0,
                                       self.indices['shot'].value() if 'shot' in data.dims else 0) for n in data.channels]
                values = np.hypot(*parts) if vector else parts[0]
                finite = values[np.isfinite(values)]
                if finite.size:
                    low, high = min(low, float(finite.min())), max(high, float(finite.max()))
            self._animation_limits = plotting.finite_limits(np.array([low, high]))
        self._animation_position = 0
        self._render_key = None

    def show_animation_frame(self):
        i, self._phase = self._animation_frames[self._animation_position]
        self.bin_slider.blockSignals(True)
        self.bin_slider.setValue(i)
        self.bin_slider.blockSignals(False)
        self.bin_changed()
        self.frame_label.setText(f"{self.animation_mode.currentText()} · frame {self._animation_position+1}/{len(self._animation_frames)}\n"
                                 f"f = {self.batch.frequency[i]:.9g} Hz · phase = {np.rad2deg(self._phase):.5g}°")

    def toggle_play(self):
        if self.timer.isActive():
            self.stop_play()
            return
        try:
            if self._animation_frames is None:
                self.prepare_animation()
            self.show_animation_frame()
            self.tabs.setCurrentIndex(2)
            self.timer.start(round(1000/self.fps.value()))
            self.play.setText("Pause")
        except ValueError as exc:
            self._animation_frames = None
            self.message.setText(str(exc))

    def stop_play(self):
        self.timer.stop()
        if hasattr(self, "play"):
            self.play.setText("Play")

    def advance(self):
        if self._animation_position+1 >= len(self._animation_frames):
            if not self.loop.isChecked():
                self.stop_play()
                return
            self._animation_position = 0
        else:
            self._animation_position += 1
        self.show_animation_frame()

    def step_frame(self):
        self.stop_play()
        try:
            if self._animation_frames is None:
                self.prepare_animation()
                self.show_animation_frame()
            else:
                self.advance()
            self.tabs.setCurrentIndex(2)
        except ValueError as exc:
            self._animation_frames = None
            self.message.setText(str(exc))

    def reset_view(self):
        self.stop_play()
        # Reset navigation while preserving estimator settings and interval.
        self.trace_view.set_interval(self.editor.interval())
        for axis in self.trace_view.axes:
            axis.relim()
            axis.autoscale(enable=True, axis="y")
        self.trace_view.toolbar.update()
        self.trace_view.canvas.draw_idle()
        self._render_key = None
        self.draw_spectrum()
        self.draw_results()

    def update_appearance(self, appearance):
        self.appearance = appearance
        self.trace_view.fig.set_facecolor(colors(appearance)["bg"])
        for ax in self.trace_view.axes:
            plotting.style(ax, colors(appearance))
        for line in self.trace_view.lines:
            line.set_color(colors(appearance)["accent"])
        self.trace_view.canvas.draw_idle()
        self._render_key = None
        self.draw_spectrum()
        self.draw_results()

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            self.message.setText("Cancel processing or wait for it to finish before exiting.")
            return
        self.stop_play()
        self.collect()
        super().reject()
