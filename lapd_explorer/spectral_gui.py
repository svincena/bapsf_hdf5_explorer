"""Browser spectral workflow; estimators and result coordinates live in spectral.py."""
import copy
import os
import threading
import numpy as np
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from . import spectral as analysis, plotting
from . import spectrogram
from .spectrogram_gui import Controls as SpectrogramControls
from .trace_picker import TracePicker
from .appearance import FacilityLogo, colors
from .langmuir_gui import IntervalEditor, TraceView
from .widgets import ScientificDoubleSpinBox, Worker, VectorArrowControls, combo, colormap_combo, spin, form_layout


def number(value=0., low=-1e15, high=1e15, decimals=6):
    box = ScientificDoubleSpinBox()
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
        self.spectrogram_result = None
        self._spectrogram_requested = False
        self.spectrogram_timer = C.QTimer(self)
        self.spectrogram_timer.setSingleShot(True)
        self.spectrogram_timer.setInterval(250)
        self.spectrogram_timer.timeout.connect(lambda: self.process_spectrogram(show=False))
        self.cancel_event = threading.Event()
        self._render_key = None
        self._animation_frames = None
        self._animation_position = 0
        self._phase = None
        self.timer = C.QTimer(self)
        self.timer.timeout.connect(self.advance)
        layout = W.QVBoxLayout(self)
        header = W.QHBoxLayout()
        title = W.QLabel("Spectral analysis")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        self.facility_logo = FacilityLogo(appearance, height=46)
        header.addWidget(self.facility_logo)
        layout.addLayout(header)
        description = " · ".join(f"{c}: {n}" for c, n in zip("AB", names))
        note = W.QLabel(description + (" · Vector components" if vector else " · Scalar inputs") +
                        "\nUses the browser's currently processed data. " + (" → ".join(data.history) or "Original data") +
                        "\nS_AB = conj(A) × B; positive cross-phase means B leads A. "
                        "Welch PSD uses one-sided density; covariance has a separate lag axis.")
        note.setWordWrap(True)
        layout.addWidget(note)
        split = W.QSplitter()
        layout.addWidget(split, 1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        self.control_tabs = W.QTabWidget()
        self.controls = W.QWidget()
        form = form_layout(self.controls)
        scroll.setWidget(self.controls)
        self.control_tabs.setMinimumWidth(315)
        self.control_tabs.addTab(scroll, "Estimate")
        split.addWidget(self.control_tabs)
        self.trace_picker = TracePicker(data, selection, self)
        self.indices = self.trace_picker.indices
        self.trace_picker.changed.connect(self.selection_changed)
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
        self.fft_workers = spin(1, max(os.cpu_count() or 1, settings.fft_workers), settings.fft_workers)
        self.fft_workers.setToolTip("Maximum threads for batched FFTs. Use 1 for serial FFTs; more workers may help large datasets but can slow small jobs.")
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
                           ("fft_workers", "FFT workers"),
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
        playback = W.QHBoxLayout()
        playback.addWidget(W.QLabel("Playback"))
        self.playback_slider = W.QSlider(C.Qt.Horizontal)
        self.playback_slider.setAccessibleName("Animation playback frame")
        self.playback_slider.setRange(0, 0)
        self.playback_slider.setEnabled(False)
        self.playback_slider.setToolTip("Seek a frequency or phase frame. Dragging or using the arrow/Home/End keys pauses playback; Play continues from the selected frame.")
        self.playback_slider.valueChanged.connect(self.seek_animation)
        self.playback_slider.sliderPressed.connect(self.seek_current_animation_frame)
        playback.addWidget(self.playback_slider, 1)
        self.playback_position = W.QLabel("—")
        playback.addWidget(self.playback_position)
        self.tabs.widget(2).layout().addLayout(playback)

        display = W.QGroupBox("Cached result display")
        df = form_layout(display)
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
        self.cmap = colormap_combo(parent.cmap.currentText() if hasattr(parent, "cmap") else "viridis")
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
        self.arrow_style = VectorArrowControls(parent.arrow_style.settings() if hasattr(parent, "arrow_style") else None)
        self.arrow_style.setEnabled(vector and len(data.spatial_dims) == 2)
        df.addRow(self.arrow_style)
        self.arrow_style.changed.connect(self.display_changed)
        df.addRow(self.lock)
        display_scroll = W.QScrollArea()
        display_scroll.setWidgetResizable(True)
        display_scroll.setWidget(display)
        self.control_tabs.addTab(display_scroll, "Display")

        animation = W.QGroupBox("Animation")
        af = form_layout(animation)
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
        trace_scroll = W.QScrollArea()
        trace_scroll.setWidgetResizable(True)
        trace_scroll.setWidget(self.trace_picker)
        self.control_tabs.addTab(trace_scroll, "Trace")
        self.spectrogram_controls = SpectrogramControls(self.fs, len(data.coords["time"]), names)
        self.spectrogram_controls.export.setEnabled(False)
        spectrogram_scroll = W.QScrollArea()
        spectrogram_scroll.setWidgetResizable(True)
        spectrogram_scroll.setWidget(self.spectrogram_controls)
        self.control_tabs.addTab(spectrogram_scroll, "Spectrogram")
        self.spectrogram_fig, self.spectrogram_canvas, self.spectrogram_toolbar = self.plot_tab("Spectrogram")
        self.spectrogram_tab = self.tabs.count()-1
        if data.spatial_dims:
            self.location_fig, self.location_canvas, self.location_toolbar = self.plot_tab("Locations — click to choose")
            self.location_canvas.mpl_connect("button_press_event", self.location_clicked)
        self.spectrogram_controls.estimate_changed.connect(self.spectrogram_settings_changed)
        self.spectrogram_controls.display_changed.connect(self.draw_spectrogram)
        self.spectrogram_controls.export_requested.connect(self.export_spectrogram)
        self.spectrogram_controls.auto_update.toggled.connect(self.schedule_spectrogram)
        self.tabs.currentChanged.connect(self.plot_tab_changed)
        layout.addWidget(self.trace_picker.navigation)
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
        self.spectrogram_button = W.QPushButton("Make spectrogram")
        self.spectrogram_button.clicked.connect(lambda: self.process_spectrogram())
        buttons.insertWidget(2, self.spectrogram_button)
        self.action_buttons.append(self.spectrogram_button)
        self.trace_picker.failed.connect(self.message.setText)
        self.cancel_button = W.QPushButton("Cancel processing")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_event.set)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.progress.connect(self.processing_progress)
        self.trace_view.selected.connect(self.editor.set_interval)
        self.editor.changed.connect(self.window_changed)
        for widget in (self.nperseg, self.nfft, self.overlap, self.max_lag):
            widget.valueChanged.connect(self.settings_changed)
        self.fft_workers.valueChanged.connect(lambda value: setattr(self.settings, "fft_workers", value))
        for widget in (self.window, self.detrend, self.average):
            widget.currentTextChanged.connect(self.settings_changed)
        self.average_shots.toggled.connect(self.settings_changed)
        self.quantity.currentTextChanged.connect(self.quantity_changed)
        for widget in (self.representation, self.cmap, self.interpretation, self.amplitude, self.animation_mode):
            widget.currentTextChanged.connect(self.display_changed)
        self.cmap.currentTextChanged.connect(self.draw_locations)
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
                              max_lag=self.max_lag.value(), fft_workers=self.fft_workers.value())
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
            self.indices["shot"].setEnabled(True)
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
        self.draw_locations()
        self.update_playback_slider()
        self.invalidate_spectrogram()

    def processing_progress(self, done, total):
        unit = "windows" if getattr(self, "_job_kind", "spectrum") == "spectrogram" else "locations/shots"
        self.message.setText(f"Processed {done:,} / {total:,} {unit} · {total-done:,} remaining…")

    def plot_tab_changed(self, value):
        if value == self.spectrogram_tab:
            self.control_tabs.setCurrentIndex(4)

    def spectrogram_settings_changed(self):
        self.spectrogram_result = None
        self.draw_spectrogram()
        self.schedule_spectrogram()

    def invalidate_spectrogram(self):
        result = self.spectrogram_result
        if result is not None and (result.index != self.index() or
                                  result.settings != self.spectrogram_controls.settings(self.editor.interval())):
            self.spectrogram_result = None
            self.draw_spectrogram()
        self.schedule_spectrogram()

    def schedule_spectrogram(self, *args):
        if (self._spectrogram_requested and self.spectrogram_result is None
                and self.spectrogram_controls.auto_update.isChecked()
                and not (self.worker and self.worker.isRunning()) and self.isVisible()):
            self.spectrogram_timer.start()
        else:
            self.spectrogram_timer.stop()

    def process_spectrogram(self, *, show=True):
        if self.worker and self.worker.isRunning():
            return
        self.spectrogram_timer.stop()
        try:
            self.trace_view.flush_limits()
            settings = self.spectrogram_controls.settings(self.editor.interval())
            index = self.index()
            spectrogram.validate(self.data, self.names, settings, index)
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self._spectrogram_requested = True
        self._spectrogram_show_on_ready = show
        self.stop_play()
        self.cancel_event.clear()
        self._job_kind = "spectrogram"
        self.control_tabs.setEnabled(False)
        self.tabs.setEnabled(False)
        for button in self.action_buttons:
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.message.setText("Computing the selected trace's spectrogram…")
        self.worker = Worker(lambda: spectrogram.process(self.data, self.names, settings, index,
                                                        self.progress.emit, self.cancel_event.is_set), self)
        self.worker.result.connect(self.spectrogram_ready)
        self.worker.failed.connect(self.spectrogram_failed)
        self.worker.finished.connect(self.job_finished)
        self.worker.start()

    def spectrogram_failed(self, message):
        # Invalid settings and canceled work wait for an explicit retry.
        self._spectrogram_requested = False
        self.message.setText(message)

    def spectrogram_ready(self, result):
        if (result.index != self.index() or
                result.settings != self.spectrogram_controls.settings(self.editor.interval())):
            return
        self.spectrogram_result = result
        if self.spectrogram_controls.sides.currentIndex() == 1 and self.spectrogram_controls.frequency_min.value() == 0:
            self.spectrogram_controls.frequency_min.blockSignals(True)
            self.spectrogram_controls.frequency_min.setValue(result.frequency[0])
            self.spectrogram_controls.frequency_min.blockSignals(False)
        if result.settings.sides == "onesided" and self.spectrogram_controls.frequency_min.value() < 0:
            self.spectrogram_controls.frequency_min.blockSignals(True)
            self.spectrogram_controls.frequency_min.setValue(0)
            self.spectrogram_controls.frequency_min.blockSignals(False)
        self.draw_spectrogram()
        if getattr(self, "_spectrogram_show_on_ready", True):
            self.tabs.setCurrentIndex(self.spectrogram_tab)
        self.message.setText(f"Spectrogram: {len(result.time):,} windows × {len(result.frequency):,} frequency bins. "
                             "Display changes use cached results; trace cycling updates automatically when enabled.")

    def draw_spectrogram(self, *args):
        if not hasattr(self, "spectrogram_canvas"):
            return
        self.spectrogram_controls.export.setEnabled(self.spectrogram_result is not None)
        if self.spectrogram_result is None:
            self.spectrogram_fig.clear()
            ax = self.spectrogram_fig.add_subplot(111)
            plotting.style(ax, colors(self.appearance))
            ax.set_axis_off()
            ax.text(.5, .5, "Select a trace, set the interval, then choose Make spectrogram.", ha="center", va="center", transform=ax.transAxes, color=colors(self.appearance)["fg"])
            self.spectrogram_canvas.draw_idle()
            return
        try:
            self.spectrogram_controls.render(self.spectrogram_result, self.spectrogram_fig, self.appearance)
            self.spectrogram_toolbar.update()
            self.spectrogram_canvas.draw_idle()
        except ValueError as exc:
            self.message.setText(str(exc))

    def export_spectrogram(self):
        if self.spectrogram_result is None:
            self.message.setText("Make a spectrogram before saving its data.")
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Save spectrogram data", "spectrogram.npz", "NumPy archive (*.npz)")
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path += ".npz"
        try:
            from .app import atomic_save
            atomic_save(path, lambda temporary: spectrogram.save(temporary, self.spectrogram_result))
            self.message.setText(f"Saved spectrogram data to {path}")
        except (OSError, ValueError) as exc:
            self.message.setText(str(exc))

    def draw_locations(self):
        if not hasattr(self, "location_canvas"):
            return
        self.location_fig.clear()
        self.location_fig.set_facecolor(colors(self.appearance)["bg"])
        self.location_ax = ax = self.location_fig.add_subplot(111)
        plotting.style(ax, colors(self.appearance))
        case = self.indices["case"].value() if "case" in self.indices else 0
        shot = self.indices["shot"].value() if "shot" in self.indices else 0
        middle = np.mean(self.editor.interval())
        sample = int(np.argmin(abs(self.data.coords["time"]-middle)))
        values = self.data.selected(self.names[0], case, shot)[..., sample]
        dims = self.data.spatial_dims
        if len(dims) == 2:
            y, x = (self.data.coords[d] for d in dims)
            # Singleton spatial axes need an explicit drawable cell width.
            def edges(v):
                return (v[0]-.5, v[0]+.5) if len(v) == 1 else (v[0]-(v[1]-v[0])/2, v[-1]+(v[-1]-v[-2])/2)
            if len(x) > 1 and len(y) > 1:
                image = ax.pcolormesh(x, y, values, shading="nearest", cmap=self.cmap.currentText())
            else:
                image = ax.imshow(values, origin="lower", aspect="auto", extent=(*edges(x), *edges(y)), cmap=self.cmap.currentText())
            bar = self.location_fig.colorbar(image, ax=ax, label=self.data.units)
            plotting.style(bar.ax, colors(self.appearance))
            ax.plot(x[self.indices[dims[1]].value()], y[self.indices[dims[0]].value()], marker="o", ms=10, mec="white", mfc="none", mew=2)
            ax.set(xlabel=f"{dims[1]} ({self.data.spatial_units})", ylabel=f"{dims[0]} ({self.data.spatial_units})")
        else:
            d = dims[0]
            ax.plot(self.data.coords[d], values)
            ax.axvline(self.data.coords[d][self.indices[d].value()], ls="--")
            ax.set(xlabel=f"{d} ({self.data.spatial_units})", ylabel=f"{self.names[0]} ({self.data.units})")
        ax.set_title(f"Click a location · {self.names[0]} at {self.data.coords['time'][sample]*1000:g} ms\n"
                     f"case index {case} · shot index {shot}")
        self.location_toolbar.update()
        self.location_canvas.draw_idle()

    def location_clicked(self, event):
        if event.inaxes != getattr(self, "location_ax", None) or self.location_toolbar.mode or event.xdata is None:
            return
        dims = self.data.spatial_dims
        positions = [event.ydata, event.xdata] if len(dims) == 2 else [event.xdata]
        index = list(self.index())
        for d, value in zip(dims, positions):
            if value is None or not np.isfinite(value):
                return
            index[self.data.dims.index(d)] = int(np.argmin(abs(self.data.coords[d]-value)))
        self.trace_picker.set_index(index)

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
        self.spectrogram_timer.stop()
        self.cancel_event.clear()
        self._job_kind = "spectrum"
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
        self.schedule_spectrogram()

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
        self.update_playback_slider()

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
        self.update_playback_slider()

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
            key += (tuple(self.arrow_style.settings().items()),)
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
                    arrow_cmap=self.parent().arrow_cmap.currentText() if hasattr(self.parent(), "arrow_cmap") else "Solid white",
                    arrow_settings=self.arrow_style.settings())
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
        index = list(self.index())
        for d, value in zip(dims, positions):
            if value is None or not np.isfinite(value):
                return
            index[self.data.dims.index(d)] = int(np.argmin(abs(self.data.coords[d]-value)))
        self.trace_picker.set_index(index)

    def animation_indices(self):
        if self.batch is None:
            raise ValueError("Process All before animating spatial fields.")
        if "covariance" in self.quantity.currentText().lower():
            raise ValueError("Covariance uses lag, not frequency; choose a spectral quantity for animation.")
        if self.animation_mode.currentIndex() == 1:
            if self.interpretation.currentText() == "Quantity representation":
                raise ValueError("Phase animation needs Phase projection or Vector components.")
            return np.arange(self.phase_steps.value())
        return self.batch.frequency_indices(self.frequency_start.value(), self.frequency_stop.value(), self.frequency_step.value())

    def update_playback_slider(self):
        # Discover the sequence without computing its fixed color limits until
        # the user actually plays or seeks. Long scans stay lazy.
        try:
            count = len(self.animation_indices())
        except ValueError:
            count = 0
        blocker = C.QSignalBlocker(self.playback_slider)
        self.playback_slider.setRange(0, max(0, count-1))
        self.playback_slider.setPageStep(max(1, count//10))
        self.playback_slider.setValue(self._animation_position if self._animation_frames is not None else 0)
        self.playback_slider.setEnabled(count > 1)
        del blocker
        if self._animation_frames is None:
            self.playback_position.setText(f"1 / {count:,}" if count else "—")
            self.frame_label.setText("Drag Playback to choose a frame, or use Play / Step." if count else
                                     "Process All and choose a frequency or phase animation to enable Playback.")

    def prepare_animation(self):
        indices = self.animation_indices()
        if self.animation_mode.currentIndex() == 1:
            self._animation_frames = [(self.bin_slider.value(), np.deg2rad(self.phase.value()) + 2*np.pi*i/self.phase_steps.value())
                                      for i in indices]
        else:
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
        self.update_playback_slider()

    def show_animation_frame(self):
        i, self._phase = self._animation_frames[self._animation_position]
        self.bin_slider.blockSignals(True)
        self.bin_slider.setValue(i)
        self.bin_slider.blockSignals(False)
        self.bin_changed()
        blocker = C.QSignalBlocker(self.playback_slider)
        self.playback_slider.setValue(self._animation_position)
        del blocker
        self.playback_position.setText(f"{self._animation_position+1:,} / {len(self._animation_frames):,}")
        self.frame_label.setText(f"{self.animation_mode.currentText()} · frame {self._animation_position+1}/{len(self._animation_frames)}\n"
                                 f"f = {self.batch.frequency[i]:.9g} Hz · phase = {np.rad2deg(self._phase):.5g}°")

    def seek_current_animation_frame(self):
        self.seek_animation(self.playback_slider.value())

    def seek_animation(self, position):
        self.stop_play()
        try:
            if self._animation_frames is None:
                self.prepare_animation()
            self._animation_position = min(max(0, position), len(self._animation_frames)-1)
            self.show_animation_frame()
        except ValueError as exc:
            self._animation_frames = None
            self.update_playback_slider()
            self.message.setText(str(exc))

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
        self.draw_locations()
        self.draw_spectrogram()

    def update_appearance(self, appearance):
        self.appearance = appearance
        self.facility_logo.set_appearance(appearance)
        self.trace_view.fig.set_facecolor(colors(appearance)["bg"])
        for ax in self.trace_view.axes:
            plotting.style(ax, colors(appearance))
        for line in self.trace_view.lines:
            line.set_color(colors(appearance)["accent"])
        self.trace_view.canvas.draw_idle()
        self._render_key = None
        self.draw_spectrum()
        self.draw_results()
        self.draw_locations()
        self.draw_spectrogram()

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            self.message.setText("Cancel processing or wait for it to finish before exiting.")
            return
        self.stop_play()
        self.spectrogram_timer.stop()
        self.collect()
        self.spectrogram_timer.stop()
        super().reject()
