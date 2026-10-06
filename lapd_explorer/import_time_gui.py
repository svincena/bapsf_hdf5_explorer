"""Bounded single-trace preview for temporal import limits."""
import numpy as np
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.widgets import SpanSelector
from . import io, plotting
from .appearance import colors
from .temporal import sample_slice, preview_record
from .widgets import Worker, ScientificDoubleSpinBox, combo, spin


class ImportTimeDialog(W.QDialog):
    def __init__(self, path, sources, metadata_options, metadata, t0, options, appearance="Light", parent=None):
        super().__init__(parent)
        self.path, self.sources, self.metadata_options = path, sources, metadata_options
        self.metadata, self.t0, self.options = metadata, t0, options
        self._source_index = 0
        self.worker = None
        self._pending = self._closing = self._syncing = False
        self._version = 0
        self._history_initialized = False
        self._limits_valid = True
        self.setWindowTitle("Preview trace and choose temporal import limits")
        self.resize(1120, 760)
        layout = W.QVBoxLayout(self)
        note = W.QLabel("Zoom, pan, or drag a horizontal interval to select the import range. "
                        "Refine the original sample indices or times below. Only one display trace is read; "
                        "zooming reloads the visible interval with more detail.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.channel = combo([s[0] for s in sources])
        layout.addWidget(self.channel)
        split = W.QSplitter()
        layout.addWidget(split, 1)
        controls = W.QWidget()
        form = W.QFormLayout(controls)
        self.form = form
        form.setRowWrapPolicy(W.QFormLayout.WrapAllRows)
        self.first, self.last = spin(0, 2**31-1), spin(0, 2**31-1)
        self.start, self.stop = ScientificDoubleSpinBox(), ScientificDoubleSpinBox()
        for box in (self.start, self.stop):
            box.setDecimals(12)
            box.setRange(-1e15, 1e15)
        form.addRow("First original sample (0-based)", self.first)
        form.addRow("Last original sample (inclusive)", self.last)
        form.addRow("Start time (s, inclusive)", self.start)
        form.addRow("End time (s, inclusive)", self.stop)
        self.mode = combo(["Record index", "Global shot number", "Nearest spatial coordinate"])
        self.index = spin(0, 2**31-1)
        self.shot = W.QLineEdit()
        self.coordinates = [W.QLineEdit() for _ in range(3)]
        self.repeat = spin(0, 2**31-1)
        form.addRow("Preview record selector", self.mode)
        form.addRow("Original record index", self.index)
        form.addRow("Global shot number", self.shot)
        for axis, box in zip("xyz", self.coordinates):
            box.setPlaceholderText("Ignore this coordinate")
            form.addRow(f"Nearest {axis} (motion units)", box)
        form.addRow("Occurrence at coordinate (0-based)", self.repeat)
        self.reload_button = W.QPushButton("Load selected trace")
        self.reload_button.clicked.connect(self.request_preview)
        form.addRow(self.reload_button)
        self.record_label = W.QLabel()
        self.record_label.setWordWrap(True)
        form.addRow(self.record_label)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        split.addWidget(scroll)
        plot = W.QWidget()
        plot_layout = W.QVBoxLayout(plot)
        self.fig = plotting.figure(appearance)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        self.axis = self.fig.add_subplot(111)
        theme = colors(appearance)
        plotting.style(self.axis, theme)
        self.axis.set_xlabel("Time (s)")
        self.line, = self.axis.plot([], [])
        self.selector = SpanSelector(self.axis, self.select_span, "horizontal", useblit=True,
                                     props=dict(facecolor=theme["accent"], alpha=.25),
                                     interactive=True, drag_from_anywhere=True)
        split.addWidget(plot)
        split.setSizes([310, 790])
        self.message = W.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Cancel)
        self.apply_button = buttons.addButton("Use these limits", W.QDialogButtonBox.AcceptRole)
        self.apply_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = C.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.request_preview)
        self.initialize_metadata(metadata, options)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.channel.currentIndexChanged.connect(self.request_preview)
        self.first.valueChanged.connect(self.from_samples)
        self.last.valueChanged.connect(self.from_samples)
        self.start.valueChanged.connect(self.from_times)
        self.stop.valueChanged.connect(self.from_times)
        self.axis.callbacks.connect("xlim_changed", self.from_plot)
        self.mode_changed()
        self.request_preview()

    def initialize_metadata(self, metadata, options):
        self.metadata = metadata
        n, dt = metadata["samples"], metadata["dt"]
        selection = sample_slice(n, dt, self.t0, **options)
        for box in (self.first, self.last):
            box.setRange(0, n-1)
        for box in (self.start, self.stop):
            box.setRange(self.t0, self.t0 + (n-1)*dt)
            box.setSingleStep(dt)
        self.set_limits(selection.start, selection.stop-1, update_plot=False)
        index, row = preview_record(metadata)
        self.index.setValue(index)
        if metadata["shots"] is not None:
            self.shot.setText(str(metadata["shots"][row]))
        xyz = metadata["xyz"]
        if xyz is not None and np.all(np.isfinite(xyz[row])):
            for box, value in zip(self.coordinates, xyz[row]):
                box.setText(f"{value:.12g}")
        self.mode.model().item(1).setEnabled(metadata["shots"] is not None)
        self.mode.model().item(2).setEnabled(bool(xyz is not None and np.any(np.all(np.isfinite(xyz), axis=1))))

    def mode_changed(self):
        mode = self.mode.currentIndex()
        self.index.setEnabled(mode == 0)
        self.shot.setEnabled(mode == 1)
        self.form.setRowVisible(self.index, mode == 0)
        self.form.setRowVisible(self.shot, mode == 1)
        for box in self.coordinates:
            box.setEnabled(mode == 2)
            self.form.setRowVisible(box, mode == 2)
        self.repeat.setEnabled(mode == 2)
        self.form.setRowVisible(self.repeat, mode == 2)

    def sample_limits(self):
        return self.first.value(), self.last.value()

    def set_limits(self, first, last, update_plot=True):
        dt, n = self.metadata["dt"], self.metadata["samples"]
        first, last = int(np.clip(first, 0, n-1)), int(np.clip(last, 0, n-1))
        self._limits_valid = first < last
        self._syncing = True
        try:
            for box, value in zip((self.first, self.last, self.start, self.stop),
                                  (first, last, self.t0 + first*dt, self.t0 + last*dt)):
                blocked = box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(blocked)
            self.selector.extents = (self.t0 + first*dt, self.t0 + last*dt)
            if update_plot and first < last:
                self.axis.set_xlim(self.t0 + first*dt, self.t0 + last*dt)
                self.canvas.draw_idle()
        finally:
            self._syncing = False

    def from_samples(self):
        if self._syncing:
            return
        self.set_limits(*self.sample_limits())
        if not self._limits_valid:
            self.message.setText("The last sample must be greater than the first sample.")
            self.apply_button.setEnabled(False)
            self.timer.stop()
            return
        self.timer.start()

    def from_times(self):
        if self._syncing:
            return
        try:
            selection = sample_slice(self.metadata["samples"], self.metadata["dt"], self.t0,
                                     time_limits=(self.start.value(), self.stop.value()))
        except ValueError as exc:
            self._limits_valid = False
            self.apply_button.setEnabled(False)
            self.message.setText(str(exc))
            self.timer.stop()
            return
        self.set_limits(selection.start, selection.stop-1)
        self.timer.start()

    def select_span(self, lo, hi):
        dt, n = self.metadata["dt"], self.metadata["samples"]
        lo, hi = sorted((lo, hi))
        first = int(np.clip(np.ceil((lo-self.t0)/dt - 1e-8), 0, n-2))
        last = int(np.clip(np.floor((hi-self.t0)/dt + 1e-8), first+1, n-1))
        self.set_limits(first, last)
        self.timer.start()

    def from_plot(self, axis):
        if not self._syncing:
            self.select_span(*axis.get_xlim())

    def request_preview(self, *args):
        self.timer.stop()
        self._version += 1
        if self.worker and self.worker.isRunning():
            self._pending = True
            return
        try:
            source_index = self.channel.currentIndex()
            source = self.sources[source_index]
            metadata = self.metadata if source_index == self._source_index else None
            limits = self.sample_limits()
            mode = ("index", "shot", "position")[self.mode.currentIndex()]
            selection = dict(mode=mode, index=self.index.value(),
                             shot=int(self.shot.text()) if mode == "shot" else None,
                             position=[float(box.text()) if box.text().strip() else None for box in self.coordinates],
                             repeat=self.repeat.value())
            version = self._version
            self.apply_button.setEnabled(False)
            self.message.setText("Reading one preview trace…")
            def read():
                meta = metadata or io.import_metadata(self.path, spec=source[1], dataset_path=source[2], **self.metadata_options)
                selected = selection if metadata is not None else {}
                index, row = preview_record(meta, decimals=meta.get("coordinate_decimals", 4), **selected)
                first, last = limits if metadata is not None else (0, meta["samples"]-1)
                sample_slice(meta["samples"], meta["dt"], self.t0, sample_limits=(first, last))
                time, values = io.preview_trace(self.path, meta, index, first, last, self.t0,
                                                spec=source[1], dataset_path=source[2])
                return meta, index, row, time, values
            def ready(result):
                if version != self._version or self._closing:
                    return
                meta, index, row, time, values = result
                if source_index != self._source_index:
                    self._syncing = True
                    self.initialize_metadata(meta, {})
                    self._syncing = False
                    self._source_index = source_index
                    self.toolbar.update()
                    self._history_initialized = False
                self.metadata = meta
                self.line.set_data(time, values)
                self.axis.set_ylabel(meta["units"])
                self.axis.relim()
                self.axis.autoscale_view(scalex=False)
                self.set_limits(*self.sample_limits())
                label = f"Original record index {index}"
                if meta["shots"] is not None:
                    label += f" · global shot {meta['shots'][row]}"
                if meta["xyz"] is not None and np.all(np.isfinite(meta["xyz"][row])):
                    label += "\n(x, y, z) = " + str(tuple(float(v) for v in meta["xyz"][row]))
                self.record_label.setText(label)
                self.message.setText(f"Display: {len(time):,} points from one trace. Limits refer to original samples; "
                                      "display thinning can hide fast changes. Zoom in for finer detail.")
                if not self._history_initialized:
                    self._syncing = True
                    self.axis.set_xlim(self.t0, self.t0+(meta["samples"]-1)*meta["dt"])
                    self.toolbar.push_current()
                    self._history_initialized = True
                    self._syncing = False
                    self.set_limits(*self.sample_limits())
                self.canvas.draw_idle()
            def failed(message):
                if version == self._version:
                    self.message.setText(message)
            self.worker = Worker(read, self)
            self.worker.result.connect(ready)
            self.worker.failed.connect(failed)
            self.worker.finished.connect(self.finished_read)
            self.worker.start()
        except (ValueError, TypeError) as exc:
            self.message.setText(str(exc))

    def finished_read(self):
        if self._closing:
            super().reject()
        elif self._pending:
            self._pending = False
            self.request_preview()
        else:
            self.apply_button.setEnabled(self._limits_valid)

    def accept(self):
        if not self._limits_valid:
            self.message.setText("Enter valid, ordered time/sample limits.")
            return
        try:
            sample_slice(self.metadata["samples"], self.metadata["dt"], self.t0,
                         sample_limits=self.sample_limits(), decimation=self.options.get("decimation", 1))
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self.timer.stop()
        super().accept()

    def reject(self):
        self.timer.stop()
        if self.worker and self.worker.isRunning():
            self._closing = True
            self._pending = False
            self.message.setText("Finishing the current preview read…")
            return
        super().reject()
