"""PyQt6 workbench. HDF5 and full-array transforms run on a worker thread."""

import json
from pathlib import Path
import traceback

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets as Q
import pyqtgraph as pg

from .analysis import filter_signal, subtract_baseline
from .data import acquisition_reshape, demo_data, motion_reshape
from .reader import inspect_file, load_channel_data
from .theme import STYLESHEET


class Worker(QtCore.QThread):
    result = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.result.emit(self.function())
        except Exception:
            self.failed.emit(traceback.format_exc())


def button(text, callback, primary=False):
    widget = Q.QPushButton(text)
    if primary:
        widget.setObjectName("primary")
    widget.clicked.connect(callback)
    return widget


def note(text):
    widget = Q.QLabel(text)
    widget.setObjectName("muted")
    widget.setWordWrap(True)
    return widget


def spin(value=0, maximum=1_000_000_000):
    widget = Q.QSpinBox()
    widget.setRange(0, maximum)
    widget.setValue(value)
    return widget


def edges(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - .5, values[0] + .5])
    return np.r_[values[0] - (values[1] - values[0]) / 2,
                 (values[:-1] + values[1:]) / 2, values[-1] + (values[-1] - values[-2]) / 2]


class ExplorerWindow(Q.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BaPSF Explorer")
        self.resize(1400, 930)
        self.setMinimumSize(1080, 740)
        self.setStyleSheet(STYLESHEET)
        pg.setConfigOptions(background="#121c27", foreground="#a9bacb", antialias=True)
        self.catalog = self.raw = self.base = self.data = None
        self.worker = None
        self.selectors = {}
        self._updating = False
        self._plot_cache = None
        self._spatial_axes = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.next_frame)
        self._build()

    def _build(self):
        self.root = Q.QWidget()
        self.setCentralWidget(self.root)
        layout = Q.QVBoxLayout(self.root)
        layout.setContentsMargins(22, 18, 22, 12)
        layout.setSpacing(16)
        header = Q.QHBoxLayout()
        brand = Q.QVBoxLayout()
        eyebrow = Q.QLabel("BASIC PLASMA SCIENCE FACILITY  /  DATA WORKBENCH")
        eyebrow.setObjectName("eyebrow")
        title = Q.QLabel("BaPSF Explorer")
        title.setObjectName("brand")
        brand.addWidget(eyebrow)
        brand.addWidget(title)
        header.addLayout(brand)
        header.addStretch()
        header.addWidget(button("Open HDF5…", self.open_dialog, True))
        header.addWidget(button("Explore demo", self.load_demo))
        self.export_button = button("Export data…", self.export_data)
        header.addWidget(self.export_button)
        layout.addLayout(header)
        self.file_label = note("Open an experiment or explore the synthetic demo to get started.")
        self.file_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.file_label)
        split = Q.QSplitter()
        layout.addWidget(split, 1)
        self.sidebar = Q.QTabWidget()
        self.sidebar.setMinimumWidth(345)
        self.sidebar.setMaximumWidth(440)
        split.addWidget(self.sidebar)
        self._source_panel()
        self._layout_panel()
        self._analysis_panel()
        right = Q.QWidget()
        main = Q.QVBoxLayout(right)
        main.setContentsMargins(16, 0, 0, 0)
        cards = Q.QHBoxLayout()
        self.metrics = []
        for text in ("NO CHANNEL\nOpen a file", "DIMENSIONS\n—", "TIME WINDOW\n—"):
            label = Q.QLabel(text)
            label.setObjectName("metric")
            cards.addWidget(label, 1)
            self.metrics.append(label)
        main.addLayout(cards)
        self.selection_box = Q.QGroupBox("Selection")
        self.selection_layout = Q.QGridLayout(self.selection_box)
        main.addWidget(self.selection_box)
        self.plots = Q.QTabWidget()
        main.addWidget(self.plots, 1)
        self.wave = pg.PlotWidget()
        self.wave.showGrid(x=True, y=True, alpha=.15)
        self.wave.addLegend(offset=(12, 12))
        self.raw_curve = self.wave.plot(pen=pg.mkPen("#63788d", width=1), name="Original")
        self.trace = self.wave.plot(pen=pg.mkPen("#56dbc7", width=1.6), name="Current")
        for curve in (self.raw_curve, self.trace):
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#e8b768", width=1))
        self.cursor.sigPositionChangeFinished.connect(self.cursor_moved)
        self.wave.addItem(self.cursor, ignoreBounds=True)
        self.plots.addTab(self.wave, "Time traces")
        spatial_page = Q.QWidget()
        spatial_layout = Q.QVBoxLayout(spatial_page)
        axes_row = Q.QHBoxLayout()
        self.horizontal = Q.QComboBox()
        self.vertical = Q.QComboBox()
        for label, combo in (("Horizontal", self.horizontal), ("Vertical", self.vertical)):
            axes_row.addWidget(Q.QLabel(label))
            axes_row.addWidget(combo)
            combo.currentIndexChanged.connect(self.refresh)
        axes_row.addStretch()
        self.lock_scale = Q.QCheckBox("Lock scale during playback")
        self.lock_scale.setChecked(True)
        self.lock_scale.toggled.connect(self.refresh)
        axes_row.addWidget(self.lock_scale)
        spatial_layout.addLayout(axes_row)
        self.spatial = pg.GraphicsLayoutWidget()
        self.spatial_plot = self.spatial.addPlot()
        self.spatial_plot.showGrid(x=True, y=True, alpha=.12)
        self.profile = self.spatial_plot.plot(pen=pg.mkPen("#56dbc7", width=2), symbol="o", symbolSize=5,
                                              symbolBrush="#56dbc7")
        self.mesh = pg.PColorMeshItem(colorMap=pg.colormap.get("viridis"), enableAutoLevels=False)
        self.spatial_plot.addItem(self.mesh)
        self.colorbar = pg.ColorBarItem(values=(-1, 1), colorMap=pg.colormap.get("viridis"), interactive=False)
        self.colorbar.setImageItem(self.mesh, insert_in=self.spatial_plot)
        spatial_layout.addWidget(self.spatial, 1)
        self.spatial_note = note("Group records by motion coordinates to view spatial profiles.")
        spatial_layout.addWidget(self.spatial_note)
        self.plots.addTab(spatial_page, "Spatial explorer")
        self.metadata = Q.QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.plots.addTab(self.metadata, "Metadata & history")
        transport = Q.QHBoxLayout()
        self.play_button = button("▶ Play", self.toggle_play)
        self.frame = Q.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame.setRange(0, 0)
        self.frame.valueChanged.connect(self.update_frame)
        self.frame_label = Q.QLabel("t = —")
        self.frame_label.setMinimumWidth(160)
        self.frame_step = spin(1, 1_000_000)
        self.frame_step.setMinimum(1)
        self.frame_step.setMaximumWidth(85)
        transport.addWidget(self.play_button)
        transport.addWidget(self.frame, 1)
        transport.addWidget(self.frame_label)
        transport.addWidget(Q.QLabel("Samples/frame"))
        transport.addWidget(self.frame_step)
        main.addLayout(transport)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.statusBar().showMessage("Ready • HDF5 files open read-only")
        self.export_button.setEnabled(False)

    def _panel(self, title):
        scroll = Q.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(Q.QFrame.Shape.NoFrame)
        content = Q.QWidget()
        layout = Q.QVBoxLayout(content)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(13)
        scroll.setWidget(content)
        self.sidebar.addTab(scroll, title)
        return layout

    def _source_panel(self):
        panel = self._panel("Source")
        panel.addWidget(note("Select a channel and a bounded read. Row and sample ranges are zero-based; stop is exclusive."))
        group = Q.QGroupBox("Digitizer channel")
        form = Q.QFormLayout(group)
        self.channels = Q.QComboBox()
        self.channels.setSizeAdjustPolicy(Q.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.channels.currentIndexChanged.connect(self.channel_changed)
        self.channel_name = Q.QLineEdit()
        form.addRow("Channel", self.channels)
        form.addRow("Display name", self.channel_name)
        panel.addWidget(group)
        group = Q.QGroupBox("Read range")
        form = Q.QFormLayout(group)
        self.row_start, self.row_stop = spin(), spin(50)
        self.sample_start, self.sample_stop = spin(), spin(600)
        for label, field in (("First row", self.row_start), ("Stop row", self.row_stop),
                             ("First sample", self.sample_start), ("Stop sample", self.sample_stop)):
            form.addRow(label, field)
            field.valueChanged.connect(self.update_estimate)
        self.read_estimate = note("Load a catalog to choose a range.")
        form.addRow(self.read_estimate)
        form.addRow(button("Use all records", self.all_records))
        panel.addWidget(group)
        group = Q.QGroupBox("Motion association")
        form = Q.QFormLayout(group)
        self.motion = Q.QComboBox()
        self.motion.addItem("No motion", None)
        self.position_source = Q.QComboBox()
        self.position_source.addItems(["Target grid", "Measured positions"])
        form.addRow("Device / config", self.motion)
        form.addRow("Coordinates", self.position_source)
        form.addRow(note("Shots are matched by bapsflib. Unmatched records are reported after reading."))
        panel.addWidget(group)
        self.load_button = button("Load channel", self.load_channel, True)
        self.load_button.setEnabled(False)
        panel.addWidget(self.load_button)
        panel.addStretch()

    def _layout_panel(self):
        panel = self._panel("Dimensions")
        panel.addWidget(note("Keep raw acquisition rows or organize them into labeled dimensions. Reshaping resets analysis to the loaded data."))
        self.layout_mode = Q.QComboBox()
        self.layout_mode.addItems(["Motion coordinates", "Acquisition order", "Raw records"])
        panel.addWidget(self.layout_mode)
        group = Q.QGroupBox("Motion grouping")
        form = Q.QFormLayout(group)
        self.motion_axes = Q.QLineEdit()
        self.motion_axes.setPlaceholderText("Automatic, or y,x")
        self.rounding = spin(4, 9)
        self.case_coordinate = Q.QComboBox()
        self.case_coordinate.addItem("None", None)
        form.addRow("Axes, slow → fast", self.motion_axes)
        form.addRow("Decimal places", self.rounding)
        form.addRow("Case coordinate", self.case_coordinate)
        form.addRow(note("Repeated visits become the shot axis. Missing cells remain NaN. Use target coordinates to avoid motor-position jitter."))
        panel.addWidget(group)
        group = Q.QGroupBox("Acquisition order")
        form = Q.QVBoxLayout(group)
        self.dimensions = Q.QLineEdit()
        self.dimensions.setPlaceholderText("y=13,x=17,case=2,shot=4")
        form.addWidget(self.dimensions)
        form.addWidget(note("Example: y=3,x=10,case=2,shot=5. Rightmost axis varies fastest. The product must match loaded rows. Coordinates here are indices."))
        panel.addWidget(group)
        panel.addWidget(button("Apply dimensions", self.reshape_data, True))
        self.shape_note = note("No data loaded.")
        panel.addWidget(self.shape_note)
        panel.addStretch()

    def _analysis_panel(self):
        panel = self._panel("Analysis")
        panel.addWidget(note("Apply baseline subtraction, then filtering, to each trace before averaging. Every Apply starts from the unprocessed data."))
        group = Q.QGroupBox("Baseline subtraction")
        form = Q.QFormLayout(group)
        self.baseline = Q.QCheckBox("Subtract interval mean")
        self.baseline_start = Q.QLineEdit("0")
        self.baseline_stop = Q.QLineEdit("0.05")
        form.addRow(self.baseline)
        form.addRow("Start (ms)", self.baseline_start)
        form.addRow("End (ms)", self.baseline_stop)
        panel.addWidget(group)
        group = Q.QGroupBox("Time-domain filter")
        form = Q.QFormLayout(group)
        self.filter_kind = Q.QComboBox()
        self.filter_kind.addItems(["None", "Lowpass", "Highpass", "Bandpass"])
        self.cutoff = Q.QLineEdit("10000")
        self.filter_order = spin(4, 12)
        self.filter_order.setMinimum(1)
        form.addRow("Filter", self.filter_kind)
        form.addRow("Cutoffs (Hz)", self.cutoff)
        form.addRow("Order", self.filter_order)
        form.addRow(note("Butterworth, forward–backward zero phase. Bandpass: enter low,high. End effects depend on the selected time window."))
        panel.addWidget(group)
        panel.addWidget(button("Apply analysis", self.apply_analysis, True))
        panel.addWidget(button("Reset analysis", self.reset_analysis))
        self.analysis_note = note("Original data is retained for comparison and reset.")
        panel.addWidget(self.analysis_note)
        panel.addStretch()

    def run_task(self, message, function, callback):
        if self.worker is not None:
            return
        self.stop_play()
        self.root.setEnabled(False)
        self.statusBar().showMessage(message)
        self.worker = Worker(function, self)
        self.worker.result.connect(callback)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.task_finished)
        self.worker.start()

    def task_finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        self.root.setEnabled(True)

    def error(self, message):
        self.statusBar().showMessage("Operation failed; previous data retained.")
        box = Q.QMessageBox(self)
        box.setWindowTitle("Unable to complete operation")
        box.setText(message.strip().splitlines()[-1])
        box.setDetailedText(message)
        box.exec()

    def open_dialog(self):
        path, _ = Q.QFileDialog.getOpenFileName(self, "Open BaPSF data", "", "HDF5 files (*.hdf5 *.h5);;All files (*)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        self.run_task("Inspecting HDF5 metadata…", lambda: inspect_file(path), self.catalog_loaded)

    def catalog_loaded(self, catalog):
        self.catalog = catalog
        self._previous_channel = None
        self.channels.blockSignals(True)
        self.channels.clear()
        for ch in catalog.channels:
            self.channels.addItem(f"{ch.label}  ·  B{ch.board} / CH{ch.channel}", ch)
            self.channels.setItemData(self.channels.count() - 1, f"{ch.digitizer} / {ch.config} / {ch.adc}", QtCore.Qt.ItemDataRole.ToolTipRole)
        self.channels.blockSignals(False)
        self.motion.clear()
        self.motion.addItem("No motion", None)
        for item in catalog.motions:
            self.motion.addItem(f"{item[0]} / {item[1]}", item)
        if len(catalog.motions) == 1:
            self.motion.setCurrentIndex(1)
        self.channel_changed()
        self.file_label.setText(f"Catalog: {catalog.path}  •  Select Load channel to replace the displayed data.")
        self.load_button.setEnabled(True)
        self.statusBar().showMessage(f"Found {len(catalog.channels)} channels and {len(catalog.motions)} motion configurations.")

    def channel_changed(self):
        ch = self.channels.currentData()
        if ch is None:
            return
        previous = getattr(self, "_previous_channel", None)
        old_rows = (self.row_start.value(), self.row_stop.value())
        old_samples = (self.sample_start.value(), self.sample_stop.value())
        self.channel_name.setText(ch.label)
        for field in (self.row_start, self.row_stop):
            field.setMaximum(ch.records)
        for field in (self.sample_start, self.sample_stop):
            field.setMaximum(ch.samples)
        for start, stop, old, total, prior_total in (
            (self.row_start, self.row_stop, old_rows, ch.records, previous.records if previous else None),
            (self.sample_start, self.sample_stop, old_samples, ch.samples, previous.samples if previous else None),
        ):
            # A full-range selection follows the new channel's own length.
            # Valid custom ranges survive channel changes.
            if previous is None or old == (0, prior_total) or not 0 <= old[0] < old[1] <= total:
                start.setValue(0)
                stop.setValue(total)
            else:
                start.setValue(old[0])
                stop.setValue(old[1])
        self._previous_channel = ch
        self.update_estimate()

    def all_records(self):
        ch = self.channels.currentData()
        if ch:
            self.row_start.setValue(0)
            self.row_stop.setValue(ch.records)

    def update_estimate(self):
        if not hasattr(self, "read_estimate"):
            return
        n = max(0, self.row_stop.value() - self.row_start.value())
        nt = max(0, self.sample_stop.value() - self.sample_start.value())
        self.read_estimate.setText(f"{n:,} records × {nt:,} samples\nSignal ≈ {n * nt * 4 / 1024**2:.1f} MiB; analysis uses additional memory.")

    def load_channel(self):
        if self.catalog is None:
            return
        ch = self.channels.currentData()
        path = self.catalog.path
        kwargs = dict(rows=slice(self.row_start.value(), self.row_stop.value()),
                      samples=slice(self.sample_start.value(), self.sample_stop.value()),
                      motion=self.motion.currentData(),
                      position_source="target" if self.position_source.currentIndex() == 0 else "measured",
                      name=self.channel_name.text().strip() or ch.label)
        self.run_task("Reading channel and deriving motion dimensions…",
                      lambda: load_channel_data(path, ch, **kwargs), self.channel_loaded)

    def channel_loaded(self, result):
        raw, shaped = result
        self.layout_mode.setCurrentText("Motion coordinates" if "motion_axes" in raw.attrs else "Acquisition order")
        self.motion_axes.clear()
        self.raw_loaded(raw, prepared=shaped)

    def load_demo(self):
        self.run_task("Generating synthetic scan…", demo_data, self.demo_loaded)

    def demo_loaded(self, raw):
        self.raw_loaded(raw)
        self.dimensions.setText("y=13,x=17,case=2,shot=4")
        self.case_coordinate.setCurrentIndex(1)
        self.base = self.data = motion_reshape(raw, cases=raw.case.values)
        self.install_data()
        self.plots.setCurrentIndex(1)

    def raw_loaded(self, raw, prepared=None):
        self.raw = raw
        self.base = self.data = raw if prepared is None else prepared
        self.dimensions.setText("" if "motion_axes" in raw.attrs else f"shot={raw.sizes['record']}")
        self.case_coordinate.clear()
        self.case_coordinate.addItem("None", None)
        for name in raw.coords:
            if name == "case":
                self.case_coordinate.addItem(name, name)
        self.file_label.setText(str(raw.attrs.get("source", "")))
        self.baseline_start.setText(f"{raw.time.values[0] * 1000:g}")
        end = raw.time.values[min(raw.sizes['time'] - 1, max(1, raw.sizes['time'] // 20))]
        self.baseline_stop.setText(f"{end * 1000:g}")
        self.baseline.setChecked(False)
        self.filter_kind.setCurrentIndex(0)
        self.analysis_note.setText("Original data. No analysis applied.")
        self.install_data()
        unmatched = raw.attrs.get("unmatched_records", 0)
        total = raw.attrs.get("channel_records", raw.sizes['record'])
        shape = " × ".join(f"{d}={n}" for d, n in self.data.sizes.items())
        self.statusBar().showMessage(f"Loaded {raw.sizes['record']:,} / {total:,} channel records • {unmatched} unmatched • {shape}")

    def reshape_data(self):
        if self.raw is None:
            return
        mode = self.layout_mode.currentIndex()
        raw = self.raw
        try:
            if mode == 0:
                axes = [a.strip() for a in self.motion_axes.text().split(",") if a.strip()] or None
                decimals = self.rounding.value()
                key = self.case_coordinate.currentData()
                cases = raw[key].values if key else None
                function = lambda: motion_reshape(raw, axes, decimals=decimals, cases=cases)
            elif mode == 1:
                pairs = [part.strip().split("=") for part in self.dimensions.text().split(",")]
                dims = {k.strip(): int(v) for k, v in pairs}
                if len(dims) != len(pairs):
                    raise ValueError("Dimension names must be unique.")
                function = lambda: acquisition_reshape(raw, dims)
            else:
                function = lambda: raw
            self.run_task("Organizing labeled dimensions…", function, self.reshaped)
        except Exception:
            self.error(traceback.format_exc())

    def reshaped(self, data):
        self.base = self.data = data
        self.baseline.setChecked(False)
        self.filter_kind.setCurrentIndex(0)
        self.analysis_note.setText("Analysis reset after dimension change.")
        self.install_data()
        self.statusBar().showMessage("Dimensions applied. Select an individual shot or the shot mean.")

    def install_data(self):
        self._updating = True
        self._plot_cache = None
        while self.selection_layout.count():
            item = self.selection_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self.selectors = {}
        self.shot_mode = Q.QComboBox()
        self.repeat_dim = "shot" if "shot" in self.data.dims else "record"
        self.shot_mode.addItems(["Individual shot", f"Mean over {self.repeat_dim}"])
        self.shot_mode.setEnabled(self.repeat_dim in self.data.dims)
        if "shot" in self.data.dims:
            self.shot_mode.setCurrentIndex(1)
        self.shot_mode.currentIndexChanged.connect(self.refresh)
        self.selection_layout.addWidget(Q.QLabel("Repeats"), 0, 0)
        self.selection_layout.addWidget(self.shot_mode, 0, 1)
        dims = [d for d in self.data.dims if d != "time"]
        for i, dim in enumerate(dims):
            control = spin(0, self.data.sizes[dim] - 1)
            label = Q.QLabel(dim)
            control.valueChanged.connect(self.refresh)
            row, col = 1 + i // 3, (i % 3) * 2
            self.selection_layout.addWidget(label, row, col)
            self.selection_layout.addWidget(control, row, col + 1)
            self.selectors[dim] = (control, label)
        axes = [d for d in dims if d not in ("shot", "record", "case")]
        self.horizontal.clear()
        self.vertical.clear()
        self.horizontal.addItems(axes)
        self.vertical.addItem("None")
        self.vertical.addItems(axes)
        if "x" in axes:
            self.horizontal.setCurrentText("x")
        if "y" in axes and len(axes) > 1:
            self.vertical.setCurrentText("y")
        self.frame.setRange(0, self.data.sizes["time"] - 1)
        self.frame.setValue(0)
        self.frame_step.setValue(max(1, self.data.sizes["time"] // 300))
        self.export_button.setEnabled(True)
        self._updating = False
        self.refresh()
        self.wave.autoRange()
        self.spatial_plot.autoRange()

    def selection(self, data, keep=()):
        selectors = {dim: control.value() for dim, (control, _) in self.selectors.items()
                     if dim not in keep and not (dim == self.repeat_dim and self.shot_mode.currentIndex() == 1)}
        result = data.isel(selectors)
        if self.repeat_dim in result.dims and self.shot_mode.currentIndex() == 1:
            result = result.mean(self.repeat_dim, skipna=True, keep_attrs=True)
        return result

    def refresh(self, *_):
        if self._updating or self.data is None:
            return
        for dim, (control, label) in self.selectors.items():
            coordinate = self.data[dim].values[control.value()]
            label.setText(f"{dim}: {coordinate:g}" if isinstance(coordinate, (float, int, np.number)) else f"{dim}: {coordinate}")
            control.setEnabled(not (dim == self.repeat_dim and self.shot_mode.currentIndex() == 1))
        trace = self.selection(self.data)
        raw_trace = self.selection(self.base)
        self.trace.setData(trace.time.values * 1000, trace.values)
        self.raw_curve.setData(raw_trace.time.values * 1000, raw_trace.values)
        self.raw_curve.setVisible(self.data is not self.base)
        # A newly selected position can have a very different signal amplitude.
        self.wave.enableAutoRange(axis="y", enable=True)
        self.wave.getViewBox().updateAutoRange()
        unit = self.data.attrs.get("units", "")
        self.wave.setLabel("bottom", "Time", units="ms")
        self.wave.setLabel("left", str(self.data.name), units=unit)
        shot_text = f"Mean over {self.repeat_dim}" if self.shot_mode.currentIndex() == 1 else "Individual record"
        if "shot_id" in trace.coords and trace.shot_id.ndim == 0:
            shot_text = f"Shot ID {int(trace.shot_id)}"
        self.wave.setTitle(f"{self.data.name} · {shot_text}", color="#dce6ef", size="12pt")
        self.metrics[0].setText(f"CHANNEL\n{self.data.name}")
        self.metrics[1].setText("DIMENSIONS\n" + " × ".join(f"{d}:{n}" for d, n in self.data.sizes.items() if d != "time"))
        time = self.data.time.values
        self.metrics[2].setText(f"TIME WINDOW\n{time[0]*1000:.4g}–{time[-1]*1000:.4g} ms")
        self.shape_note.setText(str(dict(self.data.sizes)) + f"\nMissing grid cells: {self.data.attrs.get('missing_cells', 0)}")
        self.metadata.setPlainText(str(self.data) + "\n\nATTRIBUTES\n" + json.dumps(self.data.attrs, indent=2, default=str))
        x = self.horizontal.currentText()
        y = self.vertical.currentText()
        self._plot_cache = None
        if x and y != x:
            keep = (x,) if y == "None" else (x, y)
            spatial = self.selection(self.data, keep=keep).transpose(*keep, "time")
            # Fix animation scale over the selected spatial slice and all time.
            vals = spatial.values
            finite = np.isfinite(vals)
            lo, hi = (float(np.nanmin(vals)), float(np.nanmax(vals))) if finite.any() else (0., 1.)
            if lo == hi:
                lo, hi = lo - .5, hi + .5
            self._plot_cache = (spatial, keep, (lo, hi))
        self.update_frame()
        current_axes = (x, y)
        if current_axes != self._spatial_axes:
            self._spatial_axes = current_axes
            self.spatial_plot.autoRange()

    def update_frame(self, *_):
        if self.data is None or self._updating:
            return
        index = self.frame.value()
        t = float(self.data.time.values[index]) * 1000
        self.frame_label.setText(f"t = {t:.6g} ms  ·  {index}")
        self.cursor.setValue(t)
        if self._plot_cache is None:
            self.mesh.hide()
            self.profile.hide()
            self.colorbar.hide()
            self.spatial_note.setText("Apply motion/acquisition dimensions, then choose distinct spatial axes.")
            return
        spatial, keep, limits = self._plot_cache
        values = spatial.isel(time=index).values
        x = keep[0]
        self.spatial_plot.setLabel("bottom", x, units=spatial[x].attrs.get("units", "index"))
        self.spatial_plot.setTitle(f"{self.data.name} · t = {t:.6g} ms", color="#dce6ef")
        if len(keep) == 1:
            self.spatial_plot.setAspectLocked(False)
            self.mesh.hide()
            self.colorbar.hide()
            self.profile.show()
            self.profile.setData(spatial[x].values, values)
            self.spatial_plot.setLabel("left", str(self.data.name), units=self.data.attrs.get("units", ""))
            if self.lock_scale.isChecked():
                self.spatial_plot.setYRange(*limits, padding=.04)
            else:
                self.spatial_plot.enableAutoRange(axis="y")
        else:
            self.spatial_plot.setAspectLocked(True)
            self.profile.hide()
            self.mesh.show()
            self.colorbar.show()
            y = keep[1]
            xx, yy = np.meshgrid(edges(spatial[x]), edges(spatial[y]), indexing="ij")
            if not self.lock_scale.isChecked() and np.isfinite(values).any():
                limits = float(np.nanmin(values)), float(np.nanmax(values))
                if limits[0] == limits[1]:
                    limits = limits[0] - .5, limits[1] + .5
            self.mesh.setData(xx, yy, values)
            self.colorbar.setLevels(limits)
            self.spatial_plot.setLabel("left", y, units=spatial[y].attrs.get("units", "index"))
            self.colorbar.axis.setLabel(str(self.data.name), units=self.data.attrs.get("units", ""))
        self.spatial_note.setText("Spatial selectors for the plotted axes affect the time trace. Other selectors choose this spatial slice. NaN cells are unmeasured.")

    def cursor_moved(self):
        if self.data is not None:
            index = np.argmin(np.abs(self.data.time.values * 1000 - self.cursor.value()))
            self.frame.setValue(int(index))

    def toggle_play(self):
        if self.timer.isActive():
            self.stop_play()
        elif self.data is not None:
            self.timer.start()
            self.play_button.setText("❚❚ Pause")

    def stop_play(self):
        self.timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("▶ Play")

    def next_frame(self):
        self.frame.setValue((self.frame.value() + self.frame_step.value()) % (self.frame.maximum() + 1))

    def apply_analysis(self):
        if self.base is None:
            return
        try:
            baseline = (float(self.baseline_start.text()) / 1000, float(self.baseline_stop.text()) / 1000) if self.baseline.isChecked() else None
            kind = self.filter_kind.currentText().lower()
            cutoff = [float(v) for v in self.cutoff.text().split(",")] if kind != "none" else None
            order = self.filter_order.value()
            base = self.base
            def compute():
                result = subtract_baseline(base, *baseline) if baseline else base
                return filter_signal(result, kind, cutoff, order=order) if cutoff else result
            self.run_task("Applying analysis to each time trace…", compute, self.analysis_applied)
        except Exception:
            self.error(traceback.format_exc())

    def analysis_applied(self, data):
        self.data = data
        self.analysis_note.setText("Analysis applied. The gray time trace shows the original selection.")
        self.refresh()
        self.statusBar().showMessage("Analysis complete. Export includes all current dimensions and processing history.")

    def reset_analysis(self):
        if self.base is not None:
            self.stop_play()
            self.data = self.base
            self.baseline.setChecked(False)
            self.filter_kind.setCurrentIndex(0)
            self.analysis_note.setText("Original data. No analysis applied.")
            self.refresh()

    def export_data(self):
        if self.data is None:
            return
        path, _ = Q.QFileDialog.getSaveFileName(self, "Export full current data (before display averaging)", "analysis.nc", "NetCDF (*.nc)")
        if path:
            if Path(path).resolve() == Path(self.data.attrs.get("source", "")).resolve():
                self.error("Choose an export destination different from the source HDF5 file.")
                return
            data = self.data.copy(deep=False)
            data.attrs = dict(data.attrs)
            data.attrs["display_name"] = str(data.name)
            data.name = "signal"
            self.run_task("Exporting labeled data and history…", lambda: data.to_netcdf(path, engine="h5netcdf"),
                          lambda _: self.statusBar().showMessage(f"Exported {Path(path).name} • full processed array; display selections/averaging not applied."))

    def closeEvent(self, event):
        if self.worker is not None:
            self.statusBar().showMessage("A data operation is running. Close the window after it completes.")
            event.ignore()
            return
        self.stop_play()
        event.accept()
