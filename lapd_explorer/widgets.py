"""Shared Qt controls, workers, and acquisition/import dialogs."""
import threading
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
from .cancellation import ImportCanceled, check_canceled
from .import_progress import ShotProgress
from . import io
from .temporal import DOWNSAMPLING_METHODS, retained_samples
from .appearance import FacilityLogo


class Worker(C.QThread):
    result = C.Signal(object)
    failed = C.Signal(str)
    progress = C.Signal(float, float, object)
    aborted = C.Signal()

    def __init__(self, fn, parent=None, *, canceled=None):
        super().__init__(parent)
        self.fn = fn
        self._canceled = canceled

    def run(self):
        try:
            check_canceled(self._canceled)
            result = self.fn()
            check_canceled(self._canceled)
            self.result.emit(result)
        except ImportCanceled:
            self.aborted.emit()
        except Exception as exc:
            if self._canceled is not None and self._canceled():
                self.aborted.emit()
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if self._canceled is not None:
                self.fn = None



def combo(items):
    widget = W.QComboBox()
    # Long channel/configuration names should not determine the panel width.
    # The popup still shows the complete choices.
    widget.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    widget.setMinimumContentsLength(10)
    widget.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Fixed)
    widget.addItems(items)
    return widget


def colormap_combo(default="viridis", *, solid=False):
    """One palette catalog and gradient previews for every colorbar selector."""
    import numpy as np
    from matplotlib import colormaps
    from .colormaps import COLORMAPS, COLORMAP_TOOLTIPS
    widget = combo((["Solid white"] if solid else []) + COLORMAPS)
    widget.setIconSize(C.QSize(64, 14))
    widget.setMaxVisibleItems(16)
    widget.setToolTip("Choose a color map. Gradient previews run from low to high; _r reverses the colors. Hover over a choice for its intended use.")
    for index in range(widget.count()):
        name = widget.itemText(index)
        if name == "Solid white":
            pixmap = G.QPixmap(96, 14)
            pixmap.fill(C.Qt.white)
            tooltip = "Uniform white arrows without a magnitude colorbar."
        else:
            rgba = colormaps[name](np.linspace(0, 1, 96), bytes=True)
            image = G.QImage(rgba.data, 96, 1, rgba.strides[0]*96, G.QImage.Format_RGBA8888).copy()
            pixmap = G.QPixmap.fromImage(image).scaled(96, 14)
            tooltip = COLORMAP_TOOLTIPS[name]
        widget.setItemIcon(index, G.QIcon(pixmap))
        widget.setItemData(index, tooltip, C.Qt.ToolTipRole)
    widget.setCurrentText(default)
    return widget


def form_layout(parent=None):
    """Compact forms that keep labels beside fields when there is room."""
    layout = W.QFormLayout(parent)
    layout.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
    layout.setFieldGrowthPolicy(W.QFormLayout.ExpandingFieldsGrow)
    layout.setFormAlignment(C.Qt.AlignTop)
    margin = 6 if parent is not None else 0
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setHorizontalSpacing(8)
    layout.setVerticalSpacing(4)
    return layout


def add_list_field(form, label, widget):
    """Give a scrollable selector the full form width, below its label."""
    heading = W.QLabel(label)
    heading.setSizePolicy(W.QSizePolicy.Preferred, W.QSizePolicy.Fixed)
    form.addRow(heading)
    form.addRow(widget)
    widget.setMinimumHeight(110)
    widget.setMaximumHeight(170)
    widget.setHorizontalScrollMode(W.QAbstractItemView.ScrollPerPixel)


def spin(minimum=0, maximum=999999, value=0):
    widget = W.QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def numeric_edit(text=""):
    """Numeric entries need a modest width, even inside a wide form."""
    widget = W.QLineEdit(text)
    widget.setSizePolicy(W.QSizePolicy.Maximum, W.QSizePolicy.Fixed)
    widget.setMaximumWidth(150)
    return widget


class ScientificDoubleSpinBox(W.QDoubleSpinBox):
    """A double spin box that accepts decimal and ``e`` exponent notation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Other numeric inputs use Python's locale-independent float parser, so
        # keep the decimal point and exponent syntax consistent across the GUI.
        self.setLocale(C.QLocale.c())
        self.setSizePolicy(W.QSizePolicy.Maximum, W.QSizePolicy.Fixed)
        self.setToolTip("Decimal or scientific notation is accepted (for example, 1.4e5).")

    def textFromValue(self, value):
        # Python's shortest round-trip representation removes padding without
        # rounding away a sample boundary when Qt reinterprets an unchanged edit.
        text = repr(float(value))
        return text[:-2] if text.endswith(".0") else text

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(self.fontMetrics().horizontalAdvance("-0.123456789012") + 32)
        return size

    def minimumSizeHint(self):
        return self.sizeHint()

    def validate(self, text, position):
        validator = G.QDoubleValidator(
            self.minimum(), self.maximum(), self.decimals(), self
        )
        validator.setLocale(self.locale())
        validator.setNotation(G.QDoubleValidator.ScientificNotation)
        return validator.validate(text, position)

    def valueFromText(self, text):
        value, valid = self.locale().toDouble(text.strip())
        return value if valid else self.value()


class ImportDialog(W.QDialog):
    def __init__(self, path, info, parent=None):
        super().__init__(parent)
        self.path, self.info, self.dataset, self.worker = path, info, None, None
        self.guess_worker = None
        self.guess_pending = False
        self.preview_worker = None
        self.temporal_metadata = None
        self.cancel_event = threading.Event()
        self._closing = self._accept_pending = False
        self.setWindowTitle("Import acquisition")
        self.resize(1020, 780)
        layout = W.QVBoxLayout(self)
        header = W.QHBoxLayout()
        title = W.QLabel("Map your acquisition")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        appearance = parent.appearance.currentText() if parent is not None and hasattr(parent, "appearance") else "Light"
        self.facility_logo = FacilityLogo(appearance, height=46)
        header.addWidget(self.facility_logo)
        layout.addLayout(header)
        outer_layout = layout
        content = W.QWidget()
        layout = W.QVBoxLayout(content)
        time_panel = W.QWidget()
        time_layout = W.QVBoxLayout(time_panel)
        hint = W.QLabel("Select 1–3 channels, then describe the stored records. Time must be the last raw axis.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.tabs = W.QTabWidget()
        self.tabs.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Maximum)
        layout.addWidget(self.tabs)
        mapped = W.QWidget()
        form = form_layout(mapped)
        self.motion_form = form
        self.channels = W.QListWidget()
        self.channels.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        for spec in info["channels"]:
            description = (spec.get("data_type") or "").strip()
            identity = f"B{spec['board']} Ch{spec['channel']}"
            if description:
                identity += f"  •  {description}"
            item = W.QListWidgetItem(f"{identity}  •  {spec['digitizer']} / {spec['config_name']} / {spec['adc']}")
            item.setData(C.Qt.UserRole, spec)
            item.setToolTip(item.text())
            self.channels.addItem(item)
        if self.channels.count():
            self.channels.item(0).setSelected(True)
        add_list_field(form, "Digitizer channels", self.channels)
        self.motion = combo([])
        self.motion.addItem("None — manual spatial dimensions", None)
        for ctrl in info["controls"]:
            self.motion.addItem(f"{ctrl[0]} / {ctrl[1]}", ctrl)
        if self.motion.count() > 1:
            self.motion.setCurrentIndex(1)
        form.addRow("Motion control", self.motion)
        self.mapping = combo(["Motion coordinates", "Manual dimensions"])
        if self.motion.count() == 1:
            self.mapping.setCurrentIndex(1)
        form.addRow("Mapping", self.mapping)
        self.position_source = combo(["Target if available", "Measured positions"])
        form.addRow("Position coordinates", self.position_source)
        self.cases, self.repeats = spin(1, value=1), spin(1, value=1)
        form.addRow("Cases", self.cases)
        form.addRow("Shots per case", self.repeats)
        self.order = combo(["case,shot", "shot,case"])
        self.order.setToolTip("Per-position acquisition order; the last listed dimension varies fastest.")
        form.addRow("Per-position order", self.order)
        self.precision = spin(0, 8, 4)
        form.addRow("Coordinate decimals", self.precision)
        self.start, self.stop = spin(), spin()
        self.start.setToolTip("First original record index, inclusive and zero-based.")
        self.stop.setToolTip("Stop original record index, exclusive; 0 reads to the end.")
        form.addRow("First record (inclusive)", self.start)
        form.addRow("Stop record (0 = end)", self.stop)
        self.tabs.addTab(mapped, "BaPSF / bapsflib")
        raw = W.QWidget()
        rawform = form_layout(raw)
        self.raw_paths = W.QListWidget()
        self.raw_paths.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        for path_, shape in info["datasets"]:
            item = W.QListWidgetItem(f"/{path_}  {shape}")
            item.setData(C.Qt.UserRole, path_)
            item.setToolTip(item.text())
            self.raw_paths.addItem(item)
        add_list_field(rawform, "Numeric HDF5 datasets", self.raw_paths)
        self.dt = numeric_edit("1e-6")
        rawform.addRow("Sample interval (s)", self.dt)
        self.raw_units = W.QLineEdit("V")
        rawform.addRow("Signal units", self.raw_units)
        raw_note = W.QLabel("Raw import uses manual dimensions; values are read without ADC calibration.")
        raw_note.setWordWrap(True)
        rawform.addRow(raw_note)
        self.tabs.addTab(raw, "Raw HDF5")
        if not info["channels"]:
            self.tabs.setCurrentIndex(1)
        self.t0 = numeric_edit("0")
        timeform = form_layout()
        self.t0.setToolTip("User-defined time origin in seconds.")
        timeform.addRow("Time origin (s)", self.t0)
        self.space_units = W.QLineEdit("cm")
        self.space_units.setToolTip("Spatial units for raw/manual coordinates.")
        timeform.addRow("Spatial units", self.space_units)
        time_layout.addLayout(timeform)
        temporal = W.QGroupBox("Read fewer temporal samples")
        temporal_form = form_layout(temporal)
        self.time_range_mode = combo(["All samples", "Sample limits", "Time limits (s)"])
        self.time_first, self.time_last = numeric_edit("0"), numeric_edit()
        self.time_last.setPlaceholderText("End of recording")
        self.time_first.setEnabled(False)
        self.time_last.setEnabled(False)
        self.decimation = spin(1, 2**24, 1)
        temporal_form.addRow("Temporal range", self.time_range_mode)
        self.time_first.setToolTip("First original sample index or start time in seconds, inclusive.")
        self.time_last.setToolTip("Last original sample index or end time in seconds, inclusive; blank reads to the end.")
        self.decimation.setToolTip("Keep every Nth original sample; 1 keeps all samples.")
        temporal_form.addRow("First sample / time", self.time_first)
        temporal_form.addRow("Last sample / time", self.time_last)
        temporal_form.addRow("Keep every Nth sample", self.decimation)
        self.downsampling = combo(list(DOWNSAMPLING_METHODS))
        self.downsampling.setMinimumContentsLength(24)
        self.temporal_form = temporal_form
        temporal_form.addRow("Downsampling", self.downsampling)
        temporal_form.setRowVisible(self.downsampling, False)
        self.downsampling.currentIndexChanged.connect(self.update_time_summary)
        self.preview_button = W.QPushButton("Preview one trace / choose limits…")
        self.preview_button.clicked.connect(self.preview_time)
        temporal_form.addRow(self.preview_button)
        self.time_summary = W.QLabel("Limits are inclusive and refer to original samples. The imported rate is divided by N.")
        self.time_summary.setWordWrap(True)
        temporal_form.addRow(self.time_summary)
        self.alias_note = W.QLabel("No anti-alias filtering: frequencies above the reduced Nyquist limit can alias.")
        self.alias_note.setWordWrap(True)
        self.alias_note.setVisible(False)
        temporal_form.addRow(self.alias_note)
        self.time_range_mode.currentIndexChanged.connect(self.time_mode_changed)
        self.decimation.valueChanged.connect(self.update_time_summary)
        self.time_first.editingFinished.connect(self.update_time_summary)
        self.time_last.editingFinished.connect(self.update_time_summary)
        self.t0.editingFinished.connect(self.update_time_summary)
        self.channels.itemSelectionChanged.connect(self.invalidate_time_summary)
        self.raw_paths.itemSelectionChanged.connect(self.invalidate_time_summary)
        self.tabs.currentChanged.connect(self.invalidate_time_summary)
        self.dt.editingFinished.connect(self.invalidate_time_summary)
        time_layout.addWidget(temporal)
        time_layout.addStretch()
        self.manual_panel = W.QWidget()
        manual_layout = W.QVBoxLayout(self.manual_panel)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_hint = W.QLabel("Manual dimensions • slowest → fastest; omit shot for one stored trace")
        manual_hint.setWordWrap(True)
        manual_layout.addWidget(manual_hint)
        self.axes = W.QTableWidget(0, 4)
        self.axes.setHorizontalHeaderLabels(["Axis", "Size", "Start", "End"])
        self.axes.horizontalHeader().setSectionResizeMode(W.QHeaderView.Stretch)
        self.axes.setVisible(False)
        manual_layout.addWidget(self.axes)
        row = W.QHBoxLayout()
        add, remove = W.QPushButton("+ Axis"), W.QPushButton("Remove axis")
        add.clicked.connect(self.add_axis)
        remove.clicked.connect(self.remove_axis)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        manual_layout.addLayout(row)
        layout.addWidget(self.manual_panel)
        self.message = W.QLabel("A point with one stored trace needs no manual axes. Dimension sizes must multiply to the record count.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addStretch()
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Cancel)
        self.cancel_button = buttons.button(W.QDialogButtonBox.Cancel)
        self.load = buttons.addButton("Load acquisition", W.QDialogButtonBox.AcceptRole)
        self.load.clicked.connect(self.begin)
        buttons.rejected.connect(self.reject)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        time_scroll = W.QScrollArea()
        time_scroll.setWidgetResizable(True)
        time_scroll.setWidget(time_panel)
        split = W.QSplitter()
        split.addWidget(scroll)
        split.addWidget(time_scroll)
        split.setSizes([660, 340])
        outer_layout.addWidget(split, 1)
        self.import_progress = W.QProgressBar()
        self.import_progress.setRange(0, 1000)
        self.import_progress.setVisible(False)
        self.progress_detail = W.QLabel()
        self.progress_detail.setWordWrap(True)
        self.progress_detail.setVisible(False)
        outer_layout.addWidget(self.import_progress)
        outer_layout.addWidget(self.progress_detail)
        outer_layout.addWidget(buttons)
        self.guess_timer = C.QTimer(self)
        self.guess_timer.setSingleShot(True)
        self.guess_timer.setInterval(200)
        self.guess_timer.timeout.connect(self.start_guess)
        self.channels.itemSelectionChanged.connect(self.request_guess)
        self.motion.currentIndexChanged.connect(self.request_guess)
        self.mapping.currentIndexChanged.connect(self.request_guess)
        self.position_source.currentIndexChanged.connect(self.request_guess)
        self.precision.valueChanged.connect(self.request_guess)
        self.start.valueChanged.connect(self.request_guess)
        self.stop.valueChanged.connect(self.request_guess)
        self.mapping.currentIndexChanged.connect(self.update_mapping_fields)
        self.tabs.currentChanged.connect(self.update_mapping_fields)
        self.update_mapping_fields()
        C.QTimer.singleShot(0, self.request_guess)

    def update_mapping_fields(self):
        manual = self.tabs.currentIndex() == 1 or self.mapping.currentIndex() == 1
        self.manual_panel.setVisible(manual)
        for field in (self.position_source, self.cases, self.repeats, self.order, self.precision):
            self.motion_form.setRowVisible(field, not manual)

    def time_mode_changed(self):
        enabled = self.time_range_mode.currentIndex() != 0
        self.time_first.setEnabled(enabled)
        self.time_last.setEnabled(enabled)
        self.time_first.setText("0" if self.time_range_mode.currentIndex() != 2 else self.t0.text())
        self.time_last.clear()
        self.update_time_summary()

    def invalidate_time_summary(self):
        self.temporal_metadata = None
        self.time_summary.setText("Limits are inclusive and refer to original samples. Preview a trace to see the effective rate and retained sample count.")

    def temporal_options(self):
        options = dict(decimation=self.decimation.value())
        if self.decimation.value() > 1:
            options["downsampling"] = DOWNSAMPLING_METHODS[self.downsampling.currentText()]
        mode = self.time_range_mode.currentIndex()
        if mode:
            cast = int if mode == 1 else float
            first = cast(self.time_first.text())
            last = cast(self.time_last.text()) if self.time_last.text().strip() else None
            options["sample_limits" if mode == 1 else "time_limits"] = (first, last)
        return options

    def update_time_summary(self):
        active = self.decimation.value() > 1
        self.temporal_form.setRowVisible(self.downsampling, active)
        self.alias_note.setVisible(active)
        notes = {"polyphase": "Low-pass FIR filtering before resampling; zero padding at the selected interval edges.",
                 "simple": "No anti-alias filtering: frequencies above the reduced Nyquist limit can alias.",
                 "average": "Average complete blocks of N samples; use block-center times and discard an incomplete final block. Averages provide weaker anti-alias suppression than FIR."}
        self.alias_note.setText(notes[DOWNSAMPLING_METHODS[self.downsampling.currentText()]])
        if self.temporal_metadata is None:
            return
        try:
            from .temporal import sample_slice
            meta = self.temporal_metadata
            selection = sample_slice(meta["samples"], meta["dt"], float(self.t0.text()), **self.temporal_options())
            count = retained_samples(selection, DOWNSAMPLING_METHODS[self.downsampling.currentText()])
            rate = 1 / (meta["dt"] * selection.step)
            self.time_summary.setText(f"Original rate: {1/meta['dt']:g} Hz · Imported rate: {rate:g} Hz · "
                                      f"Nyquist: {rate/2:g} Hz\nRetain {count:,} / {meta['samples']:,} samples per trace "
                                      f"({count/meta['samples']:.1%} of full temporal storage).")
        except (ValueError, TypeError) as exc:
            self.time_summary.setText(str(exc))

    def preview_time(self):
        if self._closing or self._accept_pending:
            return
        try:
            if self.preview_worker and self.preview_worker.isRunning():
                return
            sources = []
            if self.tabs.currentIndex() == 0:
                for item in self.channels.selectedItems():
                    sources.append((item.text(), item.data(C.Qt.UserRole), None))
                control = self.motion.currentData() if self.mapping.currentIndex() == 0 else None
                kwargs = dict(control=control, start=self.start.value(), stop=self.stop.value() or None,
                              position_source="target" if self.position_source.currentIndex() == 0 else "measured",
                              decimals=self.precision.value())
            else:
                for item in self.raw_paths.selectedItems():
                    sources.append((item.text(), None, item.data(C.Qt.UserRole)))
                kwargs = dict(dt=float(self.dt.text()))
            if not sources:
                raise ValueError("Select a channel or dataset to preview.")
            t0, options = float(self.t0.text()), self.temporal_options()
            self.guess_timer.stop()
            self.preview_button.setEnabled(False)
            self.load.setEnabled(False)
            self.message.setText("Reading time and record metadata for one preview channel…")
            def ready(metadata):
                if self._closing or self._accept_pending:
                    return
                self.preview_worker.wait()
                try:
                    self.temporal_metadata = metadata
                    self.update_time_summary()
                    from .import_time_gui import ImportTimeDialog
                    appearance = self.parent().appearance.currentText() if self.parent() is not None and hasattr(self.parent(), "appearance") else "Light"
                    dialog = ImportTimeDialog(self.path, sources, kwargs, metadata, t0, options, appearance, self)
                    if dialog.exec() == W.QDialog.Accepted:
                        first, last = dialog.sample_limits()
                        self.temporal_metadata = dialog.metadata
                        self.time_range_mode.setCurrentIndex(1)
                        self.time_first.setText(str(first))
                        self.time_last.setText(str(last))
                        self.update_time_summary()
                    self.message.setText("Review the temporal settings before loading the acquisition.")
                except Exception as exc:
                    self.message.setText(str(exc))
                finally:
                    self.preview_button.setEnabled(True)
                    self.load.setEnabled(True)
            def failed(message):
                if self._closing or self._accept_pending:
                    return
                self.preview_button.setEnabled(True)
                self.load.setEnabled(True)
                self.message.setText(message)
            self.preview_worker = Worker(lambda: io.import_metadata(self.path, spec=sources[0][1],
                                         dataset_path=sources[0][2], canceled=self.cancel_event.is_set, **kwargs), self,
                                         canceled=self.cancel_event.is_set)
            self.preview_worker.result.connect(ready)
            self.preview_worker.failed.connect(failed)
            self.preview_worker.finished.connect(self.workers_finished)
            self.preview_worker.start()
        except Exception as exc:
            self.failed(str(exc))

    def request_guess(self, *args):
        """Debounce inputs that change the inferred acquisition layout."""
        if not self._closing and self.tabs.currentIndex() == 0 and self.load.isEnabled() and self.preview_button.isEnabled():
            self.cases.setValue(1)
            self.guess_timer.start()

    def start_guess(self):
        if self._closing or not self.load.isEnabled() or not self.preview_button.isEnabled():
            return
        if self.guess_worker and self.guess_worker.isRunning():
            self.guess_pending = True
            return
        selected = self.channels.selectedItems()
        if not selected:
            return
        spec = selected[0].data(C.Qt.UserRole)
        control = self.motion.currentData() if self.mapping.currentIndex() == 0 else None
        source = "target" if self.position_source.currentIndex() == 0 else "measured"
        start, stop = self.start.value(), self.stop.value() or None
        if stop is not None and stop <= start:
            return
        decimals = self.precision.value()
        self.message.setText("Estimating spatial points and shots from the selected channel…")
        self.guess_worker = Worker(
            lambda: io.guess_shots_per_case(
                self.path, spec, control, start, stop, source, decimals, canceled=self.cancel_event.is_set
            ),
            self, canceled=self.cancel_event.is_set,
        )
        self.guess_worker.result.connect(self.guess_ready)
        self.guess_worker.failed.connect(self.guess_failed)
        self.guess_worker.finished.connect(self.guess_finished)
        self.guess_worker.start()

    def guess_ready(self, guess):
        if self._closing or self._accept_pending:
            return
        self.cases.setValue(1)
        self.repeats.setValue(guess["shots_per_case"])
        if not self.load.isEnabled():
            return
        shape = " × ".join(map(str, guess["spatial_shape"])) or "point"
        self.message.setText(
            f"Guessed 1 case and {guess['shots_per_case']} shots per case from "
            f"{guess['records']} records across {guess['spatial_points']} spatial "
            f"points ({shape}); positions: {guess['position_field']}. You may edit "
            "the case and shot counts."
        )

    def guess_failed(self, message):
        if self._closing or self._accept_pending:
            return
        self.message.setText(
            f"Could not infer shots automatically: {message} Enter the number of "
            "cases and shots per case manually."
        )

    def guess_finished(self):
        self.workers_finished()
        if not self._closing and not self._accept_pending and self.guess_pending:
            self.guess_pending = False
            self.guess_timer.start()

    def add_axis(self):
        row = self.axes.rowCount()
        if row == 4:
            return
        self.axes.insertRow(row)
        self.axes.setCellWidget(row, 0, combo(["x", "y", "z", "case", "shot"]))
        for col, value in enumerate(["1", "0", "0"], start=1):
            self.axes.setItem(row, col, W.QTableWidgetItem(value))
        self.update_axes_height()

    def remove_axis(self):
        row = self.axes.currentRow()
        self.axes.removeRow(row if row >= 0 else self.axes.rowCount()-1)
        self.update_axes_height()

    def update_axes_height(self):
        self.axes.setVisible(self.axes.rowCount() > 0)
        height = self.axes.horizontalHeader().height() + sum(self.axes.rowHeight(r) for r in range(self.axes.rowCount()))
        self.axes.setFixedHeight(height + 2*self.axes.frameWidth())

    def begin(self):
        if self._closing or self._accept_pending or self.worker and self.worker.isRunning():
            return
        self.cancel_event.clear()
        try:
            axes = [(self.axes.cellWidget(r, 0).currentText(), int(self.axes.item(r, 1).text()),
                     float(self.axes.item(r, 2).text()), float(self.axes.item(r, 3).text()))
                    for r in range(self.axes.rowCount())]
            if any(a[1] < 1 for a in axes):
                raise ValueError("Axis sizes must be positive.")
            t0 = float(self.t0.text())
            temporal_options = self.temporal_options()
            spatial_units = self.space_units.text()
            if self.tabs.currentIndex() == 1:
                selected = self.raw_paths.selectedItems()
                if not 1 <= len(selected) <= 3:
                    raise ValueError("Select one, two or three datasets (Cmd/Ctrl-click for multiple).")
                paths = {f"C{i+1} · {item.data(C.Qt.UserRole).split('/')[-1]}": item.data(C.Qt.UserRole)
                         for i, item in enumerate(selected)}
                dt, units = float(self.dt.text()), self.raw_units.text()
                fn = lambda: io.read_raw(self.path, paths, axes, dt, t0, spatial_units, units, progress=report, canceled=self.cancel_event.is_set, **temporal_options)
            else:
                selected = self.channels.selectedItems()
                if not 1 <= len(selected) <= 3:
                    raise ValueError("Select one, two or three digitizer channels.")
                selections = {io.channel_name(item.data(C.Qt.UserRole), i+1): item.data(C.Qt.UserRole)
                              for i, item in enumerate(selected)}
                ctrl = self.motion.currentData()
                position_source = "target" if self.position_source.currentIndex() == 0 else "measured"
                use_motion = self.mapping.currentIndex() == 0
                if use_motion and ctrl is None:
                    raise ValueError("Select a motion control or use manual dimensions.")
                cases, repeats, order = self.cases.value(), self.repeats.value(), self.order.currentText()
                decimals, start, stop = self.precision.value(), self.start.value(), self.stop.value() or None
                if stop is not None and stop <= start:
                    raise ValueError("Stop record must be greater than first record.")
                def fn():
                    rec = io.read_lapd(self.path, selections, ctrl if use_motion else None, start, stop, t0, position_source,
                                       progress=report, canceled=self.cancel_event.is_set, **temporal_options)
                    return (io.map_motion(rec, cases, repeats, order, decimals, progress=report, canceled=self.cancel_event.is_set) if use_motion
                            else io.map_manual(rec, axes, spatial_units, progress=report, canceled=self.cancel_event.is_set))
            self.load.setEnabled(False)
            self.preview_button.setEnabled(False)
            self.guess_timer.stop()
            self.message.setText("Reading and mapping acquisition…")
            def report(done, total, stage):
                self.worker.progress.emit(done, total, stage)
            self.import_progress.setValue(0)
            self.import_progress.setVisible(True)
            self.progress_detail.setText("Preparing acquisition…")
            self.progress_detail.setVisible(True)
            self.worker = Worker(fn, self, canceled=self.cancel_event.is_set)
            self.worker.progress.connect(self.update_import_progress)
            self.worker.result.connect(self.loaded)
            self.worker.failed.connect(self.failed)
            self.worker.finished.connect(self.workers_finished)
            self.worker.start()
        except Exception as exc:
            self.failed(str(exc))

    def update_import_progress(self, done, total, stage):
        if self._closing or self._accept_pending:
            return
        fraction = min(1., max(0., done / total)) if total else 0.
        self.import_progress.setValue(round(fraction*1000))
        phase = stage.phase if isinstance(stage, ShotProgress) else (2 if stage == "Mapping" else 1)
        self.import_progress.setFormat(f"Stage {phase}/2 · {fraction:.0%} complete")
        if isinstance(stage, ShotProgress):
            completed, count = stage.completed, stage.total
            channel = f" · channel {stage.channel}/{stage.channels}" if stage.channels > 1 else ""
            label = stage.stage + channel
        else:
            completed, count = int(done), int(total)
            label = stage
        detail = f"{completed:,} / {count:,} shots (acquisitions) processed · {count-completed:,} remaining"
        self.progress_detail.setText(f"{label}: {detail}")

    def loaded(self, data):
        if self._closing or self.cancel_event.is_set():
            return
        self.dataset = data
        self._accept_pending = True
        self.guess_timer.stop()
        self.guess_pending = False
        # Stop auxiliary metadata work without blocking the GUI thread. A
        # Cancel click during this cleanup still discards the completed result.
        self.cancel_event.set()
        self.workers_finished()

    def failed(self, message):
        if self._closing or self._accept_pending:
            return
        self.import_progress.setVisible(False)
        self.progress_detail.setVisible(False)
        self.message.setText(message)
        self.load.setEnabled(True)
        self.preview_button.setEnabled(True)

    def workers_finished(self):
        if any(worker and worker.isRunning() for worker in (self.worker, self.guess_worker, self.preview_worker)):
            return
        if self._closing:
            self.dataset = None
            super().reject()
        elif self._accept_pending:
            super().accept()

    def reject(self):
        self._closing = True
        self._accept_pending = False
        self.cancel_event.set()
        self.dataset = None
        self.guess_timer.stop()
        self.guess_pending = False
        self.load.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.message.setText("Stopping import…")
        self.import_progress.setVisible(False)
        self.progress_detail.setVisible(False)
        # Workers stop at their next bounded read/filter/copy boundary. Keep
        # their Qt owners alive only until handles and buffers are released.
        self.workers_finished()


class SliceAxisControls(W.QGroupBox):
    """Independent vertical limits; invalid edits retain the last valid bounds."""
    changed = C.Signal()
    MODES = {"Auto — all times": "auto", "Manual": "manual", "Interactive": "interactive"}

    def __init__(self, parent=None):
        super().__init__("Slice vertical axis", parent)
        self._limits = (-1., 1.)
        layout = W.QVBoxLayout(self)
        self.mode = combo(list(self.MODES))
        layout.addWidget(self.mode)
        form = form_layout()
        self.minimum, self.maximum = numeric_edit("-1"), numeric_edit("1")
        form.addRow("Min", self.minimum)
        form.addRow("Max", self.maximum)
        layout.addLayout(form)
        self.message = W.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addStretch()
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.minimum.editingFinished.connect(self.accept_limits)
        self.maximum.editingFinished.connect(self.accept_limits)
        self.mode_changed()

    def mode_changed(self):
        manual = self.mode.currentText() == "Manual"
        self.minimum.setEnabled(manual)
        self.maximum.setEnabled(manual)
        self.message.setText({"Manual": "Enter min/max, then press Enter or leave the field.",
                              "Interactive": "Use the plot toolbar to zoom, pan, or return Home.",
                              "Auto — all times": "Min/max over all positions and times in this slice."}[self.mode.currentText()])
        if manual:
            self.accept_limits()
        else:
            self.changed.emit()

    def accept_limits(self):
        if self.mode.currentText() != "Manual":
            return
        import numpy as np
        try:
            limits = (float(self.minimum.text()), float(self.maximum.text()))
            if not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
                raise ValueError
        except ValueError:
            self.message.setText("Enter finite values with Min < Max. Previous limits remain active.")
            return
        self._limits = limits
        self.message.setText("Manual limits apply to every frame.")
        self.changed.emit()

    def show_limits(self, limits):
        if self.mode.currentText() != "Manual":
            self._limits = tuple(limits)
            self.minimum.setText(f"{limits[0]:.12g}")
            self.maximum.setText(f"{limits[1]:.12g}")

    def settings(self):
        mode = self.MODES[self.mode.currentText()]
        return dict(mode=mode, limits=self._limits) if mode == "manual" else dict(mode=mode)

    def reset(self):
        self.mode.setCurrentIndex(0)
        self.show_limits((-1., 1.))


class ColormapRangeControls(SliceAxisControls):
    """Optional fixed mesh color bounds, in displayed signal units."""
    MODES = {"Automatic": "auto", "Manual": "manual"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Mesh colormap range")

    def mode_changed(self):
        manual = self.mode.currentText() == "Manual"
        self.minimum.setEnabled(manual)
        self.maximum.setEnabled(manual)
        self.message.setText("Enter min/max, then press Enter or leave the field." if manual else
                             "Use the current automatic scaling method.")
        if manual:
            self.accept_limits()
        else:
            self.changed.emit()


class VectorArrowControls(W.QWidget):
    """Compact, expandable style controls shared by vector displays."""
    changed = C.Signal()

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        from .vector_style import arrow_style
        style = arrow_style(settings)
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle = W.QToolButton()
        self.toggle.setText("Arrow style")
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(C.Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(C.Qt.RightArrow)
        layout.addWidget(self.toggle)
        self.panel = W.QWidget()
        form = form_layout(self.panel)
        form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.panel)
        self.fields = {}
        for key, label, low, high, step, tooltip in [
            ("width", "Shaft thickness (pt)", .1, 20., .1, "Physical shaft width in points (1/72 inch), including image and movie exports."),
            ("length", "Length multiplier", .05, 20., .1, "Larger values lengthen arrows without changing data or arrow colors. At 1, the reference maximum spans 80% of the sampled grid spacing."),
            ("headwidth", "Head width / shaft", 1., 20., .25, "Arrowhead width as a multiple of shaft thickness."),
            ("headlength", "Head length / shaft", 1., 30., .25, "Arrowhead length as a multiple of shaft thickness."),
            ("alpha", "Opacity", 0., 1., .05, "Zero is invisible; one is fully opaque."),
            ("outline", "Outline thickness (pt)", 0., 5., .1, "Dark edge width in points. Set zero to remove the outline."),
        ]:
            widget = W.QDoubleSpinBox()
            widget.setDecimals(2)
            widget.setRange(low, high)
            widget.setSingleStep(step)
            widget.setValue(style[key])
            widget.setKeyboardTracking(False)
            widget.setToolTip(tooltip)
            self.fields[key] = widget
            form.addRow(label, widget)
            widget.valueChanged.connect(self.changed)
        self.stride = spin(0, 10000, int(style["stride"]))
        self.stride.setSpecialValueText("Auto")
        self.stride.setToolTip("Draw every Nth grid point along both axes. Auto aims for roughly 18–36 arrows on the longest axis; 1 draws every point. Display sampling does not alter data.")
        form.insertRow(2, "Grid stride", self.stride)
        self.stride.valueChanged.connect(self.changed)
        self.pivot = combo(["Middle", "Tail", "Tip"])
        self.pivot.setCurrentIndex(["mid", "tail", "tip"].index(style["pivot"]))
        self.pivot.setToolTip("Part of each arrow anchored to its grid point.")
        form.addRow("Pivot", self.pivot)
        self.pivot.currentIndexChanged.connect(self.changed)
        reset = W.QPushButton("Reset arrow style")
        reset.clicked.connect(self.reset)
        form.addRow(reset)
        self.panel.hide()
        self.toggle.toggled.connect(self.expand)

    def expand(self, expanded):
        self.toggle.setArrowType(C.Qt.DownArrow if expanded else C.Qt.RightArrow)
        self.panel.setVisible(expanded)

    def settings(self):
        return dict({key: widget.value() for key, widget in self.fields.items()},
                    stride=self.stride.value(), pivot=["mid", "tail", "tip"][self.pivot.currentIndex()])

    def reset(self):
        from .vector_style import DEFAULT_ARROW_STYLE
        widgets = [*self.fields.values(), self.stride, self.pivot]
        blockers = [C.QSignalBlocker(widget) for widget in widgets]
        for key, widget in self.fields.items():
            widget.setValue(DEFAULT_ARROW_STYLE[key])
        self.stride.setValue(DEFAULT_ARROW_STYLE["stride"])
        self.pivot.setCurrentIndex(0)
        del blockers
        self.changed.emit()


class SpatialAveragingDialog(W.QDialog):
    """Configure spatial neighborhoods in the displayed plane's axis order."""
    METHODS = {"Box average": "box", "Gaussian average": "gaussian",
               "Disk average": "disk", "Median filter": "median"}
    MODES = {"Reflect edges": "reflect", "Extend nearest edge": "nearest", "Periodic wrap": "wrap"}
    MISSING = {"Propagate missing values": "propagate", "Omit missing neighbors": "omit"}

    def __init__(self, settings, spatial_dims, parent=None):
        super().__init__(parent)
        from .spatial import DEFAULT_SPATIAL_AVERAGING
        settings = DEFAULT_SPATIAL_AVERAGING | settings
        self.setWindowTitle("Spatial averaging settings")
        self.setMinimumWidth(480)
        layout = W.QVBoxLayout(self)
        note = W.QLabel("Applied to each channel's 2D plane after other preprocessing, separately at every time, case and remaining shot. Sizes are grid points, not physical distances; on irregular grids the physical width varies.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.form = form_layout()
        layout.addLayout(self.form)
        self.method = combo(list(self.METHODS))
        self.method.setCurrentIndex(list(self.METHODS.values()).index(settings["method"]))
        self.form.addRow("Method", self.method)
        self.windows = [spin(1, 101, v) for v in settings["window_size"]]
        self.sigmas = [numeric_edit(str(v)) for v in settings["sigma"]]
        for dim, window, sigma in zip(spatial_dims, self.windows, self.sigmas):
            window.setSingleStep(2)
            self.form.addRow(f"{dim} window (odd grid points)", window)
            self.form.addRow(f"{dim} Gaussian σ (grid points)", sigma)
        self.radius = spin(1, 50, settings["radius"])
        self.form.addRow("Disk radius (grid points)", self.radius)
        self.boundary = combo(list(self.MODES))
        self.boundary.setCurrentIndex(list(self.MODES.values()).index(settings["mode"]))
        self.form.addRow("Boundary", self.boundary)
        self.missing = combo(list(self.MISSING))
        self.missing.setCurrentIndex(list(self.MISSING.values()).index(settings["nan_policy"]))
        self.form.addRow("Missing values", self.missing)
        hint = W.QLabel("Box and disk averages weight neighbors equally. Gaussian weights decrease with distance (support ±4σ). Median rejects isolated spikes. Propagate marks any neighborhood containing missing data as missing; omit uses finite neighbors and can fill gaps. Empty neighborhoods remain missing. Changes take effect with Apply to original data.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.error_label = W.QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.method.currentIndexChanged.connect(self.update_fields)
        self.update_fields()

    def update_fields(self):
        method = self.METHODS[self.method.currentText()]
        for widget in self.windows:
            self.form.setRowVisible(widget, method in {"box", "median"})
        for widget in self.sigmas:
            self.form.setRowVisible(widget, method == "gaussian")
        self.form.setRowVisible(self.radius, method == "disk")
        self.error_label.clear()

    def settings(self):
        method = self.METHODS[self.method.currentText()]
        result = dict(method=method, mode=self.MODES[self.boundary.currentText()],
                      nan_policy=self.MISSING[self.missing.currentText()])
        if method in {"box", "median"}:
            result["window_size"] = tuple(w.value() for w in self.windows)
        elif method == "gaussian":
            result["sigma"] = tuple(float(w.text()) for w in self.sigmas)
        else:
            result["radius"] = self.radius.value()
        return result

    def accept(self):
        import numpy as np
        from .spatial import average_spatial
        try:
            average_spatial(np.zeros((1, 1)), **self.settings())
        except (ValueError, TypeError, OverflowError) as exc:
            self.error_label.setText(str(exc))
            return
        super().accept()


class SmoothingDialog(W.QDialog):
    """Show only the controls relevant to the selected time filter."""
    METHODS = {"Moving average": "moving", "Savitzky–Golay": "savgol",
               "Gaussian": "gaussian", "Butterworth": "butterworth"}
    BUTTER_TYPES = {"Low-pass": "lowpass", "High-pass": "highpass", "Band-pass": "bandpass"}

    def __init__(self, settings, time, parent=None):
        super().__init__(parent)
        self.time = time
        self.setWindowTitle("Time smoothing settings")
        self.setMinimumWidth(460)
        layout = W.QVBoxLayout(self)
        note = W.QLabel("Applied after optional integration, independently to each channel, position, case and shot. Windows and σ are in samples.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.form = form_layout()
        layout.addLayout(self.form)
        self.method = combo(list(self.METHODS))
        self.method.setCurrentIndex(list(self.METHODS.values()).index(settings["method"]))
        self.form.addRow("Method", self.method)
        self.fields = {
            "window_size": spin(1, 1000000, settings["window_size"]),
            "polyorder": spin(0, 100, settings["polyorder"]),
            "sigma": numeric_edit(str(settings["sigma"])),
            "butter_type": combo(list(self.BUTTER_TYPES)),
            "cutoff": numeric_edit(str(settings["cutoff"])),
            "cutoff_upper": numeric_edit(str(settings.get("cutoff_upper", 10000.0))),
            "butter_order": spin(1, 20, settings["butter_order"]),
            "mode": combo(["nearest", "reflect", "mirror", "constant", "wrap"]),
            "nan_policy": combo(["propagate", "interp"]),
        }
        self.fields["mode"].setCurrentText(settings["mode"])
        self.fields["nan_policy"].setCurrentText(settings["nan_policy"])
        self.fields["butter_type"].setCurrentIndex(
            list(self.BUTTER_TYPES.values()).index(settings.get("butter_type", "lowpass")))
        for key, label in [("window_size", "Window (samples)"), ("polyorder", "Polynomial order"),
                           ("sigma", "Gaussian σ (samples)"), ("butter_type", "Filter type"),
                           ("cutoff", "Cutoff (Hz)"), ("cutoff_upper", "Upper cutoff (Hz)"),
                           ("butter_order", "Filter order"), ("mode", "Boundary mode"),
                           ("nan_policy", "Missing values")]:
            self.form.addRow(label, self.fields[key])
        self.fields["nan_policy"].setToolTip("propagate: missing values spread through the filter. interp: interpolate gaps, extend endpoints; entirely missing traces stay NaN.")
        self.fields["mode"].setToolTip("constant pads with zero; nearest extends endpoints; reflect/mirror reflect edges; wrap assumes periodic data.")
        self.hint = W.QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.error_label = W.QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.method.currentIndexChanged.connect(self.update_fields)
        self.fields["butter_type"].currentIndexChanged.connect(self.update_fields)
        self.update_fields()

    def active_fields(self):
        method = self.METHODS[self.method.currentText()]
        active = {"moving": ("window_size", "mode"), "savgol": ("window_size", "polyorder"),
                  "gaussian": ("sigma", "mode"),
                  "butterworth": ("butter_type", "cutoff", "butter_order")}[method]
        if method == "butterworth" and self.fields["butter_type"].currentText() == "Band-pass":
            active += ("cutoff_upper",)
        return active

    def update_fields(self):
        import numpy as np
        method = self.METHODS[self.method.currentText()]
        active = self.active_fields()
        self.form.labelForField(self.fields["cutoff"]).setText(
            "Lower cutoff (Hz)" if "cutoff_upper" in active else "Cutoff (Hz)")
        for key, widget in self.fields.items():
            self.form.setRowVisible(widget, key in active or key == "nan_policy")
        if method == "butterworth":
            intervals = np.diff(self.time)
            uniform = len(intervals) and np.allclose(intervals, intervals[0], rtol=1e-5, atol=abs(intervals[0])*1e-8)
            text = (f"Sampling rate comes from the data. Nyquist: {0.5/intervals[0]:g} Hz. Zero-phase filtering uses forward/backward passes."
                    if uniform else "Butterworth requires at least two uniformly spaced time samples.")
            filter_type = self.fields["butter_type"].currentText()
            text += {"Low-pass": " Keeps frequencies below the cutoff.",
                     "High-pass": " Keeps frequencies above the cutoff.",
                     "Band-pass": " Keeps frequencies between the lower and upper cutoffs. The band-pass design has twice the selected order."}[filter_type]
        elif method == "savgol":
            text = "Use an odd window no longer than the trace and larger than the polynomial order. Edges use polynomial interpolation."
        else:
            text = "Filtering uses sample spacing; on nonuniform time grids the physical smoothing width varies."
        self.hint.setText(text + " Interpolation here cannot undo missing values already propagated by integration.")
        self.error_label.clear()

    def settings(self):
        method = self.METHODS[self.method.currentText()]
        result = dict(method=method, nan_policy=self.fields["nan_policy"].currentText())
        for key in self.active_fields():
            widget = self.fields[key]
            result[key] = (widget.value() if isinstance(widget, W.QSpinBox) else
                           widget.currentText() if isinstance(widget, W.QComboBox) else float(widget.text()))
        if method == "butterworth":
            result["butter_type"] = self.BUTTER_TYPES[result["butter_type"]]
        return result

    def accept(self):
        import numpy as np
        from .smoothing import smooth_time_series
        try:
            smooth_time_series(np.zeros(len(self.time)), time=self.time, **self.settings())
        except (ValueError, TypeError) as exc:
            self.error_label.setText(str(exc))
            return
        super().accept()
