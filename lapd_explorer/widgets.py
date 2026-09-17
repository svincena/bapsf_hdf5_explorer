"""Qt workers and the acquisition import dialog."""
from PySide6 import QtCore as C, QtWidgets as W
from . import io


class Worker(C.QThread):
    result = C.Signal(object)
    failed = C.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.result.emit(self.fn())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def combo(items):
    widget = W.QComboBox()
    widget.addItems(items)
    return widget


def spin(minimum=0, maximum=999999, value=0):
    widget = W.QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


class ImportDialog(W.QDialog):
    def __init__(self, path, info, parent=None):
        super().__init__(parent)
        self.path, self.info, self.dataset, self.worker = path, info, None, None
        self.guess_worker = None
        self.guess_pending = False
        self.setWindowTitle("Import acquisition")
        self.resize(780, 760)
        layout = W.QVBoxLayout(self)
        title = W.QLabel("Map your acquisition")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        hint = W.QLabel("Select 1–3 channels, then describe the stored records. Time must be the last raw axis.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.tabs = W.QTabWidget()
        layout.addWidget(self.tabs)
        mapped = W.QWidget()
        form = W.QFormLayout(mapped)
        self.channels = W.QListWidget()
        self.channels.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        for spec in info["channels"]:
            item = W.QListWidgetItem(f"{spec['digitizer']} / {spec['config_name']} / {spec['adc']}  •  B{spec['board']} Ch{spec['channel']}")
            item.setData(C.Qt.UserRole, spec)
            self.channels.addItem(item)
        if self.channels.count():
            self.channels.item(0).setSelected(True)
        form.addRow("Digitizer channels", self.channels)
        self.motion = W.QComboBox()
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
        form.addRow("Number of cases", self.cases)
        form.addRow("Number of shots per case", self.repeats)
        self.order = combo(["case,shot", "shot,case"])
        form.addRow("Per-position order (fastest last)", self.order)
        self.precision = spin(0, 8, 4)
        form.addRow("Motion coordinate decimals", self.precision)
        self.start, self.stop = spin(), spin()
        form.addRow("First record index (inclusive)", self.start)
        form.addRow("Stop record index (exclusive; 0 = end)", self.stop)
        self.tabs.addTab(mapped, "BaPSF / bapsflib")
        raw = W.QWidget()
        rawform = W.QFormLayout(raw)
        self.raw_paths = W.QListWidget()
        self.raw_paths.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        for path_, shape in info["datasets"]:
            item = W.QListWidgetItem(f"/{path_}  {shape}")
            item.setData(C.Qt.UserRole, path_)
            self.raw_paths.addItem(item)
        rawform.addRow("Numeric HDF5 datasets", self.raw_paths)
        self.dt = W.QLineEdit("1e-6")
        rawform.addRow("Sample interval (s)", self.dt)
        self.raw_units = W.QLineEdit("V")
        rawform.addRow("Signal units", self.raw_units)
        rawform.addRow(W.QLabel("Raw import uses manual dimensions; values are read without ADC calibration."))
        self.tabs.addTab(raw, "Raw HDF5")
        if not info["channels"]:
            self.tabs.setCurrentIndex(1)
        self.t0 = W.QLineEdit("0")
        timeform = W.QFormLayout()
        timeform.addRow("Time origin (s; user-defined)", self.t0)
        self.space_units = W.QLineEdit("cm")
        timeform.addRow("Spatial units (raw/manual coordinates)", self.space_units)
        layout.addLayout(timeform)
        layout.addWidget(W.QLabel("Manual dimensions • slowest → fastest; omit shot for one stored trace"))
        self.axes = W.QTableWidget(0, 4)
        self.axes.setHorizontalHeaderLabels(["Axis", "Size", "Start", "End"])
        self.axes.horizontalHeader().setSectionResizeMode(W.QHeaderView.Stretch)
        self.axes.setMaximumHeight(150)
        layout.addWidget(self.axes)
        row = W.QHBoxLayout()
        add, remove = W.QPushButton("+ Axis"), W.QPushButton("Remove axis")
        add.clicked.connect(self.add_axis)
        remove.clicked.connect(lambda: self.axes.removeRow(self.axes.currentRow() if self.axes.currentRow() >= 0 else self.axes.rowCount()-1))
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        layout.addLayout(row)
        self.message = W.QLabel("A point with one stored trace needs no manual axes. Dimension sizes must multiply to the record count.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Cancel)
        self.load = buttons.addButton("Load acquisition", W.QDialogButtonBox.AcceptRole)
        self.load.clicked.connect(self.begin)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
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
        C.QTimer.singleShot(0, self.request_guess)

    def request_guess(self, *args):
        """Debounce inputs that change the inferred acquisition layout."""
        if self.tabs.currentIndex() == 0:
            self.cases.setValue(1)
            self.guess_timer.start()

    def start_guess(self):
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
                self.path, spec, control, start, stop, source, decimals
            ),
            self,
        )
        self.guess_worker.result.connect(self.guess_ready)
        self.guess_worker.failed.connect(self.guess_failed)
        self.guess_worker.finished.connect(self.guess_finished)
        self.guess_worker.start()

    def guess_ready(self, guess):
        self.cases.setValue(1)
        self.repeats.setValue(guess["shots_per_case"])
        shape = " × ".join(map(str, guess["spatial_shape"])) or "point"
        self.message.setText(
            f"Guessed 1 case and {guess['shots_per_case']} shots per case from "
            f"{guess['records']} records across {guess['spatial_points']} spatial "
            f"points ({shape}); positions: {guess['position_field']}. You may edit "
            "the case and shot counts."
        )

    def guess_failed(self, message):
        self.message.setText(
            f"Could not infer shots automatically: {message} Enter the number of "
            "cases and shots per case manually."
        )

    def guess_finished(self):
        if self.guess_pending:
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

    def begin(self):
        try:
            axes = [(self.axes.cellWidget(r, 0).currentText(), int(self.axes.item(r, 1).text()),
                     float(self.axes.item(r, 2).text()), float(self.axes.item(r, 3).text()))
                    for r in range(self.axes.rowCount())]
            if any(a[1] < 1 for a in axes):
                raise ValueError("Axis sizes must be positive.")
            t0 = float(self.t0.text())
            spatial_units = self.space_units.text()
            if self.tabs.currentIndex() == 1:
                selected = self.raw_paths.selectedItems()
                if not 1 <= len(selected) <= 3:
                    raise ValueError("Select one, two or three datasets (Cmd/Ctrl-click for multiple).")
                paths = {f"C{i+1} · {item.data(C.Qt.UserRole).split('/')[-1]}": item.data(C.Qt.UserRole)
                         for i, item in enumerate(selected)}
                dt, units = float(self.dt.text()), self.raw_units.text()
                fn = lambda: io.read_raw(self.path, paths, axes, dt, t0, spatial_units, units)
            else:
                selected = self.channels.selectedItems()
                if not 1 <= len(selected) <= 3:
                    raise ValueError("Select one, two or three digitizer channels.")
                selections = {f"C{i+1} · B{item.data(C.Qt.UserRole)['board']} Ch{item.data(C.Qt.UserRole)['channel']}": item.data(C.Qt.UserRole)
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
                    rec = io.read_lapd(self.path, selections, ctrl, start, stop, t0, position_source)
                    return (io.map_motion(rec, cases, repeats, order, decimals) if use_motion
                            else io.map_manual(rec, axes, spatial_units))
            self.load.setEnabled(False)
            self.message.setText("Reading and mapping acquisition…")
            self.worker = Worker(fn, self)
            self.worker.result.connect(self.loaded)
            self.worker.failed.connect(self.failed)
            self.worker.start()
        except Exception as exc:
            self.failed(str(exc))

    def loaded(self, data):
        self.dataset = data
        self.worker.wait()
        self.accept()

    def failed(self, message):
        self.message.setText(message)
        self.load.setEnabled(True)

    def reject(self):
        if ((self.worker and self.worker.isRunning())
                or (self.guess_worker and self.guess_worker.isRunning())):
            self.message.setText("The read is still running. Close after it finishes.")
            return
        super().reject()


class SmoothingDialog(W.QDialog):
    """Show only the controls relevant to the selected time filter."""
    METHODS = {"Moving average": "moving", "Savitzky–Golay": "savgol",
               "Gaussian": "gaussian", "Butterworth low-pass": "butterworth"}

    def __init__(self, settings, time, parent=None):
        super().__init__(parent)
        self.time = time
        self.setWindowTitle("Time smoothing settings")
        self.setMinimumWidth(460)
        layout = W.QVBoxLayout(self)
        note = W.QLabel("Applied after optional integration, independently to each channel, position, case and shot. Windows and σ are in samples.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.form = W.QFormLayout()
        layout.addLayout(self.form)
        self.method = combo(list(self.METHODS))
        self.method.setCurrentIndex(list(self.METHODS.values()).index(settings["method"]))
        self.form.addRow("Method", self.method)
        self.fields = {
            "window_size": spin(1, 1000000, settings["window_size"]),
            "polyorder": spin(0, 100, settings["polyorder"]),
            "sigma": W.QLineEdit(str(settings["sigma"])),
            "cutoff": W.QLineEdit(str(settings["cutoff"])),
            "butter_order": spin(1, 20, settings["butter_order"]),
            "mode": combo(["nearest", "reflect", "mirror", "constant", "wrap"]),
            "nan_policy": combo(["propagate", "interp"]),
        }
        self.fields["mode"].setCurrentText(settings["mode"])
        self.fields["nan_policy"].setCurrentText(settings["nan_policy"])
        for key, label in [("window_size", "Window (samples)"), ("polyorder", "Polynomial order"),
                           ("sigma", "Gaussian σ (samples)"), ("cutoff", "Cutoff (Hz)"),
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
        self.update_fields()

    def update_fields(self):
        import numpy as np
        method = self.METHODS[self.method.currentText()]
        active = {"moving": {"window_size", "mode"}, "savgol": {"window_size", "polyorder"},
                  "gaussian": {"sigma", "mode"}, "butterworth": {"cutoff", "butter_order"}}[method]
        for key, widget in self.fields.items():
            self.form.setRowVisible(widget, key in active or key == "nan_policy")
        if method == "butterworth":
            intervals = np.diff(self.time)
            uniform = len(intervals) and np.allclose(intervals, intervals[0], rtol=1e-5, atol=abs(intervals[0])*1e-8)
            text = (f"Sampling rate comes from the data. Nyquist: {0.5/intervals[0]:g} Hz. Zero-phase filtering uses forward/backward passes."
                    if uniform else "Butterworth requires at least two uniformly spaced time samples.")
        elif method == "savgol":
            text = "Use an odd window no longer than the trace and larger than the polynomial order. Edges use polynomial interpolation."
        else:
            text = "Filtering uses sample spacing; on nonuniform time grids the physical smoothing width varies."
        self.hint.setText(text + " Interpolation here cannot undo missing values already propagated by integration.")
        self.error_label.clear()

    def settings(self):
        method = self.METHODS[self.method.currentText()]
        result = dict(method=method, nan_policy=self.fields["nan_policy"].currentText())
        active = {"moving": ("window_size", "mode"), "savgol": ("window_size", "polyorder"),
                  "gaussian": ("sigma", "mode"), "butterworth": ("cutoff", "butter_order")}[method]
        for key in active:
            widget = self.fields[key]
            result[key] = (widget.value() if isinstance(widget, W.QSpinBox) else
                           widget.currentText() if isinstance(widget, W.QComboBox) else float(widget.text()))
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
