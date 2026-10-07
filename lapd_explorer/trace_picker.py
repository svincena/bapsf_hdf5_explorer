"""Coordinate, index, global-shot and sequential selection of stored traces."""
import numpy as np
from PySide6 import QtCore as C, QtWidgets as W
from .widgets import spin, form_layout, numeric_edit, combo, ScientificDoubleSpinBox


class TracePicker(W.QWidget):
    changed = C.Signal()
    failed = C.Signal(str)

    def __init__(self, data, selection, parent=None):
        super().__init__(parent)
        self.data = data
        self.indices, self.coordinates = {}, {}
        form = form_layout(self)
        note = W.QLabel("Select a stored trace by axis indices, nearest spatial coordinates, flat index, or global shot. Click the Locations plot to choose a position.")
        note.setWordWrap(True)
        form.addRow(note)
        for d in data.dims[:-1]:
            widget = spin(0, len(data.coords[d])-1, min(selection.get(d, 0), len(data.coords[d])-1))
            self.indices[d] = widget
            form.addRow(f"{d} index", widget)
            widget.valueChanged.connect(self.index_changed)
            if d in data.spatial_dims:
                coordinate = ScientificDoubleSpinBox()
                coordinate.setDecimals(15)
                coordinate.setRange(float(data.coords[d][0]), float(data.coords[d][-1]))
                coordinate.setKeyboardTracking(False)
                coordinate.setToolTip("Snap to the nearest stored coordinate; indices are zero-based.")
                self.coordinates[d] = coordinate
                form.addRow(f"Nearest {d} ({data.spatial_units})", coordinate)
                coordinate.editingFinished.connect(self.choose_coordinates)
        total = int(np.prod(data.shape[:-1]))
        self.flat_index = spin(0, total-1)
        self.flat_index.setToolTip("Zero-based flat index in the displayed non-time axis order (C order, last axis varies fastest).")
        self.flat_index.setEnabled(total > 1)
        form.addRow("Flat trace index", self.flat_index)
        self.flat_index.valueChanged.connect(self.choose_flat)
        self.global_shot = numeric_edit()
        self.global_shot.setParent(self)
        if data.shot_numbers is not None:
            form.addRow("Global shot number", self.global_shot)
            self.global_shot.editingFinished.connect(self.choose_global_shot)
        else:
            self.global_shot.setVisible(False)
        self.description = W.QLabel()
        self.description.setWordWrap(True)
        form.addRow(self.description)
        self.cycle_axis = combo(["Trace (all indices)"] + (["Location"] if data.spatial_dims else []) +
                                (["Shot"] if "shot" in data.dims else []) + (["Case"] if "case" in data.dims else []))
        self.navigation = W.QWidget(self)
        row = W.QHBoxLayout(self.navigation)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(W.QLabel("Cycle through"))
        row.addWidget(self.cycle_axis)
        self.previous, self.next = W.QPushButton("← Previous"), W.QPushButton("Next →")
        self.previous.clicked.connect(lambda: self.cycle(-1))
        self.next.clicked.connect(lambda: self.cycle(1))
        row.addWidget(self.previous)
        row.addWidget(self.next)
        self.wrap = W.QCheckBox("Wrap at the first / last item")
        self.wrap.setChecked(True)
        row.addWidget(self.wrap)
        self.navigation_location = W.QLabel()
        self.navigation_location.setWordWrap(True)
        row.addWidget(self.navigation_location, 1)
        self.sync()

    def index(self):
        return tuple(w.value() for w in self.indices.values())

    def set_index(self, index):
        if tuple(index) == self.index():
            self.sync()
            return
        for widget, value in zip(self.indices.values(), index):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)
        self.index_changed()

    def index_changed(self, *args):
        self.sync()
        self.changed.emit()

    def sync(self):
        index = self.index()
        flat = int(np.ravel_multi_index(index, self.data.shape[:-1])) if index else 0
        self.flat_index.blockSignals(True)
        self.flat_index.setValue(flat)
        self.flat_index.blockSignals(False)
        for d, widget in self.coordinates.items():
            widget.setValue(float(self.data.coords[d][self.indices[d].value()]))
        parts = [f"{d}={self.data.coords[d][w.value()]:g}" + (f" {self.data.spatial_units}" if d in self.data.spatial_dims else "")
                 for d, w in self.indices.items()]
        if self.data.shot_numbers is not None:
            shot = int(self.data.shot_numbers[index])
            self.global_shot.setText(str(shot))
            parts.append(f"Global shot {shot}")
        self.description.setText(f"Trace {flat:,} / {self.flat_index.maximum():,} (zero-based) · " + (", ".join(parts) or "Single time series"))

        self.navigation_location.setText(self.description.text())

    def choose_coordinates(self):
        index = list(self.index())
        for d, widget in self.coordinates.items():
            index[self.data.dims.index(d)] = int(np.argmin(abs(self.data.coords[d]-widget.value())))
        self.set_index(index)

    def choose_flat(self, value):
        self.set_index(np.unravel_index(value, self.data.shape[:-1]) if self.indices else ())

    def choose_global_shot(self):
        try:
            shot = int(self.global_shot.text())
            matches = np.argwhere(self.data.shot_numbers == shot)
            if len(matches) != 1:
                raise ValueError("Global shot must identify exactly one stored trace; use axis indices for absent or duplicate shots.")
            self.set_index(matches[0])
        except ValueError as exc:
            self.failed.emit(str(exc))

    def cycle(self, direction):
        mode = self.cycle_axis.currentText()
        dims = tuple(self.indices) if mode == "Trace (all indices)" else self.data.spatial_dims if mode == "Location" else (mode.lower(),)
        if not dims:
            return
        shape = tuple(len(self.data.coords[d]) for d in dims)
        selected = tuple(self.indices[d].value() for d in dims)
        current = int(np.ravel_multi_index(selected, shape))
        count = int(np.prod(shape))
        new = (current+direction) % count if self.wrap.isChecked() else int(np.clip(current+direction, 0, count-1))
        index = list(self.index())
        for d, value in zip(dims, np.unravel_index(new, shape)):
            index[self.data.dims.index(d)] = value
        self.set_index(index)
