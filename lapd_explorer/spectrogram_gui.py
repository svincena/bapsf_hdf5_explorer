"""Estimator and display controls for a single stored trace's spectrogram."""
import os
import numpy as np
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from . import spectrogram as analysis, plotting
from .appearance import colors
from .widgets import combo, spin, form_layout, ScientificDoubleSpinBox


def number(value=0., low=-1e15, high=1e15):
    widget = ScientificDoubleSpinBox()
    widget.setDecimals(300)
    widget.setRange(low, high)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    return widget


class Controls(W.QWidget):
    estimate_changed = C.Signal()
    display_changed = C.Signal()
    export_requested = C.Signal()

    def __init__(self, fs, samples, names, settings=None, parent=None):
        super().__init__(parent)
        self.fs = fs
        s = settings or analysis.Settings(nperseg=min(256, samples), nfft=min(256, samples),
                                         overlap=3*min(256, samples)//4)
        self._syncing = False
        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.option_tabs = W.QTabWidget()
        estimate = W.QWidget()
        form = form_layout(estimate)
        self.form = form
        self.option_tabs.addTab(estimate, "Estimation")
        note = W.QLabel("One stored trace at the selected location, case and shot. Uses the shared analysis interval; Welch shot averaging is not applied.")
        note.setWordWrap(True)
        outer.addWidget(note)
        outer.addWidget(self.option_tabs)
        self.nperseg = spin(4, 2**24, s.nperseg)
        self.duration = number(1000*s.nperseg/fs, 0.)
        self.duration.setToolTip("Segment duration in ms, rounded to the nearest whole number of samples.")
        self.overlap = spin(0, 2**24, s.overlap)
        self.overlap_percent = number(100*s.overlap/s.nperseg, 0., 99.999999)
        self.nfft = spin(4, 2**24, s.nfft)
        self.fft_workers = spin(1, max(os.cpu_count() or 1, s.fft_workers), s.fft_workers)
        self.window = combo(analysis.WINDOWS)
        self.window.setCurrentText(s.window)
        self.window_parameter = number(s.window_parameter)
        self.symmetric_window = W.QCheckBox("Symmetric window (default is periodic)")
        self.symmetric_window.setChecked(s.symmetric_window)
        self.detrend = combo(["constant", "linear", "none"])
        self.detrend.setCurrentText(s.detrend)
        self.scaling = combo(["density", "spectrum"])
        self.scaling.setCurrentText(s.scaling)
        self.scaling.setToolTip("Density: input units squared per Hz. Spectrum: input units squared; window coherent-gain normalization.")
        self.sides = combo(["One-sided", "Two-sided (centered)"])
        self.sides.setCurrentIndex(s.sides == "twosided")
        self.boundary = combo(analysis.BOUNDARIES)
        self.boundary.setCurrentText(next(k for k, v in analysis.BOUNDARIES.items() if v == s.boundary))
        self.pad_end = W.QCheckBox("Pad final incomplete hop")
        self.pad_end.setChecked(s.pad_end)
        self.pad_end.setToolTip("Include one final window when the hop does not fit exactly. Without boundary extension, missing trailing samples use zero padding.")
        for label, widget in [("Segment samples", self.nperseg), ("Segment duration (ms)", self.duration),
                              ("Overlap samples", self.overlap), ("Overlap (%)", self.overlap_percent),
                              ("FFT samples", self.nfft), ("FFT workers", self.fft_workers),
                              ("Window", self.window), ("Window parameter", self.window_parameter)]:
            form.addRow(label, widget)
        form.addRow(self.symmetric_window)
        for label, widget in [("Detrend per window", self.detrend), ("Power scaling", self.scaling),
                              ("Frequency sides", self.sides), ("Interval edges", self.boundary)]:
            form.addRow(label, widget)
        form.addRow(self.pad_end)
        self.resolution = W.QLabel()
        self.resolution.setWordWrap(True)
        form.addRow(self.resolution)
        self.auto_update = W.QCheckBox("Update spectrogram when trace / settings change")
        self.auto_update.setChecked(True)
        outer.addWidget(self.auto_update)
        self.export = W.QPushButton("Save spectrogram data…")
        self.export.setToolTip("Save time/frequency coordinates, auto/cross-power arrays and trace/settings metadata as NPZ.")
        self.export.clicked.connect(self.export_requested)
        outer.addWidget(self.export)
        display = W.QWidget()
        df = form_layout(display)
        self.option_tabs.addTab(display, "Display")
        self.quantity = combo(analysis.QUANTITIES[:1] if len(names) == 1 else analysis.QUANTITIES)
        self.quantity.setCurrentText("Cross-power" if len(names) == 2 else "Auto-power A")
        self.representation = combo(["Magnitude", "Phase", "Real", "Imaginary", "Magnitude squared"])
        self.degrees = W.QCheckBox("Phase in degrees")
        self.degrees.setChecked(True)
        self.color_scale = combo(["dB", "Linear", "Logarithmic", "Symmetric log"])
        self.reference = number(1., 1e-300)
        self.dynamic_range = number(80., .001, 1000.)
        self.linthresh = number(.001, 1e-300)
        self.cmap = combo(plotting.COLORMAPS)
        self.cmap.setCurrentText("magma")
        self.auto_color = W.QCheckBox("Automatic color limits")
        self.auto_color.setChecked(True)
        self.color_min, self.color_max = number(-80.), number(0.)
        self.frequency_min, self.frequency_max = number(0.), number(fs/2)
        self.log_frequency = W.QCheckBox("Logarithmic frequency axis")
        for label, widget in [("Quantity", self.quantity), ("Cross representation", self.representation)]:
            df.addRow(label, widget)
        df.addRow(self.degrees)
        for label, widget in [("Color scale", self.color_scale), ("dB reference", self.reference),
                              ("dB dynamic range", self.dynamic_range), ("Symmetric-log threshold", self.linthresh),
                              ("Color map", self.cmap)]:
            df.addRow(label, widget)
        df.addRow(self.auto_color)
        for label, widget in [("Color minimum", self.color_min), ("Color maximum", self.color_max),
                              ("Frequency minimum (Hz)", self.frequency_min), ("Frequency maximum (Hz)", self.frequency_max)]:
            df.addRow(label, widget)
        df.addRow(self.log_frequency)
        self.display_form = df
        self.quantity.currentTextChanged.connect(self.update_display_controls)
        self.representation.currentTextChanged.connect(self.update_display_controls)
        self.color_scale.currentTextChanged.connect(self.update_display_controls)
        self.auto_color.toggled.connect(self.update_display_controls)
        for widget in (self.reference, self.dynamic_range, self.linthresh, self.color_min, self.color_max,
                       self.frequency_min, self.frequency_max):
            widget.valueChanged.connect(self.display_changed)
        self.degrees.toggled.connect(self.display_changed)
        self.log_frequency.toggled.connect(self.display_changed)
        self.cmap.currentTextChanged.connect(self.display_changed)
        self.nperseg.valueChanged.connect(self.samples_changed)
        self.overlap.valueChanged.connect(self.samples_changed)
        self.duration.editingFinished.connect(self.duration_changed)
        self.overlap_percent.editingFinished.connect(self.percent_changed)
        self.window.currentTextChanged.connect(self.window_changed)
        for widget in (self.nfft, self.fft_workers, self.window_parameter):
            widget.valueChanged.connect(self.changed)
        for widget in (self.detrend, self.scaling, self.sides, self.boundary):
            widget.currentTextChanged.connect(self.changed)
        self.symmetric_window.toggled.connect(self.changed)
        self.pad_end.toggled.connect(self.changed)
        self.update_window_parameter()
        self.update_display_controls()
        self.update_resolution()

    def settings(self, interval):
        return analysis.Settings(interval=interval, nperseg=self.nperseg.value(), overlap=self.overlap.value(),
                                 nfft=self.nfft.value(), fft_workers=self.fft_workers.value(), window=self.window.currentText(),
                                 window_parameter=self.window_parameter.value(), symmetric_window=self.symmetric_window.isChecked(),
                                 detrend=self.detrend.currentText(), scaling=self.scaling.currentText(),
                                 sides="onesided" if self.sides.currentIndex() == 0 else "twosided",
                                 boundary=analysis.BOUNDARIES[self.boundary.currentText()], pad_end=self.pad_end.isChecked())

    def samples_changed(self):
        if self._syncing:
            return
        self._syncing = True
        self.duration.setValue(1000*self.nperseg.value()/self.fs)
        self.overlap_percent.setValue(100*self.overlap.value()/self.nperseg.value())
        self._syncing = False
        self.changed()

    def duration_changed(self):
        self.nperseg.setValue(max(4, round(self.duration.value()*self.fs/1000)))
        self.samples_changed()

    def percent_changed(self):
        self.overlap.setValue(round(self.overlap_percent.value()*self.nperseg.value()/100))
        self.samples_changed()

    def update_window_parameter(self):
        param = analysis.PARAMETERS.get(self.window.currentText())
        self.form.setRowVisible(self.window_parameter, param is not None)
        if param:
            self.form.labelForField(self.window_parameter).setText(param[0])

    def window_changed(self):
        param = analysis.PARAMETERS.get(self.window.currentText())
        if param:
            self.window_parameter.blockSignals(True)
            self.window_parameter.setValue(param[1])
            self.window_parameter.blockSignals(False)
        self.update_window_parameter()
        self.changed()

    def changed(self, *args):
        self.update_resolution()
        self.estimate_changed.emit()

    def update_resolution(self):
        hop = self.nperseg.value()-self.overlap.value()
        self.resolution.setText(f"Window: {self.nperseg.value()/self.fs:g} s · Hop: {hop/self.fs:g} s\n"
                                f"FFT bin spacing: {self.fs/self.nfft.value():g} Hz. Zero padding refines bins, not spectral resolution.")

    def update_display_controls(self, *args):
        cross = self.quantity.currentText() == "Cross-power"
        phase = cross and self.representation.currentText() == "Phase"
        self.representation.setEnabled(cross)
        self.degrees.setEnabled(phase)
        signed = cross and self.representation.currentText() in ("Phase", "Real", "Imaginary")
        for label in ("dB", "Logarithmic"):
            self.color_scale.model().item(self.color_scale.findText(label)).setEnabled(not signed)
        if signed and self.color_scale.currentText() in ("dB", "Logarithmic"):
            self.color_scale.blockSignals(True)
            self.color_scale.setCurrentText("Linear")
            self.color_scale.blockSignals(False)
        self.display_form.setRowVisible(self.reference, self.color_scale.currentText() == "dB")
        self.display_form.setRowVisible(self.dynamic_range, self.color_scale.currentText() == "dB")
        self.display_form.setRowVisible(self.linthresh, self.color_scale.currentText() == "Symmetric log")
        self.color_min.setEnabled(not self.auto_color.isChecked())
        self.color_max.setEnabled(not self.auto_color.isChecked())
        self.display_changed.emit()

    def render(self, result, fig, appearance):
        q, kind = self.quantity.currentText(), self.representation.currentText()
        scale = self.color_scale.currentText()
        values, units = analysis.color_values(result, q, kind, self.degrees.isChecked(), scale,
                                              self.reference.value(), self.dynamic_range.value())
        low, high = self.frequency_min.value(), self.frequency_max.value()
        if not low < high:
            raise ValueError("Frequency display minimum must be below its maximum.")
        if self.log_frequency.isChecked():
            if result.settings.sides == "twosided":
                raise ValueError("Use a one-sided spectrum for a logarithmic frequency axis.")
            low = max(low, result.frequency[1])
        if not np.any((result.frequency >= low) & (result.frequency <= high)):
            raise ValueError("Frequency display limits contain no computed bins.")
        shown = values[(result.frequency >= low) & (result.frequency <= high)]
        finite = shown[np.isfinite(shown)]
        if self.auto_color.isChecked():
            vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (-1., 1.)
            if scale == "dB":
                vmin = vmax-self.dynamic_range.value()
            elif q == "Cross-power" and kind == "Phase":
                vmin, vmax = (-180., 180.) if self.degrees.isChecked() else (-np.pi, np.pi)
            elif q == "Cross-power" and kind in ("Real", "Imaginary"):
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
            if vmin == vmax:
                if scale == "Logarithmic":
                    vmin, vmax = max(vmax, np.finfo(float).tiny)/10, max(vmax, np.finfo(float).tiny)*10
                else:
                    vmin, vmax = plotting.finite_limits(np.array([vmin, vmax]))
        else:
            vmin, vmax = self.color_min.value(), self.color_max.value()
            if not vmin < vmax:
                raise ValueError("Color minimum must be below its maximum.")
        empty_log = scale == "Logarithmic" and not np.any(finite > 0)
        if scale == "Logarithmic":
            positive = finite[finite > 0]
            if empty_log and self.auto_color.isChecked():
                vmin, vmax = 1., 10.
            if self.auto_color.isChecked() and positive.size:
                vmin = positive.min()
                if vmin == vmax:
                    vmin = vmax/10
            if vmin <= 0:
                raise ValueError("Logarithmic color limits must be positive.")
            norm = LogNorm(vmin, vmax)
            values = np.ma.masked_less_equal(values, 0)
        elif scale == "Symmetric log":
            norm = SymLogNorm(self.linthresh.value(), vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin, vmax)
        fig.clear()
        fig.set_facecolor(colors(appearance)["bg"])
        ax = fig.add_subplot(111)
        plotting.style(ax, colors(appearance))
        # Explicit cell edges retain the requested window/hop width even for a
        # single time window; nearest shading invents a width for singleton axes.
        hop = (result.settings.nperseg-result.settings.overlap)*1000*self.fs**-1
        time = result.time*1000
        time_edges = np.r_[time-hop/2, time[-1]+hop/2]
        df = self.fs/result.settings.nfft
        frequency_edges = np.r_[result.frequency-df/2, result.frequency[-1]+df/2]
        image = ax.pcolormesh(time_edges, frequency_edges, values, cmap=self.cmap.currentText(), norm=norm, shading="flat", rasterized=True)
        if empty_log:
            ax.text(.5, .5, "No positive power for a logarithmic color scale.", ha="center", va="center",
                    transform=ax.transAxes, color=colors(appearance)["fg"])
        else:
            bar = fig.colorbar(image, ax=ax, label=units)
            plotting.style(bar.ax, colors(appearance))
        location = ", ".join(f"{d}={v:g}" + (f" {result.spatial_units}" if d in "xyz" else "") for d, v in result.location.items())
        if result.global_shot is not None:
            location += f" · Global shot {result.global_shot}"
        title = q + (f" ({kind.lower()})" if q == "Cross-power" else "")
        ax.set(xlabel="Window center time (ms)", ylabel="Frequency (Hz)",
               title=f"{title} · {', '.join(result.names)}\n{location or 'Single time series'}", ylim=(low, high))
        if self.log_frequency.isChecked():
            ax.set_yscale("log")
        self.last_axis, self.last_image = ax, image
        return ax
