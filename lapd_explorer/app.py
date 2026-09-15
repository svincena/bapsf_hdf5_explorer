"""Modern Qt desktop front end for LAPD Explorer."""
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "lapd-matplotlib"))
import sys
from pathlib import Path
import threading
import tempfile
import numpy as np
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from .model import demo, preprocess, quantity
from . import io, plotting
from .widgets import Worker, ImportDialog, combo, spin

STYLE = """
QWidget { background: #0c1422; color: #e7eef8; font-family: 'Helvetica Neue', 'Segoe UI'; font-size: 12px; }
QMainWindow { background: #0c1422; }
QLabel#brand { font-size: 24px; font-weight: 700; letter-spacing: 1px; }
QLabel#subtitle { color: #8ea4be; font-size: 12px; }
QLabel#sectionTitle { font-size: 19px; font-weight: 600; }
QGroupBox { border: 1px solid #26384e; border-radius: 10px; margin-top: 18px; padding: 15px 10px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; top: 2px; color: #8ea4be; }
QPushButton { background: #1b2b40; border: 1px solid #30465f; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover { background: #2b425c; border-color: #46d9c5; }
QPushButton:disabled { color: #617389; background: #142033; }
QPushButton#primary { background: #46d9c5; color: #0b2026; font-weight: 700; border: none; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #152438; border: 1px solid #30465f; border-radius: 5px; padding: 5px; min-height: 19px; }
QComboBox:disabled, QSpinBox:disabled { color: #617389; }
QComboBox QAbstractItemView { selection-background-color: #28505b; }
QListWidget, QTableWidget { background: #111e30; border: 1px solid #30465f; border-radius: 5px; }
QListWidget::item { padding: 7px; }
QListWidget::item:selected { background: #28505b; }
QHeaderView::section { background: #1b2b40; padding: 6px; border: none; }
QTabBar::tab { padding: 9px 18px; background: #142033; }
QTabBar::tab:selected { color: #46d9c5; border-bottom: 2px solid #46d9c5; }
QSlider::groove:horizontal { height: 5px; background: #26384e; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #46d9c5; border-radius: 7px; }
QCheckBox { spacing: 8px; }
QStatusBar { color: #8ea4be; border-top: 1px solid #26384e; }
QScrollArea { border: none; }
QToolBar { border: none; }
QProgressBar { border: 1px solid #30465f; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #46d9c5; }
"""


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAPD Explorer")
        self.resize(1480, 960)
        self.raw = self.data = None
        self.jobs = []
        self._values = None
        self._busy = False
        self.timer = C.QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.redraw_timer = C.QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.timeout.connect(self.draw)
        root = W.QWidget()
        self.setCentralWidget(root)
        layout = W.QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 10)
        top = W.QHBoxLayout()
        brandcol = W.QVBoxLayout()
        brand = W.QLabel("LAPD  /  EXPLORER")
        brand.setObjectName("brand")
        subtitle = W.QLabel("BaPSF  ·  Spatially resolved time-series analysis")
        subtitle.setObjectName("subtitle")
        brandcol.addWidget(brand)
        brandcol.addWidget(subtitle)
        top.addLayout(brandcol)
        top.addStretch()
        for text, callback in [("Open HDF5…", self.open_file), ("Demo", lambda: self.set_data(demo())),
                               ("Run info", self.show_info),
                               ("Save image…", self.save_image), ("Export MP4…", self.export_movie),
                               ("Save data…", self.save_data)]:
            button = W.QPushButton(text)
            button.clicked.connect(callback)
            top.addWidget(button)
        layout.addLayout(top)
        self.source = W.QLabel()
        self.source.setObjectName("subtitle")
        self.source.setTextInteractionFlags(C.Qt.TextSelectableByMouse)
        layout.addWidget(self.source)
        split = W.QSplitter()
        layout.addWidget(split, 1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(360)
        sidebar = W.QWidget()
        controls = W.QVBoxLayout(sidebar)
        controls.setContentsMargins(0, 0, 10, 0)
        scroll.setWidget(sidebar)
        split.addWidget(scroll)
        signal = W.QGroupBox("01   QUANTITY")
        sf = W.QFormLayout(signal)
        self.mode = combo(["Scalar", "Absolute", "Magnitude", "Vector"])
        self.components = [W.QComboBox() for _ in range(3)]
        sf.addRow("Display", self.mode)
        for label, widget in zip(["Scalar / horizontal", "Vertical component", "Third component"], self.components):
            sf.addRow(label, widget)
        note = W.QLabel("Vector arrows use horizontal and vertical components of the displayed plane. A third component contributes to magnitude.")
        note.setWordWrap(True)
        note.setObjectName("subtitle")
        sf.addRow(note)
        self.cmap = combo(plotting.COLORMAPS)
        sf.addRow("Mesh color map", self.cmap)
        self.arrow_cmap = combo(["Solid white"] + plotting.COLORMAPS)
        self.arrow_cmap.setToolTip("Color vector arrows by the magnitude of the horizontal and vertical components.")
        sf.addRow("Arrow color map", self.arrow_cmap)
        self.lock = W.QCheckBox("Fixed scale across time")
        self.lock.setChecked(True)
        sf.addRow(self.lock)
        controls.addWidget(signal)
        pp = W.QGroupBox("02   PREPROCESSING")
        pf = W.QFormLayout(pp)
        self.average = W.QCheckBox("Average stored shots")
        self.baseline = combo(["None", "Remove mean", "Linear detrend"])
        self.integrate = W.QCheckBox("Cumulative integration")
        self.gain = W.QLineEdit("1")
        pf.addRow(self.average)
        pf.addRow("Baseline", self.baseline)
        pf.addRow(self.integrate)
        pf.addRow("Gain", self.gain)
        apply = W.QPushButton("Apply to original data")
        apply.setObjectName("primary")
        apply.clicked.connect(self.apply_processing)
        reset = W.QPushButton("Reset processing")
        reset.clicked.connect(self.reset_processing)
        pf.addRow(apply)
        pf.addRow(reset)
        self.history = W.QLabel("Original data")
        self.history.setWordWrap(True)
        self.history.setObjectName("subtitle")
        pf.addRow(self.history)
        controls.addWidget(pp)
        selection = W.QGroupBox("03   DIMENSIONS & SLICES")
        df = W.QFormLayout(selection)
        self.case, self.shot = spin(), spin()
        df.addRow("Case index", self.case)
        df.addRow("Repeat index", self.shot)
        self.slice_boxes = [spin(), spin()]
        self.slice_labels = [W.QLabel("Position"), W.QLabel("Position")]
        for label, widget in zip(self.slice_labels, self.slice_boxes):
            df.addRow(label, widget)
        self.cursor = W.QLabel("Click the main spatial plot to move both slices.")
        self.cursor.setWordWrap(True)
        self.cursor.setObjectName("subtitle")
        df.addRow(self.cursor)
        controls.addWidget(selection)
        timegroup = W.QGroupBox("04   TIME & EXPORT")
        tf = W.QFormLayout(timegroup)
        self.time_unit = combo(list(plotting.TIME_UNITS))
        self.time_unit.setCurrentText("µs")
        self.sigfigs = spin(1, 12, 4)
        self.fps = spin(1, 60, 15)
        self.frame_step = spin(1, 100000, 1)
        self.export_start, self.export_stop = spin(), spin()
        tf.addRow("Time units", self.time_unit)
        tf.addRow("Significant figures", self.sigfigs)
        tf.addRow("Playback / export FPS", self.fps)
        tf.addRow("Frame stride", self.frame_step)
        tf.addRow("Export first frame", self.export_start)
        tf.addRow("Export last frame", self.export_stop)
        controls.addWidget(timegroup)
        controls.addStretch()
        workspace = W.QWidget()
        wl = W.QVBoxLayout(workspace)
        wl.setContentsMargins(8, 0, 0, 0)
        self.fig = plotting.figure()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        wl.addWidget(self.toolbar)
        wl.addWidget(self.canvas, 1)
        timeline = W.QHBoxLayout()
        self.play = W.QPushButton("▶ Play")
        self.play.clicked.connect(self.toggle_play)
        self.slider = W.QSlider(C.Qt.Horizontal)
        self.frame = spin()
        self.time_label = W.QLabel()
        self.time_label.setMinimumWidth(130)
        timeline.addWidget(self.play)
        timeline.addWidget(self.slider, 1)
        timeline.addWidget(self.frame)
        timeline.addWidget(self.time_label)
        wl.addLayout(timeline)
        split.addWidget(workspace)
        split.setStretchFactor(1, 1)
        self.canvas.mpl_connect("button_press_event", self.clicked)
        self.slider.valueChanged.connect(self.frame.setValue)
        self.frame.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.schedule_draw)
        for widget in [self.mode, *self.components]:
            widget.currentIndexChanged.connect(self.invalidate)
        for widget in [self.cmap, self.arrow_cmap, self.time_unit]:
            widget.currentIndexChanged.connect(self.schedule_draw)
        self.lock.toggled.connect(self.schedule_draw)
        for widget in [self.case, self.shot]:
            widget.valueChanged.connect(self.invalidate)
        for widget in [*self.slice_boxes, self.sigfigs]:
            widget.valueChanged.connect(self.schedule_draw)
        self.fps.valueChanged.connect(lambda n: self.timer.setInterval(round(1000/n)))
        self.set_data(demo())

    def launch(self, fn, done, message):
        self.stop_play()
        self.redraw_timer.stop()
        self._busy = True
        self.centralWidget().setEnabled(False)
        self.statusBar().showMessage(message)
        worker = Worker(fn, self)
        self.jobs.append(worker)
        def finish(result=None, error=None):
            self._busy = False
            self.centralWidget().setEnabled(True)
            if error:
                self.error(error)
            else:
                done(result)
        worker.result.connect(lambda result: finish(result=result))
        worker.failed.connect(lambda error: finish(error=error))
        worker.finished.connect(lambda: self.jobs.remove(worker))
        worker.start()

    def error(self, message):
        self.stop_play()
        self.statusBar().showMessage(message)
        W.QMessageBox.warning(self, "LAPD Explorer", message)

    def show_info(self):
        import json
        dialog = W.QDialog(self)
        dialog.setWindowTitle("Acquisition metadata & processing history")
        dialog.resize(850, 650)
        layout = W.QVBoxLayout(dialog)
        text = W.QPlainTextEdit()
        text.setReadOnly(True)
        metadata = {name: ({k: (f"{len(v)} recorded positions (preserved in data export)"
                              if k in ("measured xyz", "target xyz") else v)
                           for k, v in info.items()} if isinstance(info, dict) else info)
                    for name, info in self.data.metadata.items()}
        text.setPlainText(json.dumps(dict(source=self.data.source, dimensions=self.data.dims,
                                         shape=self.data.shape, units=self.data.units,
                                         history=self.data.history, metadata=metadata), indent=2, default=str))
        layout.addWidget(text)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def open_file(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "Open acquisition", "", "HDF5 (*.hdf5 *.h5 *.hdf);;All files (*)")
        if not path:
            return
        def inspected(info):
            if info["portable"]:
                self.launch(lambda: io.load_dataset(path), self.set_data, "Loading saved dataset…")
            else:
                dialog = ImportDialog(path, info, self)
                if dialog.exec() == W.QDialog.Accepted:
                    self.set_data(dialog.dataset)
        self.launch(lambda: io.inspect_file(path), inspected, "Discovering digitizers, channels and motion controls…")

    def set_data(self, data):
        self.stop_play()
        self.raw = self.data = data
        self.average.setChecked(False)
        self.baseline.setCurrentIndex(0)
        self.integrate.setChecked(False)
        self.gain.setText("1")
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        names = list(data.channels)
        for i, widget in enumerate(self.components):
            widget.blockSignals(True)
            widget.clear()
            widget.addItems(names if i == 0 else ["None"]+names)
            widget.setCurrentIndex(0 if i == 0 else i+1 if i < len(names) else 0)
            widget.blockSignals(False)
        self.slider.setRange(0, len(data.coords["time"])-1)
        self.frame.setRange(0, self.slider.maximum())
        self.slider.setValue(0)
        for widget in [self.export_start, self.export_stop]:
            widget.setRange(0, self.slider.maximum())
        self.export_start.setValue(0)
        self.export_stop.setValue(self.slider.maximum())
        self.refresh_dimensions()
        self.invalidate()

    def refresh_dimensions(self):
        data = self.data
        self.average.setEnabled("shot" in self.raw.dims and len(self.raw.coords["shot"]) > 1)
        for dim, widget in [("case", self.case), ("shot", self.shot)]:
            widget.setRange(0, len(data.coords.get(dim, [0]))-1)
            widget.setEnabled(widget.maximum() > 0)
        for i, (label, widget) in enumerate(zip(self.slice_labels, self.slice_boxes)):
            active = i < len(data.spatial_dims)
            widget.setEnabled(active)
            if active:
                d = data.spatial_dims[i]
                label.setText(f"{d} slice index")
                widget.setRange(0, len(data.coords[d])-1)
                widget.setValue(widget.maximum()//2)
            else:
                label.setText("—")
                widget.setRange(0, 0)
        self.source.setText(f"{data.source}   |   " + " × ".join(f"{d}: {n}" for d, n in zip(data.dims, data.shape)))
        self.history.setText(" → ".join(data.history) or "Original data")

    def apply_processing(self):
        try:
            settings = dict(average=self.average.isChecked(), baseline=self.baseline.currentText(),
                            integrate=self.integrate.isChecked(), gain=float(self.gain.text()))
            self.launch(lambda: preprocess(self.raw, **settings), self.processed, "Processing traces…")
        except ValueError as exc:
            self.error(str(exc))

    def processed(self, data):
        self.data = data
        self.refresh_dimensions()
        self.invalidate()

    def reset_processing(self):
        self.average.setChecked(False)
        self.baseline.setCurrentIndex(0)
        self.integrate.setChecked(False)
        self.gain.setText("1")
        self.processed(self.raw)

    def options(self):
        mode = self.mode.currentText()
        names = [self.components[0].currentText()]
        if mode in ("Magnitude", "Vector"):
            names += [w.currentText() for w in self.components[1:] if w.currentText() != "None"]
            if len(set(names)) != len(names):
                raise ValueError("Assign distinct channels to each vector component.")
            if len(names) < 2:
                raise ValueError("Assign at least two vector components.")
        if mode == "Vector" and len(self.data.spatial_dims) != 2:
            raise ValueError("Vector arrows require a plane. Use Magnitude for point or line acquisitions.")
        return dict(names=names, mode=mode, case=self.case.value(), shot=self.shot.value(),
                    slices=[w.value() for w in self.slice_boxes], time=self.slider.value(),
                    time_unit=self.time_unit.currentText(), sigfigs=self.sigfigs.value(),
                    lock=self.lock.isChecked(), cmap=self.cmap.currentText(), arrow_cmap=self.arrow_cmap.currentText())

    def invalidate(self, *args):
        self.arrow_cmap.setEnabled(self.mode.currentText() == "Vector")
        self._values = None
        self._render_key = None
        self.schedule_draw()

    def schedule_draw(self, *args):
        if self.data is not None and not self._busy:
            self.redraw_timer.start(20)

    def draw(self):
        if self._busy:
            return
        try:
            opts = self.options()
            if self._values is None:
                self._values = quantity(self.data, opts["names"], opts["mode"], opts["case"], opts["shot"])
            render_key = (id(self._values), repr({k: v for k, v in opts.items() if k != "time"}))
            if getattr(self, "_render_key", None) == render_key:
                self.fig._lapd_update_frame(opts["time"])
            else:
                self.main_ax = plotting.render(self.fig, self.data, opts, self._values)
                self._render_key = render_key
            self.canvas.draw_idle()
            t = self.data.coords["time"][opts["time"]] * plotting.TIME_UNITS[opts["time_unit"]]
            self.time_label.setText(f"{t:.{opts['sigfigs']}g} {opts['time_unit']}")
            position = ", ".join(f"{d}={self.data.coords[d][opts['slices'][i]]:g} {self.data.spatial_units}"
                                 for i, d in enumerate(self.data.spatial_dims))
            self.cursor.setText((position + "\n" if position else "Point measurement\n") + "Click the spatial plot to select a position.")
            status = f"{self.data.units}  •  {len(self.data.coords['time']):,} time samples  •  "
            if self.data.shot_numbers is not None:
                key = tuple(opts["case"] if d == "case" else opts["shot"] if d == "shot"
                            else opts["slices"][self.data.spatial_dims.index(d)] for d in self.data.dims[:-1])
                status += f"Global shot {self.data.shot_numbers[key]}"
            else:
                status += "No individual global shot associated with this trace"
            self.statusBar().showMessage(status)
        except ValueError as exc:
            self.stop_play()
            self._render_key = None
            self.fig.clear()
            self.fig.text(.5, .5, str(exc), ha="center", color=plotting.FG, fontsize=12, wrap=True)
            self.canvas.draw_idle()
            self.statusBar().showMessage(str(exc))

    def clicked(self, event):
        if event.inaxes != getattr(self, "main_ax", None) or self.toolbar.mode or event.xdata is None:
            return
        spatial = self.data.spatial_dims
        positions = [event.ydata, event.xdata] if len(spatial) == 2 else [event.xdata]
        for i, d in enumerate(spatial):
            self.slice_boxes[i].setValue(int(np.argmin(abs(self.data.coords[d] - positions[i]))))

    def stop_play(self):
        self.timer.stop()
        if hasattr(self, "play"):
            self.play.setText("▶ Play")

    def toggle_play(self):
        if self.timer.isActive():
            self.stop_play()
        else:
            self.timer.start(round(1000/self.fps.value()))
            self.play.setText("Ⅱ Pause")

    def advance(self):
        self.slider.setValue((self.slider.value()+self.frame_step.value()) % (self.slider.maximum()+1))

    def save_image(self):
        self.stop_play()
        try:
            self.options()
        except ValueError as exc:
            self.error(str(exc))
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Save visualization", "lapd-frame.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            try:
                self.draw()
                self.fig.savefig(path, dpi=200, facecolor=plotting.BG)
                self.statusBar().showMessage(f"Saved {path}")
            except Exception as exc:
                self.error(str(exc))

    def save_data(self):
        path, _ = W.QFileDialog.getSaveFileName(self, "Save processed dataset", "lapd-processed.h5", "HDF5 (*.h5)")
        if path:
            if Path(path).resolve() == Path(self.raw.source).resolve():
                self.error("Choose a new file to preserve the source acquisition.")
                return
            data = self.data
            def save():
                atomic_save(path, lambda tmp: io.save_dataset(tmp, data))
                return path
            self.launch(save, lambda p: self.statusBar().showMessage(f"Saved {p}"), "Saving processed data and provenance…")

    def export_movie(self):
        self.stop_play()
        try:
            opts = self.options()
            first, last = self.export_start.value(), self.export_stop.value()
            if first > last:
                raise ValueError("Export first frame must not exceed last frame.")
            frames = list(range(first, last+1, self.frame_step.value()))
            values = quantity(self.data, opts["names"], opts["mode"], opts["case"], opts["shot"])
        except ValueError as exc:
            self.error(str(exc))
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Export animation", "lapd-animation.mp4", "MP4 (*.mp4)")
        if not path:
            return
        progress = W.QProgressDialog("Rendering MP4 frames…", "Cancel", 0, 0, self)
        progress.setWindowModality(C.Qt.WindowModal)
        progress.setMinimumDuration(0)
        cancel = threading.Event()
        progress.canceled.connect(cancel.set)
        data, fps = self.data, self.fps.value()
        def export():
            def write(tmp):
                export_mp4(tmp, data, opts, frames, fps, values, cancel)
            atomic_save(path, write, suffix=".mp4")
            return path
        def done(p):
            progress.close()
            self.statusBar().showMessage(f"Exported {len(frames)} frames to {p}")
        self.launch(export, done, f"Exporting {len(frames)} frames at {fps} FPS…")
        # Also close progress when the worker fails or cancellation is observed.
        self.jobs[-1].failed.connect(lambda _: progress.close())
        progress.show()

    def closeEvent(self, event):
        if any(job.isRunning() for job in self.jobs):
            self.statusBar().showMessage("Wait for the active operation to finish before closing.")
            event.ignore()
        else:
            self.stop_play()
            super().closeEvent(event)


def atomic_save(path, writer, suffix=".h5"):
    """Replace destination only after a complete successful write."""
    fd, temp = tempfile.mkstemp(prefix=".lapd-", suffix=suffix, dir=Path(path).resolve().parent)
    os.close(fd)
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def export_mp4(path, data, opts, frames, fps, values=None, cancel=None):
    import matplotlib as mpl
    from matplotlib.animation import FFMpegWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import imageio_ffmpeg
    fig = plotting.figure()
    FigureCanvasAgg(fig)
    try:
        with mpl.rc_context({"animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()}):
            writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
            with writer.saving(fig, str(path), dpi=100):
                for frame_number, frame in enumerate(frames):
                    if cancel and cancel.is_set():
                        raise ValueError("Export canceled; destination was not changed.")
                    if frame_number == 0:
                        plotting.render(fig, data, dict(opts, time=frame), values)
                    else:
                        fig._lapd_update_frame(frame)
                    writer.grab_frame(facecolor=plotting.BG)
    finally:
        fig.clear()


def main():
    app = W.QApplication.instance() or W.QApplication(sys.argv)
    app.setApplicationName("LAPD Explorer")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        C.QTimer.singleShot(1200, app.quit)
    sys.exit(app.exec())
