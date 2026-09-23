"""Exercise the full Qt import and plotting workflow on the supplied x-line file.

Run from the project root. See docs/development.md for Windows instructions:
  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_sample.py /path/to/file.hdf5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import tempfile
from PySide6 import QtWidgets as W
from lapd_explorer.app import MainWindow, STYLE, export_mp4
from lapd_explorer.widgets import ImportDialog
from lapd_explorer.io import inspect_file
from lapd_explorer.model import preprocess

app = W.QApplication([])
app.setStyle('Fusion')
app.setStyleSheet(STYLE)
path = sys.argv[1]
info = inspect_file(path)
dialog = ImportDialog(path, info)
for i in range(dialog.channels.count()):
    dialog.channels.item(i).setSelected(True)
for i in range(dialog.motion.count()):
    if '<Hermes>' in dialog.motion.itemText(i):
        dialog.motion.setCurrentIndex(i)
dialog.repeats.setValue(5)
dialog.begin()
deadline = time.monotonic()+30
while dialog.dataset is None and time.monotonic() < deadline:
    app.processEvents()
    if dialog.load.isEnabled():
        raise RuntimeError(dialog.message.text())
    time.sleep(.01)
assert dialog.dataset is not None
assert dialog.dataset.shape == (91,5,6144)
dialog.dataset.source = Path(path).name
w = MainWindow()
w.set_data(dialog.dataset)
w.appearance.setCurrentText("Light")
w.average.setChecked(True)
w.baseline.setCurrentText('Remove mean')
w.processed(preprocess(w.raw, average=True, baseline='Remove mean'))
w.frame.setValue(1800)
w.show()
w.draw()
app.processEvents()
w.canvas.draw()
w.grab().save('docs/sample-xline.png')
t0 = time.monotonic()
for frame in range(1801,1811):
    w.frame.setValue(frame)
    w.draw()
    w.canvas.draw()
print(f'Mean frame render: {(time.monotonic()-t0)/10:.3f} s')
export_mp4(str(Path(tempfile.gettempdir())/'lapd-sample-check.mp4'), w.data, w.options(), [1000,1800,2600], 5)
print('Verified GUI import, average, baseline removal, line/time/PSD views and MP4 export.')
w.close()
