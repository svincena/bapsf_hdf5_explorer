"""Validate the supplied 51×51 plane and refresh its Light-mode screenshot.

Run from the project root. See docs/development.md for Windows instructions:
  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_plane.py /path/to/file.hdf5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import tempfile
from types import SimpleNamespace
import numpy as np
from PySide6 import QtWidgets as W
from lapd_explorer.app import MainWindow, STYLE, export_mp4
from lapd_explorer.widgets import ImportDialog
from lapd_explorer.io import inspect_file, save_dataset, load_dataset
from lapd_explorer.model import preprocess, quantity

app = W.QApplication([])
app.setStyle('Fusion')
app.setStyleSheet(STYLE)
path = sys.argv[1]
t0 = time.monotonic()
info = inspect_file(path)
dialog = ImportDialog(path, info)
for i in range(dialog.channels.count()):
    dialog.channels.item(i).setSelected(True)
dialog.motion.setCurrentIndex(1)
dialog.repeats.setValue(5)
dialog.begin()
deadline = time.monotonic()+120
while dialog.dataset is None and time.monotonic() < deadline:
    app.processEvents()
    if dialog.load.isEnabled():
        raise RuntimeError(dialog.message.text())
    time.sleep(.01)
assert dialog.dataset is not None
raw = dialog.dataset
raw.source = Path(path).name
assert raw.dims == ('y','x','shot','time')
assert raw.shape == (51,51,5,6144)
assert np.unique(raw.shot_numbers).size == 13005
np.testing.assert_allclose(np.diff(raw.coords['time']), 1e-8)
np.testing.assert_allclose(np.diff(raw.coords['x']), .7, atol=1e-4)
np.testing.assert_allclose(np.diff(raw.coords['y']), .7, atol=1e-4)
print('Imported full plane:',raw.shape,raw.dims,'in',round(time.monotonic()-t0,2),'s',flush=True)
print('X/Y extents:',raw.coords['x'][[0,-1]],raw.coords['y'][[0,-1]],flush=True)
# Compare mapped coordinates and signal values directly with actual bapsflib records.
from bapsflib import lapd
with lapd.File(path) as f:
    check=f.read_data(3,6,digitizer='SIS crate',adc='SIS 3302',config_name='Bdot_6kS_dt10ns',
                      index=[0,5000,13004],add_controls=[info['controls'][0]])
    for row in check:
        loc=tuple(np.argwhere(raw.shot_numbers == row['shotnum'])[0])
        y,x,shot=loc
        np.testing.assert_allclose(raw.channels[next(iter(raw.channels))][loc],row['signal'])
        np.testing.assert_allclose([raw.coords['x'][x],raw.coords['y'][y]],row['xyz_target'][:2],atol=5e-5)
print('Verified global shot / coordinate / signal alignment against direct reads.',flush=True)
w=MainWindow()
w.set_data(raw)
w.appearance.setCurrentText("Light")
w.average.setChecked(True)
w.baseline.setCurrentText('Remove mean')
w.processed(preprocess(raw,average=True,baseline='Remove mean'))
w.frame.setValue(2400)
w.show()
w.mode.setCurrentText('Vector')
w.cmap.setCurrentText('plasma')
w.arrow_cmap.setCurrentText('viridis')
w.draw()
assert w.main_ax.collections[0].cmap.name == 'plasma'
assert w.main_ax.collections[1].cmap.name == 'viridis'
app.processEvents()
assert w._values.shape==(51,51,6144)
np.testing.assert_allclose(w._values[20,30,2400],np.sqrt(sum(a[20,30,2400]**2 for a in w.data.channels.values())))
w.clicked(SimpleNamespace(inaxes=w.main_ax,xdata=3.5,ydata=-3.5))
assert w.slice_boxes[0].value()==20
assert w.slice_boxes[1].value()==30
w.draw()
w.canvas.draw()
app.processEvents()
w.grab().save('docs/sample-plane.png')
w.fig.savefig(str(Path(tempfile.gettempdir())/'lapd-plane-check.svg'))
# Verify animation updates artists, including vector components and both slices.
opts=w.options()
export_mp4(str(Path(tempfile.gettempdir())/'lapd-plane-check.mp4'),w.data,opts,[1800,2200,2600,3000],5,w._values)
w.frame.setValue(3000)
w.draw()
np.testing.assert_allclose(w.main_ax.collections[0].get_array(),w._values[...,3000])
print('Verified three-component magnitude, vector arrows, both linked slices, still and MP4 export.',flush=True)
w.close()
