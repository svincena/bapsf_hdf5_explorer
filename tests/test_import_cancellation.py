"""Cancel real read/filter/map work and reject results that arrive after Cancel."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lapd-matplotlib")
import threading
import time
import numpy as np
import h5py
import pytest
from bapsflib import lapd
from bapsflib._hdf.maps.tests import FauxHDFBuilder
from lapd_explorer import io
from lapd_explorer.cancellation import ImportCanceled
from lapd_explorer.temporal import resample_into, source_chunk_samples


@pytest.mark.parametrize("method,factor", [("simple", 1), ("simple", 3), ("polyphase", 4), ("average", 4)])
def test_raw_cancel_stops_after_first_chunk_and_closes_file(tmp_path, monkeypatch, method, factor):
    path = tmp_path/"raw.h5"
    with h5py.File(path, "w") as f:
        f["A"] = np.arange(4*200001).reshape(4, 200001)
        f["B"] = np.ones((4, 200001))
    canceled = threading.Event()
    opened, reads = [], []
    original_file, original_read = h5py.File, h5py.Dataset.__getitem__
    def open_file(*args, **kwargs):
        f = original_file(*args, **kwargs)
        opened.append(f)
        return f
    def read(dataset, key):
        reads.append((dataset.name, key))
        return original_read(dataset, key)
    monkeypatch.setattr(io.h5py, "File", open_file)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    def progress(done, total, stage):
        if done > 0:
            canceled.set()
    with pytest.raises(ImportCanceled):
        io.read_raw(path, {"A": "A", "B": "B"}, [("shot", 4, 0, 3)], 1e-6,
                    decimation=factor, downsampling=method, canceled=canceled.is_set, progress=progress)
    assert len(reads) == 1 and reads[0][0] == "/A"
    selection = reads[0][1][-1]
    assert selection.stop-selection.start <= 65536+20*factor
    assert all(not f.id.valid for f in opened)


def test_lapd_cancel_stops_calibrated_read_and_closes_file(tmp_path, monkeypatch):
    path = tmp_path/"lapd.hdf5"
    with FauxHDFBuilder(str(path)) as f:
        f.add_module("SIS 3301", {"sn_size": 3, "nt": 200001})
    spec = io.inspect_file(path)["channels"][0]
    canceled = threading.Event()
    reads = []
    original = lapd.File.read_data
    def read(f, *args, **kwargs):
        reads.append((f, kwargs["time_slice"]))
        return original(f, *args, **kwargs)
    monkeypatch.setattr(lapd.File, "read_data", read)
    with pytest.raises(ImportCanceled):
        io.read_lapd(path, {"A": spec}, canceled=canceled.is_set,
                     progress=lambda done, total, stage: canceled.set() if done else None)
    assert len(reads) == 1 and reads[0][1] == slice(0, 65536, 1)
    assert not reads[0][0].id.valid


def test_cancel_before_start_does_not_open_file(tmp_path):
    for reader, args in [(io.read_raw, ({"A": "A"}, [], 1e-6)), (io.read_lapd, ({},))]:
        with pytest.raises(ImportCanceled):
            reader(tmp_path/"missing.h5", *args, canceled=lambda: True)


def test_large_simple_stride_is_bounded_in_original_sample_span():
    selection = slice(7, 100000, 1000)
    output = np.empty((2, 100))
    reads = []
    def read(part):
        reads.append(part)
        return np.tile(np.arange(part.start, part.stop, part.step), (2, 1))
    resample_into(read, output, selection, "simple", chunk_samples=4096)
    np.testing.assert_array_equal(output[0], np.arange(7, 100000, 1000))
    assert all(part.stop-part.start <= 4096 for part in reads)
    assert source_chunk_samples(selection, "simple") == 65


@pytest.mark.parametrize("method", ["polyphase", "average"])
def test_cancel_after_source_read_prevents_processing_or_another_read(method):
    event = threading.Event()
    reads = []
    def read(part):
        reads.append(part)
        event.set()
        return np.ones((2, len(range(part.start, part.stop, part.step))))
    output = np.full((2, 1000), -99.)
    with pytest.raises(ImportCanceled):
        resample_into(read, output, slice(0, 4000, 4), method, canceled=event.is_set, chunk_samples=100)
    assert len(reads) == 1
    np.testing.assert_array_equal(output, -99.)


@pytest.mark.parametrize("order", ["case,shot", "shot,case"])
def test_bounded_motion_mapping_preserves_case_shot_order_and_cancels(order):
    rows, samples = 600, 8
    channels = {"A": np.arange(rows*samples).reshape(rows, samples), "B": np.ones((rows, samples))}
    records = io.Records(channels, np.arange(samples)*1e-6, np.arange(rows)+100,
                         np.zeros((rows, 3)), "source", {})
    result = io.map_motion(records, cases=3, repeats=200, repeat_order=order)
    expected = channels["A"].reshape(3, 200, samples) if order == "case,shot" else channels["A"].reshape(200, 3, samples).transpose(1, 0, 2)
    np.testing.assert_array_equal(result.channels["A"], expected)
    event, reports = threading.Event(), []
    def progress(done, total, stage):
        reports.append(done)
        if done > rows:
            event.set()
    with pytest.raises(ImportCanceled):
        io.map_motion(records, cases=3, repeats=200, repeat_order=order, progress=progress, canceled=event.is_set)
    assert rows < reports[-1] < rows*2


def test_mapping_a_single_long_record_can_cancel_between_time_chunks():
    samples = 200001
    records = io.Records({"A": np.arange(samples)[None, :]}, np.arange(samples)*1e-6, np.array([1]),
                         np.zeros((1, 3)), "source", {})
    event = threading.Event()
    reports = []
    def progress(done, total, stage):
        reports.append(done)
        if done > 1:
            event.set()
    with pytest.raises(ImportCanceled):
        io.map_motion(records, progress=progress, canceled=event.is_set)
    assert 1 < reports[-1] < 2


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait_until(app, condition, timeout=5):
    deadline = time.monotonic()+timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert condition(), "Background import did not stop promptly"


def raw_dialog(path):
    from lapd_explorer.widgets import ImportDialog
    dialog = ImportDialog(path, {"channels": [], "controls": [], "datasets": [("signal", (3, 200001))]})
    dialog.raw_paths.item(0).setSelected(True)
    dialog.add_axis()
    dialog.axes.cellWidget(0, 0).setCurrentText("shot")
    dialog.axes.item(0, 1).setText("3")
    return dialog


def test_cancel_button_stops_real_import_and_rejects_dialog(app, tmp_path, monkeypatch):
    path = tmp_path/"signal.h5"
    with h5py.File(path, "w") as f:
        f["signal"] = np.arange(3*200001).reshape(3, 200001)
    entered, release = threading.Event(), threading.Event()
    reads = []
    original = h5py.Dataset.__getitem__
    def read(dataset, key):
        reads.append(key)
        entered.set()
        assert release.wait(5)
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    dialog = raw_dialog(path)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.show()
    dialog.begin()
    try:
        wait_until(app, entered.is_set)
        dialog.cancel_button.click()
        assert dialog.cancel_event.is_set() and dialog.dataset is None
        assert "Stopping import" in dialog.message.text()
    finally:
        release.set()
    wait_until(app, lambda: not dialog.isVisible() and not dialog.worker.isRunning())
    assert dialog.dataset is None and not accepted and len(reads) == 1


def test_cancel_discards_late_completed_result(app, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    data = object()
    def read(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return data  # Simulate the final native operation completing after Cancel.
    monkeypatch.setattr(io, "read_raw", read)
    dialog = raw_dialog("unused.h5")
    dialog.show()
    dialog.begin()
    try:
        wait_until(app, entered.is_set)
        dialog.reject()
        dialog.loaded(data)  # A success signal can already be queued at the click.
        assert dialog.dataset is None
    finally:
        release.set()
    wait_until(app, lambda: not dialog.isVisible() and not dialog.worker.isRunning())
    assert dialog.dataset is None and dialog.worker.fn is None


def test_cancel_stops_metadata_guess_and_suppresses_queued_updates(app, monkeypatch):
    from lapd_explorer.widgets import ImportDialog
    entered, release = threading.Event(), threading.Event()
    guess = dict(shots_per_case=50, spatial_points=1, spatial_shape=(), records=50, position_field="xyz")
    def infer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return guess
    monkeypatch.setattr(io, "guess_shots_per_case", infer)
    info = {"channels": [dict(digitizer="D", config_name="cfg", adc="A", board=1, channel=1)],
            "controls": [], "datasets": []}
    dialog = ImportDialog("unused.h5", info)
    dialog.show()
    dialog.start_guess()
    try:
        wait_until(app, entered.is_set)
        dialog.reject()
        dialog.guess_ready(guess)
        assert dialog.repeats.value() != 50
    finally:
        release.set()
    wait_until(app, lambda: not dialog.isVisible() and not dialog.guess_worker.isRunning())
    assert not dialog.guess_timer.isActive() and dialog.dataset is None


def test_cancel_during_success_cleanup_discards_completed_dataset(app):
    from lapd_explorer.widgets import Worker
    entered, release = threading.Event(), threading.Event()
    def metadata():
        entered.set()
        assert release.wait(5)
        return {}
    dialog = raw_dialog("unused.h5")
    dialog.guess_worker = Worker(metadata, dialog, canceled=dialog.cancel_event.is_set)
    dialog.guess_worker.finished.connect(dialog.guess_finished)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.show()
    dialog.guess_worker.start()
    data = object()
    try:
        wait_until(app, entered.is_set)
        dialog.loaded(data)
        assert dialog.dataset is data and dialog.isVisible() and dialog._accept_pending
        dialog.cancel_button.click()
        assert dialog.dataset is None and not dialog._accept_pending
    finally:
        release.set()
    wait_until(app, lambda: not dialog.isVisible() and not dialog.guess_worker.isRunning())
    assert dialog.dataset is None and not accepted
