"""Read-only acquisition adapters and portable processed-data export."""
from dataclasses import dataclass
import json
import numpy as np
import h5py
from .model import Dataset
from .cancellation import check_canceled
from .temporal import sample_slice, selection_metadata, sample_times, source_chunk_samples, resample_into


@dataclass
class Records:
    channels: dict
    time: np.ndarray
    shots: np.ndarray
    xyz: np.ndarray | None
    source: str
    metadata: dict


def inspect_file(path):
    from bapsflib import lapd
    channels, controls, datasets = [], [], []
    with h5py.File(path, "r") as f:
        if f.attrs.get("lapd_explorer_format", "") == "1":
            return {"portable": True, "channels": [], "controls": [], "datasets": []}
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim and obj.dtype.kind in "fiu":
                datasets.append((name, obj.shape))
        f.visititems(visit)
    error = ""
    try:
        with lapd.File(path, mode="r") as f:
            for device, mapper in f.digitizers.items():
                for config_name, config in mapper.configs.items():
                    for adc in config.get("adc", []):
                        for board, channel_group, info in config.get(adc, []):
                            group = (channel_group,) if np.isscalar(channel_group) else channel_group
                            for channel in group:
                                spec = dict(digitizer=device, config_name=config_name,
                                            adc=adc, board=int(board), channel=int(channel))
                                spec.update(_sis_channel_description(f, mapper, spec))
                                channels.append(spec)
            for name, mapper in f.controls.items():
                if "motion" in str(mapper.contype).lower():
                    controls.extend((name, cfg) for cfg in mapper.configs)
    except Exception as exc:
        error = str(exc)
    return dict(portable=False, channels=channels, controls=controls, datasets=datasets, error=error)


def _sis_channel_description(f, mapper, spec):
    """Read the user-entered description from the selected SIS ADC configuration."""
    adc = spec["adc"]
    if adc not in ("SIS 3302", "SIS 3305"):
        return {}
    config = f[mapper.configs[spec["config_name"]]["config group path"]]
    slot = mapper.get_slot(spec["board"], adc)
    index = next((int(index) for slot_number, index in zip(
        config.attrs["SIS crate slot numbers"], config.attrs["SIS crate config indices"])
        if slot_number == slot), None)
    group_name = f"SIS crate {adc.split()[-1]} configurations[{index}]"
    if index is None or group_name not in config:
        return {}
    channel = spec["channel"]
    field = (f"FPGA {(channel-1)//4+1} Data type {(channel-1)%4+1}"
             if adc == "SIS 3305" else f"Data type {channel}")
    value = np.asarray(config[group_name].attrs.get(field, ""))
    if value.size != 1:
        return {}
    value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    description = value.strip() if isinstance(value, str) else ""
    return dict(data_type=description, data_type_field=field)


def channel_name(spec, index=None):
    """Use a nonblank file description, with the hardware identity as fallback."""
    description = (spec.get("data_type") or "").strip()
    name = description or f"B{spec['board']} Ch{spec['channel']}"
    return f"C{index} · {name}" if index is not None else name


def infer_shots_from_positions(xyz, decimals=4):
    """Return ``(shots_per_case, spatial_points, spatial_shape)``.

    The inference assumes one case.  A spatial point must occur the same number
    of times as every other point, and the positions must form a complete point,
    line, or rectangular plane after rounding.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("Motion positions must be a nonempty N×3 array.")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("Finite motion coordinates are unavailable.")
    xyz = np.round(xyz, decimals)
    varying = [axis for axis in range(3) if len(np.unique(xyz[:, axis])) > 1]
    if len(varying) > 2:
        raise ValueError("Motion varies in three spatial axes.")
    if varying:
        points = xyz[:, varying]
        unique_points, counts = np.unique(points, axis=0, return_counts=True)
        spatial_shape = tuple(len(np.unique(xyz[:, axis])) for axis in varying)
    else:
        unique_points = np.zeros((1, 0))
        counts = np.array([len(xyz)])
        spatial_shape = ()
    if len(unique_points) != int(np.prod(spatial_shape or (1,))):
        raise ValueError("Motion positions do not form a complete rectangular grid.")
    if not np.all(counts == counts[0]):
        raise ValueError("Spatial points do not contain equal numbers of shots.")
    return int(counts[0]), len(unique_points), spatial_shape


def guess_shots_per_case(path, spec, control=None, start=0, stop=None,
                         position_source="target", decimals=4, *, canceled=None):
    """Infer shots from one dataset's row count and optional motion metadata.

    Shot estimation never reads digitizer signal samples.
    """
    from bapsflib import lapd

    check_canceled(canceled)
    with lapd.File(path, mode="r") as f:
        check_canceled(canceled)
        mapper, opts, group, dataset, _ = _digitizer_dataset(f, spec)
        first, last, step = slice(start, stop).indices(dataset.shape[0])
        count = len(range(first, last, step))
        if not count:
            raise ValueError("No records in the selected range.")

        xyz = None
        field = "none (point assumed)"
        if control is not None:
            records = _record_metadata(f, mapper, opts, group, dataset.shape[0], control, start, stop, position_source, canceled=canceled)
            field, xyz = records["position_field"], records["xyz"]
            count = len(xyz)
            if not count:
                raise ValueError("No records in the selected range have matching motion metadata.")
            if not np.all(np.isfinite(xyz)):
                xyz = None
                field = "none (point assumed)"
    # Without finite motion metadata, assume a point without allocating one
    # placeholder coordinate per record. The user may enter manual dimensions.
    check_canceled(canceled)
    shots, points, shape = (count, 1, ()) if xyz is None else infer_shots_from_positions(xyz, decimals)
    check_canceled(canceled)
    return {
        "shots_per_case": shots,
        "spatial_points": points,
        "spatial_shape": shape,
        "records": count,
        "position_field": field,
    }


def _digitizer_dataset(f, spec):
    mapper = (f.digitizers[spec["digitizer"]] if spec.get("digitizer") else f.file_map.main_digitizer)
    if mapper is None:
        raise ValueError("Select a digitizer.")
    config, adc = mapper.validate_config_name_and_adc(spec.get("config_name") or None, spec.get("adc") or None)
    opts = dict(board=spec["board"], channel=spec["channel"], config_name=config, adc=adc)
    name, info = mapper.construct_dataset_name(**opts, return_info=True)
    group = f[mapper.info["group path"]]
    return mapper, opts, group, group[name], info


def _record_metadata(f, mapper, opts, group, count, control, start, stop, position_source, *, canceled=None):
    check_canceled(canceled)
    first, last, step = slice(start, stop).indices(count)
    indices = np.arange(first, last, step)
    config = mapper.configs[opts["config_name"]]["shotnum"]
    if config is None:
        shots = indices + 1
    else:
        header = group[mapper.construct_header_dataset_name(**opts)]
        shots = np.asarray(header.fields(config["dset field"][0])[first:last])
    field, measured, target, controls = "xyz", np.full((len(shots), 3), np.nan), None, None
    check_canceled(canceled)
    if control is not None and len(shots):
        records = f.read_controls([tuple(control)], shotnum=shots)
        check_canceled(canceled)
        valid = np.isin(shots, records["shotnum"])
        indices, shots = indices[valid], shots[valid]
        order = np.argsort(records["shotnum"])
        rows = order[np.searchsorted(records["shotnum"][order], shots)]
        measured = np.asarray(records["xyz"][rows], dtype=float)
        if "xyz_target" in records.dtype.names:
            target = np.asarray(records["xyz_target"][rows], dtype=float)
            if position_source == "target":
                field = "xyz_target"
        controls = records.info["controls"]
    check_canceled(canceled)
    return dict(indices=indices, shots=shots, xyz=target if field == "xyz_target" else measured,
                measured=measured, target=target, position_field=field, controls=controls)


def import_metadata(path, spec=None, dataset_path=None, dt=None, control=None,
                    start=0, stop=None, position_source="target", decimals=4, *, canceled=None):
    """Time base and record selectors for a preview; never reads signal samples."""
    check_canceled(canceled)
    if spec is None:
        with h5py.File(path, "r") as f:
            dataset = f[dataset_path]
            if dataset.ndim < 1:
                raise ValueError("A time-series dataset is required.")
            count = int(np.prod(dataset.shape[:-1]))
            if not count:
                raise ValueError("No records in this dataset.")
            sample_slice(dataset.shape[-1], dt)
            check_canceled(canceled)
            return dict(samples=dataset.shape[-1], dt=dt, indices=np.arange(count),
                        shots=None, xyz=None, shape=dataset.shape, units="raw units")
    from bapsflib import lapd
    with lapd.File(path, mode="r") as f:
        mapper, opts, group, dataset, info = _digitizer_dataset(f, spec)
        rate = info.get("clock rate")
        native_dt = None if rate is None else 1 / float(rate.to_value("Hz"))
        if native_dt is not None:
            native_dt *= float(info.get("sample average (hardware)") or 1)
        if native_dt is None or not np.isfinite(native_dt) or native_dt <= 0:
            raise ValueError("Digitizer sample interval is missing; use Raw HDF5 with an explicit interval.")
        records = _record_metadata(f, mapper, opts, group, dataset.shape[0], control, start, stop, position_source, canceled=canceled)
        if not len(records["indices"]):
            raise ValueError("No records in the selected range.")
        return dict(records, samples=dataset.shape[-1], dt=native_dt, shape=dataset.shape, units="V",
                    coordinate_decimals=decimals)


def preview_trace(path, metadata, record_index, first, last, t0=0., spec=None, dataset_path=None,
                  max_points=20000):
    """Read a bounded display trace, refining the sampling when the view zooms."""
    if not isinstance(max_points, int) or max_points < 2:
        raise ValueError("Preview point limit must be at least two.")
    sample_slice(metadata["samples"], metadata["dt"], t0, sample_limits=(first, last))
    step = max(1, int(np.ceil((last - first + 1) / max_points)))
    selection = slice(first, last + 1, step)
    if spec is None:
        key = np.unravel_index(record_index, metadata["shape"][:-1]) if len(metadata["shape"]) > 1 else ()
        with h5py.File(path, "r") as f:
            values = np.asarray(f[dataset_path][key + (selection,)], dtype=float)
    else:
        from bapsflib import lapd
        opts = {k: spec[k] for k in ("digitizer", "config_name", "adc") if spec.get(k)}
        with lapd.File(path, mode="r") as f:
            records = f.read_data(spec["board"], spec["channel"], **opts,
                                  index=record_index, time_slice=selection)
            values = np.array(records["signal"][0], copy=True)
    return t0 + np.arange(first, last + 1, step) * metadata["dt"], values


def read_lapd(path, selections, control=None, start=0, stop=None, t0=0.0, position_source="target",
              *, sample_limits=None, time_limits=None, decimation=1, downsampling="polyphase", progress=None, canceled=None):
    from bapsflib import lapd
    check_canceled(canceled)
    arrays, reference, time, dt, xyz, infos = {}, None, None, None, None, {}
    with lapd.File(path, mode="r") as f:
        infos["acquisition"] = dict(f.info)
        for channel_index, (name, spec) in enumerate(selections.items()):
            check_canceled(canceled)
            mapper, opts, group, dataset, info = _digitizer_dataset(f, spec)
            rate = info.get("clock rate")
            if rate is None:
                raise ValueError("Digitizer sample interval is missing. Use the raw HDF5 importer with an explicit interval.")
            step = float(info.get("sample average (hardware)") or 1) / float(rate.to_value("Hz"))
            if step <= 0 or not np.isfinite(step):
                raise ValueError("Invalid digitizer sample interval.")
            selection = sample_slice(dataset.shape[-1], step, t0, sample_limits, time_limits, decimation, downsampling)
            channel_time = sample_times(selection, step, t0, downsampling)
            meta = _record_metadata(f, mapper, opts, group, dataset.shape[0], control, start, stop, position_source, canceled=canceled)
            shot = meta["shots"]
            if not len(shot):
                raise ValueError("No records in the selected range.")
            if len(np.unique(shot)) != len(shot):
                raise ValueError("Duplicate global shot numbers; select a single acquisition configuration.")
            if reference is not None and (not np.array_equal(shot, reference) or not np.isclose(step, dt, rtol=1e-10, atol=0)
                    or time.shape != channel_time.shape
                    or not np.allclose(channel_time, time, rtol=1e-10, atol=0)):
                raise ValueError("Vector channels must have identical shot numbers, sample counts and time steps.")
            reference, time, dt = shot.copy(), channel_time, step
            check_canceled(canceled)
            signal = np.empty((len(shot), len(time)), dtype=np.float32)
            batch = max(1, min(256, 8_000_000 // source_chunk_samples(selection, downsampling)))
            read_opts = {key: spec[key] for key in ("digitizer", "adc", "config_name") if spec.get(key)}
            channel_info = {}
            for first in range(0, len(shot), batch):
                rows = meta["indices"][first:first + batch].tolist()
                def read(part):
                    check_canceled(canceled)
                    r = f.read_data(spec["board"], spec["channel"], **read_opts, index=rows, time_slice=part)
                    if not np.array_equal(r["shotnum"], shot[first:first + batch]):
                        raise ValueError("Digitizer shot numbers changed while reading.")
                    channel_info.update(r.info)
                    return r["signal"]
                def report(completed):
                    if progress:
                        done = first + len(rows)*completed/len(time)
                        progress((channel_index*len(shot) + done)*len(time),
                                 len(selections)*len(shot)*len(time), "Reading / resampling")
                if first == 0:
                    report(0)
                resample_into(read, signal[first:first+len(rows)], selection, downsampling, progress=report, canceled=canceled)
            channel_info["time_slice"] = selection
            arrays[name] = signal
            current_xyz = meta["xyz"]
            if xyz is not None and not np.allclose(current_xyz, xyz, equal_nan=True):
                raise ValueError("Channel motion coordinates do not match.")
            xyz = current_xyz.copy()
            infos[name] = channel_info
            if spec.get("data_type"):
                infos[name]["data type"] = spec["data_type"]
                if spec.get("data_type_field"):
                    infos[name]["data type field"] = spec["data_type_field"]
            infos[name]["controls"] = meta["controls"]
            infos[name]["position field used"] = meta["position_field"]
            infos[name]["measured xyz"] = meta["measured"].tolist()
            infos[name]["temporal selection"] = selection_metadata(selection, step, downsampling)
            if meta["target"] is not None:
                infos[name]["target xyz"] = meta["target"].tolist()
    if reference is None or len(reference) == 0:
        raise ValueError("No records in the selected range.")
    if len(np.unique(reference)) != len(reference):
        raise ValueError("Duplicate global shot numbers; select a single acquisition configuration.")
    check_canceled(canceled)
    return Records(arrays, time, reference,
                   xyz, str(path), infos)


def map_motion(records, cases=1, repeats=1, repeat_order="case,shot", decimals=4, *, progress=None, canceled=None):
    """Group by position, then explicitly map per-position acquisition order.

    Reject incomplete grids and unmatched repeat counts rather than silently
    combining missing positions or treating parameter scans as statistics.
    """
    check_canceled(canceled)
    if progress:
        progress(0, len(records.shots)*2, "Mapping")
    check_canceled(canceled)
    xyz = records.xyz
    if xyz is None or not np.all(np.isfinite(xyz)):
        raise ValueError("Finite motion coordinates are unavailable. Use manual dimensions.")
    xyz = np.round(xyz, decimals)
    axes = [i for i in range(3) if len(np.unique(xyz[:, i])) > 1]
    if len(axes) > 2:
        raise ValueError("Motion varies in 3 spatial axes. Select a planar subset before importing.")
    axes = list(reversed(axes))  # y,x; z,x; z,y
    dims = ["xyz"[i] for i in axes]
    coords = {d: np.unique(xyz[:, i]) for d, i in zip(dims, axes)}
    shape = tuple(len(coords[d]) for d in dims)
    if cases < 1 or repeats < 1:
        raise ValueError("Cases and shots per case must be positive.")
    expected = int(np.prod(shape)) * cases * repeats
    if expected != len(records.shots):
        raise ValueError(f"Grid requires {expected} records; found {len(records.shots)}. Set cases/repeats explicitly or use manual mapping.")
    check_canceled(canceled)
    output = {n: np.empty(shape + (cases, repeats, len(records.time)), dtype=a.dtype)
              for n, a in records.channels.items()}
    shotmap = np.empty(shape + (cases, repeats), dtype=records.shots.dtype)
    buckets = {}
    for row in range(len(xyz)):
        if row % 256 == 0:
            check_canceled(canceled)
        key = tuple(int(np.searchsorted(coords[d], xyz[row, i])) for d, i in zip(dims, axes))
        buckets.setdefault(key, []).append(row)
        if progress and (row % 256 == 0 or row+1 == len(xyz)):
            progress(row+1, len(xyz)*2, "Mapping")
    if len(buckets) != int(np.prod(shape)):
        raise ValueError("Motion points do not form a complete rectangular grid.")
    mapped = 0
    time_chunk = min(len(records.time), 65536)
    batch = max(1, min(256, 8_000_000 // time_chunk))
    if repeat_order not in ("case,shot", "shot,case"):
        raise ValueError("Unknown per-position acquisition order.")
    for key, rows in buckets.items():
        check_canceled(canceled)
        if len(rows) != cases*repeats:
            raise ValueError("Unequal records per position; select a balanced acquisition subset.")
        # Copy only a bounded record block, including a single-location run.
        for first in range(0, len(rows), batch):
            check_canceled(canceled)
            selected = rows[first:first+batch]
            order = np.arange(first, first+len(selected))
            ci, si = (order//repeats, order % repeats) if repeat_order == "case,shot" else (order % cases, order//cases)
            for time_first in range(0, len(records.time), time_chunk):
                check_canceled(canceled)
                time_last = min(len(records.time), time_first+time_chunk)
                for n, a in records.channels.items():
                    check_canceled(canceled)
                    output[n][key][ci, si, time_first:time_last] = a[selected, time_first:time_last]
                done = mapped+len(selected)*time_last/len(records.time)
                if progress:
                    progress(len(xyz)+done, len(xyz)*2, "Mapping")
            check_canceled(canceled)
            shotmap[key][ci, si] = records.shots[selected]
            mapped += len(selected)
    check_canceled(canceled)
    dims += ["case", "shot", "time"]
    coords.update(case=np.arange(cases), shot=np.arange(repeats), time=records.time)
    # Singleton case/repeat axes carry no extra statistical information.
    for d in ("case", "shot"):
        if len(coords[d]) == 1:
            axis = dims.index(d)
            output = {n: np.squeeze(a, axis=axis) for n, a in output.items()}
            shotmap = np.squeeze(shotmap, axis=axis)
            dims.remove(d)
            coords.pop(d)
    check_canceled(canceled)
    return Dataset(output, tuple(dims), coords, source=records.source,
                   shot_numbers=shotmap, metadata=records.metadata,
                   history=(f"Motion grid rounded to {decimals} decimals; per-position order {repeat_order}",) + _temporal_history(records))


def _temporal_history(records):
    if "temporal selection" in records.metadata:
        selection = next(iter(records.metadata["temporal selection"].values()))
    else:
        selection = next((info["temporal selection"] for info in records.metadata.values()
                          if isinstance(info, dict) and "temporal selection" in info), None)
    if selection is None:
        return ()
    return (f"Import original samples {selection['first original sample']}–"
            f"{selection['last original sample limit (inclusive)']}; downsampling factor "
            f"{selection['keep every Nth sample']}; effective rate "
            f"{selection['effective sampling rate (Hz)']:g} Hz; "
            f"method {selection.get('downsampling method', 'simple')}",)


def map_manual(records, axes, spatial_units="cm", *, progress=None, canceled=None):
    """axes = [(name, size, start, stop), ...] in slowest-to-fastest order."""
    check_canceled(canceled)
    if progress:
        progress(0, 1, "Mapping")
    check_canceled(canceled)
    dims = tuple(a[0] for a in axes) + ("time",)
    sizes = tuple(a[1] for a in axes)
    if int(np.prod(sizes)) != len(records.shots):
        raise ValueError(f"Dimension product {int(np.prod(sizes))} does not match {len(records.shots)} records.")
    coords = {name: np.linspace(start, stop, size) for name, size, start, stop in axes}
    coords["time"] = records.time
    arrays = {n: a.reshape(sizes + (len(records.time),)) for n, a in records.channels.items()}
    if progress:
        progress(1, 1, "Mapping")
    check_canceled(canceled)
    return Dataset(arrays, dims, coords, spatial_units=spatial_units, source=records.source,
                   shot_numbers=records.shots.reshape(sizes), metadata=records.metadata,
                   history=("Manual mapping, C order (last listed axis varies fastest)",) + _temporal_history(records))


def read_raw(path, paths, axes, dt, t0=0, spatial_units="cm", units="V",
             *, sample_limits=None, time_limits=None, decimation=1, downsampling="polyphase", progress=None, canceled=None):
    check_canceled(canceled)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise ValueError("Sample interval must be positive and start time finite.")
    arrays, time, temporal = {}, None, {}
    with h5py.File(path, "r") as f:
        for channel_index, (name, p) in enumerate(paths.items()):
            check_canceled(canceled)
            dataset = f[p]
            if dataset.ndim < 1:
                raise ValueError("A time-series dataset is required.")
            selection = sample_slice(dataset.shape[-1], dt, t0, sample_limits, time_limits, decimation, downsampling)
            channel_time = sample_times(selection, dt, t0, downsampling)
            if time is not None and not np.array_equal(time, channel_time):
                raise ValueError("Raw channels must have matching record/time shapes.")
            time = channel_time
            count = int(np.prod(dataset.shape[:-1]))
            a = np.empty((count, len(time)), dtype=float)
            def report(completed, row, rows):
                if progress:
                    done = row + rows*completed/len(time)
                    progress((channel_index*count + done)*len(time),
                             len(paths)*count*len(time), "Reading / resampling")
            report(0, 0, 0)
            if dataset.ndim == 1:
                resample_into(lambda part: np.asarray(dataset[part], dtype=float)[None, :], a, selection,
                              downsampling, progress=lambda done: report(done, 0, 1), canceled=canceled)
            else:
                batch = max(1, min(256, 8_000_000 // source_chunk_samples(selection, downsampling)))
                row = 0
                for prefix in np.ndindex(dataset.shape[:-2]):
                    for first in range(0, dataset.shape[-2], batch):
                        last = min(dataset.shape[-2], first + batch)
                        resample_into(lambda part: np.asarray(dataset[prefix + (slice(first, last), part)], dtype=float),
                                      a[row:row+last-first], selection, downsampling,
                                      progress=lambda done: report(done, row, last-first), canceled=canceled)
                        row += last - first
            arrays[name] = a
            temporal[name] = selection_metadata(selection, dt, downsampling)
    shape = next(iter(arrays.values())).shape
    if any(a.shape != shape for a in arrays.values()):
        raise ValueError("Raw channels must have matching record/time shapes.")
    records = Records(arrays, time, np.arange(shape[0]), None, str(path),
                      {"raw datasets": paths, "temporal selection": temporal,
                       "shot numbers": "record indices; global shot numbers unavailable"})
    result = map_manual(records, axes, spatial_units, progress=progress, canceled=canceled)
    result.units = units
    result.shot_numbers = None
    check_canceled(canceled)
    return result


def save_dataset(path, data):
    with h5py.File(path, "w") as f:
        f.attrs.update(lapd_explorer_format="1", dims=json.dumps(data.dims), units=data.units,
                       spatial_units=data.spatial_units, source=data.source,
                       history=json.dumps(data.history), metadata=json.dumps(data.metadata, default=str))
        channels = f.create_group("channels")
        for i, (name, a) in enumerate(data.channels.items()):
            ds = channels.create_dataset(str(i), data=a, compression="gzip")
            ds.attrs["name"] = name
        coords = f.create_group("coordinates")
        for d, c in data.coords.items():
            coords.create_dataset(d, data=c)
        if data.shot_numbers is not None:
            f.create_dataset("shot_numbers", data=data.shot_numbers)


def load_dataset(path):
    with h5py.File(path, "r") as f:
        if f.attrs.get("lapd_explorer_format") != "1":
            raise ValueError("Not a LAPD Explorer export.")
        return Dataset({ds.attrs["name"]: ds[:] for ds in f["channels"].values()},
                       tuple(json.loads(f.attrs["dims"])),
                       {d: c[:] for d, c in f["coordinates"].items()},
                       units=f.attrs["units"], spatial_units=f.attrs["spatial_units"],
                       source=f.attrs["source"], history=tuple(json.loads(f.attrs["history"])),
                       metadata=json.loads(f.attrs["metadata"]),
                       shot_numbers=f["shot_numbers"][()] if "shot_numbers" in f else None)
