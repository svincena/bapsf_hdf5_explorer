"""Read-only acquisition adapters and portable processed-data export."""
from dataclasses import dataclass
import json
import numpy as np
import h5py
from .model import Dataset
from .temporal import sample_slice, selection_metadata


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
                                channels.append(dict(digitizer=device, config_name=config_name,
                                                     adc=adc, board=int(board), channel=int(channel)))
            for name, mapper in f.controls.items():
                if "motion" in str(mapper.contype).lower():
                    controls.extend((name, cfg) for cfg in mapper.configs)
    except Exception as exc:
        error = str(exc)
    return dict(portable=False, channels=channels, controls=controls, datasets=datasets, error=error)


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
                         position_source="target", decimals=4):
    """Infer shots from one dataset's row count and optional motion metadata.

    Shot estimation never reads digitizer signal samples.
    """
    from bapsflib import lapd

    with lapd.File(path, mode="r") as f:
        mapper, opts, group, dataset, _ = _digitizer_dataset(f, spec)
        first, last, step = slice(start, stop).indices(dataset.shape[0])
        count = len(range(first, last, step))
        if not count:
            raise ValueError("No records in the selected range.")

        xyz = None
        field = "none (point assumed)"
        if control is not None:
            records = _record_metadata(f, mapper, opts, group, dataset.shape[0], control, start, stop, position_source)
            field, xyz = records["position_field"], records["xyz"]
            count = len(xyz)
            if not count:
                raise ValueError("No records in the selected range have matching motion metadata.")
            if not np.all(np.isfinite(xyz)):
                xyz = None
                field = "none (point assumed)"
    # Without finite motion metadata, assume a point without allocating one
    # placeholder coordinate per record. The user may enter manual dimensions.
    shots, points, shape = (count, 1, ()) if xyz is None else infer_shots_from_positions(xyz, decimals)
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


def _record_metadata(f, mapper, opts, group, count, control, start, stop, position_source):
    first, last, step = slice(start, stop).indices(count)
    indices = np.arange(first, last, step)
    config = mapper.configs[opts["config_name"]]["shotnum"]
    if config is None:
        shots = indices + 1
    else:
        header = group[mapper.construct_header_dataset_name(**opts)]
        shots = np.asarray(header.fields(config["dset field"][0])[first:last])
    field, measured, target, controls = "xyz", np.full((len(shots), 3), np.nan), None, None
    if control is not None and len(shots):
        records = f.read_controls([tuple(control)], shotnum=shots)
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
    return dict(indices=indices, shots=shots, xyz=target if field == "xyz_target" else measured,
                measured=measured, target=target, position_field=field, controls=controls)


def import_metadata(path, spec=None, dataset_path=None, dt=None, control=None,
                    start=0, stop=None, position_source="target", decimals=4):
    """Time base and record selectors for a preview; never reads signal samples."""
    if spec is None:
        with h5py.File(path, "r") as f:
            dataset = f[dataset_path]
            if dataset.ndim < 1:
                raise ValueError("A time-series dataset is required.")
            count = int(np.prod(dataset.shape[:-1]))
            if not count:
                raise ValueError("No records in this dataset.")
            sample_slice(dataset.shape[-1], dt)
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
        records = _record_metadata(f, mapper, opts, group, dataset.shape[0], control, start, stop, position_source)
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
              *, sample_limits=None, time_limits=None, decimation=1):
    from bapsflib import lapd
    arrays, reference, time, dt, xyz, infos = {}, None, None, None, None, {}
    with lapd.File(path, mode="r") as f:
        infos["acquisition"] = dict(f.info)
        for name, spec in selections.items():
            mapper, opts, group, dataset, info = _digitizer_dataset(f, spec)
            rate = info.get("clock rate")
            if rate is None:
                raise ValueError("Digitizer sample interval is missing. Use the raw HDF5 importer with an explicit interval.")
            step = float(info.get("sample average (hardware)") or 1) / float(rate.to_value("Hz"))
            if step <= 0 or not np.isfinite(step):
                raise ValueError("Invalid digitizer sample interval.")
            selection = sample_slice(dataset.shape[-1], step, t0, sample_limits, time_limits, decimation)
            channel_time = t0 + np.arange(selection.start, selection.stop, selection.step) * step
            meta = _record_metadata(f, mapper, opts, group, dataset.shape[0], control, start, stop, position_source)
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
            # Keep only a bounded batch of calibrated source records alongside
            # the final thinned array, instead of duplicating the entire channel.
            signal = None
            batch = max(1, min(256, 8_000_000 // len(time)))
            read_opts = {key: spec[key] for key in ("digitizer", "adc", "config_name") if spec.get(key)}
            for first in range(0, len(shot), batch):
                rows = meta["indices"][first:first + batch].tolist()
                r = f.read_data(spec["board"], spec["channel"], **read_opts, index=rows, time_slice=selection)
                if not np.array_equal(r["shotnum"], shot[first:first + batch]):
                    raise ValueError("Digitizer shot numbers changed while reading.")
                if signal is None:
                    signal = np.empty((len(shot), len(time)), dtype=r["signal"].dtype)
                signal[first:first + batch] = r["signal"]
                channel_info = dict(r.info)
                del r
            arrays[name] = signal
            current_xyz = meta["xyz"]
            if xyz is not None and not np.allclose(current_xyz, xyz, equal_nan=True):
                raise ValueError("Channel motion coordinates do not match.")
            xyz = current_xyz.copy()
            infos[name] = channel_info
            infos[name]["controls"] = meta["controls"]
            infos[name]["position field used"] = meta["position_field"]
            infos[name]["measured xyz"] = meta["measured"].tolist()
            infos[name]["temporal selection"] = selection_metadata(selection, step)
            if meta["target"] is not None:
                infos[name]["target xyz"] = meta["target"].tolist()
    if reference is None or len(reference) == 0:
        raise ValueError("No records in the selected range.")
    if len(np.unique(reference)) != len(reference):
        raise ValueError("Duplicate global shot numbers; select a single acquisition configuration.")
    return Records(arrays, time, reference,
                   xyz, str(path), infos)


def map_motion(records, cases=1, repeats=1, repeat_order="case,shot", decimals=4):
    """Group by position, then explicitly map per-position acquisition order.

    Reject incomplete grids and unmatched repeat counts rather than silently
    combining missing positions or treating parameter scans as statistics.
    """
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
    output = {n: np.empty(shape + (cases, repeats, len(records.time)), dtype=a.dtype)
              for n, a in records.channels.items()}
    shotmap = np.empty(shape + (cases, repeats), dtype=records.shots.dtype)
    buckets = {}
    for row in range(len(xyz)):
        key = tuple(int(np.searchsorted(coords[d], xyz[row, i])) for d, i in zip(dims, axes))
        buckets.setdefault(key, []).append(row)
    if len(buckets) != int(np.prod(shape)):
        raise ValueError("Motion points do not form a complete rectangular grid.")
    for key, rows in buckets.items():
        if len(rows) != cases*repeats:
            raise ValueError("Unequal records per position; select a balanced acquisition subset.")
        for n, a in records.channels.items():
            block = a[rows]
            if repeat_order == "case,shot":
                output[n][key] = block.reshape(cases, repeats, -1)
            elif repeat_order == "shot,case":
                output[n][key] = block.reshape(repeats, cases, -1).transpose(1, 0, 2)
            else:
                raise ValueError("Unknown per-position acquisition order.")
        block = records.shots[rows]
        shotmap[key] = (block.reshape(cases, repeats) if repeat_order == "case,shot"
                        else block.reshape(repeats, cases).T)
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
            f"{selection['last original sample limit (inclusive)']}; keep every "
            f"{selection['keep every Nth sample']} sample(s); effective rate "
            f"{selection['effective sampling rate (Hz)']:g} Hz",)


def map_manual(records, axes, spatial_units="cm"):
    """axes = [(name, size, start, stop), ...] in slowest-to-fastest order."""
    dims = tuple(a[0] for a in axes) + ("time",)
    sizes = tuple(a[1] for a in axes)
    if int(np.prod(sizes)) != len(records.shots):
        raise ValueError(f"Dimension product {int(np.prod(sizes))} does not match {len(records.shots)} records.")
    coords = {name: np.linspace(start, stop, size) for name, size, start, stop in axes}
    coords["time"] = records.time
    arrays = {n: a.reshape(sizes + (len(records.time),)) for n, a in records.channels.items()}
    return Dataset(arrays, dims, coords, spatial_units=spatial_units, source=records.source,
                   shot_numbers=records.shots.reshape(sizes), metadata=records.metadata,
                   history=("Manual mapping, C order (last listed axis varies fastest)",) + _temporal_history(records))


def read_raw(path, paths, axes, dt, t0=0, spatial_units="cm", units="V",
             *, sample_limits=None, time_limits=None, decimation=1):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise ValueError("Sample interval must be positive and start time finite.")
    arrays, time, temporal = {}, None, {}
    with h5py.File(path, "r") as f:
        for name, p in paths.items():
            dataset = f[p]
            if dataset.ndim < 1:
                raise ValueError("A time-series dataset is required.")
            selection = sample_slice(dataset.shape[-1], dt, t0, sample_limits, time_limits, decimation)
            channel_time = t0 + np.arange(selection.start, selection.stop, selection.step) * dt
            if time is not None and not np.array_equal(time, channel_time):
                raise ValueError("Raw channels must have matching record/time shapes.")
            time = channel_time
            count = int(np.prod(dataset.shape[:-1]))
            a = np.empty((count, len(time)), dtype=float)
            if dataset.ndim == 1:
                a[0] = dataset[selection]
            else:
                batch = max(1, min(256, 8_000_000 // len(time)))
                row = 0
                for prefix in np.ndindex(dataset.shape[:-2]):
                    for first in range(0, dataset.shape[-2], batch):
                        last = min(dataset.shape[-2], first + batch)
                        a[row:row + last - first] = dataset[prefix + (slice(first, last), selection)]
                        row += last - first
            arrays[name] = a
            temporal[name] = selection_metadata(selection, dt)
    shape = next(iter(arrays.values())).shape
    if any(a.shape != shape for a in arrays.values()):
        raise ValueError("Raw channels must have matching record/time shapes.")
    records = Records(arrays, time, np.arange(shape[0]), None, str(path),
                      {"raw datasets": paths, "temporal selection": temporal,
                       "shot numbers": "record indices; global shot numbers unavailable"})
    result = map_manual(records, axes, spatial_units)
    result.units = units
    result.shot_numbers = None
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
