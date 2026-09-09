"""Read-only bapsflib adapter. Each operation owns and closes its HDF5 handle."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from bapsflib import lapd
from bapsflib._hdf.maps.controls.types import ConType

from .data import record_array


@dataclass(frozen=True)
class Channel:
    digitizer: str
    config: str
    adc: str
    board: int
    channel: int
    label: str
    records: int
    samples: int

    @property
    def read_kwargs(self):
        return dict(digitizer=self.digitizer, config_name=self.config, adc=self.adc)


@dataclass
class Catalog:
    path: str
    channels: list[Channel]
    motions: list[tuple[str, object]]
    description: str


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _channel_label(file, mapper, config, adc, board, channel, specs):
    """Resolve labels only in the matching dataset or board's configuration."""
    dataset = file[specs["device dataset path"]]
    for key in ("Data type", "data type", "channel_label"):
        if key in dataset.attrs and _text(dataset.attrs[key]):
            return _text(dataset.attrs[key])
    cfg = mapper.configs[config]
    labels = specs.get("channel_labels")
    if labels:
        for b, channels, _ in cfg[adc]:
            if b == board and channel in channels:
                return _text(labels[list(channels).index(channel)])
    if "slot_info" in dir(mapper):
        group = file[cfg["config group path"]]
        slots = group.attrs.get("SIS crate slot numbers", [])
        indices = group.attrs.get("SIS crate config indices", [])
        for slot, index in zip(slots, indices):
            if mapper.slot_info.get(int(slot)) != (board, adc):
                continue
            name = f"SIS crate {adc.split()[-1]} configurations[{index}]"
            if name not in group:
                continue
            key = f"Data type {channel}"
            if adc == "SIS 3305":
                key = f"FPGA {1 + (channel - 1) // 4} Data type {1 + (channel - 1) % 4}"
            label = _text(group[name].attrs.get(key, ""))
            if label:
                return label
    return f"Board {board} / Channel {channel}"


def inspect_file(path):
    channels, motions = [], []
    with lapd.File(str(path), mode="r") as file:
        for name, mapper in file.digitizers.items():
            for config in mapper.active_configs:
                cfg = mapper.configs[config]
                for adc in cfg["adc"]:
                    for board, numbers, _ in cfg[adc]:
                        for number in numbers:
                            specs = file.get_digitizer_specs(board, number, digitizer=name, adc=adc, config_name=config)
                            channels.append(Channel(name, config, adc, int(board), int(number),
                                _channel_label(file, mapper, config, adc, board, number, specs),
                                int(specs["nshotnum"]), int(specs["nt"])))
        for name, mapper in file.controls.items():
            if mapper.contype == ConType.MOTION:
                motions.extend((name, config) for config in mapper.configs)
        description = _text(file.info.get("run description", ""))
    if not channels:
        raise ValueError("bapsflib found no supported active digitizer channels in this file.")
    return Catalog(str(Path(path).resolve()), channels, motions, description)


def _position_units(control_config):
    mg = control_config.get("MG_CONFIG", {})
    if not mg and control_config.get("meta"):
        mg = control_config["meta"][0].get("MG_CONFIG", {})
    units = {axis.get("units") for axis in mg.get("drive", {}).get("axes", {}).values()}
    return str(next(iter(units))) if len(units) == 1 and None not in units else "native"


def read_channel(path, channel, *, rows=None, samples=None, motion=None,
                 position_source="target", name=None, max_bytes=512 * 1024**2):
    rows = rows or slice(0, min(channel.records, 50))
    samples = samples or slice(0, channel.samples)
    if position_source not in ("target", "measured"):
        raise ValueError("Position source must be target or measured.")
    for selection, size in ((rows, channel.records), (samples, channel.samples)):
        if not isinstance(selection, slice) or (selection.step or 1) != 1:
            raise ValueError("Use contiguous row/sample slices; decimation requires an anti-alias filter.")
        if selection.start is not None and not 0 <= selection.start < size:
            raise ValueError("Slice start is outside the dataset.")
        if selection.stop is not None and not 0 < selection.stop <= size:
            raise ValueError("Slice stop is outside the dataset.")
    count = len(range(*rows.indices(channel.records)))
    nt = len(range(*samples.indices(channel.samples)))
    if not count or not nt:
        raise ValueError("Select at least one record and time sample.")
    if count * nt * 4 > max_bytes:
        raise ValueError("Read exceeds the 512 MiB channel limit. Select fewer records or samples.")
    with lapd.File(str(path), mode="r") as file:
        specs = file.get_digitizer_specs(channel.board, channel.channel, **channel.read_kwargs)
        # Generate time from the full specs, then slice ONCE, preserving time origin.
        # Clock metadata supports float64 generation without float32 timing jitter.
        clock = specs.get("clock rate")
        if clock is not None:
            dt = (1 / clock).to_value("s") * float(specs.get("sample average") or 1)
            time = np.arange(*samples.indices(channel.samples), dtype=float) * dt
        else:
            time = np.asarray(file.get_time_array(specs), dtype=float)[samples]
            time_path = specs.get("time_dset_path")
            units = file[time_path].attrs.get("units") if time_path else None
            if units is None:
                raise ValueError("Dedicated time dataset has no units. Add an explicit time-unit adapter before analysis.")
            from astropy import units as u
            time = (time * u.Unit(_text(units))).to_value("s")
        data = file.read_data(channel.board, channel.channel, index=rows, time_slice=samples,
                              add_controls=[motion] if motion else None, keep_bits=False,
                              intersection_set=True, **channel.read_kwargs)
        if not len(data):
            raise ValueError("No digitizer shots matched the selected motion configuration.")
        positions = {}
        position_units = "native"
        if motion:
            config = file.controls[motion[0]].configs[motion[1]]
            position_units = _position_units(config)
            source = "xyz_target" if position_source == "target" else "xyz"
            if source not in data.dtype.names:
                raise ValueError("This motion device has no target coordinates; choose measured positions.")
            fields = config.get("state values", {}).get(source, {}).get("dset field", ())
            for i, axis in enumerate("xyz"):
                if fields and (len(fields) <= i or not fields[i]):
                    continue
                values = np.asarray(data[source])[:, i]
                if np.isfinite(values).any():
                    positions[axis] = values
                    positions[f"measured_{axis}"] = np.asarray(data["xyz"])[:, i]
        attrs = {"units": str(data.info.get("signal units") or "unknown"),
                 "position_units": position_units, "source": str(Path(path).resolve()),
                 "digitizer": channel.digitizer, "configuration": channel.config,
                 "adc": channel.adc, "board": channel.board, "channel": channel.channel,
                 "dataset_path": specs["device dataset path"], "position_source": position_source,
                 "motion": str(motion), "requested_records": count,
                 "unmatched_records": count - len(data),
                 "read_selection": json.dumps({"rows": [rows.start, rows.stop], "samples": [samples.start, samples.stop]}),
                 "scaling": "bapsflib keep_bits=False; no probe calibration applied",
                 "history": "[]"}
        return record_array(np.asarray(data["signal"]), time, np.asarray(data["shotnum"]),
                            positions=positions, name=name or channel.label, attrs=attrs)
