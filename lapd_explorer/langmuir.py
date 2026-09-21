"""Browser adapter for the authoritative Langmuir core (V, A, m²)."""
from dataclasses import dataclass, field, replace
import numpy as np
from . import langmuir_analysis_core as core

QUANTITIES = {"T_e": ("te_eV", "eV"), "n_e": ("n_e_m3", "m⁻³"),
              "V_p": ("vp_V", "V"), "V_f": ("vf_V", "V")}


@dataclass
class Settings:
    voltage_channel: str = ""
    current_channel: str = ""
    voltage_factor: float = 1.
    current_factor: float = 1.
    resistance: float = 1.
    area_mm2: float = 1.
    invert: bool = False
    interval: tuple | None = None  # inclusive times, seconds
    offset_mode: str = "None"
    offset_interval: tuple | None = None
    offset_volts: float = 0.  # digitizer volts, before scaling and inversion
    zero_calibrated: bool = False
    fit: dict = field(default_factory=lambda: {
        k: v for k, v in core._DEFAULT_CONFIG.items()
        if k not in {"probe_area", "current_zero_calibrated"}})


def interval_slice(time, interval, minimum=1):
    if interval is None:
        raise ValueError("Select a time interval first.")
    start, stop = interval
    if (not np.all(np.isfinite([start, stop])) or start >= stop
            or start < time[0] or stop > time[-1]):
        raise ValueError("Interval must have start < end within the recorded time range.")
    lo = int(np.searchsorted(time, start, side="left"))
    hi = int(np.searchsorted(time, stop, side="right"))
    if hi - lo < minimum:
        raise ValueError(f"Interval needs at least {minimum} samples.")
    return slice(int(lo), hi)


def validate(data, settings, analysis=True):
    s = settings
    if s.voltage_channel not in data.channels or s.current_channel not in data.channels:
        raise ValueError("Select both V_sweep and I_sweep channels from the imported dataset.")
    if s.voltage_channel == s.current_channel:
        raise ValueError("V_sweep and I_sweep must be distinct channels.")
    v, i = data.channels[s.voltage_channel], data.channels[s.current_channel]
    if v.shape != i.shape or v.shape != data.shape:
        raise ValueError("Sweep channels must have matching dimensions and time samples.")
    if data.units != "V":
        raise ValueError("Langmuir input must be unprocessed digitizer data in volts (V).")
    for label, value in [("Voltage attenuation", s.voltage_factor),
                         ("Current attenuation", s.current_factor),
                         ("Resistance", s.resistance), ("Probe area", s.area_mm2)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive.")
    if s.offset_mode not in {"None", "Recorded interval", "Constant"}:
        raise ValueError("Unknown offset correction mode.")
    if s.offset_mode == "Constant" and not np.isfinite(s.offset_volts):
        raise ValueError("Offset must be finite (digitizer volts).")
    if s.offset_mode == "Recorded interval":
        interval_slice(data.coords["time"], s.offset_interval)
    cfg = dict(s.fit, probe_area=s.area_mm2 * 1e-6,
               current_zero_calibrated=s.zero_calibrated or s.offset_mode != "None")
    try:
        core._validated_config(cfg)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid fitting settings: {exc}") from exc
    if analysis:
        interval_slice(data.coords["time"], s.interval, minimum=7)
        if not cfg["current_zero_calibrated"]:
            raise ValueError("Calibrate the absolute current zero: select an offset correction or confirm an independently calibrated zero.")
    return cfg


def traces(data, settings, index):
    """Subtract a per-trace digitizer offset, then scale to V and A."""
    s = settings
    v = np.asarray(data.channels[s.voltage_channel][index], dtype=float)
    i = np.asarray(data.channels[s.current_channel][index], dtype=float)
    offset = 0.
    if s.offset_mode == "Constant":
        offset = s.offset_volts
    elif s.offset_mode == "Recorded interval":
        zero = i[interval_slice(data.coords["time"], s.offset_interval)]
        if not np.all(np.isfinite(zero)):
            raise ValueError("Offset region contains nonfinite current samples.")
        offset = float(np.mean(zero))
    return v * s.voltage_factor, (i - offset) * s.current_factor / s.resistance * (-1 if s.invert else 1)


def analyze(data, settings, index):
    cfg = validate(data, settings)
    v, i = traces(data, settings, index)
    window = interval_slice(data.coords["time"], settings.interval, minimum=7)
    if not np.all(np.isfinite(v[window])) or not np.all(np.isfinite(i[window])):
        raise ValueError("Analysis interval contains nonfinite samples.")
    result = core.analyze_iv_trace(v[window], i[window], config=cfg)
    values = {name: float(result[key]) for name, (key, _) in QUANTITIES.items()}
    valid = (result["ok"] and all(np.isfinite(list(values.values())))
             and values["T_e"] > 0 and values["n_e"] > 0 and values["V_f"] < values["V_p"])
    if not valid:
        raise ValueError("; ".join(result["warnings"]) or "Analysis returned missing or nonphysical values.")
    return values, result.get("model_notes", [])


@dataclass
class BatchResult:
    values: dict
    failures: list
    settings: Settings

    def dataset(self, source, name):
        """Singleton time holds an interval estimate, preserving all shot/space axes."""
        return replace(source, channels={name: self.values[name][..., None]},
                       coords=dict(source.coords, time=np.array([np.mean(self.settings.interval)])),
                       units=QUANTITIES[name][1],
                       history=source.history + (f"Langmuir {name}: one estimate per selected sweep interval",),
                       metadata=dict(source.metadata, langmuir=vars(self.settings)))


def process_all(data, settings, progress=None, canceled=None):
    import copy
    validate(data, settings)
    arrays = {name: np.full(data.shape[:-1], np.nan) for name in QUANTITIES}
    failures = []
    total = int(np.prod(data.shape[:-1]))
    for number, index in enumerate(np.ndindex(data.shape[:-1])):
        if canceled is not None and canceled():
            raise ValueError("Langmuir processing canceled; previous results retained.")
        try:
            values, _ = analyze(data, settings, index)
            for name, value in values.items():
                arrays[name][index] = value
        except Exception as exc:
            failures.append((index, str(exc)))
        if progress is not None and ((number + 1) % max(1, total // 100) == 0 or number + 1 == total):
            progress(number + 1, total)
    return BatchResult(arrays, failures, copy.deepcopy(settings))
