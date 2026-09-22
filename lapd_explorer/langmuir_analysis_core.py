"""Numerical core for swept planar Langmuir-probe I-V analysis.

The default quality checks allow the sloped saturation branches and modest
non-Maxwellian structure normally seen in experimental probe curves.  An
opt-in ideal-model mode additionally requires nearly flat saturation regions,
a straight Maxwellian semilog branch, negligible electron leakage into the ion
region, and agreement between two plasma-potential estimators.  Current must
retain an independently calibrated absolute zero; zeroing the low-bias ion
branch makes floating potential and ion current unidentifiable.
"""

from __future__ import annotations

from collections.abc import Mapping

import astropy.units as u
from astropy.constants import e, k_B, m_e
import numpy as np
from scipy.signal import savgol_filter


_MIN_CURVE_POINTS = 7
_ION_WINDOW_FRACTION = 0.25
_ION_MIN_POINTS = 5
_ION_ESTIMATE_MIN_SNR = 3.0
_SATURATION_MAX_TREND_FRACTION = 0.20
_TE_CURRENT_CEILING_FRACTION = 0.80
_TE_MIN_LOG_SPAN = 1.0
_TE_MAX_RELATIVE_SLOPE_UNCERTAINTY = 0.30
_TE_MAX_CURVATURE_LOG = 0.05
_TE_CURVATURE_SIGNIFICANCE = 3.0
_ELECTRON_SIGNAL_TO_NOISE = 5.0
_MAX_REVERSE_SWEEP_FRACTION = 0.05
_IES_METHODS = {"high_bias_median", "at_vp"}


_DEFAULT_CONFIG = {
    "iv_npts": 600,
    "voltage_bin_width": 0.05,
    "vp_smoothing": "savgol",
    "vp_smoothing_width_V": 2.5,
    "vp_savgol_order": 2,
    "ies_method": "high_bias_median",
    "te_min_points": 8,
    "te_margin_from_vp": 0.2,
    "te_current_floor_frac": 0.03,
    "te_min_eV": 0.1,
    "te_max_eV": 30.0,
    "te_min_r2": 0.95,
    "ion_min_snr": _ION_ESTIMATE_MIN_SNR,
    "probe_area": None,
    "current_zero_calibrated": False,
    "enforce_ideal_model_checks": False,
}


def density_from_electron_saturation_current(
    I_esat,
    Te,
    probe_area,
    use_abs=False,
):
    """Compute density from the planar Maxwellian random-electron flux.

    Unitless inputs are interpreted as amperes, electronvolts, and square
    metres, respectively.  ``use_abs`` is available only for an explicitly
    verified reversed-current convention; the default rejects a non-positive
    electron saturation current so polarity errors are not hidden.
    """
    if not isinstance(I_esat, u.Quantity):
        I_esat = np.asarray(I_esat) * u.A
    if not isinstance(Te, u.Quantity):
        Te = np.asarray(Te) * u.eV
    if not isinstance(probe_area, u.Quantity):
        probe_area = np.asarray(probe_area) * u.m**2

    I_esat = I_esat.to(u.A)
    probe_area = probe_area.to(u.m**2)

    if use_abs:
        I_esat = np.abs(I_esat)

    if np.any(~np.isfinite(I_esat.value)) or np.any(I_esat <= 0 * u.A):
        raise ValueError("I_esat must be finite and positive.")
    if np.any(~np.isfinite(probe_area.value)) or np.any(probe_area <= 0 * u.m**2):
        raise ValueError("probe_area must be finite and positive.")
    if np.any(~np.isfinite(Te.value)) or np.any(Te <= 0 * Te.unit):
        raise ValueError("Te must be finite and positive.")

    if Te.unit.is_equivalent(u.K):
        Te_energy = (k_B * Te).to(u.J)
    else:
        Te_energy = Te.to(u.J)

    electron_flux_speed = np.sqrt(Te_energy / (2 * np.pi * m_e))
    density = I_esat / (e.si * probe_area * electron_flux_speed)
    return density.to(u.m**-3)


def _as_float_1d(values, name, unit=None):
    if isinstance(values, u.Quantity):
        if unit is None:
            values = values.value
        else:
            values = values.to_value(unit)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _validate_trace_arrays(voltage, current):
    voltage = _as_float_1d(voltage, "V", u.V)
    current = _as_float_1d(current, "I", u.A)
    if voltage.size != current.size:
        raise ValueError("V and I must have the same length.")
    return voltage, current


def _validated_config(config):
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")

    unknown = set(config) - set(_DEFAULT_CONFIG)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown Langmuir analysis config key(s): {names}.")

    cfg = {**_DEFAULT_CONFIG, **config}

    integer_fields = {
        "iv_npts": _MIN_CURVE_POINTS,
        "vp_savgol_order": 1,
        "te_min_points": 3,
    }
    for name, minimum in integer_fields.items():
        value = cfg[name]
        if (
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        cfg[name] = int(value)

    positive_fields = (
        "voltage_bin_width",
        "vp_smoothing_width_V",
        "te_min_eV",
        "te_max_eV",
    )
    for name in positive_fields:
        value = float(cfg[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        cfg[name] = value

    cfg["te_margin_from_vp"] = float(cfg["te_margin_from_vp"])
    if not np.isfinite(cfg["te_margin_from_vp"]) or cfg["te_margin_from_vp"] < 0:
        raise ValueError("te_margin_from_vp must be finite and non-negative.")

    cfg["te_current_floor_frac"] = float(cfg["te_current_floor_frac"])
    if not 0 <= cfg["te_current_floor_frac"] < _TE_CURRENT_CEILING_FRACTION:
        raise ValueError(
            f"te_current_floor_frac must be in [0, {_TE_CURRENT_CEILING_FRACTION})."
        )

    cfg["te_min_r2"] = float(cfg["te_min_r2"])
    if not 0 <= cfg["te_min_r2"] <= 1:
        raise ValueError("te_min_r2 must be between 0 and 1.")
    if cfg["te_min_eV"] >= cfg["te_max_eV"]:
        raise ValueError("te_min_eV must be less than te_max_eV.")

    cfg["ion_min_snr"] = float(cfg["ion_min_snr"])
    if not np.isfinite(cfg["ion_min_snr"]) or cfg["ion_min_snr"] < 0:
        raise ValueError("ion_min_snr must be finite and non-negative.")

    for name in ("current_zero_calibrated", "enforce_ideal_model_checks"):
        if not isinstance(cfg[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean.")
        cfg[name] = bool(cfg[name])

    probe_area = cfg["probe_area"]
    if probe_area is not None:
        if isinstance(probe_area, u.Quantity):
            try:
                area_value = np.asarray(probe_area.to_value(u.m**2), dtype=float)
            except u.UnitConversionError as error:
                raise ValueError("probe_area must have units of area.") from error
        else:
            area_value = np.asarray(probe_area, dtype=float)
        if area_value.shape != () or not np.isfinite(area_value) or area_value <= 0:
            raise ValueError("probe_area must be a finite positive scalar.")

    smoothing = cfg["vp_smoothing"]
    smoothing = "none" if smoothing is None else str(smoothing).lower()
    if smoothing not in {"none", "moving", "savgol"}:
        raise ValueError("vp_smoothing must be None, 'moving', or 'savgol'.")
    cfg["vp_smoothing"] = smoothing

    ies_method = str(cfg["ies_method"]).lower()
    if ies_method not in _IES_METHODS:
        choices = ", ".join(sorted(_IES_METHODS))
        raise ValueError(f"ies_method must be one of: {choices}.")
    cfg["ies_method"] = ies_method
    return cfg


def bin_average_by_voltage(voltage, current, bin_width):
    """Return finite voltage-bin means sorted in strictly increasing voltage."""
    voltage, current = _validate_trace_arrays(voltage, current)
    bin_width = float(bin_width)
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive.")

    good = np.isfinite(voltage) & np.isfinite(current)
    voltage = voltage[good]
    current = current[good]
    if voltage.size < 2:
        return None, None

    order = np.argsort(voltage, kind="stable")
    voltage = voltage[order]
    current = current[order]
    bin_ids = np.floor((voltage - voltage[0]) / bin_width).astype(np.int64)
    _, inverse = np.unique(bin_ids, return_inverse=True)
    counts = np.bincount(inverse)
    Vb = np.bincount(inverse, weights=voltage) / counts
    Ib = np.bincount(inverse, weights=current) / counts

    keep = np.isfinite(Vb) & np.isfinite(Ib)
    Vb = Vb[keep]
    Ib = Ib[keep]
    keep = np.concatenate(([True], np.diff(Vb) > 0))
    Vb = Vb[keep]
    Ib = Ib[keep]
    if Vb.size < 2:
        return None, None
    return Vb, Ib


def make_monotonic_iv_curve(bias, current, npts=600, bin_width=0.05):
    """Bin a sweep and make separate analysis and fixed-size display grids."""
    voltage, current = _validate_trace_arrays(bias, current)
    if (
        isinstance(npts, (bool, np.bool_))
        or int(npts) != npts
        or npts < _MIN_CURVE_POINTS
    ):
        raise ValueError(f"npts must be an integer >= {_MIN_CURVE_POINTS}.")

    Vb, Ib = bin_average_by_voltage(voltage, current, bin_width=bin_width)
    if Vb is None or Vb.size < _MIN_CURVE_POINTS:
        return None

    V_grid = np.linspace(Vb[0], Vb[-1], int(npts))
    I_grid = np.interp(V_grid, Vb, Ib)

    typical_spacing = max(np.median(np.diff(Vb)), 0.5 * float(bin_width))
    if not np.isfinite(typical_spacing) or typical_spacing <= 0:
        return None
    analysis_npts = min(
        max(4 * Vb.size, _MIN_CURVE_POINTS),
        max(
            _MIN_CURVE_POINTS,
            int(np.ceil((Vb[-1] - Vb[0]) / typical_spacing)) + 1,
        ),
    )
    V_analysis = np.linspace(Vb[0], Vb[-1], analysis_npts)
    I_analysis = np.interp(V_analysis, Vb, Ib)

    return {
        "V_raw": voltage,
        "I_raw": current,
        "V_binned": Vb,
        "I_binned": Ib,
        "V_analysis": V_analysis,
        "I_analysis": I_analysis,
        "V_grid": V_grid,
        "I_grid": I_grid,
    }


def find_floating_potential_from_iv(voltage, current, upper_bound=None):
    """Return the highest genuinely bracketed negative-to-positive crossing."""
    voltage, current = _validate_trace_arrays(voltage, current)
    finite = np.isfinite(voltage) & np.isfinite(current)
    voltage = voltage[finite]
    current = current[finite]
    if voltage.size < 2 or np.any(np.diff(voltage) <= 0):
        return np.nan * u.V

    candidates = np.flatnonzero((current[:-1] < 0) & (current[1:] >= 0))
    if upper_bound is not None:
        if isinstance(upper_bound, u.Quantity):
            upper_bound = upper_bound.to_value(u.V)
        candidates = candidates[voltage[candidates] < float(upper_bound)]
    if candidates.size == 0:
        return np.nan * u.V

    index = candidates[-1]
    v1, v2 = voltage[index : index + 2]
    i1, i2 = current[index : index + 2]
    if i2 == i1:
        return np.nan * u.V
    return (v1 - i1 * (v2 - v1) / (i2 - i1)) * u.V


def _window_points(length, voltage_step, smoothing_width, polyorder=0):
    points = max(1, int(round(smoothing_width / voltage_step)))
    if points % 2 == 0:
        points += 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    points = max(points, minimum)
    maximum = length if length % 2 == 1 else length - 1
    return min(points, maximum)


def find_plasma_potential_from_iv(
    voltage,
    current,
    vf=None,
    smoothing="savgol",
    smoothing_width_V=2.5,
    savgol_order=2,
):
    """Estimate Vp from an interior maximum of a physically smoothed dI/dV."""
    voltage, current = _validate_trace_arrays(voltage, current)
    if (
        voltage.size < _MIN_CURVE_POINTS
        or np.any(~np.isfinite(voltage))
        or np.any(~np.isfinite(current))
    ):
        return np.nan * u.V, np.full(voltage.shape, np.nan, dtype=float)
    if np.any(np.diff(voltage) <= 0):
        return np.nan * u.V, np.full(voltage.shape, np.nan, dtype=float)

    voltage_step = float(np.median(np.diff(voltage)))
    smoothing_name = "none" if smoothing is None else str(smoothing).lower()
    edge_points = 1

    if smoothing_name == "none":
        dIdV = np.gradient(current, voltage)
    elif smoothing_name == "moving":
        window = _window_points(len(current), voltage_step, smoothing_width_V)
        kernel = np.ones(window, dtype=float) / window
        weights = np.convolve(np.ones_like(current), kernel, mode="same")
        current_smooth = np.convolve(current, kernel, mode="same") / weights
        dIdV = np.gradient(current_smooth, voltage)
        edge_points = max(1, window // 2)
    elif smoothing_name == "savgol":
        window = _window_points(
            len(current), voltage_step, smoothing_width_V, savgol_order
        )
        if window <= savgol_order:
            return np.nan * u.V, np.full(voltage.shape, np.nan, dtype=float)
        dIdV = savgol_filter(
            current,
            window_length=window,
            polyorder=savgol_order,
            deriv=1,
            delta=voltage_step,
            mode="interp",
        )
        edge_points = max(1, window // 2)
    else:
        raise ValueError("smoothing must be None, 'moving', or 'savgol'.")

    search = np.isfinite(dIdV)
    search[: edge_points + 1] = False
    search[-(edge_points + 1) :] = False
    if vf is not None:
        vf_value = vf.to_value(u.V) if isinstance(vf, u.Quantity) else float(vf)
        if np.isfinite(vf_value):
            search &= voltage > vf_value

    indices = np.flatnonzero(search)
    if indices.size < 3:
        return np.nan * u.V, dIdV
    index = indices[np.argmax(dIdV[indices])]
    if (
        index in (indices[0], indices[-1])
        or not np.isfinite(dIdV[index])
        or dIdV[index] <= 0
    ):
        return np.nan * u.V, dIdV

    post_start = min(len(voltage), index + edge_points)
    post_stop = len(voltage) - edge_points
    if post_stop - post_start < 3:
        return np.nan * u.V, dIdV
    post_slope = np.nanmedian(dIdV[post_start:post_stop])
    if np.isfinite(post_slope) and post_slope >= 0.5 * dIdV[index]:
        return np.nan * u.V, dIdV

    return voltage[index] * u.V, dIdV


def _sweep_direction_is_valid(V, bin_width):
    finite_voltage = V[np.isfinite(V)]
    if finite_voltage.size < _MIN_CURVE_POINTS:
        return False
    voltage_span = np.ptp(finite_voltage)
    if voltage_span < (_MIN_CURVE_POINTS - 1) * bin_width:
        return False

    block_count = min(25, finite_voltage.size // 3)
    block_medians = np.array(
        [np.median(block) for block in np.array_split(finite_voltage, block_count)]
    )
    increments = np.diff(block_medians)
    material = np.abs(increments) >= max(bin_width, 0.01 * voltage_span)
    increments = increments[material]
    if increments.size == 0:
        return False
    forward_travel = np.sum(increments[increments > 0])
    reverse_travel = -np.sum(increments[increments < 0])
    minority_travel = min(forward_travel, reverse_travel)
    return bool(minority_travel <= _MAX_REVERSE_SWEEP_FRACTION * voltage_span)


def _effective_sample_count(values):
    """Estimate independent samples using the initial positive autocorrelation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_values = values.size
    if n_values < 2:
        return float(n_values)

    centered = values - np.mean(values)
    variance_sum = float(np.dot(centered, centered))
    if not np.isfinite(variance_sum) or variance_sum <= 0:
        return float(n_values)

    autocorrelation = np.correlate(centered, centered, mode="full")[
        n_values - 1 :
    ] / variance_sum
    positive_lags = autocorrelation[1:]
    nonpositive = np.flatnonzero(positive_lags <= 0)
    if nonpositive.size:
        positive_lags = positive_lags[: nonpositive[0]]
    correlation_time = 1.0 + 2.0 * float(np.sum(positive_lags))
    if not np.isfinite(correlation_time) or correlation_time <= 0:
        return 1.0
    return float(np.clip(n_values / correlation_time, 1.0, n_values))


def _estimate_ion_current(voltage, current, vf, enforce_flatness=False):
    cutoff = voltage[0] + _ION_WINDOW_FRACTION * (vf - voltage[0])
    mask = np.isfinite(voltage) & np.isfinite(current) & (voltage <= cutoff)
    if np.count_nonzero(mask) < _ION_MIN_POINTS:
        return None, "too few independent points in the ion-saturation region"

    V_ion = voltage[mask]
    I_ion_samples = current[mask]
    ion_current = float(np.median(I_ion_samples))
    slope, intercept = np.polyfit(V_ion, I_ion_samples, 1)
    fitted = slope * V_ion + intercept
    residual = I_ion_samples - fitted
    ion_noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    effective_sample_count = _effective_sample_count(residual)
    median_standard_error = (
        np.sqrt(np.pi / 2.0) * ion_noise / np.sqrt(effective_sample_count)
    )
    trend_span = float(abs(slope) * np.ptp(V_ion))
    ion_magnitude = -ion_current

    if not np.isfinite(ion_current) or ion_current >= 0:
        return (
            None,
            "ion-saturation current is not negative under the configured polarity",
        )
    trend_fraction = trend_span / ion_magnitude
    trend_exceeds_limit = trend_fraction > _SATURATION_MAX_TREND_FRACTION
    if enforce_flatness and trend_exceeds_limit:
        return None, "low-bias current does not form a constant ion-saturation plateau"

    return {
        "current_A": ion_current,
        "noise_A": ion_noise,
        "uncertainty_A": float(median_standard_error),
        "snr": (
            float(ion_magnitude / median_standard_error)
            if median_standard_error > 0
            else np.inf
        ),
        "effective_sample_count": effective_sample_count,
        "slope_A_per_V": float(slope),
        "intercept_A": float(intercept),
        "trend_span_A": trend_span,
        "trend_fraction": trend_fraction,
        "trend_exceeds_limit": trend_exceeds_limit,
        "mask": mask,
    }, None


def _estimate_electron_saturation_current(
    V,
    measured_current,
    electron_current,
    vp,
    ion_noise,
    method="high_bias_median",
    enforce_flatness=False,
):
    """Estimate electron saturation at Vp or from the high-bias region."""
    if method == "at_vp":
        saturation_current = float(np.interp(vp, V, electron_current))
        if not np.isfinite(saturation_current) or saturation_current <= 0:
            return (
                None,
                "electron current at the plasma potential is not finite and positive",
            )
        if saturation_current < _ELECTRON_SIGNAL_TO_NOISE * ion_noise:
            return (
                None,
                "electron current at the plasma potential is not resolved above ion noise",
            )
        return {
            "current_A": saturation_current,
            "noise_A": np.nan,
            "slope_A_per_V": np.nan,
            "trend_span_A": np.nan,
            "trend_fraction": np.nan,
            "trend_exceeds_limit": False,
            "mask": np.zeros(V.shape, dtype=bool),
        }, None

    if method != "high_bias_median":
        raise ValueError(f"Unknown electron-saturation method {method!r}.")

    cutoff = vp + 0.5 * (V[-1] - vp)
    mask = np.isfinite(V) & np.isfinite(measured_current) & (V >= cutoff)
    if np.count_nonzero(mask) < _ION_MIN_POINTS:
        return None, "too few independent points in the electron-saturation region"

    V_sat = V[mask]
    I_sat = measured_current[mask]
    saturation_current = float(np.median(I_sat))
    if not np.isfinite(saturation_current) or saturation_current <= 0:
        return None, "electron saturation current is not finite and positive"
    if saturation_current < _ELECTRON_SIGNAL_TO_NOISE * ion_noise:
        return None, "electron saturation current is not resolved above ion noise"

    slope, intercept = np.polyfit(V_sat, I_sat, 1)
    fitted = slope * V_sat + intercept
    residual = I_sat - fitted
    residual_noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    trend_span = float(abs(slope) * np.ptp(V_sat))
    trend_fraction = trend_span / saturation_current
    trend_exceeds_limit = trend_fraction > _SATURATION_MAX_TREND_FRACTION
    if enforce_flatness and trend_exceeds_limit:
        return None, "high-bias current does not form an electron-saturation plateau"

    return {
        "current_A": saturation_current,
        "noise_A": residual_noise,
        "slope_A_per_V": float(slope),
        "trend_span_A": trend_span,
        "trend_fraction": trend_fraction,
        "trend_exceeds_limit": trend_exceeds_limit,
        "mask": mask,
    }, None


def _fit_electron_temperature(V, electron_current, I_es, vf, vp, ion_noise, cfg):
    signal_floor = max(
        cfg["te_current_floor_frac"] * I_es,
        _ELECTRON_SIGNAL_TO_NOISE * ion_noise,
        np.finfo(float).tiny,
    )
    signal_ceiling = _TE_CURRENT_CEILING_FRACTION * I_es
    eligible = (
        np.isfinite(V)
        & np.isfinite(electron_current)
        & (V >= vf)
        & (V <= vp - cfg["te_margin_from_vp"])
        & (electron_current > 0)
    )
    within_thresholds = (
        eligible
        & (electron_current >= signal_floor)
        & (electron_current <= signal_ceiling)
    )
    threshold_indices = np.flatnonzero(within_thresholds)
    mask = np.zeros(V.shape, dtype=bool)
    if threshold_indices.size:
        mask[threshold_indices[0] : threshold_indices[-1] + 1] = True
        mask &= eligible

    output = {
        "fit_mask": mask,
        "npts": int(np.count_nonzero(mask)),
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "rmse_logI": np.nan,
        "relative_slope_uncertainty": np.nan,
        "log_span": np.nan,
        "curvature_log": np.nan,
        "curvature_detected": False,
        "te_eV_candidate": np.nan,
        "passed_r2": False,
        "valid": False,
        "reason": None,
    }
    if output["npts"] < cfg["te_min_points"]:
        output["reason"] = (
            "too few independent binned samples in the electron-retarding fit region"
        )
        return output

    V_fit = V[mask]
    log_current = np.log(electron_current[mask])
    try:
        slope, intercept = np.polyfit(V_fit, log_current, 1)
    except np.linalg.LinAlgError:
        output["reason"] = "electron-temperature regression failed"
        return output

    fitted = slope * V_fit + intercept
    residual = log_current - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((log_current - np.mean(log_current)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
    rmse = float(np.sqrt(np.mean(residual**2)))
    x_variance_sum = float(np.sum((V_fit - np.mean(V_fit)) ** 2))
    if len(V_fit) > 2 and x_variance_sum > 0:
        slope_se = np.sqrt((ss_res / (len(V_fit) - 2)) / x_variance_sum)
        relative_uncertainty = slope_se / slope if slope > 0 else np.inf
    else:
        relative_uncertainty = np.inf
    te_candidate = 1.0 / slope if slope > 0 else np.nan
    log_span = float(slope * np.ptp(V_fit)) if slope > 0 else np.nan
    curvature_log = np.nan
    curvature_detected = False
    if len(V_fit) >= 5:
        try:
            quadratic, covariance = np.polyfit(V_fit, log_current, 2, cov=True)
            quadratic_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
            curvature_log = float(abs(quadratic[0]) * np.ptp(V_fit) ** 2)
            curvature_detected = bool(
                curvature_log > _TE_MAX_CURVATURE_LOG
                and abs(quadratic[0]) > _TE_CURVATURE_SIGNIFICANCE * quadratic_se
            )
        except np.linalg.LinAlgError:
            curvature_detected = True

    output.update(
        {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": r2,
            "rmse_logI": rmse,
            "relative_slope_uncertainty": float(relative_uncertainty),
            "log_span": log_span,
            "curvature_log": curvature_log,
            "curvature_detected": curvature_detected,
            "te_eV_candidate": float(te_candidate),
            "passed_r2": bool(np.isfinite(r2) and r2 >= cfg["te_min_r2"]),
            "vstart": float(V_fit[0]),
            "vstop": float(V_fit[-1]),
        }
    )

    checks = (
        (
            np.isfinite(slope) and slope > 0,
            "electron-temperature slope is not finite and positive",
        ),
        (
            np.isfinite(te_candidate)
            and cfg["te_min_eV"] <= te_candidate <= cfg["te_max_eV"],
            "electron temperature lies outside the configured physical range",
        ),
        (output["passed_r2"], "electron-temperature fit failed the R^2 threshold"),
        (
            np.isfinite(log_span) and log_span >= _TE_MIN_LOG_SPAN,
            "electron-temperature fit spans less than one e-fold in current",
        ),
        (
            np.isfinite(relative_uncertainty)
            and relative_uncertainty <= _TE_MAX_RELATIVE_SLOPE_UNCERTAINTY,
            "electron-temperature slope uncertainty is too large",
        ),
        (
            not cfg["enforce_ideal_model_checks"] or not curvature_detected,
            "electron-retarding semilog branch has significant curvature",
        ),
    )
    for passed, reason in checks:
        if not passed:
            output["reason"] = reason
            return output

    output["valid"] = True
    return output


def _empty_result(trace_label):
    return {
        "trace_label": trace_label,
        "ok": False,
        "warnings": [],
        "model_notes": [],
        "te_eV": np.nan,
        "vp_V": np.nan,
        "vp_fit_V": np.nan,
        "vp_derivative_V": np.nan,
        "vf_V": np.nan,
        "ies_A": np.nan,
        "ies_method": None,
        "iis_A": np.nan,
        "n_e_m3": np.nan,
        "ion_current_A": np.nan,
        "ion_noise_A": np.nan,
        "ion_current_uncertainty_A": np.nan,
        "ion_current_snr": np.nan,
        "ion_effective_sample_count": np.nan,
        "ion_fit_slope_A_per_V": np.nan,
        "electron_saturation_slope_A_per_V": np.nan,
        "ion_saturation_trend_fraction": np.nan,
        "electron_saturation_trend_fraction": np.nan,
        "electron_leakage_fraction": np.nan,
        "te_fit_r2": np.nan,
        "te_fit_rmse": np.nan,
        "te_fit_npts": 0,
        "te_fit_vstart": np.nan,
        "te_fit_vstop": np.nan,
        "te_fit_slope": np.nan,
        "te_fit_intercept": np.nan,
        "te_fit_relative_uncertainty": np.nan,
        "te_fit_log_span": np.nan,
        "te_fit_curvature_log": np.nan,
        "te_fit_curvature_flagged": False,
        "te_fit_valid": False,
        # Compatibility aliases. I0 is now the measured median ion-region
        # current, not the old artificial positivity shift; one fit is attempted.
        "te_fit_i0": np.nan,
        "te_fit_subtract_i0": True,
        "te_fit_passed_r2": False,
        "te_fit_candidate_count": 0,
        "iv_voltage_grid": None,
        "iv_current_grid": None,
        "iv_didv_grid": None,
        "iv_fit_mask": None,
        "diagnostic_data": None,
    }


def analyze_iv_trace(
    bias_values,
    current_values,
    config=None,
    include_diagnostic_data=False,
    trace_label=None,
):
    """Analyze one trace and return every estimate that could be calculated.

    ``ok`` means that the measured sweep and numerical fit passed the enabled
    quality checks. Experimental deviations from the ideal planar-Maxwellian
    model are recorded in ``model_notes`` unless
    ``enforce_ideal_model_checks`` is true. A quality-rejected trace can still
    contain finite scalar estimates so callers can plot and inspect the fit;
    ``warnings`` explains why it was not accepted. Density may be NaN when
    ``probe_area`` is absent.
    """
    cfg = _validated_config(config)
    result = _empty_result(trace_label)
    result["ies_method"] = cfg["ies_method"]
    bias, current = _validate_trace_arrays(bias_values, current_values)

    if not _sweep_direction_is_valid(bias, cfg["voltage_bin_width"]):
        result["warnings"].append(
            "voltage samples do not contain one dominant sweep direction"
        )
        return result

    iv = make_monotonic_iv_curve(
        bias,
        current,
        npts=cfg["iv_npts"],
        bin_width=cfg["voltage_bin_width"],
    )
    if iv is None:
        result["warnings"].append(
            "could not construct a sufficiently sampled I-V curve"
        )
        return result

    Vb = iv["V_binned"]
    Ib = iv["I_binned"]
    Va = iv["V_analysis"]
    Ia = iv["I_analysis"]
    Vg = iv["V_grid"]
    Ig = iv["I_grid"]
    result["iv_voltage_grid"] = Vg
    result["iv_current_grid"] = Ig
    result["iv_fit_mask"] = np.zeros(Vg.shape, dtype=np.uint8)

    if include_diagnostic_data:
        result["diagnostic_data"] = {
            "V_raw": iv["V_raw"],
            "I_raw": iv["I_raw"],
            "V_binned": Vb,
            "I_binned": Ib,
        }

    if not cfg["current_zero_calibrated"]:
        result["warnings"].append(
            "absolute current zero is not independently calibrated"
        )
        return result

    preliminary_vf = find_floating_potential_from_iv(Vb, Ib)
    if not np.isfinite(preliminary_vf.to_value(u.V)):
        result["warnings"].append("no bracketed negative-to-positive current crossing")
        return result

    Vp, dIdV_analysis = find_plasma_potential_from_iv(
        Va,
        Ia,
        vf=preliminary_vf,
        smoothing=cfg["vp_smoothing"],
        smoothing_width_V=cfg["vp_smoothing_width_V"],
        savgol_order=cfg["vp_savgol_order"],
    )
    result["iv_didv_grid"] = np.interp(Vg, Va, dIdV_analysis)
    if not np.isfinite(Vp.to_value(u.V)):
        result["warnings"].append("no interior plasma-potential derivative peak")
        return result

    Vf = find_floating_potential_from_iv(Vb, Ib, upper_bound=Vp)
    vf_value = Vf.to_value(u.V)
    vp_value = Vp.to_value(u.V)
    if not np.isfinite(vf_value) or not vf_value < vp_value:
        result["warnings"].append(
            "floating and plasma potentials are not physically ordered"
        )
        return result
    result["vf_V"] = vf_value
    result["vp_derivative_V"] = vp_value
    # The derivative estimate is a reportable plasma-potential measurement even
    # if a later Maxwellian consistency check rejects the overall trace.
    result["vp_V"] = vp_value

    ion, ion_error = _estimate_ion_current(
        Vb,
        Ib,
        vf_value,
        enforce_flatness=cfg["enforce_ideal_model_checks"],
    )
    if ion is None:
        result["warnings"].append(ion_error)
        return result
    result["ion_current_A"] = ion["current_A"]
    result["ion_noise_A"] = ion["noise_A"]
    result["ion_current_uncertainty_A"] = ion["uncertainty_A"]
    result["ion_current_snr"] = ion["snr"]
    result["ion_effective_sample_count"] = ion["effective_sample_count"]
    result["ion_fit_slope_A_per_V"] = ion["slope_A_per_V"]
    result["ion_saturation_trend_fraction"] = ion["trend_fraction"]
    result["iis_A"] = -ion["current_A"]
    result["te_fit_i0"] = ion["current_A"]
    if ion["snr"] < cfg["ion_min_snr"]:
        result["warnings"].append(
            "ion-saturation current estimate is not resolved above its uncertainty"
        )
        return result
    if ion["trend_exceeds_limit"]:
        result["model_notes"].append(
            "ion-saturation branch is sloped; median region current was used"
        )

    ion_voltage_max = Vb[ion["mask"]][-1]
    derivative_ion_mask = Va <= ion_voltage_max
    derivative_baseline = np.median(dIdV_analysis[derivative_ion_mask])
    derivative_noise = 1.4826 * np.median(
        np.abs(
            dIdV_analysis[derivative_ion_mask]
            - np.median(dIdV_analysis[derivative_ion_mask])
        )
    )
    derivative_peak = float(np.interp(vp_value, Va, dIdV_analysis))
    if (
        not np.isfinite(derivative_peak)
        or derivative_peak
        <= derivative_baseline + _ELECTRON_SIGNAL_TO_NOISE * derivative_noise
    ):
        result["warnings"].append(
            "plasma-potential derivative peak is not resolved above ion-region noise"
        )
        return result

    supported = Vb >= vf_value
    supported_spacing = np.diff(Vb[supported])
    typical_spacing = np.median(np.diff(Vb))
    maximum_gap = max(
        5 * typical_spacing,
        0.5 * cfg["vp_smoothing_width_V"],
    )
    if supported_spacing.size == 0 or np.max(supported_spacing) > maximum_gap:
        result["warnings"].append(
            "measured voltage bins do not continuously support the electron branch"
        )
        return result

    electron_current = Ib - ion["current_A"]
    electron_saturation, saturation_error = _estimate_electron_saturation_current(
        Vb,
        Ib,
        electron_current,
        vp_value,
        ion["noise_A"],
        method=cfg["ies_method"],
        enforce_flatness=cfg["enforce_ideal_model_checks"],
    )
    if electron_saturation is None:
        result["warnings"].append(saturation_error)
        return result
    I_es = electron_saturation["current_A"]
    result["ies_A"] = I_es
    result["electron_saturation_slope_A_per_V"] = electron_saturation["slope_A_per_V"]
    result["electron_saturation_trend_fraction"] = electron_saturation[
        "trend_fraction"
    ]
    if electron_saturation["trend_exceeds_limit"]:
        result["model_notes"].append(
            "electron-saturation branch is sloped; median region current was used"
        )

    te_fit = _fit_electron_temperature(
        Vb,
        electron_current,
        I_es,
        vf_value,
        vp_value,
        ion["noise_A"],
        cfg,
    )
    result["te_fit_npts"] = te_fit["npts"]
    result["te_fit_r2"] = te_fit["r2"]
    result["te_fit_rmse"] = te_fit["rmse_logI"]
    result["te_fit_slope"] = te_fit["slope"]
    result["te_fit_intercept"] = te_fit["intercept"]
    result["te_fit_vstart"] = te_fit.get("vstart", np.nan)
    result["te_fit_vstop"] = te_fit.get("vstop", np.nan)
    result["te_fit_relative_uncertainty"] = te_fit["relative_slope_uncertainty"]
    result["te_fit_log_span"] = te_fit["log_span"]
    result["te_fit_curvature_log"] = te_fit["curvature_log"]
    result["te_fit_curvature_flagged"] = te_fit["curvature_detected"]
    result["te_fit_passed_r2"] = te_fit["passed_r2"]
    result["te_fit_candidate_count"] = 1 if te_fit["npts"] else 0
    result["te_fit_valid"] = te_fit["valid"]
    if te_fit["curvature_detected"]:
        result["model_notes"].append(
            "electron-retarding semilog branch has significant curvature"
        )
    # Retain the numerical fit result independently of the quality decision.
    # This prevents a failed diagnostic gate from erasing an otherwise finite
    # fit and leaving diagnostic/profile plots entirely blank.
    result["te_eV"] = te_fit["te_eV_candidate"]

    if np.isfinite(te_fit["slope"]) and te_fit["slope"] > 0:
        fitted_vp = (np.log(I_es) - te_fit["intercept"]) / te_fit["slope"]
        if np.isfinite(fitted_vp):
            result["vp_fit_V"] = float(fitted_vp)

    probe_area = cfg["probe_area"]
    if (
        probe_area is not None
        and np.isfinite(result["te_eV"])
        and result["te_eV"] > 0
    ):
        density = density_from_electron_saturation_current(
            I_es,
            result["te_eV"],
            probe_area,
        )
        result["n_e_m3"] = density.to_value(u.m**-3)

    if te_fit["npts"]:
        selected_voltage = Vb[te_fit["fit_mask"]]
        if selected_voltage.size:
            result["iv_fit_mask"] = (
                (Vg >= selected_voltage[0]) & (Vg <= selected_voltage[-1])
            ).astype(np.uint8)

    if np.isfinite(te_fit["slope"]) and np.isfinite(te_fit["intercept"]):
        electron_leakage = np.exp(
            te_fit["slope"] * ion_voltage_max + te_fit["intercept"]
        )
        result["electron_leakage_fraction"] = electron_leakage / result["iis_A"]
        if result["electron_leakage_fraction"] > 0.02:
            message = "negative-bias coverage may include electron current in the ion estimate"
            if cfg["enforce_ideal_model_checks"]:
                result["warnings"].append(
                    "negative-bias coverage is insufficient to isolate ion saturation"
                )
                return result
            result["model_notes"].append(message)

    if not te_fit["valid"]:
        result["warnings"].append(te_fit["reason"])
        return result

    fitted_vp = result["vp_fit_V"]
    vp_tolerance = max(
        2 * cfg["voltage_bin_width"],
        min(0.5 * cfg["vp_smoothing_width_V"], 0.5 * te_fit["te_eV_candidate"]),
    )
    vp_estimators_disagree = (
        not np.isfinite(fitted_vp)
        or not vf_value < fitted_vp < Vb[-1]
        or abs(fitted_vp - vp_value) > vp_tolerance
    )
    if vp_estimators_disagree:
        message = "derivative and retarding/saturation plasma-potential estimates disagree"
        if cfg["enforce_ideal_model_checks"]:
            result["warnings"].append(message)
            return result
        result["model_notes"].append(message)

    # Use the fit extrapolation only when it is consistent with the measured
    # derivative knee; otherwise retain the derivative-based Vp estimate.
    if not vp_estimators_disagree:
        result["vp_V"] = float(fitted_vp)
    if probe_area is None:
        result["warnings"].append(
            "missing probe_area; electron density was not calculated"
        )

    result["ok"] = True
    return result
