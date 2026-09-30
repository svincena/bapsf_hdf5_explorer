"""Spatial neighborhood filters that never mix time, case, or repeat axes."""
import operator
import numpy as np
from scipy.ndimage import (convolve, gaussian_filter, uniform_filter,
                           median_filter, maximum_filter, generic_filter)
from .smoothing import _integer


DEFAULT_SPATIAL_AVERAGING = dict(method="box", window_size=(3, 3), sigma=(1., 1.),
                                 radius=1, mode="reflect", nan_policy="propagate")


def _pair(value, name):
    values = (value, value) if np.isscalar(value) else tuple(value)
    if len(values) != 2:
        raise ValueError(f"{name} must be a scalar or a pair in spatial-axis order.")
    return values


def average_spatial(A, method="box", *, spatial_axes=(0, 1), window_size=(3, 3),
                    sigma=(1., 1.), radius=1, mode="reflect", nan_policy="propagate"):
    """Filter each 2D plane, retaining shape and leaving the input untouched.

    Widths are grid points, not physical distances; irregular coordinate grids
    therefore have variable physical smoothing widths. Box/median windows are
    odd; Gaussian support extends to round(4*sigma) on each axis. Disk radius is
    in grid points. Reflect, nearest, and periodic wrap boundaries are supported.
    Nonfinite values are missing: propagate masks the entire neighborhood;
    omit uses only finite neighbors (renormalizing averages) and can fill gaps.
    Neighborhoods with no finite samples remain NaN. Median is a robust filter,
    not an arithmetic average.
    """
    a = np.asarray(A)
    if a.ndim < 2 or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise ValueError("Spatial averaging requires a real numeric array with at least two dimensions.")
    axes = tuple(operator.index(axis) for axis in spatial_axes)
    if len(axes) != 2 or any(not -a.ndim <= axis < a.ndim for axis in axes):
        raise ValueError("Supply two valid spatial axes.")
    axes = tuple(axis % a.ndim for axis in axes)
    if axes[0] == axes[1]:
        raise ValueError("Spatial axes must be distinct.")
    if any(n == 0 for n in a.shape):
        raise ValueError("Spatial averaging requires nonempty data.")
    if mode not in {"reflect", "nearest", "wrap"}:
        raise ValueError("Boundary mode must be reflect, nearest, or wrap.")
    if nan_policy not in {"propagate", "omit"}:
        raise ValueError("Missing-value policy must be propagate or omit.")
    b = np.moveaxis(np.asarray(a, dtype=float), axes, (0, 1))
    extra = (1,) * (b.ndim - 2)
    if method in {"box", "median"}:
        widths = tuple(_integer(v, "Window size", 1) for v in _pair(window_size, "Window size"))
        if any(v % 2 == 0 for v in widths):
            raise ValueError("Spatial windows must be odd on both axes.")
        size = widths + extra
        footprint = None
        def linear(values):
            return uniform_filter(values, size=size, mode=mode)
    elif method == "gaussian":
        sigmas = tuple(float(v) for v in _pair(sigma, "Gaussian sigma"))
        if any(not np.isfinite(v) or v <= 0 for v in sigmas):
            raise ValueError("Gaussian sigma must be finite and positive on both axes.")
        size = tuple(2 * int(4*v + .5) + 1 for v in sigmas) + extra
        footprint = None
        def linear(values):
            return gaussian_filter(values, sigma=sigmas + (0.,) * (b.ndim - 2), mode=mode)
    elif method == "disk":
        radius = _integer(radius, "Disk radius", 1)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        footprint = (x*x + y*y <= radius*radius).reshape((2*radius+1, 2*radius+1) + extra)
        size = None
        weights = footprint / footprint.sum()
        def linear(values):
            return convolve(values, weights, mode=mode)
    else:
        raise ValueError("Unknown spatial method; use box, gaussian, disk, or median.")
    valid = np.isfinite(b)
    clean = np.where(valid, b, 0.)
    if method == "median":
        if nan_policy == "omit" and not valid.all():
            def finite_median(values):
                finite = values[np.isfinite(values)]
                return np.median(finite) if finite.size else np.nan
            result = generic_filter(np.where(valid, b, np.nan), finite_median, size=size, mode=mode)
        else:
            result = median_filter(clean, size=size, mode=mode)
    else:
        result = linear(clean)
        if nan_policy == "omit" and not valid.all():
            weight = linear(valid.astype(float))
            # Running sums may leave tiny roundoff where support is empty.
            supported = maximum_filter(valid, size=size, footprint=footprint, mode=mode)
            result = np.divide(result, weight, out=np.full_like(result, np.nan),
                               where=supported & (weight > 0))
    if nan_policy == "propagate" and not valid.all():
        affected = maximum_filter(~valid, size=size, footprint=footprint, mode=mode)
        result[affected] = np.nan
    return np.moveaxis(result, (0, 1), axes)
