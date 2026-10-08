"""Shared vector display settings, independent of data and Qt."""
import numpy as np


DEFAULT_ARROW_STYLE = dict(width=1.5, length=1., stride=0, headwidth=3.,
                           headlength=5., alpha=.95, pivot="mid", outline=.3)


def arrow_style(settings=None):
    style = DEFAULT_ARROW_STYLE | (settings or {})
    for name in ("width", "length", "headwidth", "headlength"):
        if not np.isfinite(style[name]) or style[name] <= 0:
            raise ValueError(f"Arrow {name} must be finite and positive.")
    if not np.isfinite(style["alpha"]) or not 0 <= style["alpha"] <= 1:
        raise ValueError("Arrow opacity must be between zero and one.")
    if not np.isfinite(style["outline"]) or style["outline"] < 0:
        raise ValueError("Arrow outline thickness must be finite and nonnegative.")
    if not np.isfinite(style["stride"]) or style["stride"] < 0 or int(style["stride"]) != style["stride"]:
        raise ValueError("Arrow grid stride must be a nonnegative integer.")
    if style["pivot"] not in ("tail", "mid", "tip"):
        raise ValueError("Unknown arrow pivot.")
    return style


def quiver_kwargs(style):
    # Fixed physical width keeps shaft and outline sizes consistent at export
    # resolutions. Length remains independent of thickness and head geometry.
    return dict(units="inches", width=style["width"]/72, alpha=style["alpha"],
                pivot=style["pivot"], angles="xy", scale_units="xy",
                headwidth=style["headwidth"], headlength=style["headlength"],
                headaxislength=.9*style["headlength"],
                edgecolors="#172334", linewidths=style["outline"])


def arrow_stride(shape, style):
    return int(style["stride"]) or max(1, max(shape)//18)


def arrow_scale(x, y, maximum, step, style):
    spacing = min((np.min(np.diff(c)) for c in (x, y) if len(c) > 1), default=1.)
    # A stride larger than the entire grid leaves a single arrow, and should
    # not give it a reference length thousands of times wider than the plot.
    step = min(step, max(len(x)-1, len(y)-1, 1))
    return maximum/(spacing*step*.8*style["length"]) if maximum > 0 else 1.


def maximum_magnitude(u, v):
    """Find an in-plane reference without allocating a full-size magnitude."""
    maximum = 0.
    chunks = np.nditer([u, v], flags=["external_loop", "buffered", "zerosize_ok"],
                       op_flags=[["readonly"], ["readonly"]], buffersize=65536)
    for a, b in chunks:
        values = np.hypot(a, b)
        finite = values[np.isfinite(values)]
        if finite.size:
            maximum = max(maximum, float(finite.max()))
    return maximum
