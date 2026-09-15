"""Shared renderer for interactive views, stills and movies."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from .model import quantity, spectrum

BG, PANEL, FG, MUTED, ACCENT = "#0c1422", "#111e30", "#e7eef8", "#8ea4be", "#46d9c5"
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo",
             "RdBu_r", "coolwarm", "Spectral", "seismic", "gray", "Greys"]
TIME_UNITS = {"s": 1, "ms": 1e3, "µs": 1e6, "ns": 1e9}


def figure():
    return Figure(figsize=(12, 7), dpi=100, facecolor=BG, layout="constrained")


def style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2a3b51")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(alpha=.12, color=MUTED)


def render(fig, data, opts, values=None):
    """Render selected state; return main axes for click-to-slice interaction."""
    fig.clear()
    names, mode = opts["names"], opts["mode"]
    values = quantity(data, names, mode, opts["case"], opts["shot"]) if values is None else values
    spatial = data.spatial_dims
    coords = [data.coords[d] for d in spatial]
    index = tuple(min(opts["slices"][i], len(c)-1) for i, c in enumerate(coords))
    ti = min(opts["time"], len(data.coords["time"])-1)
    scale, unit = TIME_UNITS[opts["time_unit"]], opts["time_unit"]
    t = data.coords["time"] * scale
    trace = values[index] if spatial else values
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, :2])
    trace_ax = fig.add_subplot(gs[1, :2])
    side1, side2 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    for a in (ax, trace_ax, side1, side2):
        style(a)
    title = f"{mode}  ·  t = {t[ti]:.{opts['sigfigs']}g} {unit}"
    ax.set_title(title, loc="left", fontsize=13, pad=12, color=FG)
    if len(spatial) == 2:
        vertical, horizontal = coords
        frame = values[..., ti]
        finite = values[np.isfinite(values)] if opts["lock"] else frame[np.isfinite(frame)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
        if lo == hi:
            lo, hi = lo-.5, hi+.5
        mesh = ax.pcolormesh(horizontal, vertical, frame, cmap=opts["cmap"],
                             shading="nearest", vmin=lo, vmax=hi, rasterized=True)
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")
        cb = fig.colorbar(mesh, ax=ax, pad=.02, fraction=.035)
        cb.set_label(data.units, color=MUTED)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(f"{spatial[1]} ({data.spatial_units})")
        ax.set_ylabel(f"{spatial[0]} ({data.spatial_units})")
        ax.axvline(horizontal[index[1]], color=FG, lw=.8, ls="--", alpha=.8)
        ax.axhline(vertical[index[0]], color=FG, lw=.8, ls="--", alpha=.8)
        if mode == "Vector":
            # Components explicitly assigned to horizontal/vertical directions.
            u = data.selected(names[0], opts["case"], opts["shot"])[..., ti]
            v = data.selected(names[1], opts["case"], opts["shot"])[..., ti]
            step = max(1, max(frame.shape)//18)
            arrow_cmap = opts.get("arrow_cmap", "Solid white")
            arrow_args = dict(alpha=.9, pivot="mid", angles="xy", scale_units="xy")
            if arrow_cmap == "Solid white":
                quiver = ax.quiver(horizontal[::step], vertical[::step], u[::step, ::step], v[::step, ::step],
                                   color="white", **arrow_args)
            else:
                arrow_magnitude = np.hypot(u, v)
                if opts["lock"]:
                    all_magnitudes = np.hypot(data.selected(names[0], opts["case"], opts["shot"]),
                                              data.selected(names[1], opts["case"], opts["shot"]))
                    maximum = float(np.nanmax(all_magnitudes))
                else:
                    maximum = float(np.nanmax(arrow_magnitude))
                arrow_norm = Normalize(0, maximum if np.isfinite(maximum) and maximum > 0 else 1)
                quiver = ax.quiver(horizontal[::step], vertical[::step], u[::step, ::step], v[::step, ::step],
                                   arrow_magnitude[::step, ::step], cmap=arrow_cmap, norm=arrow_norm, **arrow_args)
                arrow_cb = fig.colorbar(quiver, ax=ax, pad=.09, fraction=.035)
                arrow_cb.set_label(f"In-plane magnitude ({data.units})", color=MUTED)
                arrow_cb.ax.tick_params(colors=MUTED, labelsize=8)
        h_slice, = side1.plot(horizontal, frame[index[0], :], color=ACCENT)
        side1.set(title=f"{spatial[1]} slice · {spatial[0]}={vertical[index[0]]:g}",
                  xlabel=f"{spatial[1]} ({data.spatial_units})", ylabel=data.units)
        v_slice, = side2.plot(vertical, frame[:, index[1]], color="#91aaff")
        side2.set(title=f"{spatial[0]} slice · {spatial[1]}={horizontal[index[1]]:g}",
                  xlabel=f"{spatial[0]} ({data.spatial_units})", ylabel=data.units)
    elif len(spatial) == 1:
        spatial_line, = ax.plot(coords[0], values[:, ti], color=ACCENT, lw=2)
        ax.axvline(coords[0][index[0]], color=FG, lw=.8, ls="--")
        ax.set(xlabel=f"{spatial[0]} ({data.spatial_units})", ylabel=data.units)
        if opts["lock"]:
            finite = values[np.isfinite(values)]
            if finite.size:
                lo, hi = finite.min(), finite.max()
                pad = max((hi-lo)*.05, 1e-12)
                ax.set_ylim(lo-pad, hi+pad)
        preview_step = max(1, len(t)//1000)
        side1.pcolormesh(t[::preview_step], coords[0], values[:, ::preview_step], shading="nearest", cmap=opts["cmap"])
        side1.grid(False)
        side1.set(title="Position × time", xlabel=f"Time ({unit})", ylabel=f"{spatial[0]} ({data.spatial_units})")
        overview_cursor = side1.axvline(t[ti], color=FG, lw=.8)
        draw_psd(side2, trace, data)
    else:
        for name in names:
            ax.plot(t, data.selected(name, opts["case"], opts["shot"]), label=name, lw=1.2)
        point_cursor = ax.axvline(t[ti], color=ACCENT, lw=1)
        ax.set(xlabel=f"Time ({unit})", ylabel=data.units)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=FG, edgecolor=PANEL)
        draw_psd(side1, trace, data)
        side2.axis("off")
        finite = trace[np.isfinite(trace)]
        stats = (f"Mean    {finite.mean():.5g}\nRMS     {np.sqrt(np.mean(finite**2)):.5g}\n"
                 f"Peak–peak    {np.ptp(finite):.5g}" if finite.size else "No finite samples")
        side2.text(.06, .9, "TRACE STATISTICS\n\n"+stats+f"\n\nUnits: {data.units}",
                   transform=side2.transAxes, color=FG, va="top", linespacing=1.8)
    trace_ax.plot(t, trace, color=ACCENT, lw=1.4)
    trace_cursor = trace_ax.axvline(t[ti], color="#ffc577", lw=1)
    trace_dot = trace_ax.scatter([t[ti]], [trace[ti]], color="#ffc577", s=20, zorder=3)
    trace_ax.set(title="Time trace at cursor", xlabel=f"Time ({unit})", ylabel=data.units)
    for a in (ax, trace_ax, side1, side2):
        a.xaxis.label.set_color(MUTED)
        a.yaxis.label.set_color(MUTED)
        a.title.set_color(FG)
    def update_frame(frame_index):
        ax.set_title(f"{mode}  ·  t = {t[frame_index]:.{opts['sigfigs']}g} {unit}",
                     loc="left", fontsize=13, pad=12, color=FG)
        trace_cursor.set_xdata([t[frame_index]]*2)
        trace_dot.set_offsets([[t[frame_index], trace[frame_index]]])
        if len(spatial) == 2:
            frame = values[..., frame_index]
            mesh.set_array(frame)
            if not opts["lock"]:
                finite = frame[np.isfinite(frame)]
                if finite.size:
                    lo, hi = finite.min(), finite.max()
                    mesh.set_clim(lo if lo != hi else lo-.5, hi if lo != hi else hi+.5)
            h_slice.set_ydata(frame[index[0], :])
            v_slice.set_ydata(frame[:, index[1]])
            for side in (side1, side2):
                side.relim()
                side.autoscale_view()
            if mode == "Vector":
                u = data.selected(names[0], opts["case"], opts["shot"])[..., frame_index]
                v = data.selected(names[1], opts["case"], opts["shot"])[..., frame_index]
                if arrow_cmap == "Solid white":
                    quiver.set_UVC(u[::step, ::step], v[::step, ::step])
                else:
                    magnitude = np.hypot(u, v)
                    quiver.set_UVC(u[::step, ::step], v[::step, ::step], magnitude[::step, ::step])
                    if not opts["lock"]:
                        maximum = float(np.nanmax(magnitude))
                        quiver.set_clim(0, maximum if np.isfinite(maximum) and maximum > 0 else 1)
        elif len(spatial) == 1:
            spatial_line.set_ydata(values[:, frame_index])
            overview_cursor.set_xdata([t[frame_index]]*2)
            if not opts["lock"]:
                ax.relim()
                ax.autoscale_view()
        else:
            point_cursor.set_xdata([t[frame_index]]*2)
    fig._lapd_update_frame = update_frame
    return ax


def draw_psd(ax, trace, data):
    try:
        f, p = spectrum(trace, data.coords["time"])
        ax.semilogy(f[1:]/1e3, np.maximum(p[1:], np.finfo(float).tiny), color="#91aaff")
        ax.set(title="Welch power spectrum", xlabel="Frequency (kHz)", ylabel=f"{data.units}²/Hz")
    except ValueError as exc:
        ax.text(.1, .5, str(exc), wrap=True, transform=ax.transAxes, color=MUTED)
