"""Vector appearance, data fidelity, and stable scaling during playback/export."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from lapd_explorer import plotting
from lapd_explorer.model import Dataset
from lapd_explorer.vector_style import arrow_style, maximum_magnitude


def plane():
    u = np.broadcast_to([0., 3., 6.], (7, 9, 3)).copy()
    v = np.broadcast_to([0., 4., 8.], (7, 9, 3)).copy()
    return Dataset({"U": u, "V": v}, ("y", "x", "time"),
                   dict(y=np.arange(7.), x=np.arange(9.), time=np.arange(3.)))


def options(style, **kwargs):
    return dict(dict(names=["U", "V"], mode="Vector", case=0, shot=0,
                     slices=[2, 3], time=0, time_unit="s", sigfigs=4, lock=True,
                     cmap="viridis", arrow_cmap="plasma", arrow_style=style), **kwargs)


def test_vector_style_preserves_components_colors_and_fixed_playback_scale():
    data = plane()
    before = {name: a.copy() for name, a in data.channels.items()}
    style = dict(width=3., length=2., stride=3, headwidth=4., headlength=6.,
                 alpha=.6, pivot="tip", outline=.8)
    fig = plotting.figure()
    canvas = FigureCanvasAgg(fig)
    ax = plotting.render(fig, data, options(style))
    arrows = ax.collections[1]
    canvas.draw()  # A zero first frame must still have a finite, stable scale.
    assert arrows.N == 9
    assert arrows.width == pytest.approx(3/72) and arrows.units == "inches"
    assert arrows.headwidth == 4 and arrows.headlength == 6
    assert arrows.get_alpha() == .6 and arrows.pivot == "tip"
    assert arrows.get_linewidths() == pytest.approx([.8])
    assert arrows.scale == pytest.approx(10/(3*.8*2))
    scale = arrows.scale
    fig._lapd_update_frame(2)
    canvas.draw()
    assert ax.collections[1] is arrows and arrows.scale == scale
    np.testing.assert_array_equal(arrows.U, 6.)
    np.testing.assert_array_equal(arrows.V, 8.)
    np.testing.assert_array_equal(arrows.get_array(), 10.)
    assert arrows.get_clim() == (0., 10.)
    # Physical thickness is the same number of points at different export DPI.
    width_pixels = arrows.get_transform().transform((arrows.width, 0))[0]
    fig.set_dpi(fig.dpi*2)
    canvas.draw()
    assert arrows.get_transform().transform((arrows.width, 0))[0] == pytest.approx(width_pixels*2)
    for name in before:
        np.testing.assert_array_equal(data.channels[name], before[name])


def test_length_and_thickness_are_independent_and_unlocked_frames_rescale():
    data = plane()
    fig = plotting.figure()
    FigureCanvasAgg(fig)
    ax = plotting.render(fig, data, options(dict(length=1., width=1.), time=1, lock=False))
    first_scale = ax.collections[1].scale
    ax = plotting.render(fig, data, options(dict(length=2., width=4.), time=1, lock=False))
    arrows = ax.collections[1]
    assert arrows.scale == pytest.approx(first_scale/2)
    assert arrows.width == pytest.approx(4/72)
    np.testing.assert_array_equal(arrows.U, 3.)
    np.testing.assert_array_equal(arrows.get_array(), 5.)
    fig._lapd_update_frame(2)
    assert arrows.scale == pytest.approx(first_scale)
    assert arrows.get_clim() == (0., 10.)


def test_derived_vector_style_and_animation_keep_fixed_reference():
    data = plane()
    frame = Dataset({n: a[..., 1:2] for n, a in data.channels.items()}, data.dims,
                    dict(data.coords, time=np.array([0.])))
    fig = plotting.figure()
    canvas = FigureCanvasAgg(fig)
    ax = plotting.render_derived(fig, frame, "U", vector=True, slices=[2, 3],
                                 limits=(0., 10.), arrow_cmap="cividis",
                                 arrow_settings=dict(width=2., length=1.5, stride=2, pivot="tail"))
    arrows = ax.collections[1]
    canvas.draw()
    assert arrows.N == 20 and arrows.pivot == "tail"
    assert arrows.width == pytest.approx(2/72)
    assert arrows.scale == pytest.approx(10/(2*.8*1.5))
    fig._lapd_update_derived(frame, "Next phase")
    assert ax.collections[1] is arrows
    assert arrows.scale == pytest.approx(10/(2*.8*1.5))
    np.testing.assert_array_equal(arrows.get_array(), 5.)


def test_reference_maximum_uses_bounded_chunks_for_strided_arrays(monkeypatch):
    u = np.arange(100*100*31., dtype=float).reshape(100, 100, 31)[:, :, ::2]
    u[0, 0, 0] = np.nan
    v = np.zeros_like(u)
    original = np.hypot
    sizes = []
    def hypot(a, b):
        sizes.append(a.size)
        return original(a, b)
    monkeypatch.setattr(np, "hypot", hypot)
    assert maximum_magnitude(u, v) == 100*100*31-1
    assert max(sizes) <= 65536 and len(sizes) > 1


@pytest.mark.parametrize("style", [dict(width=0), dict(alpha=np.nan), dict(stride=1.5)])
def test_invalid_vector_style_is_rejected(style):
    with pytest.raises(ValueError, match="Arrow"):
        arrow_style(style)
