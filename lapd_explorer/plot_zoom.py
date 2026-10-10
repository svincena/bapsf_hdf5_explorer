"""Retain user navigation when a plot is rebuilt for another stored trace."""
import numpy as np


class PlotZoom:
    def __init__(self):
        self.clear()

    def clear(self):
        self.context = None
        self.defaults = []
        self.limits = []

    @staticmethod
    def axes(fig):
        return [ax for ax in fig.axes if ax.get_label() != "<colorbar>" and ax.has_data()]

    def capture(self, fig, context):
        if context != self.context:
            self.clear()
            self.context = context
            return
        axes = self.axes(fig)
        if len(axes) != len(self.defaults):
            return  # Loading/empty placeholders must not erase a retained view.
        self.limits = []
        for ax, defaults in zip(axes, self.defaults):
            view = {}
            for dim in ("x", "y"):
                limits = getattr(ax, f"get_{dim}lim")()
                if not np.allclose(limits, defaults[dim], rtol=1e-9, atol=0):
                    view[dim] = limits
            self.limits.append(view)

    def restore(self, fig, toolbar):
        axes = self.axes(fig)
        self.defaults = [dict(x=ax.get_xlim(), y=ax.get_ylim()) for ax in axes]
        toolbar.update()
        toolbar.push_current()  # Home always restores the new trace's full view.
        restored = False
        if len(axes) == len(self.limits):
            for ax, view in zip(axes, self.limits):
                for dim, limits in view.items():
                    getattr(ax, f"set_{dim}lim")(*limits, emit=False)
                    restored = True
        if restored:
            toolbar.push_current()

    def updated(self, fig):
        # An in-place animation may autoscale its slice plots. Those new
        # automatic bounds are defaults, not a zoom selected by the user.
        for ax, defaults, view in zip(self.axes(fig), self.defaults, self.limits):
            for dim in ("x", "y"):
                if dim not in view:
                    defaults[dim] = getattr(ax, f"get_{dim}lim")()
