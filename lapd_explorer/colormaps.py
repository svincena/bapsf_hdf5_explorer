"""Shared scientific palettes for mesh, arrow, and analysis colorbars."""

# Keep viridis first so existing defaults remain unchanged. Pair each palette
# with its reverse, and retain the names used in saved plotting options.
COLORMAP_GROUPS = {
    "Perceptually uniform sequential": ("viridis", "plasma", "inferno", "magma", "cividis"),
    "Sequential": ("Blues", "Greens", "Oranges", "Reds", "Purples", "YlGnBu", "YlOrRd", "BuGn", "PuBuGn", "hot", "afmhot", "copper"),
    "Diverging": ("RdBu", "coolwarm", "seismic", "BrBG", "PiYG", "PuOr", "PRGn", "RdYlBu", "RdYlGn", "Spectral"),
    "Cyclic": ("twilight", "twilight_shifted", "hsv"),
    "Grayscale": ("gray", "Greys", "bone", "binary"),
    "Rainbow": ("turbo", "jet"),
}
COLORMAPS = [variant for names in COLORMAP_GROUPS.values() for name in names
             for variant in (name, name + "_r")]
COLORMAP_TOOLTIPS = {}
for group, names in COLORMAP_GROUPS.items():
    usage = {
        "Perceptually uniform sequential": "Ordered magnitudes and power; even changes in perceived lightness.",
        "Sequential": "Ordered magnitudes and power.",
        "Diverging": "Signed values and deviations around a meaningful midpoint; set symmetric limits when appropriate.",
        "Cyclic": "Wrapped phase and angles; use a full-cycle color range.",
        "Grayscale": "Monochrome displays and printing.",
        "Rainbow": "Familiar multicolor display; uneven lightness can emphasize artificial boundaries.",
    }[group]
    for name in names:
        for variant in (name, name + "_r"):
            COLORMAP_TOOLTIPS[variant] = f"{variant} · {group}\n{usage}" + ("\nReversed color order." if variant.endswith("_r") else "")
