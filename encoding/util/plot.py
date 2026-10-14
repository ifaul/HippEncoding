"""Utilities to make the plotting life easier
"""

import nibabel as nib
import numpy as np
from brainspace.mesh.mesh_io import read_surface
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
from neuromaps.transforms import fsaverage_to_fsaverage, fsaverage_to_fslr
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
from surfplot import Plot
from surfplot.utils import threshold as surf_threshold

_image_cache = {}


def upsample_fsaverage(values: np.ndarray, method: str = "linear") -> np.ndarray:
    dataL = values[:40962]
    dataR = values[40962:]
    gifL = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(dataL),))
    gifR = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(dataR),))
    gifLn, gifRn = fsaverage_to_fsaverage(
        (gifL, gifR), "164k", hemi=("L", "R"), method=method
    )
    resampled_data = np.concatenate((gifLn.agg_data(), gifRn.agg_data()))
    return resampled_data


def get_surfplot(
    surface: str = "fsaverage",
    density: str = "41k",
    brightness: float = 0.7,
    sulc_alpha: float = 0.5,
    add_sulc: bool = False,
    surf_lh_fn: str = None,
    surf_rh_fn: str = None,
    **kwargs,
) -> Plot:
    """Get a basic Plot to add layers to."""

    fetch_func = fetch_fsaverage if surface == "fsaverage" else fetch_fslr
    surfaces = fetch_func(data_dir="mats", density=density)
    if surf_lh_fn is None or surf_rh_fn is None:
        surf_lh_fn, surf_rh_fn = surfaces["inflated"]
    sulc_lh_fn, sulc_rh_fn = surfaces["sulc"]

    if surf_lh_fn not in _image_cache:
        _image_cache[surf_lh_fn] = read_surface(str(surf_lh_fn))
    if surf_rh_fn not in _image_cache:
        _image_cache[surf_rh_fn] = read_surface(str(surf_rh_fn))
    if sulc_lh_fn not in _image_cache:
        _image_cache[sulc_lh_fn] = nib.load(str(sulc_lh_fn))
    if sulc_rh_fn not in _image_cache:
        _image_cache[sulc_rh_fn] = nib.load(str(sulc_rh_fn))

    surf_lh = _image_cache[surf_lh_fn]
    surf_rh = _image_cache[surf_rh_fn]
    sulc_lh = _image_cache[sulc_lh_fn]
    sulc_rh = _image_cache[sulc_rh_fn]

    p = Plot(surf_lh=surf_lh, surf_rh=surf_rh, brightness=brightness, **kwargs)
    if add_sulc:
        p.add_layer(
            {"left": sulc_lh, "right": sulc_rh},
            cmap="binary_r",
            cbar=False,
            alpha=sulc_alpha,
        )
    return p


def surface_plot(
    values: np.ndarray,
    title: str = None,
    cmap: str = "coolwarm",
    cbar: bool = True,
    cbar_label: str = None,
    vmin: float = None,
    vmax: float = None,
    threshold: float = None,
    symmetric: bool = True,
    transform: str = None,
    atlas=None,
    atlas_mode: str = "outline",
    fig=None,
    ax=None,
    zeroNan=True,
    **kwargs,
):

    if vmax is None:
        vals = np.abs(values) if symmetric else values
        vmax = np.quantile(vals, 0.995)
    if vmin is None:
        vmin = 0
        if symmetric:
            vmin = -vmax
    elif vmin == "quantile":
        vals = np.abs(values) if symmetric else values
        vmin = np.quantile(vals, 0.75)

    p = get_surfplot(**kwargs)
    if threshold is not None:
        if isinstance(threshold, float):
            threshold = vmin if threshold == "vmin" else threshold
            values = surf_threshold(values, threshold)
        elif isinstance(threshold, np.ndarray):
            values = values.copy()
            values[threshold] = 0

    if transform == "fsaverage_to_fslr":
        n_verts = values.size // 2
        gifL = GiftiImage(darrays=(GiftiDataArray(values[:n_verts]),))
        gifR = GiftiImage(darrays=(GiftiDataArray(values[n_verts:]),))
        gifL, gifR = fsaverage_to_fslr((gifL, gifR))
        values = {"left": gifL, "right": gifR}

    p.add_layer(
        values,
        cmap=cmap,
        cbar=cbar,
        cbar_label=cbar_label,
        color_range=(vmin, vmax),
        zero_transparent=zeroNan,
    )

    if atlas is not None and atlas_mode == "outline":
        parc_mask = atlas
        if not isinstance(atlas, np.ndarray):
            parc_mask = atlas.label_img
        p.add_layer(parc_mask, cmap="gray", as_outline=True, cbar=False)

    if fig is None and ax is None:
        fig = p.build()
        if title is not None:
            fig.suptitle(title)
    else:
        # copied from source code of Plot.build() so i can use my own fig/ax.
        plotter = p.render()
        plotter._check_offscreen()
        x = plotter.to_numpy(transparent_bg=True, scale=(2, 2))

        if ax is None:
            figsize = tuple((np.array(p.size) / 100) + 1)
            ax = fig.subplots(figsize=figsize)

        ax.imshow(x)
        ax.axis("off")

        if cbar:
            p._add_colorbars(fig=fig, ax=ax)

        if title is not None:
            ax.set_title(title)

    return fig
