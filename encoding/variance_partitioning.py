import importlib
import os
import sys
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from IPython.display import display
from nilearn.surface import vol_to_surf
import cortex
from config import DATA_DIR

sys.path.insert(0, "/Users/ianfaul/Code/HippEncoding/hippomaps")
import hippomaps as hm

def partition_variance(subject, modelname, corr_threshold):
    # =============================================================================
    # 1. Configuration & Paths
    # =============================================================================
    xfmname = f"{subject}_auto"

    DATA_DIR = "/Volumes/Aiolo/deep-fMRI-dataset"
    results_dir = os.path.join(DATA_DIR, "derivatives", "results", modelname)
    pycortex_db = os.path.join(DATA_DIR, "derivatives", "pycortex-db")
    cortex.database.default_filestore = pycortex_db

    mask_path = os.path.join(pycortex_db, subject, "transforms", xfmname, "mask_thick.nii.gz")
    hippunfold_dir = os.path.join(DATA_DIR, "derivatives", "HippUnfold")

    feature_spaces = ["acoustic", "speech", "language"]

    # Define discrete color palette: 0=None, 1=Acoustic, 2=Speech, 3=Language
    feature_colors = ["#E41A1C", "#4DAF4A", "#377EB8"]
    cmap_name = "hipp_winner_cmap"
    cortical_cmap_name = "hipp_winner_volume_cmap"

    try:
        plt.colormaps.register(name=cmap_name, cmap=ListedColormap(feature_colors))
    except ValueError:
        # Already registered in current Python session
        pass

    try:
        plt.colormaps.register(
            name=cortical_cmap_name,
            cmap=ListedColormap(["#00000000", *feature_colors]),
        )
    except ValueError:
        # Already registered in current Python session
        pass

    # Pycortex embeds its own PNG colormap catalog in the viewer HTML; a
    # Matplotlib-only registration leaves the browser without this texture.
    cortical_cmap = plt.get_cmap(cortical_cmap_name)
    cmap_path = os.path.join(
        cortex.options.config.get("webgl", "colormaps"),
        f"{cortical_cmap_name}.png",
    )
    cortex.utils.add_cmap(cortical_cmap, cortical_cmap_name)
    webgl_view = importlib.import_module("cortex.webgl.view")
    webgl_view.colormaps = [
        (name, image) for name, image in webgl_view.colormaps
        if name != cortical_cmap_name
    ]
    webgl_view.colormaps.append(
        (cortical_cmap_name, webgl_view.serve.make_base64(cmap_path))
    )

    # =============================================================================
    # 2. Load Correlations & Compute Cortical Voxel Winners
    # =============================================================================
    print("Loading model correlations...")
    joint_corrs_path = os.path.join(results_dir, "joint", subject, "corrs.npz")
    with np.load(joint_corrs_path) as f:
        joint_corrs = np.squeeze(f["corrs"])

    indiv_corrs = {}
    indiv_r2 = {}

    for space in feature_spaces:
        path = os.path.join(results_dir, space, subject, "corrs.npz")
        with np.load(path) as f:
            r = np.squeeze(f["corrs"])
            indiv_corrs[space] = np.nan_to_num(r, nan=0.0)
            # Fractional variance explained: R^2 = max(0, r)^2
            indiv_r2[space] = np.maximum(0.0, indiv_corrs[space]) ** 2

    # Stack R^2 matrices: shape (3, N_voxels)
    r2_stack = np.vstack([indiv_r2[s] for s in feature_spaces])

    # Winner selection: argmax over feature spaces (0=Acoustic, 1=Speech, 2=Language)
    # Add 1 so labels match: 1=Acoustic, 2=Speech, 3=Language
    voxel_winners = np.argmax(r2_stack, axis=0) + 1

    # Mask voxels where model fit is not statistically meaningful
    significant_mask = (joint_corrs >= corr_threshold) & (np.max(r2_stack, axis=0) > 0)
    voxel_winners[~significant_mask] = 0

    print(f"Cortical Voxel Summary (threshold r >= {corr_threshold}):")
    print(f"  Significant Voxels: {np.sum(significant_mask):,} / {len(joint_corrs):,}")
    for i, space in enumerate(feature_spaces, start=1):
        count = np.sum(voxel_winners == i)
        print(f"  {space.capitalize():8s} Winner: {count:,} voxels ({count / max(1, np.sum(significant_mask)) * 100:.1f}%)")

    # =============================================================================
    # 3. Cortical Surface Visualization (Pycortex)
    # =============================================================================
    print("\nLaunching Pycortex cortical volume viewer...")
    cortex_vol = cortex.Volume(
        voxel_winners.astype(np.float32),
        subject=subject,
        xfmname=xfmname,
        mask="thick",
        vmin=0,
        vmax=3,
        cmap=cortical_cmap_name,
    )

    cortex.webshow(cortex_vol, port=8889, open_browser=False)

    # =============================================================================
    # 4. Generate 3D NIfTI Volumes for Hippocampal Surface Projection
    # =============================================================================
    print("\nReconstructing 3D NIfTI volumes for hippocampal projection...")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)
    # Fortran-order indexing matches Pycortex's transposed NIfTI mask storage.
    flat_mask_indices = np.flatnonzero(mask_data.ravel(order="F"))

    def export_unmasked_nifti(data_1d, out_path):
        vol = np.zeros(mask_data.shape, dtype=np.float32, order="F")
        data_1d = np.asarray(data_1d).reshape(-1)
        if data_1d.size != flat_mask_indices.size:
            raise ValueError(
                f"Expected {flat_mask_indices.size} masked values, got {data_1d.size}"
            )
        vol.ravel(order="F")[flat_mask_indices] = data_1d
        img = nib.Nifti1Image(vol, affine=mask_img.affine, header=mask_img.header)
        nib.save(img, out_path)
        return out_path

    nii_paths = {}
    temp_dir = os.path.join(results_dir, "temp_nifti")
    os.makedirs(temp_dir, exist_ok=True)

    # Export continuous joint correlation and individual R^2 volumes
    nii_paths["joint"] = export_unmasked_nifti(joint_corrs, os.path.join(temp_dir, "joint_corrs.nii.gz"))
    for space in feature_spaces:
        nii_paths[space] = export_unmasked_nifti(indiv_r2[space], os.path.join(temp_dir, f"{space}_r2.nii.gz"))

    # =============================================================================
    # 5. Project onto Hippocampal Surfaces & Compute Surface Winners
    # =============================================================================
    print("Projecting continuous feature volumes onto hippocampal surfaces...")
    den = "0p5mm"
    hemis = ["L", "R"]
    hipp_surf_winners = []

    for hemi in hemis:
        # Subject-specific midthickness mesh
        surf_filename = f"sub-{subject}_hemi-{hemi}_space-T1w_den-{den}_label-hipp_midthickness.surf.gii"
        surf_path = os.path.join(hippunfold_dir, f"sub-{subject}", "surf", surf_filename)

        # 1. Resample joint model correlation (float array)
        surf_joint_r = vol_to_surf(
            img=nii_paths["joint"],
            surf_mesh=surf_path,
            radius=1.0,
            interpolation="nearest",
            kind="line",
            n_samples=5,
        )
        surf_joint_r = np.nan_to_num(surf_joint_r, nan=0.0)

        # 2. Resample individual feature space R^2 volumes
        surf_r2_list = []
        for space in feature_spaces:
            surf_r2 = vol_to_surf(
                img=nii_paths[space],
                surf_mesh=surf_path,
                radius=1.0,
                interpolation="nearest",
                kind="line",
                n_samples=5,
            )
            surf_r2_list.append(np.nan_to_num(surf_r2, nan=0.0))

        surf_r2_stack = np.vstack(surf_r2_list)  # Shape: (3, N_vertices)

        # 3. Determine winning feature space: 1=Acoustic, 2=Speech, 3=Language
        # Cast to float so non-significant vertices can hold np.nan
        hemi_winners = (np.argmax(surf_r2_stack, axis=0) + 1).astype(np.float32)

        # Mask vertices failing threshold as NaN so nan_color renders them
        hemi_sig = (surf_joint_r >= corr_threshold) & (np.max(surf_r2_stack, axis=0) > 0)
        hemi_winners[~hemi_sig] = np.nan

        hipp_surf_winners.append(hemi_winners)

# =============================================================================
# 6. Hippocampus Visualization (HippoMaps Canonical Fold/Unfold)
# =============================================================================
    print("Rendering canonical folded and unfolded hippocampal maps...")

    # Format shape strictly as Vx2xF (V vertices, 2 hemis, 1 feature)
    cdata = np.stack(hipp_surf_winners, axis=1)  # Shape: (V, 2)
    cdata = cdata[:, :, np.newaxis]              # Shape: (V, 2, 1)

    with tempfile.TemporaryDirectory() as temp_dir:
        surface_image_path = os.path.join(temp_dir, "hippocampal_surface.png")
        hm.plotting.surfplot_canonical_foldunfold(
            cdata=cdata,
            den=den,
            hemis=hemis,
            labels=["hipp"],
            unfoldAPrescale=True,
            tighten_cwindow=False,                # Must be False to prevent percentile clipping
            color_range=(1, 3),                   # Replaces vmin/vmax for BrainSpace
            cmap=cmap_name,                       # Registered 3-color map
            nan_color=(0.85, 0.85, 0.85, 1.0),    # Light grey for non-significant/unmasked
            color_bar=None,                       # Suppress continuous color bar
            screenshot=True,
            filename=surface_image_path,
            suppress_warnings=True,
        )
        surface_image = plt.imread(surface_image_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(surface_image)
    ax.axis("off")

    # Custom categorical legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#D3D3D3", label="Not Significant"),
        plt.Rectangle((0, 0), 1, 1, color="#E41A1C", label="Acoustic"),
        plt.Rectangle((0, 0), 1, 1, color="#4DAF4A", label="Speech"),
        plt.Rectangle((0, 0), 1, 1, color="#377EB8", label="Language"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.05),
    )

    out_plot_path = os.path.join(results_dir, f"{subject}_hippocampal_feature_winners.png")
    fig.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    display(fig)
    plt.close(fig)

    print(f"Saved corrected hippocampal winner map to: {out_plot_path}")