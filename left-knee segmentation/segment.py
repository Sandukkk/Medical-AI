import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology

path = "/Users/sandhyabaral/Desktop/workspace/Projects/left-knee segmentation/3702_left_knee.nii"  

img = nib.load(path)
data = img.get_fdata()
header = img.header
affine = img.affine

print("=" * 60)
print("SHAPE & DATA TYPE")
print("=" * 60)
print(f"Shape (voxels):        {data.shape}")
print(f"On-disk dtype:         {header.get_data_dtype()}")
print(f"In-memory dtype:       {data.dtype}")

print("\n" + "=" * 60)
print("SPACING & ORIENTATION")
print("=" * 60)
print(f"Voxel spacing (mm):    {header.get_zooms()}")
print(f"Affine matrix:\n{affine}")
print(f"Orientation (axcodes): {nib.aff2axcodes(affine)}")
# axcodes tell you which array axis is L/R, A/P, S/I —
# e.g. ('R','A','S') means axis0=sagittal, axis1=coronal, axis2=axial

print("\n" + "=" * 60)
print("INTENSITY STATS (helps identify modality: CT vs MRI)")
print("=" * 60)
print(f"Min:                   {np.min(data):.2f}")
print(f"Max:                   {np.max(data):.2f}")
print(f"Mean:                  {np.mean(data):.2f}")
print(f"Std:                   {np.std(data):.2f}")
percentiles = [1, 5, 25, 50, 75, 95, 99]
for p in percentiles:
    print(f"  {p}th percentile:     {np.percentile(data, p):.2f}")
print(f"NaNs present:          {np.isnan(data).any()}")
print(f"Negative values present: {(data < 0).any()}  "
      f"(CT air/background is usually ~ -1000; MRI is usually >= 0)")

print("\n" + "=" * 60)
print("HEADER EXTRAS (scaling, units, description)")
print("=" * 60)
print(f"Scl slope/inter:       {header['scl_slope']}, {header['scl_inter']}")
print(f"xyz units:             {header.get_xyzt_units()}")
print(f"Descrip field:         {header['descrip']}")

# Histogram of intensities
plt.figure(figsize=(6, 4))
plt.hist(data.flatten(), bins=100)
plt.title("Intensity histogram (whole volume)")
plt.xlabel("Voxel value")
plt.ylabel("Count")
plt.show()

# Standard radiological convention: axial = along Superior-Inferior (S) axis,
# coronal = along Anterior-Posterior (A) axis.
axcodes = nib.aff2axcodes(affine)
axis_map = {code[-1]: i for i, code in enumerate(axcodes)}  # last letter: S,P,R etc
# axcodes gives one of R/L, A/P, S/I per axis
si_axis = next(i for i, c in enumerate(axcodes) if c in ("S", "I"))
ap_axis = next(i for i, c in enumerate(axcodes) if c in ("A", "P"))
print(f"\nAxial slices vary along array axis {si_axis} (size {data.shape[si_axis]})")
print(f"Coronal slices vary along array axis {ap_axis} (size {data.shape[ap_axis]})")

# Show a middle slice from each view
def get_slice(vol, axis, index):
    idx = [slice(None)] * 3
    idx[axis] = index
    return vol[tuple(idx)]

axial_mid = get_slice(data, si_axis, data.shape[si_axis] // 2)
coronal_mid = get_slice(data, ap_axis, data.shape[ap_axis] // 2)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(axial_mid.T, cmap="gray", origin="lower")
axes[0].set_title("Axial (mid slice)")
axes[1].imshow(coronal_mid.T, cmap="gray", origin="lower")
axes[1].set_title("Coronal (mid slice)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# BONE SEGMENTATION (HU thresholding + morphological cleanup)
BONE_HU_THRESHOLD = 200  # cancellous bone starts ~150-200 HU, cortical much higher

bone_mask = data > BONE_HU_THRESHOLD

# Remove small noise (vessels, calcifications outside bone) and close gaps
# in cortical rim so marrow cavities get filled.
bone_mask = morphology.binary_closing(bone_mask, morphology.ball(2))
bone_mask = morphology.remove_small_objects(bone_mask, min_size=500)

# Fill holes per-axial-slice so trabecular/marrow interior is included
# inside the cortical shell.
for i in range(bone_mask.shape[si_axis]):
    idx = [slice(None)] * 3
    idx[si_axis] = i
    idx = tuple(idx)
    bone_mask[idx] = ndimage.binary_fill_holes(bone_mask[idx])

# Keep only the largest connected components (femur, tibia, patella, fibula).
labeled, num_components = ndimage.label(bone_mask)
if num_components > 0:
    sizes = ndimage.sum(bone_mask, labeled, range(1, num_components + 1))
    keep_n = min(4, num_components)
    largest_labels = np.argsort(sizes)[::-1][:keep_n] + 1
    bone_mask = np.isin(labeled, largest_labels)

print(f"\nBone voxels: {bone_mask.sum()} / {bone_mask.size} "
      f"({100 * bone_mask.sum() / bone_mask.size:.2f}%)")

# Overlay bone mask on axial and coronal mid slices
axial_mask_mid = get_slice(bone_mask, si_axis, data.shape[si_axis] // 2)
coronal_mask_mid = get_slice(bone_mask, ap_axis, data.shape[ap_axis] // 2)

# CT bone window (level 400, width 1800) so bone/soft-tissue contrast is
# actually visible — plain autoscale gets washed out by the -3024 padding value.
BONE_VMIN, BONE_VMAX = 400 - 900, 400 + 900

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(axial_mid.T, cmap="gray", origin="lower", vmin=BONE_VMIN, vmax=BONE_VMAX)
axes[0].imshow(np.ma.masked_where(~axial_mask_mid.T, axial_mask_mid.T),
               cmap="autumn", alpha=0.5, origin="lower")
axes[0].set_title("Axial — bone segmentation")
axes[1].imshow(coronal_mid.T, cmap="gray", origin="lower", vmin=BONE_VMIN, vmax=BONE_VMAX)
axes[1].imshow(np.ma.masked_where(~coronal_mask_mid.T, coronal_mask_mid.T),
               cmap="autumn", alpha=0.5, origin="lower")
axes[1].set_title("Coronal — bone segmentation")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Save the mask as its own .nii for use elsewhere (e.g. 3D viewers, further analysis)
mask_img = nib.Nifti1Image(bone_mask.astype(np.uint8), affine)
nib.save(mask_img, path.replace(".nii", "_bone_mask.nii"))
print(f"Saved bone mask to {path.replace('.nii', '_bone_mask.nii')}")

# bone mask overlaid, instead of only seeing the one mid-slice above.
from matplotlib.widgets import Slider

def slice_viewer(volume, mask, axis, title, vmin=BONE_VMIN, vmax=BONE_VMAX):
    n_slices = volume.shape[axis]
    idx0 = n_slices // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)

    img_slice = get_slice(volume, axis, idx0)
    mask_slice = get_slice(mask, axis, idx0)

    im = ax.imshow(img_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    overlay = ax.imshow(np.ma.masked_where(~mask_slice.T, mask_slice.T),
                         cmap="autumn", alpha=0.5, origin="lower")
    ax.set_title(f"{title} — slice {idx0}/{n_slices - 1}")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n_slices - 1, valinit=idx0, valstep=1)

    def update(val):
        i = int(slider.val)
        img_slice = get_slice(volume, axis, i)
        mask_slice = get_slice(mask, axis, i)
        im.set_data(img_slice.T)
        overlay.set_data(np.ma.masked_where(~mask_slice.T, mask_slice.T))
        ax.set_title(f"{title} — slice {i}/{n_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# Uncomment to browse slice-by-slice (drag the slider at the bottom of the window):
# slice_viewer(data, bone_mask, si_axis, "Axial")
# slice_viewer(data, bone_mask, ap_axis, "Coronal")
