import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import morphology

INPUT_PATH = "/Users/sandhyabaral/Desktop/workspace/Projects/Medical-Image-Task---Computer-Vision/Segmentaion task/3702_left_knee.nii"

BONE_HU_THRESHOLD = 200      # HU value cutoff separating bone from soft tissue
MORPH_CLOSE_RADIUS = 2       # closes small gaps in the cortical shell
MIN_OBJECT_SIZE = 500        # drops noise-sized speckles
MIN_EXTENT = 0.10            # voxel_count / bbox_volume; drops thin external
                              # objects (positioning straps, table edge) which
                              # are long but sparse, unlike solid bone
FUSED_SPAN_FRACTION = 0.7    # if the largest bone component spans more than
                              # this fraction of the volume's depth, femur and
                              # tibia are touching and must be split

OUTPUT_DIR = os.path.dirname(INPUT_PATH)
BASENAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]


def get_slice(vol, axis, index):
    idx = [slice(None)] * 3
    idx[axis] = index
    return vol[tuple(idx)]


def component_extent(mask, bbox_slice):
    bbox_shape = tuple(s.stop - s.start for s in bbox_slice)
    bbox_vol = np.prod(bbox_shape)
    return mask.sum() / bbox_vol


def split_fused_bone(fused_mask, si_axis, superior_sign):
    """Split a single femur+tibia blob at the joint by finding the narrowest
    cross-section (the bone-on-bone contact bridge) between the wide femoral
    condyle and tibial plateau regions, using prominence-based trough
    detection on the cross-sectional area profile."""
    other_axes = tuple(a for a in range(3) if a != si_axis)
    areas = fused_mask.sum(axis=other_axes).astype(float)

    n_slices = len(areas)
    margin = int(n_slices * 0.15)
    search_region = areas[margin:n_slices - margin]

    peaks, props = find_peaks(-search_region, prominence=200)
    if len(peaks) == 0:
        raise RuntimeError("Could not locate a joint constriction to split femur/tibia. "
                            "Try adjusting BONE_HU_THRESHOLD.")
    waist_slice = peaks[np.argmax(props["prominences"])] + margin

    # Build boolean masks via broadcasting along si_axis
    shape = [1, 1, 1]
    shape[si_axis] = n_slices
    slice_idx = np.arange(n_slices).reshape(shape)

    if superior_sign == 1:
        femur_mask = fused_mask & (slice_idx > waist_slice)
        tibia_mask = fused_mask & (slice_idx <= waist_slice)
    else:
        femur_mask = fused_mask & (slice_idx < waist_slice)
        tibia_mask = fused_mask & (slice_idx >= waist_slice)

    return femur_mask, tibia_mask, waist_slice


def main():
    img = nib.load(INPUT_PATH)
    data = img.get_fdata()
    affine = img.affine

    axcodes = nib.aff2axcodes(affine)
    si_axis, si_code = next((i, c) for i, c in enumerate(axcodes) if c in ("S", "I"))
    superior_sign = 1 if si_code == "S" else -1

    # --- Threshold bone by Hounsfield Unit (CT only) ---
    bone_mask = data > BONE_HU_THRESHOLD

    # --- Morphological cleanup ---
    bone_mask = morphology.binary_closing(bone_mask, morphology.ball(MORPH_CLOSE_RADIUS))
    bone_mask = morphology.remove_small_objects(bone_mask, min_size=MIN_OBJECT_SIZE)

    # Fill the marrow cavity per axial slice so each bone is solid, not hollow
    for i in range(bone_mask.shape[si_axis]):
        idx = [slice(None)] * 3
        idx[si_axis] = i
        idx = tuple(idx)
        bone_mask[idx] = ndimage.binary_fill_holes(bone_mask[idx])

    # --- Connected components, then drop non-bone artifacts ---
    # Positioning straps / table edges threshold above 200 HU too, but they are
    # thin and sparse relative to their bounding box, unlike solid bone.
    labeled, num_components = ndimage.label(bone_mask, structure=np.ones((3, 3, 3)))
    if num_components < 1:
        raise RuntimeError("No bone-like structures found above the HU threshold.")

    objs = ndimage.find_objects(labeled)
    sizes = ndimage.sum(bone_mask, labeled, range(1, num_components + 1))

    bone_labels = []
    for lbl in range(1, num_components + 1):
        comp_mask = labeled == lbl
        extent = component_extent(comp_mask, objs[lbl - 1])
        if extent > MIN_EXTENT:
            bone_labels.append(lbl)

    if len(bone_labels) < 1:
        raise RuntimeError("All components were filtered out as non-bone artifacts. "
                            "Lower MIN_EXTENT and re-check visually.")

    bone_labels.sort(key=lambda l: sizes[l - 1], reverse=True)

    largest_label = bone_labels[0]
    largest_mask = labeled == largest_label
    z_indices = np.where(largest_mask)[si_axis]
    z_span_fraction = (z_indices.max() - z_indices.min() + 1) / bone_mask.shape[si_axis]

    if z_span_fraction >= FUSED_SPAN_FRACTION:
        # Femur and tibia are touching at the joint -> split the single blob
        print(f"Largest bone component spans {z_span_fraction:.0%} of the volume depth "
              f"-> treating as fused femur+tibia, splitting at the joint.")
        femur_mask, tibia_mask, waist_slice = split_fused_bone(largest_mask, si_axis, superior_sign)
        print(f"Joint (waist) located at slice index {waist_slice} along the S/I axis.")
    else:
        # Femur and tibia are already separate components -> take the two largest
        if len(bone_labels) < 2:
            raise RuntimeError("Only one bone component found and it doesn't span enough "
                                "of the volume to be a fused femur+tibia. Check BONE_HU_THRESHOLD.")
        top_two = bone_labels[:2]
        centroids = ndimage.center_of_mass(bone_mask, labeled, top_two)
        scores = [c[si_axis] * superior_sign for c in centroids]
        if scores[0] > scores[1]:
            femur_label, tibia_label = top_two[0], top_two[1]
        else:
            femur_label, tibia_label = top_two[1], top_two[0]
        femur_mask = labeled == femur_label
        tibia_mask = labeled == tibia_label

    print(f"Femur voxels: {femur_mask.sum()}")
    print(f"Tibia voxels: {tibia_mask.sum()}")

    # --- Save outputs ---
    combined = np.zeros(bone_mask.shape, dtype=np.uint8)
    combined[femur_mask] = 1
    combined[tibia_mask] = 2

    nib.save(nib.Nifti1Image(combined, affine),
              os.path.join(OUTPUT_DIR, f"{BASENAME}_femur_tibia_labels.nii.gz"))
    nib.save(nib.Nifti1Image(femur_mask.astype(np.uint8), affine),
              os.path.join(OUTPUT_DIR, f"{BASENAME}_femur.nii.gz"))
    nib.save(nib.Nifti1Image(tibia_mask.astype(np.uint8), affine),
              os.path.join(OUTPUT_DIR, f"{BASENAME}_tibia.nii.gz"))

    print("Saved:")
    print(f"  {BASENAME}_femur.nii.gz")
    print(f"  {BASENAME}_tibia.nii.gz")
    print(f"  {BASENAME}_femur_tibia_labels.nii.gz  (0=background, 1=femur, 2=tibia)")


if __name__ == "__main__":
    main()
