This repo contains the code file for medical-images.

## Femur and Tibia Segmentation from Left Knee CT

### Dataset

The input is `3702_left_knee.nii`, a 3D volume of size `512 x 512 x 216`
voxels, with in-plane resolution of ~0.87mm and slice thickness of 2mm. The
modality was not stated in the filename or metadata, so it was determined
directly from the intensity distribution: the histogram shows a large spike
of voxels at exactly -1000 (the Hounsfield Unit value of air) and a hard
floor at -3024 (a padding value used outside the CT reconstruction circle).
Both are specific to **CT** imaging and rule out MRI, whose intensities are
scanner-dependent and always non-negative. This identification determined the
overall approach — on CT, bone is the densest tissue present, so it can be
isolated directly via Hounsfield Unit thresholding, without a trained model.

The volume's orientation (`L`, `P`, `S`, read from the NIfTI affine matrix)
was used to map array axes to anatomical planes: axis 2 corresponds to the
axial (Superior/Inferior) direction, axis 1 to the coronal
(Anterior/Posterior) direction. This mapping is what allows the code to
correctly extract axial and coronal slices from the raw 3D array.

*(insert a figure here: raw axial/coronal slices and the intensity histogram
from `segment.py`)*

### Step 1: Whole-bone segmentation

`segment.py` implements the first stage: threshold all voxels above 200 HU
(above soft tissue, within the cancellous/cortical bone range), then apply
morphological closing, small-object removal, and per-axial-slice hole filling
so each bone is represented as a solid region rather than a hollow shell. The
largest connected components are retained as the combined bone mask.

Initial visual inspection appeared to show an incomplete segmentation, but
this was a display artifact rather than a segmentation error: default
autoscaling in `matplotlib` stretched intensity values across the full
-3024 to 1769 range, reducing contrast between bone and soft tissue. Applying
a standard CT bone window (level 400, width 1800) resolved this — the
segmentation mask matched the visibly dense bone regions in both axial and
coronal views.

*(insert a figure here: before/after bone-window comparison; also reference
`animation.gif` in this folder, which shows the bone mask across the full
slice stack)*

### Step 2: Separating femur from tibia

The second stage required femur and tibia as two distinct labeled
structures, using classical image processing only (no machine learning).
Two issues were identified and resolved during this stage.

**Non-bone artifact.** Selecting the two largest connected components
initially produced components that each spanned the full 216-slice depth of
the volume — anatomically inconsistent, since femur and tibia meet at a
single joint rather than each running the full scan length. Inspection of
individual components showed the cause: a thin diagonal structure located
outside the leg, consistent with a positioning strap or support device on
the CT table, dense enough to exceed the 200 HU threshold. This was
addressed by filtering components on shape rather than size alone, using
*extent* (voxel count divided by bounding-box volume): solid bone occupies a
substantial fraction of its bounding box, while a thin elongated object does
not. This filter removed the artifact.

**Femur-tibia contact.** After removing the artifact, the largest remaining
component was still a single connected object, since the femoral condyles
and tibial plateau are in close contact in this scan. Connected-component
labeling cannot separate objects that are physically touching. This was
resolved by computing the cross-sectional area of the fused component at
every axial slice along its length. The resulting profile is consistent with
knee anatomy: narrow at the tibial shaft, widening toward the tibial
plateau, reaching a local minimum at the joint contact point, widening again
into the femoral condyles, then narrowing into the femoral shaft.
Prominence-based trough detection on this profile locates the joint line
automatically; voxels on either side are labeled femur and tibia
respectively.

*(insert a figure here: coronal view showing femur/tibia split at the joint
line; optionally a short recording of `view_segmentation.py` scrolling
through axial slices with the overlay)*

### Output

The complete pipeline is implemented in `segment_femur_tibia.py`, with
`view_segmentation.py` provided to review the result slice by slice.
Outputs: `3702_left_knee_femur.nii.gz`, `3702_left_knee_tibia.nii.gz`, and a
combined `3702_left_knee_femur_tibia_labels.nii.gz` (0 = background,
1 = femur, 2 = tibia). The full pipeline uses thresholding, morphological
operations, connected-component analysis, and signal-processing peak
detection — no deep learning is involved.
