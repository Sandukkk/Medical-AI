import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

CT_PATH = "3702_left_knee.nii"
LABELS_PATH = "3702_left_knee_femur_tibia_labels.nii.gz"  # 0=background, 1=femur, 2=tibia

VMIN, VMAX = -500, 1300  # CT bone window
CMAP = ListedColormap(["none", "red", "blue"])


def get_slice(vol, axis, index):
    idx = [slice(None)] * 3
    idx[axis] = index
    return vol[tuple(idx)]


def slice_viewer(volume, labels, axis, title):
    n_slices = volume.shape[axis]
    idx0 = n_slices // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)

    img_slice = get_slice(volume, axis, idx0)
    lbl_slice = get_slice(labels, axis, idx0)

    im = ax.imshow(img_slice.T, cmap="gray", origin="lower", vmin=VMIN, vmax=VMAX)
    overlay = ax.imshow(np.ma.masked_where(lbl_slice.T == 0, lbl_slice.T),
                         cmap=CMAP, alpha=0.6, origin="lower", vmin=0, vmax=2)
    ax.set_title(f"{title} — slice {idx0}/{n_slices - 1}  (red=femur, blue=tibia)")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n_slices - 1, valinit=idx0, valstep=1)

    def update(val):
        i = int(slider.val)
        img_slice = get_slice(volume, axis, i)
        lbl_slice = get_slice(labels, axis, i)
        im.set_data(img_slice.T)
        overlay.set_data(np.ma.masked_where(lbl_slice.T == 0, lbl_slice.T))
        ax.set_title(f"{title} — slice {i}/{n_slices - 1}  (red=femur, blue=tibia)")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    img = nib.load(CT_PATH)
    data = img.get_fdata()
    labels = nib.load(LABELS_PATH).get_fdata().astype(int)
    affine = img.affine

    axcodes = nib.aff2axcodes(affine)
    si_axis = next(i for i, c in enumerate(axcodes) if c in ("S", "I"))
    ap_axis = next(i for i, c in enumerate(axcodes) if c in ("A", "P"))

    print("Close each window to move to the next view.")
    slice_viewer(data, labels, si_axis, "Axial")
    slice_viewer(data, labels, ap_axis, "Coronal")


if __name__ == "__main__":
    main()
