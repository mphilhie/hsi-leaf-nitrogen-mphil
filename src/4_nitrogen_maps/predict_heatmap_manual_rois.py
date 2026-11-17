# hs_sam_predict_heatmap_manualROI.py
# - Draw MULTIPLE manual ROIs interactively
# - Mean of all ROIs used as reference for SAM
# - Predict only on SAM mask
# - Save SAM image, mask overlay, and nitrogen heatmap (.png) + optional .npy

# -------- Backend BEFORE pyplot import (TkAgg for on-screen) --------
import matplotlib as mpl
mpl.use('TkAgg')
# --------------------------------------------------------------------

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from scipy import ndimage as ndi
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath

# ========================= USER SETTINGS =========================
OUTPUT_BASE               = "2)Blue_Berry/Back4/40_A"
PREFIXES                  = ("n2suz_120d_2025-06-10_004",)
MODEL_PATH                = "./Raw_BEST.joblib"

PREDICTION_NAME           = "Nitrogen (%)"
HEATMAP_VMIN_OVERRIDE     = 0.0
HEATMAP_VMAX_OVERRIDE     = 7.0
HEATMAP_PERCENTILES       = (2.0, 98.0)

ROTATE_K                  = -1
SAM_THRESHOLD = 0.14
SAM_CLOSE_RADIUS = 3
SAM_MIN_LEAF_AREA_PX = 2500
SAM_ERODE_ON_EDGE = False


ALPHA_MASK_ON_RGB         = 0.25
ALPHA_HEATMAP_ON_RGB      = 0.85
# =================================================================


def safe_division(a, b):
    return np.divide(a, b, where=(b != 0), out=np.zeros_like(a, dtype=np.float32))


def _nearest_band_index(target_nm, all_wavs):
    return int(np.argmin(np.abs(all_wavs - float(target_nm))))


def _nearest_band_indices(target_wavs, all_wavs):
    idxs = []
    for tw in np.array(target_wavs).ravel():
        idxs.append(int(np.argmin(np.abs(all_wavs - float(tw)))))
    return np.array(idxs, dtype=int)


def min_max_scale_spectra(X):
    X = X.astype(np.float32, copy=False)
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)


def overlay_basepath():
    base = os.path.splitext(os.path.basename(OUTPUT_BASE))[0]
    out_dir = os.path.dirname(OUTPUT_BASE)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)


# --------------------- Load hyperspectral cube ---------------------
def load_and_process_hyperspectral(base_path: str, part_number: str):
    hdr_file   = f'{base_path}/{part_number}/results/REFLECTANCE_{part_number}.hdr'
    data_hdr   = f'{base_path}/{part_number}/capture/{part_number}.hdr'
    white_hdr  = f'{base_path}/{part_number}/capture/WHITEREF_{part_number}.hdr'
    dark_hdr   = f'{base_path}/{part_number}/capture/DARKREF_{part_number}.hdr'

    img = envi.open(hdr_file)
    wavelengths = np.array(img.bands.centers, dtype=np.float32)

    data  = np.array(envi.open(data_hdr,  f'{base_path}/{part_number}/capture/{part_number}.raw').load())
    white = np.array(envi.open(white_hdr, f'{base_path}/{part_number}/capture/WHITEREF_{part_number}.raw').load())
    dark  = np.array(envi.open(dark_hdr,  f'{base_path}/{part_number}/capture/DARKREF_{part_number}.raw').load())

    corrected = safe_division(data - dark, white - dark).astype(np.float32, copy=False)
    return corrected, wavelengths


def visualize_rgb(cube: np.ndarray, wavelengths: np.ndarray):
    r = _nearest_band_index(650.0, wavelengths)
    g = _nearest_band_index(550.0, wavelengths)
    b = _nearest_band_index(425.0, wavelengths)
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)
    return rgb


# --------------------- Manual MULTI-ROI selection ---------------------
def draw_multiple_rois(image: np.ndarray, cube: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("Hold ALT and click to draw ROI polygons. Close near start.\nPress 'q' to finish.")
    rois, current, line, dots, alt = [], [], None, [], False

    def near_start(p, pts, px=5):
        return bool(pts) and (np.linalg.norm(np.array(p) - np.array(pts[0])) < px)

    def on_click(e):
        nonlocal line, current, dots
        if alt and e.button == 1 and e.inaxes is ax:
            if current and near_start((e.xdata, e.ydata), current):
                current.append(current[0])
                rois.append(current.copy())
                ax.add_patch(Polygon(current, closed=True, edgecolor='r', facecolor='none', lw=2))
                current.clear()
                [d.remove() for d in dots]; dots.clear(); line = None
                fig.canvas.draw()
                print(f"ROI {len(rois)} added.")
            else:
                current.append((e.xdata, e.ydata))
                d, = ax.plot(e.xdata, e.ydata, 'ro'); dots.append(d)
                if line:
                    line.set_data([p[0] for p in current], [p[1] for p in current])
                else:
                    line, = ax.plot([p[0] for p in current], [p[1] for p in current], 'r-')
                fig.canvas.draw()

    def on_key_press(e):
        nonlocal alt
        if e.key == 'alt': alt = True
        elif e.key == 'q': plt.close(fig)

    def on_key_release(e):
        nonlocal alt
        if e.key == 'alt': alt = False

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    plt.show()

    if not rois:
        raise RuntimeError("No ROI drawn.")

    # Build combined mask and mean reference
    H, W = image.shape[:2]
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pts = np.stack((x.flatten(), y.flatten()), axis=-1)
    combined_mask = np.zeros((H, W), dtype=bool)

    for i, poly in enumerate(rois, start=1):
        path = MplPath(poly)
        mask_flat = path.contains_points(pts).reshape(H, W)
        combined_mask |= mask_flat

    ref = cube[combined_mask, :].mean(axis=0)
    print(f"✅ Total ROIs: {len(rois)} → combined reference pixels: {combined_mask.sum()}")

    return ref


# --------------------------- SAM mask ---------------------------
def build_sam_mask(cube: np.ndarray, ref_spectrum: np.ndarray):
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float32, copy=False)
    ref = ref_spectrum / (np.linalg.norm(ref_spectrum) + 1e-12)

    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    sam = np.arccos(np.clip((flat @ ref) / norms.ravel(), -1.0, 1.0))
    sam_img = sam.reshape(H, W)

    m = sam_img < SAM_THRESHOLD
    rad = max(1, int(SAM_CLOSE_RADIUS))
    yy, xx = np.ogrid[-rad:rad+1, -rad:rad+1]
    disk = (xx*xx + yy*yy) <= rad*rad
    m = ndi.binary_closing(m, structure=disk)
    m = ndi.binary_fill_holes(m)

    labeled, nlab = ndi.label(m)
    if nlab > 0:
        sizes = ndi.sum(m, labeled, index=range(1, nlab+1))
        keep = np.zeros_like(m, dtype=bool)
        for lab, sz in enumerate(sizes, start=1):
            if sz >= SAM_MIN_LEAF_AREA_PX:
                keep |= (labeled == lab)
        m = keep

    if SAM_ERODE_ON_EDGE:
        m = ndi.binary_erosion(m, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool))

    return m, sam_img


# ---------------------- Model wrapper ----------------------
class PredictWrapper:
    def __init__(self, bundle, wavelengths=None, use_bundle_preproc=True):
        self.b = bundle
        self.wavelengths = wavelengths
        self.use_bundle_preproc = use_bundle_preproc
        self.model = self.b.get("ols", None)
        if self.model is None or not hasattr(self.model, "predict"):
            raise ValueError("Bundle missing usable final model under key 'ols'.")

        self.idx = None
        if "selected_indices" in self.b and self.b["selected_indices"] is not None:
            self.idx = np.asarray(self.b["selected_indices"]).astype(int)
        elif "feature_cols" in self.b and self.b["feature_cols"] is not None:
            fc = np.asarray(self.b["feature_cols"])
            if fc.dtype.kind in ("i", "u"):
                self.idx = fc.astype(int)
            elif fc.dtype.kind in ("f",):
                self.idx = _nearest_band_indices(fc, self.wavelengths)
        self.pre = self.b.get("preprocessing", None)

    def predict(self, X_scaled):
        Xt = X_scaled
        if self.idx is not None:
            Xt = Xt[:, self.idx]
        if self.use_bundle_preproc and self.pre is not None and hasattr(self.pre, "transform"):
            Xt = self.pre.transform(Xt)
        return self.model.predict(Xt)


def load_model(path: str, wavelengths: np.ndarray) -> PredictWrapper:
    bundle = joblib.load(path)
    print(f"✅ Loaded model bundle from: {path}")
    return PredictWrapper(bundle, wavelengths=wavelengths, use_bundle_preproc=True)


# ---------------------- Prediction ----------------------
def predict_on_sam_mask(cube, wavelengths, predictor, sam_mask):
    H, W, B = cube.shape
    npx = int(sam_mask.sum())
    if npx == 0:
        raise RuntimeError("SAM mask is empty.")

    pix = cube[sam_mask, :]
    pix_scaled = min_max_scale_spectra(pix)
    preds = predictor.predict(pix_scaled).astype(np.float32)
    pred_map = np.full((H, W), np.nan, dtype=np.float32)
    pred_map[sam_mask] = preds

    if HEATMAP_VMIN_OVERRIDE is not None and HEATMAP_VMAX_OVERRIDE is not None:
        vmin, vmax = HEATMAP_VMIN_OVERRIDE, HEATMAP_VMAX_OVERRIDE
    else:
        lo, hi = HEATMAP_PERCENTILES
        vmin = np.nanpercentile(preds, lo)
        vmax = np.nanpercentile(preds, hi)
    return pred_map, (vmin, vmax)


# ---------------------- Visualization ----------------------
def save_outputs(rgb, sam_img, sam_mask, pred_map, clim, part_number):
    base = overlay_basepath()
    vmin, vmax = clim

    plt.figure(figsize=(8, 8))
    im = plt.imshow(sam_img, cmap="hot", vmin=0, vmax=np.pi/2)
    plt.colorbar(im, label="Spectral Angle (rad)")
    plt.title("SAM Image vs ROI mean")
    plt.axis('off')
    plt.savefig(f"{base}_SAM_image.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    im = plt.imshow(sam_mask, cmap="Greens")
    im.set_alpha(np.where(sam_mask, ALPHA_MASK_ON_RGB, 0.0))
    plt.title("SAM Mask Overlay")
    plt.axis('off')
    plt.savefig(f"{base}_SAM_mask_overlay.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    im = plt.imshow(pred_map, cmap="viridis", vmin=vmin, vmax=vmax)
    im.set_alpha(np.where(sam_mask, ALPHA_HEATMAP_ON_RGB, 0.0))
    plt.colorbar(im, label=PREDICTION_NAME)
    plt.title(f"{PREDICTION_NAME} — SAM Heatmap ({part_number})")
    plt.axis('off')
    plt.savefig(f"{base}_SAM_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"🧩 Saved all SAM-based heatmaps under {os.path.dirname(base)}")


# ---------------------- Pipeline ----------------------
def process_file(base_path: str, part_number: str):
    cube, wavelengths = load_and_process_hyperspectral(base_path, part_number)
    if ROTATE_K != 0:
        cube = np.rot90(cube, k=ROTATE_K)
    rgb = visualize_rgb(cube, wavelengths)

    print("🎨 Draw multiple ROIs for reference...")
    ref_spec = draw_multiple_rois(rgb, cube)

    print(f"Building SAM mask (threshold={SAM_THRESHOLD})...")
    sam_mask, sam_img = build_sam_mask(cube, ref_spec)

    predictor = load_model(MODEL_PATH, wavelengths)
    pred_map, clim = predict_on_sam_mask(cube, wavelengths, predictor, sam_mask)
    save_outputs(rgb, sam_img, sam_mask, pred_map, clim, part_number)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    im = plt.imshow(pred_map, cmap="viridis", vmin=clim[0], vmax=clim[1])
    im.set_alpha(np.where(sam_mask, ALPHA_HEATMAP_ON_RGB, 0.0))
    plt.axis('off')
    plt.colorbar(im).set_label(PREDICTION_NAME)
    plt.title(f"{PREDICTION_NAME} — SAM Heatmap ({part_number})")
    plt.show()


def main():
    base_root = './'
    for name in os.listdir(base_root):
        folder = os.path.join(base_root, name)
        if name.startswith(PREFIXES) and os.path.isdir(folder):
            print(f"Processing: {name}")
            process_file(base_root, name)


if __name__ == "__main__":
    main()
