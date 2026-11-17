# hs_roi_sam_predict_heatmap.py
# - Same SAM-based ROI splitting workflow
# - For each detected leaf (component), collect ALL pixels
# - Row-wise min–max scale each pixel spectrum
# - Feed into your trained model and generate a prediction heatmap per leaf
# - No CSV/Excel writing

# ------------------ Backend must be set BEFORE pyplot import ------------------
import matplotlib as mpl
mpl.use('TkAgg')
# ------------------------------------------------------------------------------

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi

# ================== USER SETTINGS ==================
OUTPUT_BASE     = "2)Blue_Berry/Back4/10_A"   # base path (no extension) for all outputs
PREFIXES        = ("n2suzzy_a_2025-06-10_003",)    # folders under 1_DataSets to process

# --- Model / prediction ---
MODEL_PATH          = "./Raw_BEST.joblib"    # your trained model bundle
PREDICTION_NAME     = "Nitrogen (%)"         # label for the colorbar/title
SAVE_NUMPY_ARRAYS   = True                   # also save per-leaf predictions as .npy

# --- SAM / component split ---
SAM_THRESHOLD   = 0.10
MIN_AREA_PX     = 1000
ROTATE_K        = -1
TARGET_COMPONENTS = 10
OPEN_RADIUS = 1
MIN_PEAK_DISTANCE = 8

# mask refinement knobs
REFINE_CLOSE_RADIUS = 3
REFINE_OPEN_RADIUS  = 0
USE_CONVEX_HINT     = True
CONVEX_BLEND        = 0.30

# optional quality check
USE_SOLIDITY_CHECK  = False
MIN_SOLIDITY        = 0.80

# ---- Heatmap colour scaling (set these to override percentiles) ----
HEATMAP_VMIN_OVERRIDE = 0.0     # set to None to auto-compute; else a float (e.g., 0.0)
HEATMAP_VMAX_OVERRIDE = 7.0    # set >7.0 to show beyond 7%; else None for auto
HEATMAP_PERCENTILES   = (2.0, 98.0)  # used only when both overrides are None

# ===================================================

print("Current backend:", mpl.get_backend())

# ------------------ UTILITIES ------------------
def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array([[1]], dtype=bool)
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (xx*xx + yy*yy) <= radius*radius

def refine_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    m = ndi.binary_fill_holes(m)
    if REFINE_CLOSE_RADIUS > 0:
        m = ndi.binary_closing(m, structure=_disk(REFINE_CLOSE_RADIUS))
    if USE_CONVEX_HINT:
        try:
            from skimage.morphology import convex_hull_image
            hull = convex_hull_image(m)
            if CONVEX_BLEND > 0:
                m_dil = ndi.binary_dilation(m, structure=_disk(1))
                m = ((hull & m_dil) | m) if CONVEX_BLEND >= 0.5 else (m | (hull & m))
        except Exception:
            pass
    if REFINE_OPEN_RADIUS > 0:
        m = ndi.binary_opening(m, structure=_disk(REFINE_OPEN_RADIUS))
    return m

def _solidity(mask: np.ndarray) -> float:
    if not USE_SOLIDITY_CHECK:
        return 1.0
    try:
        from skimage.morphology import convex_hull_image
        hull = convex_hull_image(mask.astype(bool))
        a = float(mask.sum()); h = float(hull.sum())
        return (a / h) if h > 0 else 0.0
    except Exception:
        return 1.0

def to_4connectivity_structure():
    return np.array([[0,1,0],
                     [1,1,1],
                     [0,1,0]], dtype=bool)

def binary_opening_break_bridges(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    se = (xx*xx + yy*yy) <= radius*radius
    opened = ndi.binary_opening(mask, structure=se)
    return opened.astype(bool)

def split_touching_watershed(mask: np.ndarray, min_peak_distance: int = 8) -> np.ndarray:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, labels=mask, min_distance=min_peak_distance)
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    labels = watershed(-distance, markers, mask=mask)
    return labels

def split_touching_without_skimage(mask: np.ndarray, erode_iter: int = 1) -> np.ndarray:
    if erode_iter <= 0:
        eroded = mask.copy()
    else:
        se = to_4connectivity_structure()
        eroded = ndi.binary_erosion(mask, structure=se, iterations=erode_iter)
    labeled, _ = ndi.label(eroded, structure=to_4connectivity_structure())
    if erode_iter > 0:
        se = to_4connectivity_structure()
        rec = np.zeros_like(mask, dtype=bool)
        for lab in range(1, labeled.max()+1):
            comp = (labeled == lab)
            comp = ndi.binary_dilation(comp, structure=se, iterations=erode_iter)
            rec |= comp
        return ndi.label(rec, structure=to_4connectivity_structure())[0]
    return labeled

def safe_division(a, b):
    return np.divide(a, b, where=(b != 0), out=np.zeros_like(a))

def min_max_scale_spectra(data: np.ndarray) -> np.ndarray:
    """Row-wise [0,1] scaling per pixel spectrum."""
    data = data.astype(np.float32, copy=False)
    min_vals = data.min(axis=1, keepdims=True)
    max_vals = data.max(axis=1, keepdims=True)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)

def overlay_basepath():
    base = os.path.splitext(os.path.basename(OUTPUT_BASE))[0]
    out_dir = os.path.dirname(OUTPUT_BASE)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)

def mask_centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (float('nan'), float('nan'))
    return float(ys.mean()), float(xs.mean())

# ------------------ HYPERSPECTRAL LOADING ------------------
def load_and_process_hyperspectral(base_path: str, part_number: str):
    hdr_file   = f'{base_path}/{part_number}/results/REFLECTANCE_{part_number}.hdr'
    data_hdr   = f'{base_path}/{part_number}/capture/{part_number}.hdr'
    white_hdr  = f'{base_path}/{part_number}/capture/WHITEREF_{part_number}.hdr'
    dark_hdr   = f'{base_path}/{part_number}/capture/DARKREF_{part_number}.hdr'

    img = envi.open(hdr_file)
    wavelengths = np.array(img.bands.centers)

    data  = np.array(envi.open(data_hdr,  f'{base_path}/{part_number}/capture/{part_number}.raw').load())
    white = np.array(envi.open(white_hdr, f'{base_path}/{part_number}/capture/WHITEREF_{part_number}.raw').load())
    dark  = np.array(envi.open(dark_hdr,  f'{base_path}/{part_number}/capture/DARKREF_{part_number}.raw').load())

    corrected = safe_division(data - dark, white - dark)  # shape: (H, W, B)
    return corrected, wavelengths

def visualize_rgb(cube: np.ndarray, wavelengths: np.ndarray):
    r = np.argmin(np.abs(wavelengths - 650))
    g = np.argmin(np.abs(wavelengths - 550))
    b = np.argmin(np.abs(wavelengths - 425))
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)
    return rgb

# ------------------ ROI DRAWING & SAM ------------------
def draw_rois(image: np.ndarray, cube: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("Hold 'Alt' to draw ROI. Click to place points. Close near start. Press 'q' to finish.")
    rois, poly, line, dots, alt = [], [], None, [], False

    def near_start(p, pts, px=5):
        return bool(pts) and (np.linalg.norm(np.array(p) - np.array(pts[0])) < px)

    def on_click(e):
        nonlocal line, poly, dots
        if alt and e.button == 1 and e.inaxes is ax:
            if poly and near_start((e.xdata, e.ydata), poly):
                poly.append(poly[0])
                rois.append(poly.copy())
                ax.add_patch(Polygon(poly, closed=True, edgecolor='r', facecolor='none', lw=2))
                poly.clear()
                [d.remove() for d in dots]; dots.clear(); line = None
                fig.canvas.draw()
                print(f"ROI {len(rois)} added.")
            else:
                poly.append((e.xdata, e.ydata))
                d, = ax.plot(e.xdata, e.ydata, 'ro'); dots.append(d)
                if line:
                    line.set_data([p[0] for p in poly], [p[1] for p in poly])
                else:
                    line, = ax.plot([p[0] for p in poly], [p[1] for p in poly], 'r-')
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
        raise RuntimeError("No ROI drawn. Hold Alt and click to draw, then close near the start point.")

    roi_poly = rois[0]
    H, W = image.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pts = np.stack((x.flatten(), y.flatten()), axis=-1)  # (x, y)
    roi_path = MplPath(roi_poly)
    mask_flat = roi_path.contains_points(pts).reshape(H, W)
    mask[mask_flat] = True

    # Reference = mean spectrum inside hand-drawn polygon
    ref = cube[mask, :].mean(axis=0)

    # Full-image SAM
    flat = cube.reshape(-1, cube.shape[-1])
    anorm = np.linalg.norm(flat, axis=-1)
    bnorm = np.linalg.norm(ref)
    dot = flat @ ref
    ang = np.arccos(np.clip(dot / (anorm * bnorm + 1e-10), -1, 1))
    sam = ang.reshape(H, W)

    # Hysteresis-ish grow if available
    try:
        from skimage.morphology import reconstruction
        LOW = SAM_THRESHOLD - 0.03
        HIGH = SAM_THRESHOLD + 0.06
        seeds = (sam < LOW)
        allow = (sam < HIGH)
        sam_mask = reconstruction(seeds.astype(np.uint8),
                                  allow.astype(np.uint8),
                                  method='dilation').astype(bool)
    except Exception:
        sam_mask = (sam < SAM_THRESHOLD).astype(bool)

    sam_mask = ndi.binary_closing(sam_mask, structure=_disk(2))
    sam_mask = binary_opening_break_bridges(sam_mask, radius=OPEN_RADIUS)

    plt.figure(figsize=(8, 8))
    plt.imshow(sam, cmap='hot', vmin=0, vmax=np.pi / 2)
    plt.colorbar(label='Spectral Angle (rad)')
    plt.title('SAM Image vs ROI mean')
    plt.axis('off'); plt.show()

    # Tighten and split
    sam_mask = (sam < SAM_THRESHOLD).astype(bool)
    sam_mask = ndi.binary_closing(sam_mask, structure=_disk(2))
    sam_mask = binary_opening_break_bridges(sam_mask, radius=OPEN_RADIUS)

    try:
        labeled = split_touching_watershed(sam_mask, min_peak_distance=MIN_PEAK_DISTANCE)
    except Exception:  # scikit-image not installed
        labeled = split_touching_without_skimage(sam_mask, erode_iter=1)

    # Filter, refine, keep top TARGET_COMPONENTS by area
    areas = []
    for i in range(1, int(labeled.max()) + 1):
        area = np.sum(labeled == i)
        if area >= MIN_AREA_PX:
            areas.append((i, area))
    areas.sort(key=lambda x: x[1], reverse=True)

    masks = []
    for i, _area in areas[:TARGET_COMPONENTS]:
        m = (labeled == i)
        m = refine_component(m)
        if m.sum() >= MIN_AREA_PX and _solidity(m) >= (MIN_SOLIDITY if USE_SOLIDITY_CHECK else 0.0):
            masks.append(m)

    if len(masks) < TARGET_COMPONENTS:
        print(f"⚠️ Only {len(masks)} components found (target={TARGET_COMPONENTS}). "
              f"Consider lowering SAM_THRESHOLD or OPEN_RADIUS.")

    print(f"After splitting: {len(masks)} components kept (min area={MIN_AREA_PX})")

    # Quick previews
    for i, m in enumerate(masks, 1):
        plt.figure(figsize=(5, 5))
        plt.imshow(m, cmap='gray')
        plt.title(f'ROI {i} mask preview')
        plt.axis('off')
        plt.show()

    return masks

# ------------------ Overlays ------------------
def visualize_rois(image, fruit_masks, part_number, save_combined=True):
    basepath = overlay_basepath()
    if save_combined:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        cmap = plt.cm.get_cmap("tab20", max(20, len(fruit_masks)))
        for i, m in enumerate(fruit_masks, 1):
            plt.contour(m.astype(float), levels=[0.5], linewidths=2, linestyles='-',
                        colors=[cmap((i-1) % cmap.N)])
            cy, cx = mask_centroid(m)
            plt.text(cx, cy, f"ROI {i}", ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))
        plt.title(f"Detected ROI masks — {part_number}")
        plt.axis('off')
        combined_path = f"{basepath}_visualized_rois.png"
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        plt.savefig(combined_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"🖼️ Saved combined overlay → {combined_path}")

# ------------------ Heatmap generation ------------------
def predict_leaf_heatmaps(cube: np.ndarray,
                          wavelengths: np.ndarray,
                          masks: list,
                          predictor: PredictWrapper,
                          part_number: str):
    """
    For each mask (leaf), predict value per pixel and save a heatmap.
    Also returns a combined prediction map for all leaves.
    """
    H, W, B = cube.shape
    base = overlay_basepath()
    os.makedirs(os.path.dirname(base), exist_ok=True)

    all_preds_for_clim = []
    per_leaf_pred_maps = []
    combined_pred_map = np.full((H, W), np.nan, dtype=np.float32)

    for idx, m in enumerate(masks, 1):
        coords = np.where(m)
        if coords[0].size == 0:
            per_leaf_pred_maps.append(np.full((H, W), np.nan, dtype=np.float32))
            continue

        # Row-wise min–max per pixel
        roi_pix = cube[m, :]                        # (Npx, B)
        roi_scaled = min_max_scale_spectra(roi_pix) # (Npx, B)

        # Predict
        preds = predictor.predict(roi_scaled).astype(np.float32).reshape(-1)

        # Save into maps
        pred_map = np.full((H, W), np.nan, dtype=np.float32)
        pred_map[m] = preds
        per_leaf_pred_maps.append(pred_map)

        # Combined map gets latest assignment for each pixel (OK: masks don’t overlap after split)
        combined_pred_map[m] = preds

        all_preds_for_clim.append(preds)

        # Stats
        print(f"ROI {idx}: pixels={preds.size}  {PREDICTION_NAME} min/mean/max = "
              f"{np.nanmin(preds):.4f} / {np.nanmean(preds):.4f} / {np.nanmax(preds):.4f}")

        # Optional numpy dump
        if SAVE_NUMPY_ARRAYS:
            np.save(f"{base}_ROI{idx}_pred.npy", pred_map)

    # Decide vmin/vmax
    if HEATMAP_VMIN_OVERRIDE is not None and HEATMAP_VMAX_OVERRIDE is not None:
        vmin, vmax = HEATMAP_VMIN_OVERRIDE, HEATMAP_VMAX_OVERRIDE
    elif len(all_preds_for_clim) > 0:
        concat = np.concatenate(all_preds_for_clim)
        lo, hi = HEATMAP_PERCENTILES
        vmin = np.nanpercentile(concat, lo) if HEATMAP_VMIN_OVERRIDE is None else HEATMAP_VMIN_OVERRIDE
        vmax = np.nanpercentile(concat, hi) if HEATMAP_VMAX_OVERRIDE is None else HEATMAP_VMAX_OVERRIDE
    else:
        vmin = vmax = None

    # Save per-leaf plots
    for idx, pred_map in enumerate(per_leaf_pred_maps, 1):
        plt.figure(figsize=(8, 8))
        im = plt.imshow(pred_map, cmap="viridis", vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im)
        cb.set_label(PREDICTION_NAME)
        plt.title(f"{PREDICTION_NAME} Heatmap — ROI {idx}")
        plt.axis('off')
        out_path = f"{base}_ROI{idx}_heatmap.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"🔥 Saved heatmap → {out_path}")

    return combined_pred_map, (vmin, vmax)

def save_combined_heatmap(rgb: np.ndarray,
                          masks: list,
                          combined_pred_map: np.ndarray,
                          clim: tuple,
                          part_number: str):
    base = overlay_basepath()
    vmin, vmax = clim if clim else (None, None)

    # Build a single union mask for alpha
    union_mask = np.zeros(combined_pred_map.shape, dtype=bool)
    for m in masks:
        union_mask |= m.astype(bool)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)  # background
    im = plt.imshow(combined_pred_map, cmap="viridis", vmin=vmin, vmax=vmax)
    im.set_alpha(np.where(union_mask, 0.85, 0.0))  # only visible on leaves

    # Draw contours & labels
    cmap = plt.cm.get_cmap("tab20", max(20, len(masks)))
    for i, m in enumerate(masks, 1):
        plt.contour(m.astype(float), levels=[0.5], linewidths=1.8,
                    linestyles='-', colors=[cmap((i-1) % cmap.N)])
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label(PREDICTION_NAME)
    plt.title(f"{PREDICTION_NAME} — Combined Heatmap ({part_number})")
    plt.axis('off')
    out_path = f"{base}_ALL_heatmap.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"🧩 Saved combined all-leaves heatmap → {out_path}")


# ------------------ Model loading & prediction (wired to your bundle) ------------------
def _nearest_band_indices(target_wavs, all_wavs):
    idxs = []
    for tw in np.array(target_wavs).ravel():
        idxs.append(int(np.argmin(np.abs(all_wavs - float(tw)))))
    return np.array(idxs, dtype=int)

class PredictWrapper:
    """
    Uses your bundle keys exactly:
      - 'selected_indices' (preferred) or 'feature_cols' to subset columns
      - 'preprocessing' (if present) to transform features
      - 'ols' as final regressor
    Also supports 'feature_cols' as either integers (indices) or floats (nm wavelengths).
    """
    def __init__(self, bundle, wavelengths=None, use_bundle_preproc=True):
        self.b = bundle
        self.wavelengths = wavelengths
        self.use_bundle_preproc = use_bundle_preproc

        # quick sanity: find model
        self.model = self.b.get("ols", None)
        if self.model is None or not hasattr(self.model, "predict"):
            raise ValueError("Bundle missing usable final model under key 'ols'.")

        # discover column selection source
        self.idx = None
        if "selected_indices" in self.b and self.b["selected_indices"] is not None:
            self.idx = np.asarray(self.b["selected_indices"]).astype(int)
            self._idx_source = "selected_indices"
        elif "feature_cols" in self.b and self.b["feature_cols"] is not None:
            fc = np.asarray(self.b["feature_cols"])
            if fc.dtype.kind in ("i", "u"):               # ints -> direct indices
                self.idx = fc.astype(int)
                self._idx_source = "feature_cols[idx]"
            elif fc.dtype.kind in ("f"):                  # floats -> wavelengths
                if self.wavelengths is None:
                    raise ValueError("feature_cols appear to be wavelengths (floats) "
                                     "but cube wavelengths are not available.")
                self.idx = _nearest_band_indices(fc, self.wavelengths)
                self._idx_source = "feature_cols[wavelengths→idx]"
            else:
                raise ValueError("feature_cols has unsupported dtype; expected ints or floats.")
        else:
            self._idx_source = None  # no subsetting stored

        # optional preprocessor
        self.pre = self.b.get("preprocessing", None)

    def _subset(self, X):
        if self.idx is not None:
            X = X[:, self.idx]
            print(f"→ Applied column subset via {self._idx_source}: {X.shape[1]} features")
        return X

    def _preprocess(self, X):
        if self.use_bundle_preproc and self.pre is not None and hasattr(self.pre, "transform"):
            X = self.pre.transform(X)
            print(f"→ Applied bundle 'preprocessing' → shape {X.shape}")
        return X

    def predict(self, X_scaled):
        # 1) subset columns to training feature set
        Xt = self._subset(X_scaled)
        # 2) apply train-time preprocessing (if any)
        Xt = self._preprocess(Xt)

        # 3) final safeguard: align to model.n_features_in_ if available
        need = getattr(self.model, "n_features_in_", None)
        if need is not None and Xt.shape[1] != need:
            msg = (f"Model expects {need} features but has {Xt.shape[1]}. "
                   f"Check 'selected_indices'/'feature_cols' and 'preprocessing'.")
            raise ValueError(msg)

        return self.model.predict(Xt)

def load_model(path: str, wavelengths: np.ndarray) -> PredictWrapper:
    bundle = joblib.load(path)
    print(f"✅ Loaded model bundle from: {path}")
    # pass wavelengths so we can map feature_cols in nm → nearest band indices
    return PredictWrapper(bundle, wavelengths=wavelengths, use_bundle_preproc=True)

# ------------------ PIPELINE ------------------
def process_file(base_path: str, part_number: str):
    cube, wavelengths = load_and_process_hyperspectral(base_path, part_number)
    if ROTATE_K != 0:
        cube = np.rot90(cube, k=ROTATE_K)
    rgb = visualize_rgb(cube, wavelengths)
    masks = draw_rois(rgb, cube)
    print(f"Saving ROI overlays for {part_number}…")
    visualize_rois(rgb, masks, part_number, save_combined=True)

    print(f"Predicting per-leaf heatmaps for {part_number} …")
    predictor = load_model(MODEL_PATH, wavelengths)

    # Get both per-leaf heatmaps and color limits
    combined_pred_map, clim = predict_leaf_heatmaps(cube, wavelengths, masks, predictor, part_number)

    # 🔹 Save one combined heatmap of all leaves on the RGB
    save_combined_heatmap(rgb, masks, combined_pred_map, clim, part_number)

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
