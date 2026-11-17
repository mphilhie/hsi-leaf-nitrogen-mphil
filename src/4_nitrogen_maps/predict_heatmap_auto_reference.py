# hs_sam_predict_heatmap.py
# - Auto-select reference spectrum from brightest-green patch
# - Build SAM mask (no NDVI/brightness fallback)
# - Predict only on SAM pixels with row-wise min–max scaling
# - Save SAM image, mask overlay, and nitrogen heatmap (.png) + optional .npy

# -------- Backend BEFORE pyplot import (TkAgg for on-screen; use 'Agg' for headless) --------
import matplotlib as mpl
mpl.use('TkAgg')
# --------------------------------------------------------------------------------------------

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from scipy import ndimage as ndi

# ========================= USER SETTINGS =========================
# I/O
OUTPUT_BASE               = "2)Blue_Berry/Back4/10_A"     # base path (no extension) for outputs
PREFIXES                  = ("n2suzzy_a_2025-06-10_003",) # dataset folders to process (relative to base_root)
MODEL_PATH                = "./Raw_BEST.joblib"

# Prediction & display
PREDICTION_NAME           = "Nitrogen (%)"
SAVE_NUMPY_ARRAYS         = True
HEATMAP_VMIN_OVERRIDE     = 0.0    # None → auto by percentiles
HEATMAP_VMAX_OVERRIDE     = 7.0    # None → auto by percentiles
HEATMAP_PERCENTILES       = (2.0, 98.0)

# Geometry / rotation
ROTATE_K                  = -1      # 0=no rotation; -1=90° CW, 1=90° CCW, etc.

# Auto reference (brightest-green patch)
GREEN_NM                  = 550.0   # band used to find "greenest" pixels
GREEN_TOP_PERCENT         = 2.0     # take top X% brightest in green band
REF_PATCH_MIN_SIZE        = 500    # minimum pixels for a valid ref blob
REF_PATCH_ERODE_DILATE    = 1       # smooth the ref blob a bit

# SAM mask creation
SAM_THRESHOLD             = 0.15    # lower = stricter; typical 0.08–0.15
SAM_CLOSE_RADIUS          = 2       # morphological smoothing of SAM mask
SAM_MIN_LEAF_AREA_PX      = 5000    # drop tiny blobs from SAM mask
SAM_ERODE_ON_EDGE         = True    # contract mask slightly to avoid edges

# Visualization alphas
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
    """Row-wise [0,1] scaling per pixel spectrum. X: (Npx, B)."""
    X = X.astype(np.float32, copy=False)
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)

def overlay_basepath():
    base = os.path.splitext(os.path.basename(OUTPUT_BASE))[0]
    out_dir = os.path.dirname(OUTPUT_BASE)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)

# ---------------------------- Loading ----------------------------
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

    corrected = safe_division(data - dark, white - dark).astype(np.float32, copy=False)  # (H, W, B)
    return corrected, wavelengths

def visualize_rgb(cube: np.ndarray, wavelengths: np.ndarray):
    r = _nearest_band_index(650.0, wavelengths)
    g = _nearest_band_index(550.0, wavelengths)
    b = _nearest_band_index(425.0, wavelengths)
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)
    return rgb

# --------------------- Auto reference selection ------------------
def auto_reference_from_green(cube: np.ndarray, wavelengths: np.ndarray):
    """
    Picks the largest contiguous blob among the top GREEN_TOP_PERCENT %
    brightest pixels in the GREEN_NM band, then returns its mean spectrum.
    """
    H, W, _ = cube.shape
    g_idx = _nearest_band_index(GREEN_NM, wavelengths)
    green = cube[:, :, g_idx]

    # Threshold by top percentile
    thr = np.percentile(green, 100.0 - float(GREEN_TOP_PERCENT))
    mask = green >= thr

    # Clean up & keep largest blob
    if REF_PATCH_ERODE_DILATE > 0:
        se = np.ones((3, 3), dtype=bool)
        mask = ndi.binary_opening(mask, structure=se, iterations=REF_PATCH_ERODE_DILATE)
        mask = ndi.binary_closing(mask, structure=se, iterations=REF_PATCH_ERODE_DILATE)
        mask = ndi.binary_fill_holes(mask)

    labeled, nlab = ndi.label(mask)
    if nlab == 0:
        raise RuntimeError("Auto-reference selection failed: no bright-green region found. "
                           "Try increasing GREEN_TOP_PERCENT or check rotation.")
    sizes = ndi.sum(mask, labeled, index=range(1, nlab+1))
    best_lab = int(np.argmax(sizes)) + 1
    best_area = int(sizes[best_lab - 1])

    if best_area < REF_PATCH_MIN_SIZE:
        print(f"⚠️ Reference blob small: {best_area} px (min {REF_PATCH_MIN_SIZE}). Continuing anyway.")

    ref_mask = (labeled == best_lab)
    ref_spec = cube[ref_mask, :].mean(axis=0)

    # Save a quick diagnostic
    base = overlay_basepath()
    plt.figure(figsize=(8, 8))
    plt.imshow(green, cmap="Greens")
    plt.contour(ref_mask.astype(float), levels=[0.5], colors=['k'], linewidths=2)
    plt.title("Auto Reference Region (brightest-green blob)")
    out = f"{base}_auto_ref_region.png"
    plt.axis('off'); plt.savefig(out, bbox_inches='tight', dpi=300); plt.close()
    print(f"🧭 Saved auto-reference diagnostic → {out}")

    return ref_spec

# --------------------------- SAM mask ----------------------------
def build_sam_mask(cube: np.ndarray, ref_spectrum: np.ndarray):
    """
    Compute Spectral Angle Mapper vs reference and produce a binary SAM mask.
    Returns (sam_mask, sam_image)
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float32, copy=False)

    ref = ref_spectrum.astype(np.float32, copy=False)
    ref_norm = np.linalg.norm(ref) + 1e-12
    ref = ref / ref_norm

    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    dots = flat @ ref
    sam = np.arccos(np.clip(dots / norms.ravel(), -1.0, 1.0))  # [N]
    sam_img = sam.reshape(H, W)

    # Threshold & morphology
    m = sam_img < float(SAM_THRESHOLD)

    rad = max(1, int(SAM_CLOSE_RADIUS))
    yy, xx = np.ogrid[-rad:rad+1, -rad:rad+1]
    disk = (xx*xx + yy*yy) <= rad*rad

    m = ndi.binary_closing(m, structure=disk)
    m = ndi.binary_fill_holes(m)

    # Remove small blobs
    labeled, nlab = ndi.label(m)
    if nlab > 0:
        sizes = ndi.sum(m, labeled, index=range(1, nlab+1))
        keep = np.zeros_like(m, dtype=bool)
        for lab, sz in enumerate(sizes, start=1):
            if sz >= SAM_MIN_LEAF_AREA_PX:
                keep |= (labeled == lab)
        m = keep

    # Edge safety
    if SAM_ERODE_ON_EDGE:
        m = ndi.binary_erosion(m, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool))

    return m, sam_img

# ---------------------- Model wrapper (bundle) -------------------
class PredictWrapper:
    """
    Expects bundle keys:
      - 'selected_indices' (preferred) or 'feature_cols' (ints=idx, floats=nm)
      - 'preprocessing' (optional transformer with .transform)
      - 'ols' (final regressor)
    """
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
            self._idx_source = "selected_indices"
        elif "feature_cols" in self.b and self.b["feature_cols"] is not None:
            fc = np.asarray(self.b["feature_cols"])
            if fc.dtype.kind in ("i", "u"):
                self.idx = fc.astype(int)
                self._idx_source = "feature_cols[idx]"
            elif fc.dtype.kind in ("f",):
                if self.wavelengths is None:
                    raise ValueError("feature_cols are wavelengths but cube wavelengths not provided.")
                self.idx = _nearest_band_indices(fc, self.wavelengths)
                self._idx_source = "feature_cols[wavelengths→idx]"
            else:
                raise ValueError("feature_cols dtype unsupported (expect int indices or float wavelengths).")
        else:
            self._idx_source = None  # no subsetting

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
        Xt = self._subset(X_scaled)
        Xt = self._preprocess(Xt)
        need = getattr(self.model, "n_features_in_", None)
        if need is not None and Xt.shape[1] != need:
            raise ValueError(f"Model expects {need} features, got {Xt.shape[1]}. "
                             f"Check 'selected_indices'/'feature_cols' and 'preprocessing'.")
        return self.model.predict(Xt)

def load_model(path: str, wavelengths: np.ndarray) -> PredictWrapper:
    bundle = joblib.load(path)
    print(f"✅ Loaded model bundle from: {path}")
    return PredictWrapper(bundle, wavelengths=wavelengths, use_bundle_preproc=True)

# -------------------- Prediction & saving ------------------------
def predict_on_sam_mask(cube: np.ndarray,
                        wavelengths: np.ndarray,
                        predictor: PredictWrapper,
                        sam_mask: np.ndarray,
                        part_number: str):
    """
    Predict per-pixel only on sam_mask. Returns (pred_map, (vmin, vmax)).
    """
    H, W, B = cube.shape
    base = overlay_basepath()
    os.makedirs(os.path.dirname(base), exist_ok=True)

    npx = int(sam_mask.sum())
    if npx == 0:
        raise RuntimeError("SAM mask is empty. Adjust SAM_THRESHOLD or reference selection settings.")

    # Prepare row-wise scaled spectra & predict
    pix = cube[sam_mask, :]                  # (Npx, B)
    pix_scaled = min_max_scale_spectra(pix)  # row-wise [0,1]
    preds = predictor.predict(pix_scaled).astype(np.float32).reshape(-1)

    print(f"{PREDICTION_NAME} on SAM pixels → min/mean/max = "
          f"{np.nanmin(preds):.4f} / {np.nanmean(preds):.4f} / {np.nanmax(preds):.4f}")

    pred_map = np.full((H, W), np.nan, dtype=np.float32)
    pred_map[sam_mask] = preds

    # Color limits
    if HEATMAP_VMIN_OVERRIDE is not None and HEATMAP_VMAX_OVERRIDE is not None:
        vmin, vmax = HEATMAP_VMIN_OVERRIDE, HEATMAP_VMAX_OVERRIDE
    else:
        lo, hi = HEATMAP_PERCENTILES
        vmin = np.nanpercentile(preds, lo) if HEATMAP_VMIN_OVERRIDE is None else HEATMAP_VMIN_OVERRIDE
        vmax = np.nanpercentile(preds, hi) if HEATMAP_VMAX_OVERRIDE is None else HEATMAP_VMAX_OVERRIDE

    if SAVE_NUMPY_ARRAYS:
        np.save(f"{base}_SAM_pred.npy", pred_map)

    return pred_map, (vmin, vmax)

def save_sam_and_heatmaps(rgb, sam_img, sam_mask, pred_map, clim, part_number):
    base = overlay_basepath()

    # SAM image
    plt.figure(figsize=(9, 9))
    im = plt.imshow(sam_img, cmap="hot", vmin=0, vmax=np.pi/2)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Spectral Angle (rad)")
    plt.title("SAM Image vs Auto-Ref")
    plt.axis('off')
    out = f"{base}_SAM_image.png"
    plt.savefig(out, bbox_inches='tight', dpi=300); plt.close()
    print(f"🧪 Saved SAM image → {out}")

    # Mask overlay on RGB
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    m = sam_mask.astype(float); m[~sam_mask] = np.nan
    im = plt.imshow(m, cmap="Greens")
    im.set_alpha(np.where(sam_mask, ALPHA_MASK_ON_RGB, 0.0))
    plt.title(f"SAM Mask Overlay ({part_number})")
    plt.axis('off')
    out = f"{base}_SAM_mask_overlay.png"
    plt.savefig(out, bbox_inches='tight', dpi=300); plt.close()
    print(f"🟢 Saved SAM mask overlay → {out}")

    # Heatmap on RGB
    vmin, vmax = clim
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    im = plt.imshow(pred_map, cmap="viridis", vmin=vmin, vmax=vmax)
    im.set_alpha(np.where(sam_mask, ALPHA_HEATMAP_ON_RGB, 0.0))
    cb = plt.colorbar(im, fraction=0.046, pad=0.04); cb.set_label(PREDICTION_NAME)
    plt.title(f"{PREDICTION_NAME} — SAM Heatmap ({part_number})")
    plt.axis('off')
    out = f"{base}_SAM_heatmap.png"
    plt.savefig(out, bbox_inches='tight', dpi=300); plt.close()
    print(f"🔥 Saved SAM heatmap → {out}")

# ----------------------------- Pipeline --------------------------
def process_file(base_path: str, part_number: str):
    cube, wavelengths = load_and_process_hyperspectral(base_path, part_number)
    if ROTATE_K != 0:
        cube = np.rot90(cube, k=ROTATE_K)
    rgb = visualize_rgb(cube, wavelengths)

    print(f"Auto-selecting reference spectrum from brightest-green patch …")
    ref_spec = auto_reference_from_green(cube, wavelengths)

    print(f"Building SAM mask with threshold={SAM_THRESHOLD} …")
    sam_mask, sam_img = build_sam_mask(cube, ref_spec)

    print(f"Predicting on SAM pixels for {part_number} …")
    predictor = load_model(MODEL_PATH, wavelengths)

    pred_map, clim = predict_on_sam_mask(cube, wavelengths, predictor, sam_mask, part_number)
    save_sam_and_heatmaps(rgb, sam_img, sam_mask, pred_map, clim, part_number)

    # Optional on-screen view
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    im = plt.imshow(pred_map, cmap="viridis", vmin=clim[0], vmax=clim[1])
    im.set_alpha(np.where(sam_mask, ALPHA_HEATMAP_ON_RGB, 0.0))
    plt.title(f"{PREDICTION_NAME} — SAM ({part_number})")
    plt.axis('off')
    plt.colorbar(im).set_label(PREDICTION_NAME)
    plt.tight_layout()
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
