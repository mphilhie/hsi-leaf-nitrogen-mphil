# hs_roi_sam_extract.py
# - Loads Specim-style hyperspectral cubes
# - Lets you draw ONE polygon ROI to define a reference spectrum
# - Computes SAM, thresholds, and segments connected components as ROIs
# - Shows masks one-by-one (preview), saves combined + per-ROI labeled overlays
# - Exports spectra depending on SAVE_MEAN_ONLY:
#     True  -> only per-leaf mean spectra (mean of raw reflectance) to *_MEAN.xlsx
#     False -> *_RAW.xlsx (all raw per ROI), *_SCALED.xlsx (all scaled per ROI),
#              *_MEAN.xlsx (one sheet, all ROI means)

# ------------------ Backend must be set BEFORE pyplot import ------------------
import matplotlib as mpl
mpl.use('TkAgg')
# ------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi

# ================== USER SETTINGS ==================
OUTPUT_BASE     = "5)Raw/120/"   # base path (no extension) for all outputs
SAVE_MEAN_ONLY  = True                 # True: only per-leaf mean spectra; False: raw + scaled + mean
PREFIXES        = ("n2suz_120a_2025-06-10_001",)  # tuple of prefixes to process
SAM_THRESHOLD   =  0.20             # lower → stricter matching (fewer pixels)
MIN_AREA_PX     = 1200                   # drop small blobs
ROTATE_K        = -1                    # rotate cube 90° clockwise (set 0 if not needed)
TARGET_COMPONENTS = 9             # keep at most this many components
OPEN_RADIUS = 2                        # light opening to break thin bridges (1–2)

# mask refinement knobs
REFINE_CLOSE_RADIUS = 3                 # close small gaps along edges (2–3 typical)
REFINE_OPEN_RADIUS  = 0               # remove tiny spurs/noise (0–1 typical)
USE_CONVEX_HINT     = True              # try convex hull assist if available
CONVEX_BLEND        = 0.30              # fraction to blend hull with mask (0..1)

# optional quality check
USE_SOLIDITY_CHECK  = False             # set True to filter “bitten” masks
MIN_SOLIDITY        = 0.70              # 0.75–0.85 typical
# ===================================================

print("Current backend:", mpl.get_backend())

# ------------------ UTILITIES ------------------
def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array([[1]], dtype=bool)
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (xx*xx + yy*yy) <= radius*radius

def refine_component(mask: np.ndarray) -> np.ndarray:
    """
    Post-process a single leaf mask to avoid 'half leaves':
      - fill holes
      - closing to bridge tiny gaps
      - optional convex-hull hint blended back
      - light opening to remove spurs
    """
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
    data = data.astype(np.float32, copy=False)
    min_vals = data.min(axis=1).reshape(-1, 1)
    max_vals = data.max(axis=1).reshape(-1, 1)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)

def save_excel(filename: str, roi_dict: dict, wavelengths: np.ndarray, label="RAW"):
    if not roi_dict:
        print(f"⚠️ Nothing to save for {label}.")
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    col_names = [f"λ{w:.2f}" for w in wavelengths]
    with pd.ExcelWriter(filename, engine="openpyxl", mode="w") as writer:
        for key, arr in roi_dict.items():
            pd.DataFrame(arr, columns=col_names).to_excel(writer, sheet_name=key, index=False)
    print(f"✅ Saved {label} spectra → {filename}")

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

    corrected = safe_division(data - dark, white - dark)
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
    plt.figure(figsize=(8, 8))
    plt.imshow(sam, cmap='hot', vmin=0, vmax=np.pi / 2)
    plt.colorbar(label='Spectral Angle (rad)')
    plt.title('SAM Image vs ROI mean')
    plt.axis('off'); plt.show()

    # ---- Threshold (tighter reduces merges) ----
    sam_mask = (sam < SAM_THRESHOLD).astype(bool)
    sam_mask = ndi.binary_closing(sam_mask, structure=_disk(2))
    # ---- Light opening to break thin bridges ----
    sam_mask = binary_opening_break_bridges(sam_mask, radius=OPEN_RADIUS)

    # ---- Split touching blobs (prefer watershed; fallback if skimage missing) ----
    try:
        labeled, _ = ndi.label(sam_mask)
    except Exception:
        labeled = split_touching_without_skimage(sam_mask, erode_iter=1)

    # ---- Collect components and sort by area (largest first) ----
    areas = []
    for i in range(1, int(labeled.max()) + 1):
        area = np.sum(labeled == i)
        if area >= MIN_AREA_PX:
            areas.append((i, area))
    areas.sort(key=lambda x: x[1], reverse=True)

    # ---- Keep exactly TARGET_COMPONENTS (or fewer if not available) + refine ----
    masks = []
    for i, _area in areas[:TARGET_COMPONENTS]:
        m = (labeled == i)
        m = refine_component(m)  # <<< refine to avoid cut leaves
        if m.sum() >= MIN_AREA_PX and _solidity(m) >= (MIN_SOLIDITY if USE_SOLIDITY_CHECK else 0.0):
            masks.append(m)

    if len(masks) < TARGET_COMPONENTS:
        print(f"⚠️ Only {len(masks)} components found (target={TARGET_COMPONENTS}). "
              f"Consider lowering SAM_THRESHOLD or OPEN_RADIUS.")

    print(f"After splitting: {len(masks)} components kept (min area={MIN_AREA_PX})")

    # Show masks one-by-one (preview), like original code
    for i, m in enumerate(masks, 1):
        plt.figure(figsize=(5, 5))
        plt.imshow(m, cmap='gray')
        plt.title(f'ROI {i} mask preview')
        plt.axis('off')
        plt.show()

    return masks

# ------------------ Labeled overlays (combined + per-ROI) ------------------
def visualize_rois(image, fruit_masks, part_number, save_per_roi=True, save_combined=True, interactive=False):
    basepath = overlay_basepath()

    # -------- combined overlay --------
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
        if interactive: plt.show()
        else: plt.close()
        print(f"🖼️ Saved combined overlay → {combined_path}")


# ------------------ ROI PROCESSING & EXPORT ------------------
class ROINavigation:
    def __init__(self, masks, cube, wavelengths, part_number):
        self.masks = masks
        self.cube = cube
        self.wavelengths = wavelengths
        self.part_number = part_number
        self.export()

    def export(self):
        raw_dict, scaled_dict, mean_dict = {}, {}, {}
        for idx, m in enumerate(self.masks, 1):

            roi = self.cube[m, :]  # raw reflectance (Npix × B)
            scaled = min_max_scale_spectra(roi)  # row-wise min–max (exported only when SAVE_MEAN_ONLY=False)

            mean_spec = roi.mean(axis=0).reshape(1, -1)  # ✅ MEAN FROM RAW

            key = f"{self.part_number}_ROI{idx}"
            raw_dict[key] = roi
            scaled_dict[key] = scaled
            mean_dict[key] = mean_spec

            print(f"✅ ROI {idx}: N={roi.shape[0]} → raw mean computed")

        base = overlay_basepath()
        os.makedirs(os.path.dirname(base), exist_ok=True)

        if SAVE_MEAN_ONLY:
            # single-sheet MEAN
            roi_names = list(mean_dict.keys())
            mat = np.vstack([mean_dict[name] for name in roi_names])
            col_names = [f"λ{w:.2f}" for w in self.wavelengths]
            df = pd.DataFrame(mat, index=roi_names, columns=col_names)
            df.index.name = "ROI"
            out_path = f"{base}_MEAN.xlsx"
            df.to_excel(out_path, sheet_name="ScaledMean", engine="openpyxl")
            print(f"✅ Wrote all ROI mean spectra to single sheet → {out_path}")
        else:
            # RAW + SCALED (multi-sheet each)
            save_excel(f"{base}_RAW.xlsx",    raw_dict,    self.wavelengths, "RAW")
            save_excel(f"{base}_SCALED.xlsx", scaled_dict, self.wavelengths, "SCALED")
            # MEAN (single sheet)
            roi_names = list(mean_dict.keys())
            mat = np.vstack([mean_dict[name] for name in roi_names])
            col_names = [f"λ{w:.2f}" for w in self.wavelengths]
            df = pd.DataFrame(mat, index=roi_names, columns=col_names)
            df.index.name = "ROI"
            out_path = f"{base}_MEAN.xlsx"
            df.to_excel(out_path, sheet_name="ScaledMean", engine="openpyxl")
            print(f"✅ Wrote all ROI mean spectra to single sheet → {out_path}")

# ------------------ PIPELINE ------------------
def process_file(base_path: str, part_number: str):
    cube, wavelengths = load_and_process_hyperspectral(base_path, part_number)
    if ROTATE_K != 0:
        cube = np.rot90(cube, k=ROTATE_K)
    rgb = visualize_rgb(cube, wavelengths)
    masks = draw_rois(rgb, cube)
    print(f"Saving ROI overlays for {part_number}…")
    visualize_rois(rgb, masks, part_number, save_per_roi=True, save_combined=True, interactive=False)
    print(f"Exporting spectra for {part_number} (SAVE_MEAN_ONLY={SAVE_MEAN_ONLY})…")
    ROINavigation(masks, cube, wavelengths, part_number)
    plt.show()

def main():
    base = './'
    for name in os.listdir(base):
        folder = os.path.join(base, name)
        if name.startswith(PREFIXES) and os.path.isdir(folder):
            print(f"Processing: {name}")
            process_file(base, name)

if __name__ == "__main__":
    main()
