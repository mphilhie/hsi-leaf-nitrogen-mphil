# --- hs_roi_sam_mean_from_scaled_only.py ---
# Workflow:
# 1) For each target folder (first 20, configurable), load Specim cube
# 2) Draw ONE polygon ROI to define a reference spectrum (interactive)
# 3) Compute SAM, segment components (leaves), keep components ≥ MIN_AREA
# 4) For each component: scale rows (min–max per pixel), then MEAN across pixels
# 5) Append all means into ONE Excel file: a single sheet "ScaledMean"
#    Rows are labeled "<folder>_ROI<i>" and columns are wavelengths

import os
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # interactive ROI drawing
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy.ndimage import label

# ------------------ USER SETTINGS ------------------
BASE_DIR        = r'./'     # root containing your folders
PREFIX_BASE     = 'N2final27_2025-05-31_'           # base name; folders like ..._001 ..._020
START_IDX       = 1                                  # first index (inclusive)
END_IDX         = 59                                # last  index (inclusive)
OUTPUT_XLSX     = r'2)S40/All_MEAN_thirdx.xlsx'             # single Excel destination (one sheet)
SHEET_NAME      ='Mean'

SAM_THRESHOLD   = 0.20    # lower = stricter match (fewer pixels)
MIN_AREA_PX     = 50      # ignore tiny blobs
ROTATE_K        = -1      # rotate cube 90° clockwise; set 0 if not needed
# ---------------------------------------------------

print("Current backend:", mpl.get_backend())

# ------------------ UTILS ------------------
def safe_division(a, b):
    return np.divide(a, b, where=(b != 0), out=np.zeros_like(a))

def min_max_scale_rows(arr: np.ndarray) -> np.ndarray:
    """Row-wise min-max scaling to [0,1]. Constant rows → zeros."""
    arr = arr.astype(np.float32, copy=False)
    mn = arr.min(axis=1, keepdims=True)
    mx = arr.max(axis=1, keepdims=True)
    return (arr - mn) / (mx - mn + 1e-8)

def load_cube(base_path: str, part: str):
    """Load Specim-style capture and compute reflectance; return cube (H,W,B) and wavelengths (B,)."""
    hdr_reflect = f'{base_path}/{part}/results/REFLECTANCE_{part}.hdr'
    hdr_data    = f'{base_path}/{part}/capture/{part}.hdr'
    hdr_white   = f'{base_path}/{part}/capture/WHITEREF_{part}.hdr'
    hdr_dark    = f'{base_path}/{part}/capture/DARKREF_{part}.hdr'

    img = envi.open(hdr_reflect)
    wavelengths = np.array(img.bands.centers)

    data  = np.array(envi.open(hdr_data,  f'{base_path}/{part}/capture/{part}.raw').load())
    white = np.array(envi.open(hdr_white, f'{base_path}/{part}/capture/WHITEREF_{part}.raw').load())
    dark  = np.array(envi.open(hdr_dark,  f'{base_path}/{part}/capture/DARKREF_{part}.raw').load())

    cube = safe_division(data - dark, white - dark)
    return cube, wavelengths

def rgb_preview(cube: np.ndarray, wavelengths: np.ndarray):
    """Make an RGB for display using ~650/550/425 nm."""
    r = np.argmin(np.abs(wavelengths - 650))
    g = np.argmin(np.abs(wavelengths - 550))
    b = np.argmin(np.abs(wavelengths - 425))
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)
    return rgb

def draw_reference_polygon(image: np.ndarray):
    """Interactive: Hold Alt to draw, click near start to close, 'q' to finish."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("Hold 'Alt' to draw ROI; click near start to close; press 'q' to finish.")
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
        raise RuntimeError("No ROI drawn.")
    return rois[0]

def compute_sam_masks(cube: np.ndarray, poly: list, sam_threshold: float, min_area_px: int):
    """Use polygon mean as reference, compute SAM, threshold, label, and return masks list."""
    H, W, B = cube.shape
    # polygon → mask
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pts = np.stack((x.ravel(), y.ravel()), axis=-1)  # (x,y)
    roi_path = MplPath(poly)
    ref_mask = roi_path.contains_points(pts).reshape(H, W)
    # reference spectrum
    ref = cube[ref_mask, :].mean(axis=0)

    # SAM
    flat = cube.reshape(-1, B)
    a_norm = np.linalg.norm(flat, axis=-1)
    b_norm = np.linalg.norm(ref)
    dot = flat @ ref
    sam = np.arccos(np.clip(dot / (a_norm * b_norm + 1e-10), -1, 1)).reshape(H, W)

    # Preview SAM (optional; comment out if running many files quickly)
    plt.figure(figsize=(8, 8))
    plt.imshow(sam, cmap='hot', vmin=0, vmax=np.pi/2)
    plt.colorbar(label='Spectral Angle (rad)')
    plt.title('SAM map vs ROI mean')
    plt.axis('off'); plt.show()

    # threshold + connected components
    sam_mask = (sam < sam_threshold)
    labeled, num = label(sam_mask, structure=np.ones((3, 3), dtype=int))

    masks = []
    for i in range(1, num + 1):
        m = (labeled == i)
        if m.sum() >= min_area_px:
            masks.append(m)

    # Preview masks (optional; comment out for speed)
    for i, m in enumerate(masks, 1):
        plt.figure(figsize=(5, 5))
        plt.imshow(m, cmap='gray')
        plt.title(f'ROI {i} mask preview')
        plt.axis('off'); plt.show()

    return masks

def write_means_single_sheet(out_path: str, wavelengths: np.ndarray, rows: list):
    """
    rows: list of tuples (sample_name, 1×B mean array)
    Writes/updates a single sheet 'ScaledMean'. If file exists, it appends by reading + rewrite.
    """
    import pandas as pd
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = [f"λ{w:.2f}" for w in wavelengths]

    # Build a DataFrame for new rows
    new_df = pd.DataFrame(
        np.vstack([r[1] for r in rows]).astype(np.float32),
        index=[r[0] for r in rows],
        columns=cols
    )
    new_df.index.name = "Sample"

    if os.path.exists(out_path):
        try:
            old = pd.read_excel(out_path, sheet_name=SHEET_NAME)
            old = old.set_index("Sample")
            combined = pd.concat([old, new_df], axis=0)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    # Rewrite single sheet
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as w:
        combined.to_excel(w, sheet_name=SHEET_NAME)
    print(f"✅ Wrote {len(rows)} new rows (total {combined.shape[0]}) → {out_path}")

# ------------------ MAIN BATCH ------------------
def process_one_folder(base_dir: str, folder: str, agg_rows: list, wavelengths_ref_holder: dict):
    """Process one folder: interactively pick ROI ref, segment, compute mean-from-scaled per ROI; append to agg_rows."""
    cube, wavelengths = load_cube(base_dir, folder)
    if ROTATE_K != 0:
        cube = np.rot90(cube, k=ROTATE_K)

    # keep wavelengths to use for column names (consistent)
    if "w" not in wavelengths_ref_holder:
        wavelengths_ref_holder["w"] = wavelengths

    # RGB for drawing
    rgb = rgb_preview(cube, wavelengths)

    # Draw one polygon reference and get component masks
    poly = draw_reference_polygon(rgb)
    masks = compute_sam_masks(cube, poly, SAM_THRESHOLD, MIN_AREA_PX)

    # For each component: scale rows then mean → append to aggregated rows
    for i, m in enumerate(masks, 1):
        roi = cube[m, :]                         # (Npix, B)
        scaled = min_max_scale_rows(roi)         # row-wise min–max
        mean_spec = scaled.mean(axis=0)          # (B,)
        sample_name = f"{folder}_ROI{i}"
        agg_rows.append((sample_name, mean_spec))
        print(f"  • {sample_name}: pixels={roi.shape[0]} (mean of scaled computed)")


def list_first_n_folders(base_dir: str, prefix_base: str, start_idx: int, end_idx: int):
    """
    Use an explicit list of folder names (instead of auto-generating them).
    Works perfectly when folder names don't follow a simple date prefix pattern.
    """
    folder_names = [
        "N2final27_2025-05-31_001","N2final27_2025-05-31_004","N2final27_2025-05-31_006",
        "N2final27_2025-05-31_008","N2final27_2025-05-31_010","N2final27_2025-05-31_012",
        "N2final27_2025-05-31_014","N2final27_2025-05-31_016","N2final27_2025-05-31_018",
        "N2final27_2025-05-31_020","N2final27_2025-05-31_022","N2final27_2025-05-31_024",
        "N2final27_2025-05-31_026","N2final27_2025-05-31_028","N2final27_2025-05-31_030",
        "N2final27_2025-05-31_032","N2final27_2025-05-31_034","N2final27_2025-05-31_036",
        "N2final27_2025-05-31_038","N2final27_2025-05-31_040","N2final27_2025-05-31_042",
        "N2final27_2025-05-31_044","N2final27_2025-05-31_046","N2final27_2025-05-31_048",
        "N2final27_2025-05-31_050","N2final27_2025-05-31_052","N2final27_2025-05-31_054",
        "N2final27_2025-05-31_056","N2final27_2025-05-31_058","N2final27_2025-05-31_060",
        "N2final27_2025-05-31_062","N2final27_2025-05-31_064","N2final27_2025-05-31_066",
        "N2final27_2025-05-31_068","N2final27_2025-05-31_070","N2final27_2025-05-31_072",
        "N2final27_2025-05-31_074","N2final27_2025-05-31_076","N2final27_2025-05-31_078",
        "N2final27_2025-05-31_080","N2final27_2025-05-31_082","N2final27_2025-05-31_084",
        "N2final27_2025-05-31_086","N2final27_2025-05-31_088","N2final27_2025-05-31_090",
        "N2final27_2025-05-31_092","N2final27_2025-05-31_094","N2final27_2025-05-31_097",
        "N2final27_2025-05-31_099","N2final27_2025-05-31_101","N2final27_2025-05-31_103",
        "N2final27_2025-05-31_105","N2final27_2025-05-31_107","N2final27_2025-05-31_109",
        "N2final27_2025-05-31_111","N2final27_2025-05-31_113","N2final27_2025-05-31_115",
        "N2final27_2025-05-31_117","N2final27_2025-05-31_119","N2final27_2025-05-31_121",
        "N2final27_2025-05-31_123","N2final27_2025-05-31_125","N2final27_2025-05-31_127",
        "N2final27_2025-05-31_129","N2final27_2025-05-31_131","N2final27_2025-05-31_133",
        "N2final27_2025-05-31_135","N2final27_2025-05-31_137","N2final27_2025-05-31_139",
        "N2final27_2025-05-31_141","N2final27_2025-05-31_143","N2final27_2025-05-31_145",
        "N2final27_2025-05-31_147","N2final27_2025-05-31_149","N2final27_2025-05-31_151",
        "N2final27_2025-05-31_153","N2final27_2025-05-31_155","N2final27_2025-05-31_157"
    ]

    out = []
    for name in folder_names:
        if os.path.isdir(os.path.join(base_dir, name)):
            out.append(name)
    return out

def main():
    # collect the listed folders that exist under BASE_DIR
    targets = list_first_n_folders(BASE_DIR, PREFIX_BASE, START_IDX, END_IDX)
    if not targets:
        print("⚠️ No matching folders found.")
        return

    print("Targets:", targets)
    aggregated_rows = []               # will store tuples (sample_name, mean_spectrum)
    wavelengths_holder = {}            # to store wavelengths once

    for folder in targets:
        print(f"\nProcessing: {folder}")
        process_one_folder(BASE_DIR, folder, aggregated_rows, wavelengths_holder)

    if aggregated_rows and "w" in wavelengths_holder:
        write_means_single_sheet(OUTPUT_XLSX, wavelengths_holder["w"], aggregated_rows)
    else:
        print("⚠️ Nothing to write.")

if __name__ == "__main__":
    main()
