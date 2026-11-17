import os
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from matplotlib.patches import Polygon
from matplotlib.path import Path


global_excel_file_name = "4)Test/Test.xlsx"
print("Current backend:", mpl.get_backend())

# Global variable for wavelengths
band_wavelengths = None


class ROINavigation:
    def __init__(self, rois, image, corrected_nparr, band_wavelengths, part_number):
        self.rois = rois
        self.image = image
        self.corrected_nparr = corrected_nparr
        self.band_wavelengths = band_wavelengths
        self.current_idx = 0
        self.part_number = part_number
        self.process_all_rois()

    def process_all_rois(self):
        """Automatically process all ROIs using polygon masks and write their spectra to Excel."""
        all_sheets = {}

        for idx, roi in enumerate(self.rois):
            # ✅ Create binary mask using polygon directly (no extra rotation)
            roi_mask = np.zeros(self.image.shape[:2], dtype=bool)
            roi_path = Path(roi[:, ::-1])  # Path expects (x, y), so flip columns

            y_coords, x_coords = np.meshgrid(
                np.arange(self.image.shape[0]),
                np.arange(self.image.shape[1]),
                indexing='ij'
            )
            coords = np.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
            roi_mask_flat = roi_path.contains_points(coords)
            roi_mask[y_coords, x_coords] = roi_mask_flat.reshape(self.image.shape[:2])

            # ✅ Visualize the mask
            plt.figure(figsize=(5, 5))
            plt.imshow(roi_mask, cmap='gray')
            plt.title(f"Polygon Mask for ROI {idx + 1}")
            plt.axis('off')
            plt.show()

            roi_spectra = self.corrected_nparr[roi_mask]
            print(roi_spectra.shape)

            if roi_spectra.shape[0] == 0:
                print(f"⚠️ ROI {idx + 1} has no valid pixels.")
                continue

            sheet_name = f"{self.part_number}_ROI{idx + 1}"

            all_sheets[sheet_name] = {
                "raw": roi_spectra,
            }

            print(f"✅ Processed ROI {idx + 1} with {roi_spectra.shape[0]} pixels")

        # Save all ROI results to Excel
        save_all_rois_to_excel(global_excel_file_name, all_sheets, self.band_wavelengths)


def safe_division(numerator, denominator):
    """Safely divide two arrays, avoiding division by zero."""
    return np.divide(numerator, denominator, where=(denominator != 0), out=np.zeros_like(numerator))

def load_and_process_hyperspectral(base_path, part_number):
    """Load and process hyperspectral data to calculate reflectance."""
    hdr_file = f'{base_path}/{part_number}/results/REFLECTANCE_{part_number}.hdr'
    data_ref_hdr_path = f'{base_path}/{part_number}/capture/{part_number}.hdr'
    white_ref_hdr_path = f'{base_path}/{part_number}/capture/WHITEREF_{part_number}.hdr'
    dark_ref_hdr_path = f'{base_path}/{part_number}/capture/DARKREF_{part_number}.hdr'

    img = envi.open(hdr_file)
    global band_wavelengths
    band_wavelengths = np.array(img.bands.centers)


    data_nparr = np.array(envi.open(data_ref_hdr_path, f'{base_path}/{part_number}/capture/{part_number}.raw').load())
    white_nparr = np.array(envi.open(white_ref_hdr_path, f'{base_path}/{part_number}/capture/WHITEREF_{part_number}.raw').load())
    dark_nparr = np.array(envi.open(dark_ref_hdr_path, f'{base_path}/{part_number}/capture/DARKREF_{part_number}.raw').load())

    corrected_nparr = safe_division(data_nparr - dark_nparr, white_nparr - dark_nparr)
    return corrected_nparr, band_wavelengths


def spectral_angle(a, b):
    """Compute spectral angle (in radians) between two spectra."""
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    dot_product = np.sum(a * b, axis=-1)
    angle = np.arccos(np.clip(dot_product / (a_norm * b_norm + 1e-10), -1, 1))
    return angle
def draw_rois(image, corrected_nparr, threshold=0.5):
    """Draw ROIs on an image and extract coordinates matching NDVI threshold."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("Hold 'Alt' to draw ROIs. Press 'd' to delete, 'q' to quit.")
    rois, current_polygon, line, dots = [], [], None, []
    alt_pressed = False

    def is_close_to_start(point, vertices, threshold=5):
        if not vertices:
            return False
        return np.linalg.norm(np.array(point) - np.array(vertices[0])) < threshold

    def on_click(event):
        nonlocal line, current_polygon, dots
        if alt_pressed and event.button == 1 and event.inaxes is ax:
            if current_polygon and is_close_to_start((event.xdata, event.ydata), current_polygon):
                current_polygon.append(current_polygon[0])
                rois.append(current_polygon.copy())
                polygon = Polygon(current_polygon, closed=True, edgecolor='r', facecolor='none', lw=2)
                ax.add_patch(polygon)
                current_polygon.clear()
                for dot in dots:
                    dot.remove()
                dots.clear()
                line = None
                fig.canvas.draw()
                print(f"ROI {len(rois)} added.")
            else:
                current_polygon.append((event.xdata, event.ydata))
                dot, = ax.plot(event.xdata, event.ydata, 'ro')  # Mark the point
                dots.append(dot)  # Append to dots
                if line is not None:
                    line.set_data([p[0] for p in current_polygon], [p[1] for p in current_polygon])
                else:
                    line, = ax.plot([p[0] for p in current_polygon], [p[1] for p in current_polygon], 'r-')
                fig.canvas.draw()

    def on_key_press(event):
        nonlocal alt_pressed
        if event.key == 'alt':
            alt_pressed = True
        elif event.key == 'd' and rois:
            rois.pop()
            ax.patches.pop()
            fig.canvas.draw()
            print("Most recent ROI deleted.")
        elif event.key == 'q':
            plt.close(fig)

    def on_key_release(event):
        nonlocal alt_pressed
        if event.key == 'alt':
            alt_pressed = False

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    plt.show()

    roi = rois[0]

    # Create a binary mask for the ROI
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_path = Path(roi)
    y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    points = np.stack((x.flatten(), y.flatten()), axis=-1)
    mask[roi_path.contains_points(points).reshape(image.shape[:2])] = 255
    roi_mask = mask.astype(bool)

    # Extract all spectra inside ROI
    spectra = corrected_nparr[roi_mask, :]  # (num_pixels_in_roi, num_bands)

    # Define a reference spectrum (for example: mean of ROI spectra)
    reference_spectrum = np.mean(spectra, axis=0)

    # Compute spectral angle for entire image
    image_spectra = corrected_nparr.reshape(-1, corrected_nparr.shape[-1])
    sam_angles = spectral_angle(image_spectra, reference_spectrum)
    sam_image = sam_angles.reshape(image.shape[:2])

    # Visualize full SAM image to select a good threshold
    plt.figure(figsize=(8, 8))
    plt.imshow(sam_image, cmap='hot', vmin=0, vmax=np.pi / 2)
    plt.colorbar(label='Spectral Angle (radians)')
    plt.title('Full SAM Image')
    plt.axis('off')
    plt.show()

    # Threshold: pixels with small angle (good match)
    sam_threshold = 0.2  # start with 0.2 first!
    sam_mask = sam_image < sam_threshold

    # Label connected components (fruits)
    from scipy.ndimage import label
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(sam_mask, structure=structure)

    print(f"Detected {num_features} initial components")

    # Extract coordinates per fruit with area filtering and sorting
    min_area = 100  # adjust this value based on your image!

    fruit_infos = []

    for i in range(1, num_features + 1):
        fruit_mask = labeled == i
        matching_indices = np.argwhere(fruit_mask)

        # Filter small blobs
        if len(matching_indices) < min_area:
            continue  # skip small blob

        # Compute centroid
        centroid_y = np.mean(matching_indices[:, 0])
        centroid_x = np.mean(matching_indices[:, 1])

        fruit_infos.append({
            'indices': matching_indices,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y
        })

    print(f"After filtering: {len(fruit_infos)} fruits")

    # Sort by row (y), then by column (x)
    row_tolerance = 20  # tune this if needed
    fruit_infos_sorted = sorted(fruit_infos, key=lambda f: (round(f['centroid_y'] / row_tolerance), f['centroid_x']))

    # Save sorted coordinates
    extracted_coordinates = []

    for i, fruit in enumerate(fruit_infos_sorted):
        matching_indices = fruit['indices']
        extracted_coordinates.append(matching_indices)

        # Optional visualize
        plt.figure(figsize=(6, 6))
        fruit_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        fruit_mask[matching_indices[:, 0], matching_indices[:, 1]] = 1

        plt.imshow(fruit_mask, cmap='gray')
        plt.title(f'Sorted Fruit {i + 1}')
        plt.axis('off')
        plt.show()

    return extracted_coordinates

def visualize_corrected_nparr_as_rgb(corrected_nparr, band_wavelengths):
    """Generate RGB visualization of hyperspectral data."""
    red_idx = np.argmin(np.abs(band_wavelengths - 650))
    green_idx = np.argmin(np.abs(band_wavelengths - 550))
    blue_idx = np.argmin(np.abs(band_wavelengths - 425))
    rgb_image = np.stack([
        corrected_nparr[:, :, red_idx],
        corrected_nparr[:, :, green_idx],
        corrected_nparr[:, :, blue_idx]
    ], axis=-1)
    return (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())


def visualize_rois(image, rois):
    """
    Visualize the image with all ROIs rotated 90 degrees clockwise.

    Args:
        image (numpy.ndarray): The RGB image.
        rois (list of numpy.ndarray): List of ROIs, where each ROI is an array of points [(x1, y1), (x2, y2), ...].
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Rotate and draw each ROI
    for idx, roi in enumerate(rois):
        rotated_roi = roi
        plt.plot(rotated_roi[:, 1], rotated_roi[:, 0], label=f"ROI {idx + 1}", linewidth=2)  # Adjusted axes order

    plt.legend(loc="upper right")
    plt.title("Visualized ROIs (Rotated 90° Clockwise) on Image")
    plt.axis("off")

    base_filename = os.path.splitext(os.path.basename(global_excel_file_name))[0]
    plot_filename = f"{base_filename}_visualized_rois.png"

    # Optional: Save to the same directory as the Excel file
    output_dir = os.path.dirname(global_excel_file_name)
    output_path = os.path.join(output_dir, plot_filename)

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def process_file(file_path, part_number):
    """Process a single hyperspectral file."""
    # Load hyperspectral data
    corrected_nparr, band_wavelengths = load_and_process_hyperspectral(file_path, part_number)
    corrected_nparr = np.rot90(corrected_nparr, k=-1)

    # Generate and display RGB image
    rgb_image = visualize_corrected_nparr_as_rgb(corrected_nparr, band_wavelengths)

    # Draw ROIs and process them
    rois = draw_rois(rgb_image, corrected_nparr)
    print(f"Visualizing ROIs for {os.path.basename(file_path)}:")
    visualize_rois(rgb_image, rois)
    print(f"Launching ROI navigation for {os.path.basename(file_path)}:")
    ROINavigation(rois, rgb_image, corrected_nparr, band_wavelengths,part_number)
    plt.show()


def save_all_rois_to_excel(excel_path, all_sheets, wavelengths, flag=False):
    import pandas as pd

    col_names = [f"λ{w:.2f}" for w in wavelengths]
    file_exists = os.path.exists(excel_path)

    if flag and file_exists:
        print(f"\n⚠️ File exists and flag is True. Skipping file creation: {excel_path}\n")
        return

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    total = len(all_sheets)
    print(f"\n📝 Writing {total} ROI sheets to: {excel_path}\n")

    mode = "a" if file_exists else "w"

    # Dynamically build ExcelWriter arguments
    writer_args = {
        "path": excel_path,
        "engine": "openpyxl",
        "mode": mode,
    }
    if mode == "a":
        writer_args["if_sheet_exists"] = "new"

    with pd.ExcelWriter(**writer_args) as writer:
        for idx, (sheet_name, data) in enumerate(all_sheets.items(), start=1):
            raw = data["raw"]
            startrow = 0

            # RAW SPECTRA
            pd.DataFrame({"Section": ["RAW SPECTRA"]}).to_excel(
                writer, sheet_name=sheet_name, index=False, header=False, startrow=startrow
            )
            startrow += 1
            pd.DataFrame(raw, columns=col_names).to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=startrow
            )
            startrow += len(raw) + 2
            print(f" [{idx}/{total}] Saved ROI sheet: {sheet_name}")

    print(f"\n✅ All ROI sheets saved successfully to '{excel_path}'\n")


def main():
    base_path = './'  # Set your base folder path here


    # Loop through all image folders in base_path
    for file_name in os.listdir(base_path):
        if file_name.startswith('n2suzzy_a_2025-06-10_003'):
            file_path = os.path.join(base_path, file_name)
            print(f"Processing file: {file_name}")
            process_file(base_path, file_name)


if __name__ == "__main__":
    main()
