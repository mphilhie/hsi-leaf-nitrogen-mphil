import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_each_roi_scaled_spectra(excel_file, output_dir="40_B"):
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Load all sheet names
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names

    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        spectra = df.values
        wavelengths = np.linspace(397.32, 1003.58, spectra.shape[1])

        plt.figure(figsize=(12, 5))
        for i in range(spectra.shape[0]):
            plt.plot(wavelengths, spectra[i], alpha=0.4)

        plt.title(f"Scaled Spectra – {sheet}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Scaled Reflectance")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{sheet}_scaled_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"✅ Saved: {plot_path}")

if __name__ == "__main__":
    plot_each_roi_scaled_spectra("My_Data/40/40A.xlsx")
