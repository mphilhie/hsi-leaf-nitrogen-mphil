import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg', 'Agg', etc., if you have it installed

import matplotlib.pyplot as plt

def plot_all_scaled_spectra(excel_file, sheet_name="Mean Spectra", save_path="All_Scaled_Spectra.png"):
    # Load mean spectra
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Extract ROI names and spectra
    roi_names = df["ROI"]
    spectra = df.drop(columns=["ROI"]).values
    wavelengths = np.linspace(397.32, 1003.58, spectra.shape[1])

    # Plot
    plt.figure(figsize=(14, 6))
    for i in range(len(spectra)):
        plt.plot(wavelengths, spectra[i], label=roi_names[i], alpha=0.6)

    plt.title("Mean Scaled Spectra Across All ROIs")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scaled Reflectance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Plot saved as: {save_path}")


if __name__ == "__main__":
    plot_all_scaled_spectra("Scale_Data/120/120D_Mean_Spectra_Summary.xlsx")
