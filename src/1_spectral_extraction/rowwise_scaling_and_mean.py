import numpy as np
import pandas as pd
import os
output_dir = "Scale_Data_Back/100/"
filename = "All_Scaled_Spectra.xlsx"
filename2 = "Mean_Spectra_Summary.xlsx"
prefix = "100D_Back"
def read_raw_spectra(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
    df = df.dropna(how='all')
    df.columns = [str(col) for col in df.columns]
    return df

def scale_row_wise(data):
    min_vals = data.min(axis=1).reshape(-1, 1)
    max_vals = data.max(axis=1).reshape(-1, 1)
    scaled = (data - min_vals) / (max_vals - min_vals + 1e-8)
    return scaled

def process_all_rois(excel_file):
    sheet_names = pd.ExcelFile(excel_file).sheet_names
    wavelengths = np.linspace(397.32, 1003.58, pd.read_excel(excel_file, sheet_name=sheet_names[0], skiprows=1).shape[1])
    col_names = [f"λ{w:.2f}" for w in wavelengths]

    # Excel writers
    output_file_path = os.path.join(output_dir, prefix + filename)
    scaled_writer = pd.ExcelWriter(output_file_path, engine="xlsxwriter")
    mean_list = []
    mean_sheet_names = []

    for sheet in sheet_names:
        print(f"📄 Processing: {sheet}")
        raw_df = read_raw_spectra(excel_file, sheet)
        raw_data = raw_df.values.astype(np.float32)

        scaled_data = scale_row_wise(raw_data)
        scaled_df = pd.DataFrame(scaled_data, columns=col_names)
        scaled_df.to_excel(scaled_writer, index=False, sheet_name=sheet)

        mean_spectrum = np.mean(scaled_data, axis=0)
        mean_list.append(mean_spectrum)
        mean_sheet_names.append(sheet)
    # Save all scaled spectra
    scaled_writer.close()
    print(f"✅ Saved mean spectra summary into: {output_file_path}")

    # Save all means into one sheet
    mean_df = pd.DataFrame(mean_list, columns=col_names)
    mean_df.insert(0, "ROI", mean_sheet_names)
    output_file_path2 = os.path.join(output_dir,prefix +  filename2)
    with pd.ExcelWriter(output_file_path2, engine="xlsxwriter") as mean_writer:
        mean_df.to_excel(mean_writer, index=False, sheet_name="Mean Spectra")
    print(f"✅ Saved mean spectra summary into: {output_file_path2}")


if __name__ == "__main__":
    excel_file = "My_Data_3-Back/100/100D_back.xlsx"
    process_all_rois(excel_file)
