import pandas as pd

input_file = "Without-Scaling-summary.xlsx"
output_file = "Without-Scaling-summary_FINAL_PRESENTATION.xlsx"

xls = pd.ExcelFile(input_file)
all_rows = []

# ---------- helper: format mean ± std ----------
def format_pm(value):
    if pd.isna(value):
        return value
    s = str(value)
    if "±" in s:
        mean, std = s.split("±")
        try:
            mean = float(mean.strip())
            std = float(std.strip())
            return f"{mean:.3f} ± {std:.3f}"
        except ValueError:
            return s
    else:
        # numeric only (fallback)
        try:
            return f"{float(s):.3f}"
        except ValueError:
            return s

# ----------------------------------------------

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    # Drop unnamed index columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalise column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("²", "2", regex=False)
    )

    required_cols = [
        "Method",
        "Test R2",
        "Train R2",
        "Val R2",
        "Test RMSE",
        "Test MAE",
        "Test RPD"
    ]

    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Skipping sheet '{sheet}'")
        continue

    out = df[required_cols].copy()

    # ✅ Build final Method = Model | Preprocessing
    out["Method"] = sheet + " | " + out["Method"].astype(str)

    # ✅ Format all metric columns to mean ± std (3 decimals)
    for col in required_cols:
        if col != "Method":
            out[col] = out[col].apply(format_pm)

    all_rows.append(out)

# ---------- Combine ----------
final_df = pd.concat(all_rows, ignore_index=True)

# ---------- Sort by MEAN Test R² ----------
final_df["_sort"] = (
    final_df["Test R2"]
    .str.split("±")
    .str[0]
    .astype(float)
)

final_df = final_df.sort_values(by="_sort", ascending=False)
final_df = final_df.drop(columns="_sort")

# ---------- Save ----------
final_df.to_excel(
    output_file,
    sheet_name="Mean_±_Std",
    index=False
)

print("✅ DONE: mean ± std formatted to 3 decimals (both values)")
