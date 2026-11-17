import pandas as pd
from pathlib import Path

# ---- CONFIG ----
INPUT = "results_dataset3_model_comparison.csv"
OUTPUT = "results_dataset3_model_comparison_combined.csv"
DEC = 3  # number of decimals to round

# ---- LOAD ----
df = pd.read_csv(INPUT)

# Normalize headers: replace Unicode 'R²' with 'R2', strip spaces
norm_cols = {c: c.replace('R²', 'R2').strip() for c in df.columns}
df.rename(columns=norm_cols, inplace=True)

# Helper to make "mean ± sd" text safely
def mean_pm_sd(mean_col, sd_col, fallback="–"):
    if mean_col in df.columns and sd_col in df.columns:
        return df[mean_col].round(DEC).astype(str) + " ± " + df[sd_col].round(DEC).astype(str)
    else:
        # graceful fallback if a pair is missing
        return pd.Series([fallback] * len(df))

# Build combined display columns
out = pd.DataFrame()
out["Model"] = df["Model"]

# R2 blocks
out["Test R2"]        = mean_pm_sd("Test R2 Mean",  "Test R2 Std")
out["Validation R2"]  = mean_pm_sd("Val R2 Mean",   "Val R2 Std")
out["Training R2"]    = mean_pm_sd("Train R2 Mean", "Train R2 Std")

# Error metrics (from TEST split to match your figure)
out["RMSE"]           = mean_pm_sd("Test RMSE Mean","Test RMSE Std")
out["MAE"]            = mean_pm_sd("Test MAE Mean", "Test MAE Std")
out["RPD"]            = mean_pm_sd("Test RPD Mean", "Test RPD Std")

# LCB (single value column, keep as rounded numeric string if present)
if "LCB" in df.columns:
    out["LCB"] = df["LCB"].round(DEC).astype(str)

# Save UTF-8 CSV
out.to_csv(OUTPUT, index=False)
print(f"✅ Wrote: {Path(OUTPUT).resolve()}")
