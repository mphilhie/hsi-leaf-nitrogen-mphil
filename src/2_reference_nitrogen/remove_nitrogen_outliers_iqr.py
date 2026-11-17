import pandas as pd

# Load data
df = pd.read_csv(
    "My_Data_Final_Front- 21_10_2025(Sheet1) (1).csv"
)

# -------------------------------
# Outlier removal using IQR on N%
# -------------------------------
Q1 = df["N%"].quantile(0.25)
Q3 = df["N%"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[
    (df["N%"] >= lower_bound) &
    (df["N%"] <= upper_bound)
]

# -------------------------------
# Save as CSV
# -------------------------------
output_path = "My_Data_Npercent_Outliers_Removed.csv"
df_clean.to_csv(output_path, index=False)

print("✅ Original rows :", len(df))
print("✅ Remaining rows:", len(df_clean))
print("✅ Removed rows  :", len(df) - len(df_clean))
print("✅ Saved to:", output_path)
