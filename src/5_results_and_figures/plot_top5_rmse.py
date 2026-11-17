import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg', 'Agg', etc., if you have it installed

import matplotlib.pyplot as plt
import numpy as np

# === Load the Excel file ===
file_path = "Sum_Back.xlsx"  # 🔁 Replace with your file path
df = pd.read_excel(file_path)

# === Clean column names ===
df.columns = df.columns.str.strip()

# === Keep only the mean part of the RMSE columns (ignore ± std) ===
rmse_cols = ["Train RMSE", "Val RMSE", "Test RMSE"]
for col in rmse_cols:
    df[col] = df[col].astype(str).str.split("±").str[0].astype(float)

# === Sort by Test R² and keep top 5 ===
df_top5 = df.sort_values(by="Test R²", ascending=False).head(5).reset_index(drop=True)

# === Grouped bar chart setup ===
labels = df_top5["Method & Model"]
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

rects1 = ax.bar(x - width, df_top5["Train RMSE"], width, label='Train RMSE', color='red')
rects2 = ax.bar(x,         df_top5["Val RMSE"],   width, label='Val RMSE',   color='yellow')
rects3 = ax.bar(x + width, df_top5["Test RMSE"],  width, label='Test RMSE',  color='blue')

# === Add text labels ===
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# === Final formatting ===
ax.set_ylabel('RMSE Score')
ax.set_title('Top 5 Models-Front: Train, Validation, and Test RMSE')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# === Show or Save ===
plt.savefig("top5_RMSE_comparison_back.png", dpi=300)  # Optional: save instead
plt.show()
