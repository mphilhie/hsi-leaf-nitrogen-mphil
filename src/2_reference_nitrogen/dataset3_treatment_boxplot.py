# ------------------------------------------------------------
# Boxplot of Nitrogen Treatments (Dataset 3) with Tukey HSD letters
# ------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib as mpl
mpl.use('TkAgg')
# --- Step 1: Load leaf N% per treatment ---
# Input CSV (not included; see data/README.md) with columns:
#   Treatment  - nitrogen treatment (mg L-1): 40, 60, 80, 100 or 120
#   N_percent  - leaf nitrogen concentration (% dry mass, Dumas combustion)
DATA_CSV = "Dataset3_leaf_nitrogen.csv"
df = pd.read_csv(DATA_CSV)
df["Treatment"] = df["Treatment"].astype(int)

# --- Step 2: Tukey HSD to get grouping letters ---
tukey = pairwise_tukeyhsd(df["N_percent"], df["Treatment"], alpha=0.05)
print(tukey.summary())

# Determine compact letter display (manual grouping from Tukey)
# 40 and 60 share 'a', 80 and 100 share 'b', 120 is 'c'
letters = {40: "a", 60: "a", 80: "b", 100: "b", 120: "c"}

# --- Step 3: Plot ---
sns.set(style="whitegrid", font="serif", context="talk")
plt.figure(figsize=(8, 6))

palette = sns.color_palette("crest", 5)
ax = sns.boxplot(x="Treatment", y="N_percent", data=df, palette=palette,
                 width=0.6, showmeans=True,
                 meanprops={"marker": "o", "markerfacecolor": "black",
                            "markeredgecolor": "black", "markersize": 6})

# Annotate Tukey letters above boxes
positions = sorted(df["Treatment"].unique())
for i, pos in enumerate(positions):
    y = df[df["Treatment"] == pos]["N_percent"].max() + 0.25
    plt.text(i, y, letters[pos], ha="center", va="bottom",
             fontsize=16, fontweight="bold")

# --- Step 4: Labeling ---
plt.title("Effect of Nitrogen Treatment on Leaf Nitrogen Concentration", fontsize=16, pad=20)
plt.xlabel("Nitrogen Treatment (mg L$^{-1}$)", fontsize=14)
plt.ylabel("Leaf Nitrogen Concentration (%)", fontsize=14)
plt.tight_layout()

# --- Step 5: Save and show ---
plt.savefig("Dataset3_Nitrogen_Treatments_Boxplot.png", dpi=300)
plt.show()
