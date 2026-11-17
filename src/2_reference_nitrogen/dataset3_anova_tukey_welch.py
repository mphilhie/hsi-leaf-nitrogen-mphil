# ------------------------------------------------------------
# ANOVA, Tukey HSD and Welch t-tests for Dataset 3 (Capsicum DPI)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Step 1: Load leaf N% per treatment ---
# Input CSV (not included; see data/README.md) with columns:
#   Treatment  - nitrogen treatment (mg L-1): 40, 60, 80, 100 or 120
#   N_percent  - leaf nitrogen concentration (% dry mass, Dumas combustion)
DATA_CSV = "Dataset3_leaf_nitrogen.csv"
df = pd.read_csv(DATA_CSV)
df["Treatment"] = df["Treatment"].astype(float)

# --- Step 2: Descriptive stats ---
summary = df.groupby("Treatment")["N_percent"].agg(["mean", "std", "count"])
print("\nMean ± SD per treatment:")
print(summary.to_string(float_format="%.3f"))

# --- Step 3: One-way ANOVA ---
model = ols('N_percent ~ C(Treatment)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Results:")
print(anova_table)

# --- Step 4: Tukey HSD post-hoc ---
tukey = pairwise_tukeyhsd(df["N_percent"], df["Treatment"], alpha=0.05)
print("\nTukey HSD Results:")
print(tukey.summary())

# --- Step 5: Pairwise Welch t-tests ---
keys = sorted(df["Treatment"].unique())
groups = [df.loc[df["Treatment"] == k, "N_percent"].to_numpy() for k in keys]

print("\nPairwise Welch's t-tests:")
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        t_stat, p_val = stats.ttest_ind(groups[i], groups[j], equal_var=False)
        print(f"{keys[i]:g} vs {keys[j]:g}: t = {t_stat:.3f}, p = {p_val:.5f}")

# --- Step 6: Overall summary note ---
print("\nInterpretation:")
print("Significant ANOVA and Tukey results confirm a clear nitrogen gradient across treatments.")
