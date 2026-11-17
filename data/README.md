# Data

The data are not included in this repository. They are available from the author
on reasonable request, subject to the data-sharing conditions of the Future Food
Systems CRC and NSW DPIRD.

## Expected formats

**Hyperspectral images** (Specim folder layout), one folder per image:

```
<image_name>/
  capture/<image_name>.hdr, .raw
  capture/WHITEREF_<image_name>.hdr, .raw
  capture/DARKREF_<image_name>.hdr, .raw
  results/REFLECTANCE_<image_name>.hdr      (used for band wavelengths)
```

**Modelling table** (CSV), one row per leaf:

| Column | Content |
| --- | --- |
| `Sample` or `Name` (optional) | Leaf identifier; dropped before modelling |
| `λ397.32` ... `λ1003.58` | Mean reflectance at each band (one column per band) |
| `N%` | Leaf nitrogen concentration from Dumas combustion (% dry mass) |

**Dataset 3 reference nitrogen** (`Dataset3_leaf_nitrogen.csv`), used by
`src/2_reference_nitrogen/dataset3_anova_tukey_welch.py` and
`dataset3_treatment_boxplot.py`, one row per leaf:

| Column | Content |
| --- | --- |
| `Treatment` | Nitrogen treatment (mg L⁻¹): 40, 60, 80, 100 or 120 |
| `N_percent` | Leaf nitrogen concentration (% dry mass, Dumas combustion) |
