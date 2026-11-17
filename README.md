# Hyperspectral imaging for leaf nitrogen estimation (MPhil thesis code)

[![DOI](https://zenodo.org/badge/1461447190.svg)](https://doi.org/10.5281/zenodo.24311610)

Python code used in the MPhil thesis *Hyperspectral Imaging for Monitoring Crop
Quality in Protected Cropping Systems* (Kalana M. W. Ranasinghe, Western Sydney
University, 2026). The code extracts leaf reflectance spectra from close-range
hyperspectral images (Specim, approx. 397–1004 nm), compares nine regression
models for predicting leaf nitrogen concentration (N%, Dumas combustion), and
produces pixel-wise nitrogen maps from the selected model.

The research was carried out under the Future Food Systems CRC project
"Automated Crop Monitoring for Protected Cropping Systems".

> **Status:** this repository archives the code as it was used for the thesis.
> The scripts are research scripts: file paths, folder prefixes and thresholds
> are set at the top of each file and must be edited before running.

## Repository structure

```
notebooks/
  nitrogen_prediction_models.ipynb   Model comparison (Google Colab): PLSR, SVR, XGBoost,
                                     Random Forest, Ridge, Decision Tree, Lasso, KNN,
                                     AdaBoost x 13 preprocessing methods; Lasso wavelength
                                     selection; saving the best model bundle (.joblib)
src/
  1_spectral_extraction/             Hyperspectral cube -> leaf spectra
    multi_leaf_spectrum_extractor*.py   Reflectance calibration, SAM-based leaf
                                        segmentation, per-leaf mean spectra
    batch_mean_spectrum_extractor.py    Batch version over many image folders
    manual_roi_extractor.py             Manual polygon ROI extraction
    rowwise_scaling_and_mean.py         Row-wise min-max scaling and mean spectra
  2_reference_nitrogen/              Reference N% data (Dataset 3)
    dataset3_anova_tukey_welch.py       One-way ANOVA, Tukey HSD, Welch t-tests
    dataset3_plant_mean_anova.py        ANOVA / Tukey HSD on plant means
    dataset3_treatment_boxplot.py       Boxplot with Tukey grouping letters
    remove_nitrogen_outliers_iqr.py     1.5 x IQR outlier removal on N%
  3_modelling/
    plsr_preprocessing_comparison.py    Stand-alone PLSR x preprocessing comparison
  4_nitrogen_maps/                   Pixel-wise nitrogen prediction maps
    predict_leaf_heatmaps.py            Per-leaf heatmaps (SAM segmentation)
    predict_heatmap_manual_rois.py      Heatmap with manually drawn reference ROIs
    predict_heatmap_auto_reference.py   Heatmap with automatic reference selection
  5_results_and_figures/             Results tables and figures
data/
  README.md                          Expected input format (data not included)
```

## Workflow

1. **Reflectance calibration.** Raw cubes are converted to reflectance with the
   white and dark references: `(raw - dark) / (white - dark)`.
2. **Leaf segmentation and spectra.** A reference spectrum is taken from a
   hand-drawn polygon; the Spectral Angle Mapper (SAM) and connected-component
   labelling segment individual leaves; the mean spectrum of each leaf is exported
   (`src/1_spectral_extraction`). This step is interactive (matplotlib, TkAgg).
3. **Reference nitrogen.** Leaf N% from Dumas combustion is joined to the spectra;
   treatment effects in Dataset 3 are tested with ANOVA and Tukey HSD
   (`src/2_reference_nitrogen`).
4. **Model comparison** (`notebooks/nitrogen_prediction_models.ipynb`), see below.
5. **Nitrogen maps.** The saved model bundle is applied to every leaf pixel to
   produce nitrogen heatmaps (`src/4_nitrogen_maps`).

## Model-comparison settings (for reproducibility)

| Setting | Value in the code |
| --- | --- |
| Train/test partitioning | 10 repeated random 90:10 splits, `train_test_split(test_size=0.1, random_state=seed)`, seeds 0–9 |
| Hyperparameter tuning | `GridSearchCV(cv=5, scoring="neg_mean_squared_error")` on each training set; `cv=5` gives unshuffled 5-fold `KFold`, so folds are deterministic |
| Validation predictions | `cross_val_predict(best_model, X_train, y_train, cv=5)` |
| Estimator seeds | Random Forest, Decision Tree, AdaBoost: `random_state=seed`; other models are deterministic or use library defaults |
| Preprocessing | Raw, MSC, SNV, Savitzky–Golay, 1st/2nd derivative, log 1st/2nd derivative, and MSC combinations (13 in total); Savitzky–Golay window 11, polynomial order 2 |
| Feature scaling | Models were trained on unscaled reflectance; the Lasso variants in the last notebook cells use `StandardScaler` inside a `Pipeline` |
| Metrics | R², RMSE, MAE (train, 5-fold validation, test) and RPD = SD(y_test) / RMSE(test), reported as mean ± SD over the 10 splits |

Hyperparameter grids:

| Model | Grid |
| --- | --- |
| PLSR | `n_components`: 1–20 |
| SVR (RBF kernel) | `C`: 0.01, 0.1, 1, 10, 100, 1000; `gamma`: scale, auto |
| XGBoost | `n_estimators`: 100; `max_depth`: 3, 5, 7; `learning_rate`: 0.01, 0.1; `subsample`: 0.8; `colsample_bytree`: 0.8 |
| Random Forest | `n_estimators`: 100, 200; `max_depth`: 5, 10, None; `max_features`: sqrt, log2 |
| Ridge | `alpha`: 0.01, 0.1, 1, 10, 100 |
| Decision Tree | `max_depth`: 3, 5, 10, None; `min_samples_split`: 2, 5, 10; `min_samples_leaf`: 1, 2, 4 |
| Lasso (`max_iter=10000`) | `alpha`: 0.0001, 0.001, 0.01, 0.1, 1, 10 |
| KNN | `n_neighbors`: 3, 5, 7, 9; `weights`: uniform, distance; `p`: 1, 2 |
| AdaBoost | `n_estimators`: 50, 100, 200; `learning_rate`: 0.01, 0.1, 1.0 |

**Lasso wavelength selection and model bundle.** In the Lasso workflow, wavelengths
with non-zero Lasso coefficients are selected on each training set and an ordinary
least squares model (`LinearRegression`) is refitted on the selected wavelengths.
The bundle saved for nitrogen mapping (`*_BEST.joblib`) contains the preprocessing
name, the selected wavelength indices, the selected `alpha` and the refitted OLS
model.

**Known limitations.** MSC uses the mean spectrum of the whole dataset as its
reference and is applied before the train/test split. The leaf segmentation step
is interactive, so ROI placement is not fully scripted.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

The notebook was run on Google Colab. Remove or skip the `google.colab` drive-mount
cell to run it locally, and set the CSV file names at the top of each cell.

## Data availability

The hyperspectral images and reference nitrogen data are not included in this
repository. They are available from the author on reasonable request, subject to
the data-sharing conditions of the Future Food Systems CRC and NSW DPIRD. The
expected file formats are described in [`data/README.md`](data/README.md).

## Citation

If you use this code, please cite the thesis and this software archive (see
[`CITATION.cff`](CITATION.cff)):

> Ranasinghe, K. M. W. (2026). *Hyperspectral imaging pipeline for leaf nitrogen
> estimation (MPhil thesis code)* (Version 1.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.24311611

## Acknowledgements

Supervisors: Prof. Oula Ghannoum, A/Prof. Yi Guo and Dr Ethan Shi (Western Sydney
University). Dr Shi assisted with the Python code used for model development.

## License

MIT, see [`LICENSE`](LICENSE).
