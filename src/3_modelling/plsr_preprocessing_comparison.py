import os
import shutil
import warnings
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


# ------------ Load CSV ------------
df = pd.read_csv("Blueberry_F.csv")
df = df.drop(columns=["Name"], errors="ignore")
destination_path = "./"
# Drop "N%" then drop first and last of the remaining feature columns
X_raw = df.drop(columns=["N%"]).iloc[:, 1:-1].values
y = df["N%"].values.astype(float)

# ------------ Utility: finite check ------------
def ensure_finite(X, tag="X"):
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

# ------------ Safe Savitzky-Golay helper ------------
def _valid_savgol_params(n_features, window_length, polyorder):
    # window must be odd, > polyorder, and <= n_features
    w = min(window_length, n_features)
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        w = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
    w = min(w, n_features if n_features % 2 == 1 else n_features - 1)
    if w < 3:
        w = 3 if n_features >= 3 else n_features | 1  # best effort
    return max(w, polyorder + 1 | 1)

def apply_savgol(X, params, deriv=0):
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    w = _valid_savgol_params(n_features, params['window_length'], params['polyorder'])
    return savgol_filter(X, window_length=w, polyorder=params['polyorder'], deriv=deriv, axis=1)

# ------------ Robust preprocessing funcs ------------
def apply_msc(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    ref = np.mean(X, axis=0)
    ref = ref - ref.mean()  # center ref
    out = np.empty_like(X, dtype=float)
    denom = np.dot(ref, ref) + eps
    for i, x in enumerate(X):
        x_c = x - x.mean()
        slope = np.dot(x_c, ref) / denom
        intercept = x.mean() - slope * ref.mean()  # zero-ish
        out[i] = (x - intercept) / (slope + eps)
    return out

def apply_snv(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

def apply_first_derivative(X, params):
    return apply_savgol(X, params, deriv=1)

def apply_second_derivative(X, params):
    return apply_savgol(X, params, deriv=2)

def apply_log_first_derivative(X, params):
    X = np.log1p(np.clip(X, 0, None))
    return apply_first_derivative(X, params)

def apply_log_second_derivative(X, params):
    X = np.log1p(np.clip(X, 0, None))
    return apply_second_derivative(X, params)

# ------------ Parameters ------------
savgol_params = {"window_length": 11, "polyorder": 2}

preprocessing_methods = {
    "Raw": lambda X: X,
    "MSC": apply_msc,
    "SNV": apply_snv,
    "Savitzky-Golay": lambda X: apply_savgol(X, savgol_params),
    "First Derivative": lambda X: apply_first_derivative(X, savgol_params),
    "Second Derivative": lambda X: apply_second_derivative(X, savgol_params),
    "Log First Derivative": lambda X: apply_log_first_derivative(X, savgol_params),
    "Log Second Derivative": lambda X: apply_log_second_derivative(X, savgol_params),
    "MSC + First Derivative": lambda X: apply_first_derivative(apply_msc(X), savgol_params),
    "MSC + Second Derivative": lambda X: apply_second_derivative(apply_msc(X), savgol_params),
    "MSC + Savitzky-Golay": lambda X: apply_savgol(apply_msc(X), savgol_params),
    "MSC + Log First Derivative": lambda X: apply_log_first_derivative(apply_msc(X), savgol_params),
    "MSC + Log Second Derivative": lambda X: apply_log_second_derivative(apply_msc(X), savgol_params),
}

# ------------ Modeling config ------------
cv_folds = 5
repeats = 10
summary_data = {}
output_filename = "PLSR_Full_Preprocessing_Comparison_Back.xlsx"

with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
    for method_name, preprocess_func in preprocessing_methods.items():
        print(f"🔁 Running: {method_name}")
        results = []

        # Preprocess and sanitize
        X_processed = preprocess_func(X_raw)
        X_processed = ensure_finite(X_processed, tag=method_name)

        for seed in range(repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.1, random_state=seed
            )

            # Cap n_components safely per inner CV fold
            n_features = X_train.shape[1]
            n_train = X_train.shape[0]
            n_train_per_fold = max(1, int(np.floor((cv_folds - 1) / cv_folds * n_train)))
            n_comp_max = max(1, min(n_features, n_train_per_fold))

            n_grid = list(range(1, min(20, n_comp_max) + 1))  # <= 20 and <= feasible
            grid = GridSearchCV(
                PLSRegression(),
                {'n_components': n_grid},
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                return_train_score=True,
                error_score='raise'
            )

            # Fit
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            # Predictions
            y_train_pred = best_model.predict(X_train).ravel()
            y_val_pred = cross_val_predict(best_model, X_train, y_train, cv=cv_folds).ravel()
            y_test_pred = best_model.predict(X_test).ravel()

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_train, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_train, y_val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_train, y_val_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # RPD (std/ RMSE) – guard zero std
            ytest_std = np.std(y_test)
            test_rpd = (ytest_std / test_rmse) if test_rmse > 0 else np.nan

            results.append({
                "Repeat": seed + 1,
                "Train R²": train_r2,
                "Train RMSE": train_rmse,
                "Train MAE": train_mae,
                "Val R²": val_r2,
                "Val RMSE": val_rmse,
                "Val MAE": val_mae,
                "Test R²": test_r2,
                "Test RMSE": test_rmse,
                "Test MAE": test_mae,
                "Test RPD": test_rpd,
                "Best n_components": grid.best_params_["n_components"],
            })

        df_result = pd.DataFrame(results)

        # Mean/Std rows (don’t average repeats or n_components together)
        stats_cols = ["Train R²","Train RMSE","Train MAE","Val R²","Val RMSE","Val MAE",
                      "Test R²","Test RMSE","Test MAE","Test RPD"]
        df_mean = df_result[stats_cols].mean()
        df_std  = df_result[stats_cols].std()

        # Write per-method sheet
        df_out = df_result.copy()
        df_out.loc[len(df_out)] = [np.nan] + list(df_mean.values) + [np.nan]  # Mean row
        df_out.loc[len(df_out)] = [np.nan] + list(df_std.values)  + [np.nan]  # Std row
        df_out.to_excel(writer, sheet_name=method_name[:31], index=False)

        # For summary sheet
        summary_data[method_name] = (df_mean.round(6).astype(str) + " ± " + df_std.round(4).astype(str))

    # Summary
    pd.DataFrame(summary_data).T.to_excel(writer, sheet_name="Summary")

print(f"✅ All preprocessing comparisons saved to {output_filename}")

# Safe move (optional)
try:
    target = os.path.join(destination_path, output_filename)
    shutil.move(output_filename, target)
    print(f"✅ PLSR File moved to {target}")
except Exception as e:
    print(f"ℹ️ Skipped move (set destination_path). Reason: {e}")
