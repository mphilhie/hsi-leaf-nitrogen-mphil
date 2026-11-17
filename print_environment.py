"""Print the Python and package versions used to run this code."""
import sys
from importlib.metadata import version, PackageNotFoundError

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib",
            "seaborn", "statsmodels", "spectral", "scikit-image", "joblib",
            "openpyxl", "xlsxwriter"]

print(f"Python {sys.version.split()[0]}")
for p in PACKAGES:
    try:
        print(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print(f"{p}: not installed")
