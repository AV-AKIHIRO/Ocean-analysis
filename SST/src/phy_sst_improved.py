
"""
Improved SST preprocessing for Sunny et al. (2023) five-ocean box model.

Purpose
-------
1. Load COBE/COBE-2 monthly SST.
2. Reproduce the five-ocean basin averages using the geographic boundaries
   stated in Sunny et al. Appendix A.
3. Fit monthly linear SST trends over Jan-2000 to Dec-2018.
4. Compare reproduced coefficients with Appendix D of the original paper.
5. Save machine-readable results and validation plots.

Important limitation
--------------------
The paper specifies the Indonesian archipelago as part of the Indian/Pacific
boundary, while the simple rules below use the paper's explicit meridians
(20E, 140E, 70W) plus 60S for the Southern Ocean. This is therefore a
"paper-boundary approximation", not a pixel-perfect recreation of the
original coastline mask. The purpose is to improve substantially over the
previous 280E/360E and 50S masks and to quantify the remaining mismatch.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# Configuration
# =========================
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "sst_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PATHS = [
    REPO_ROOT / "data" / "sst.mon.mean.nc",
    REPO_ROOT / "sst.mon.mean.nc",
    Path.home() / "Downloads" / "sst.mon.mean.nc",
    Path("/kaggle/input/datasets/aryanvaghasiya/cobe2-sst-dataset/sst.mon.mean.nc"),
    Path("sst.mon.mean.nc"),
]

FILE_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
START_DATE = "2000-01-01"
END_DATE = "2018-12-31"

OCEANS = ["NA", "SA", "SO", "PO", "IO"]

# Paper Appendix D values, temperature in K, t = months from Jan-2000.
PAPER_TRENDS = {
    "NA": {"slope": 0.00170, "intercept_K": 291.90},
    "SA": {"slope": -0.00015, "intercept_K": 290.30},
    "SO": {"slope": -0.00056, "intercept_K": 272.29},
    "PO": {"slope": 0.00155, "intercept_K": 292.88},
    "IO": {"slope": 0.00026, "intercept_K": 291.57},
}

# Original paper geographic split:
# 70 W = 290 E when longitude is stored on 0..360.
WEST_ATLANTIC = 290.0
EAST_ATLANTIC = 20.0
INDIAN_PACIFIC = 140.0
SOUTHERN_LAT = -60.0


def detect_temperature_units(da: xr.DataArray) -> str:
    """Return 'C' or 'K' using metadata and fallback value inspection."""
    units = str(da.attrs.get("units", "")).lower()

    if any(token in units for token in ["kelvin", "degk", " k"]):
        return "K"
    if any(token in units for token in ["celsius", "degc", "degree_c", "°c"]):
        return "C"

    # Fallback for datasets lacking reliable unit metadata.
    sample = float(da.mean(skipna=True).values)
    return "K" if sample > 150 else "C"


def to_kelvin(da: xr.DataArray) -> xr.DataArray:
    units = detect_temperature_units(da)
    if units == "K":
        return da
    out = da + 273.15
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "K"
    return out


def load_sst(path: Path) -> xr.DataArray:
    """Load, validate, and subset the SST variable."""
    if not path.exists():
        raise FileNotFoundError(
            f"SST file not found: {path}\n"
            "Update FILE_PATH to the location of sst.mon.mean.nc."
        )

    ds = xr.open_dataset(path)

    if "sst" not in ds:
        raise KeyError(f"Expected variable 'sst'; found: {list(ds.data_vars)}")

    sst = ds["sst"]

    required_dims = {"time", "lat", "lon"}
    if not required_dims.issubset(sst.dims):
        raise ValueError(f"SST must have dimensions {required_dims}; got {sst.dims}")

    sst = sst.sel(time=slice(START_DATE, END_DATE))

    if sst.sizes["time"] != 228:
        raise ValueError(
            f"Expected 228 monthly observations for 2000-01 through 2018-12; "
            f"got {sst.sizes['time']}."
        )

    return to_kelvin(sst)


def paper_boundary_masks(sst: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    Implement the paper's explicit large-scale geographic boundaries.

    Longitudes are assumed to be 0..360.
    The Atlantic therefore wraps across 360/0:
      290E (=70W) -> 360 -> 0 -> 20E.

    Southern Ocean is defined first because the paper includes all water
    south of 60S in that box.
    """
    lat = sst["lat"]
    lon = sst["lon"]

    southern = lat <= SOUTHERN_LAT

    atlantic = (lon >= WEST_ATLANTIC) | (lon < EAST_ATLANTIC)

    na = (lat > 0) & atlantic & ~southern
    sa = (lat <= 0) & atlantic & ~southern

    io = (lat > SOUTHERN_LAT) & (lat <= 0) & (lon >= EAST_ATLANTIC) & (lon < INDIAN_PACIFIC)
    po = (lat > SOUTHERN_LAT) & (lat <= 0) & (lon >= INDIAN_PACIFIC) & (lon < WEST_ATLANTIC)

    return {"NA": na, "SA": sa, "SO": southern, "PO": po, "IO": io}


def previous_masks(sst: xr.DataArray) -> dict[str, xr.DataArray]:
    """Recreate the masks used in the current notebook for comparison."""
    lat = sst["lat"]
    lon = sst["lon"]

    return {
        "NA": (lat > 0) & (lon >= 280) & (lon <= 360),
        "SA": (lat <= 0) & (lon >= 280) & (lon <= 360),
        "SO": lat <= -50,
        "IO": (lon >= 20) & (lon <= 120) & (lat > -50),
        "PO": (lon >= 120) & (lon <= 280) & (lat > -50),
    }


def basin_means(sst: xr.DataArray, masks: dict[str, xr.DataArray]) -> pd.DataFrame:
    """Spatial mean SST by basin, monthly."""
    records = []
    for ocean in OCEANS:
        basin = sst.where(masks[ocean])
        ts = basin.mean(dim=("lat", "lon"), skipna=True)

        values = ts.values.astype(float)
        if np.all(np.isnan(values)):
            raise ValueError(f"No valid SST cells found for {ocean}.")

        records.append(pd.Series(values, index=pd.to_datetime(ts["time"].values), name=ocean))

    out = pd.concat(records, axis=1)
    out.index.name = "month"

    return out


def fit_trends(monthly_K: pd.DataFrame) -> pd.DataFrame:
    """Fit T(t) = slope * t + intercept, with t in months from Jan-2000."""
    t = np.arange(len(monthly_K), dtype=float).reshape(-1, 1)
    rows = []

    for ocean in OCEANS:
        y = monthly_K[ocean].to_numpy(dtype=float)

        if np.any(~np.isfinite(y)):
            raise ValueError(f"Non-finite SST values remain for {ocean}.")

        model = LinearRegression().fit(t, y)
        pred = model.predict(t)
        residual = y - pred

        rows.append({
            "ocean": ocean,
            "slope_K_per_month": float(model.coef_[0]),
            "intercept_K": float(model.intercept_),
            "r2": float(model.score(t, y)),
            "rmse_K": float(np.sqrt(np.mean(residual**2))),
            "paper_slope_K_per_month": PAPER_TRENDS[ocean]["slope"],
            "paper_intercept_K": PAPER_TRENDS[ocean]["intercept_K"],
            "slope_abs_diff": float(model.coef_[0] - PAPER_TRENDS[ocean]["slope"]),
            "intercept_abs_diff_K": float(model.intercept_ - PAPER_TRENDS[ocean]["intercept_K"]),
        })

    result = pd.DataFrame(rows)
    result["slope_pct_error"] = (
        100 * result["slope_abs_diff"] / result["paper_slope_K_per_month"].abs()
    ).replace([np.inf, -np.inf], np.nan)

    return result


def save_plots(monthly_K: pd.DataFrame, trend_table: pd.DataFrame, prefix: str = "sst"):
    """Save actual-vs-trend plots and basin time series."""
    t = np.arange(len(monthly_K), dtype=float)

    # All time series
    fig, ax = plt.subplots(figsize=(12, 6))
    for ocean in OCEANS:
        ax.plot(monthly_K.index, monthly_K[ocean] - 273.15, label=ocean)
    ax.set_title("SST by Ocean Basin — Paper Geographic Boundaries")
    ax.set_xlabel("Date")
    ax.set_ylabel("SST (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sst_basin_timeseries.png", dpi=300)
    plt.close(fig)

    # Actual vs fitted
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(11, 18), sharex=True)
    for ax, ocean in zip(axes, OCEANS):
        row = trend_table.loc[trend_table["ocean"] == ocean].iloc[0]
        pred_K = row["slope_K_per_month"] * t + row["intercept_K"]

        ax.plot(monthly_K.index, monthly_K[ocean] - 273.15,
                label="COBE-2 basin mean")
        ax.plot(monthly_K.index, pred_K - 273.15,
                linestyle="--", label=f"Linear fit R²={row['r2']:.4f}")
        ax.set_title(ocean)
        ax.set_ylabel("°C")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sst_actual_vs_trend_paper_boundaries.png", dpi=300)
    plt.close(fig)


def main():
    sst = load_sst(FILE_PATH)

    print("SST dimensions:", sst.sizes)
    print("SST units:", sst.attrs.get("units"))
    print("Time range:", str(sst["time"].values[0]), "to", str(sst["time"].values[-1]))

    # 1) Existing implementation — baseline.
    old = basin_means(sst, previous_masks(sst))
    old.to_csv(OUTPUT_DIR / "sst_previous_masks.csv")

    # 2) Improved paper-boundary approximation.
    improved = basin_means(sst, paper_boundary_masks(sst))
    improved.to_csv(OUTPUT_DIR / "sst_monthly_paper_boundaries_K.csv")

    # Convenient °C version for inspection.
    improved_C = improved - 273.15
    improved_C.to_csv(OUTPUT_DIR / "sst_monthly_paper_boundaries_C.csv")

    # 3) Trend fit.
    trends = fit_trends(improved)
    trends.to_csv(OUTPUT_DIR / "sst_trend_comparison.csv", index=False)

    # 4) Plots.
    save_plots(improved, trends)

    # 5) Summary.
    print("\n=== Trend comparison ===")
    print(
        trends[
            [
                "ocean",
                "slope_K_per_month",
                "paper_slope_K_per_month",
                "slope_pct_error",
                "intercept_K",
                "paper_intercept_K",
                "r2",
                "rmse_K",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
