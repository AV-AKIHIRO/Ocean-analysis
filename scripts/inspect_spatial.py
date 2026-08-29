import os
import xarray as xr
import numpy as np

def inspect_spatial():
    data_dir = "/mnt/c/Users/Areen Vaghasiya/OneDrive - iiit-b/Courses/Areen SEM9/Ocean analysis/data/precipitation_evaporation_CM_SAF_2000_TO_2014"
    sample_file = os.path.join(data_dir, "EMPmm20000101000000213SCPOS01GL.nc")

    if not os.path.exists(sample_file):
        print(f"Error: File not found at {sample_file}")
        return

    ds = xr.open_dataset(sample_file)
    budg = ds['budg'].isel(time=0) # Get the first time step

    print("=== Coordinates ===")
    print("Lat range:", budg.lat.min().item(), "to", budg.lat.max().item(), "size:", len(budg.lat))
    print("Lon range:", budg.lon.min().item(), "to", budg.lon.max().item(), "size:", len(budg.lon))

    # Total grid cells
    total_cells = budg.size
    nan_cells = np.isnan(budg.values).sum()
    valid_cells = total_cells - nan_cells

    print("\n=== Grid Cell Count ===")
    print(f"Total cells: {total_cells}")
    print(f"NaN (typically land/ice) cells: {nan_cells} ({nan_cells/total_cells*100:.2f}%)")
    print(f"Valid (ocean) cells: {valid_cells} ({valid_cells/total_cells*100:.2f}%)")

    # Let's inspect the data range of valid values
    valid_values = budg.values[~np.isnan(budg.values)]
    print("\n=== Data Range of Valid Cells ===")
    print(f"Min: {valid_values.min():.4f}")
    print(f"Max: {valid_values.max():.4f}")
    print(f"Mean: {valid_values.mean():.4f}")
    print(f"Std Dev: {valid_values.std():.4f}")

    # Let's check lat/lon step sizes
    lat_diffs = np.diff(budg.lat.values)
    lon_diffs = np.diff(budg.lon.values)
    print("\n=== Grid Spacing ===")
    print("Lat steps:", np.unique(lat_diffs))
    print("Lon steps:", np.unique(lon_diffs))

if __name__ == "__main__":
    inspect_spatial()
