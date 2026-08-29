import os
import xarray as xr

def inspect_netcdf():
    # Define data directory and sample file path
    data_dir = "/mnt/c/Users/Areen Vaghasiya/OneDrive - iiit-b/Courses/Areen SEM9/Ocean analysis/data/precipitation_evaporation_CM_SAF_2000_TO_2014"
    sample_file = os.path.join(data_dir, "EMPmm20000101000000213SCPOS01GL.nc")

    print(f"Attempting to open sample NetCDF file: {sample_file}\n")
    if not os.path.exists(sample_file):
        print(f"Error: File not found at {sample_file}")
        return

    try:
        # Load dataset using xarray
        ds = xr.open_dataset(sample_file)
        
        print("=========================================")
        print("1. DATASET GENERAL INFO")
        print("=========================================")
        print(ds)
        print("\n")

        print("=========================================")
        print("2. DIMENSIONS")
        print("=========================================")
        for dim_name, dim_size in ds.dims.items():
            print(f"Dimension: {dim_name} (Size: {dim_size})")
        print("\n")

        print("=========================================")
        print("3. COORDINATE VARIABLES")
        print("=========================================")
        for coord_name in ds.coords:
            coord_var = ds[coord_name]
            print(f"Coordinate: {coord_name}")
            print(f"  Shape: {coord_var.shape}")
            print(f"  Units: {coord_var.attrs.get('units', 'No units attribute')}")
            print(f"  Range: {coord_var.values.min()} to {coord_var.values.max()}")
        print("\n")

        print("=========================================")
        print("4. DATA VARIABLES")
        print("=========================================")
        for var_name in ds.data_vars:
            var = ds[var_name]
            print(f"Variable: {var_name}")
            print(f"  Shape: {var.shape}")
            print(f"  Data Type: {var.dtype}")
            print("  Attributes:")
            for attr_name, attr_val in var.attrs.items():
                print(f"    {attr_name}: {attr_val}")
        print("\n")

        print("=========================================")
        print("5. GLOBAL ATTRIBUTES")
        print("=========================================")
        for attr_name, attr_val in ds.attrs.items():
            print(f"  {attr_name}: {attr_val}")
        print("\n")

    except Exception as e:
        print(f"An error occurred while reading the NetCDF file: {e}")

if __name__ == "__main__":
    inspect_netcdf()
