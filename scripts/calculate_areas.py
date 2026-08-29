import os
import xarray as xr
import numpy as np

def calculate_cell_areas(lat, lon):
    R = 6.371e6 # Earth radius in meters
    dlat = 0.5 * np.pi / 180.0
    dlon = 0.5 * np.pi / 180.0
    lat_rad = np.deg2rad(lat)
    cell_areas = (R**2) * np.cos(lat_rad) * dlat * dlon
    cell_areas_2d = np.tile(cell_areas[:, np.newaxis], (1, len(lon)))
    return cell_areas_2d

def test_refined_boundaries():
    data_dir = "/mnt/c/Users/Areen Vaghasiya/OneDrive - iiit-b/Courses/Areen SEM9/Ocean analysis/data/precipitation_evaporation_CM_SAF_2000_TO_2014"
    sample_file = os.path.join(data_dir, "EMPmm20000101000000213SCPOS01GL.nc")
    
    ds = xr.open_dataset(sample_file)
    budg = ds['budg'].isel(time=0)
    
    lat = budg.lat.values
    lon = budg.lon.values
    
    cell_areas = calculate_cell_areas(lat, lon)
    ocean_mask = ~np.isnan(budg.values)
    
    # 2D coordinate arrays
    lat_2d = np.tile(lat[:, np.newaxis], (1, len(lon)))
    lon_2d = np.tile(lon[np.newaxis, :], (len(lat), 1))
    
    # Southern Ocean (SO)
    mask_SO = (lat_2d <= -60.0) & ocean_mask
    
    # South Atlantic (SA)
    mask_SA = (lat_2d > -60.0) & (lat_2d <= 0.0) & (lon_2d >= -70.0) & (lon_2d < 20.0) & ocean_mask
    
    # Indian Ocean (IO)
    # 1. IO is south of Asia, so lat <= 30.0
    # 2. South of Equator (-60 < lat <= 0): longitude between 20E and 140E
    # 3. North of Equator (0 < lat <= 30): longitude between 20E and 100E (Bay of Bengal is west of 100E)
    mask_IO = (lat_2d > -60.0) & (lat_2d <= 30.0) & (
        ((lat_2d <= 0.0) & (lon_2d >= 20.0) & (lon_2d < 140.0)) |
        ((lat_2d > 0.0) & (lon_2d >= 20.0) & (lon_2d < 100.0))
    ) & ocean_mask
    
    # North Atlantic (NA)
    # Bounded in south by Equator (lat > 0.0)
    # Bounded in north by grid limits (includes Arctic)
    # Bounded on west by the Americas:
    #   - For 0 < lat <= 10: lon >= -80
    #   - For 10 < lat <= 20: lon >= -90
    #   - For lat > 20: lon >= -100
    #   - For lat > 60: we also include Bering Strait (lon < -169 or lon >= 20)
    # Bounded on east by Europe/Africa (lon < 20)
    # Note: North of 60N, since the Arctic is subsumed, NA includes all longitudes except the Pacific side of Bering Strait
    cond_NA_tropics1 = (lat_2d > 0.0) & (lat_2d <= 10.0) & (lon_2d >= -80.0) & (lon_2d < 20.0)
    cond_NA_tropics2 = (lat_2d > 10.0) & (lat_2d <= 20.0) & (lon_2d >= -90.0) & (lon_2d < 20.0)
    cond_NA_mid = (lat_2d > 20.0) & (lat_2d <= 60.0) & (lon_2d >= -100.0) & (lon_2d < 20.0)
    cond_NA_high = (lat_2d > 60.0) & (((lon_2d >= -180.0) & (lon_2d < -169.0)) | (lon_2d >= 20.0) | ((lon_2d >= -100.0) & (lon_2d < 20.0)))
    
    mask_NA = (cond_NA_tropics1 | cond_NA_tropics2 | cond_NA_mid | cond_NA_high) & ocean_mask
    
    # Pacific Ocean (PO)
    # All remaining ocean cells
    mask_PO = (lat_2d > -60.0) & ~mask_SA & ~mask_NA & ~mask_IO & ocean_mask
    
    # Compute areas
    area_SO = np.sum(cell_areas[mask_SO]) / 1e12
    area_SA = np.sum(cell_areas[mask_SA]) / 1e12
    area_NA = np.sum(cell_areas[mask_NA]) / 1e12
    area_IO = np.sum(cell_areas[mask_IO]) / 1e12
    area_PO = np.sum(cell_areas[mask_PO]) / 1e12
    
    print("=== Paper Target Areas (10^12 m^2) ===")
    print("NA: 41.49")
    print("SA: 40.27")
    print("SO: 20.33")
    print("PO: 165.25")
    print("IO: 70.56")
    
    print("\n=== Refined Computed Areas (10^12 m^2) ===")
    print(f"NA: {area_NA:.2f} (Diff: {area_NA - 41.49:.2f})")
    print(f"SA: {area_SA:.2f} (Diff: {area_SA - 40.27:.2f})")
    print(f"SO: {area_SO:.2f} (Diff: {area_SO - 20.33:.2f})")
    print(f"PO: {area_PO:.2f} (Diff: {area_PO - 165.25:.2f})")
    print(f"IO: {area_IO:.2f} (Diff: {area_IO - 70.56:.2f})")
    
    total_paper = 41.49 + 40.27 + 20.33 + 165.25 + 70.56
    total_computed = area_NA + area_SA + area_SO + area_PO + area_IO
    print(f"\nTotal Paper Area: {total_paper:.2f}")
    print(f"Total Computed Area: {total_computed:.2f} (Diff: {total_computed - total_paper:.2f})")

if __name__ == "__main__":
    test_refined_boundaries()
