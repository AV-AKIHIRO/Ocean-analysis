import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_cell_areas(lat, lon):
    # Constants
    R = 6.371e6 # Earth radius in meters
    dlat = 0.5 * np.pi / 180.0 # 0.5 degrees in radians
    dlon = 0.5 * np.pi / 180.0 # 0.5 degrees in radians
    
    # Area of grid cell as a function of latitude: R^2 * cos(lat) * dlat * dlon
    lat_rad = np.deg2rad(lat)
    cell_areas = (R**2) * np.cos(lat_rad) * dlat * dlon
    
    # Broadcast to 2D shape (lat, lon)
    cell_areas_2d = np.tile(cell_areas[:, np.newaxis], (1, len(lon)))
    return cell_areas_2d

def save_comparison_plot(df_sub, paper_vals, title_suffix, plot_path, artifact_plot_path):
    plt.figure(figsize=(15, 10))
    
    basins_info = [
        ('F_NA', 'North Atlantic (NA)', 'green', paper_vals['F_NA']),
        ('F_SA', 'South Atlantic (SA)', 'purple', paper_vals['F_SA']),
        ('F_SO', 'Southern Ocean (SO)', 'pink', paper_vals['F_SO']),
        ('F_PO', 'Pacific Ocean (PO)', 'gold', paper_vals['F_PO']),
        ('F_IO', 'Indian Ocean (IO)', 'cyan', paper_vals['F_IO'])
    ]
    
    for i, (col, name, color, paper_val) in enumerate(basins_info, 1):
        plt.subplot(3, 2, i)
        # Plot observational time series
        plt.plot(df_sub.index, df_sub[col], label='Observational (HOAPS)', color=color, linewidth=1.2)
        # Draw dashed line for paper's constant value
        plt.axhline(y=paper_val, color='red', linestyle='--', linewidth=1.2, label=f"Paper constant ({paper_val} Sv)")
        # Calculate observed average over this specific period
        obs_mean = df_sub[col].mean()
        # Draw dotted line for observed average constant
        plt.axhline(y=obs_mean, color='blue', linestyle='-.', linewidth=1.2, label=f"Observed mean ({obs_mean:.4f} Sv)")
        
        plt.title(f"Freshwater Flux (E - P) - {name} ({title_suffix})")
        plt.ylabel("Flux (Sv)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.savefig(artifact_plot_path, bbox_inches='tight', dpi=150)
    plt.close()

def process_fluxes():
    dir_1987_1999 = "/mnt/c/Users/Areen Vaghasiya/OneDrive - iiit-b/Courses/Areen SEM9/Ocean analysis/data/precipitation_evaporation_CM_SAF_1987_TO_1999"
    dir_2000_2014 = "/mnt/c/Users/Areen Vaghasiya/OneDrive - iiit-b/Courses/Areen SEM9/Ocean analysis/data/precipitation_evaporation_CM_SAF_2000_TO_2014"
    
    files_1987_1999 = glob.glob(os.path.join(dir_1987_1999, "EMPmm*.nc"))
    files_2000_2014 = glob.glob(os.path.join(dir_2000_2014, "EMPmm*.nc"))
    
    # Combine and sort files by filename to ensure chronological ordering
    all_files = sorted(files_1987_1999 + files_2000_2014, key=lambda x: os.path.basename(x))
    
    if len(all_files) == 0:
        print("Error: No NetCDF files found!")
        return

    print(f"Found {len(all_files)} total NetCDF files (1987-2014) to process.")
    
    # Open first file to extract coordinates and compute cell areas
    with xr.open_dataset(all_files[0]) as ds:
        lat = ds.lat.values
        lon = ds.lon.values
        
    cell_areas = calculate_cell_areas(lat, lon)
    
    # 2D coordinates
    lat_2d = np.tile(lat[:, np.newaxis], (1, len(lon)))
    lon_2d = np.tile(lon[np.newaxis, :], (len(lat), 1))
    
    # Define coordinate conditions for the ocean masks north of 60S
    cond_SA = (lat_2d > -60.0) & (lat_2d <= 0.0) & (lon_2d >= -70.0) & (lon_2d < 20.0)
    
    cond_IO = (lat_2d > -60.0) & (lat_2d <= 30.0) & (
        ((lat_2d <= 0.0) & (lon_2d >= 20.0) & (lon_2d < 140.0)) |
        ((lat_2d > 0.0) & (lon_2d >= 20.0) & (lon_2d < 100.0))
    )
    
    cond_NA_tropics1 = (lat_2d > 0.0) & (lat_2d <= 10.0) & (lon_2d >= -80.0) & (lon_2d < 20.0)
    cond_NA_tropics2 = (lat_2d > 10.0) & (lat_2d <= 20.0) & (lon_2d >= -90.0) & (lon_2d < 20.0)
    cond_NA_mid = (lat_2d > 20.0) & (lat_2d <= 60.0) & (lon_2d >= -100.0) & (lon_2d < 20.0)
    cond_NA_high = (lat_2d > 60.0) & (((lon_2d >= -180.0) & (lon_2d < -169.0)) | (lon_2d >= 20.0) | ((lon_2d >= -100.0) & (lon_2d < 20.0)))
    cond_NA = (cond_NA_tropics1 | cond_NA_tropics2 | cond_NA_mid | cond_NA_high)
    
    # Lists to store the time series data
    dates = []
    flux_NA = []
    flux_SA = []
    flux_SO = []
    flux_PO = []
    flux_IO = []
    
    # Process each NetCDF file
    for idx, filepath in enumerate(all_files):
        with xr.open_dataset(filepath) as ds:
            # Extract data variable 'budg' (E - P, mm/d)
            budg = ds['budg'].isel(time=0).values
            time_val = pd.to_datetime(ds.time.values[0])
            date_str = time_val.strftime('%Y-%m-%d')
            dates.append(date_str)
            
            # Mask of non-NaN ocean cells
            ocean_mask = ~np.isnan(budg)
            
            # Apply basin definitions
            mask_SO = (lat_2d <= -60.0) & ocean_mask
            mask_SA = cond_SA & ocean_mask
            mask_IO = cond_IO & ocean_mask
            mask_NA = cond_NA & ocean_mask
            mask_PO = (lat_2d > -60.0) & ~mask_SA & ~mask_NA & ~mask_IO & ocean_mask
            
            # Conversion: mm/d to m/s is 1e-3 / 86400
            # E - P (budg) is already evaporation - precipitation -> positive value represents net water loss.
            # Volume flux in m^3/s = (E - P in m/s) * cell_area (in m^2)
            # Volume flux in Sv = volume flux / 1e6
            conversion_factor = 1.0 * 1e-3 / (86400.0 * 1e6)
            
            # Calculate sum of cell fluxes (Sv) for each basin
            f_SO = np.sum(budg[mask_SO] * cell_areas[mask_SO]) * conversion_factor
            f_SA = np.sum(budg[mask_SA] * cell_areas[mask_SA]) * conversion_factor
            f_IO = np.sum(budg[mask_IO] * cell_areas[mask_IO]) * conversion_factor
            f_NA = np.sum(budg[mask_NA] * cell_areas[mask_NA]) * conversion_factor
            f_PO = np.sum(budg[mask_PO] * cell_areas[mask_PO]) * conversion_factor
            
            flux_SO.append(f_SO)
            flux_SA.append(f_SA)
            flux_IO.append(f_IO)
            flux_NA.append(f_NA)
            flux_PO.append(f_PO)
            
        if (idx + 1) % 30 == 0 or (idx + 1) == len(all_files):
            print(f"Processed {idx + 1}/{len(all_files)} files...")
            
    # Save to DataFrame
    df = pd.DataFrame({
        'date': dates,
        'F_NA': flux_NA,
        'F_SA': flux_SA,
        'F_SO': flux_SO,
        'F_PO': flux_PO,
        'F_IO': flux_IO
    })
    
    # Set date as index for plotting and statistics
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Create output directory
    output_dir = "/mnt/c/Users/Areen Vaghasiya/OneDrive - iiit-b/Courses/Areen SEM9/Ocean analysis/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files
    csv_path_full = os.path.join(output_dir, "processed_freshwater_fluxes_1987_2014.csv")
    df.to_csv(csv_path_full)
    print(f"\nSaved combined 1987-2014 CSV output to: {csv_path_full}")
    
    # Save subset 2000-2014 CSV (backwards compatibility)
    df_2000_2014 = df.loc['2000':'2014']
    csv_path_sub = os.path.join(output_dir, "processed_freshwater_fluxes.csv")
    df_2000_2014.to_csv(csv_path_sub)
    print(f"Saved 2000-2014 CSV output subset to: {csv_path_sub}")
    
    # Constants from paper (for plotting comparison line)
    paper_vals = {
        'F_NA': 0.475,
        'F_SA': 0.526,
        'F_SO': -0.209,
        'F_PO': 0.064,
        'F_IO': 2.3495
    }
    
    # Create the three plots:
    # 1. 2000 to 2014
    df_plot1 = df.loc['2000':'2014']
    plot_path1 = os.path.join(output_dir, "freshwater_flux_comparison_2000_2014.png")
    art_path1 = "/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63/freshwater_flux_comparison_2000_2014.png"
    save_comparison_plot(df_plot1, paper_vals, "2000-2014", plot_path1, art_path1)
    
    # Also save as the original comparison plot file for compatibility
    plot_path_compat = os.path.join(output_dir, "freshwater_flux_comparison.png")
    art_path_compat = "/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63/freshwater_flux_comparison.png"
    save_comparison_plot(df_plot1, paper_vals, "2000-2014", plot_path_compat, art_path_compat)
    
    # 2. 1987 to 1999
    df_plot2 = df.loc['1987':'1999']
    plot_path2 = os.path.join(output_dir, "freshwater_flux_comparison_1987_1999.png")
    art_path2 = "/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63/freshwater_flux_comparison_1987_1999.png"
    save_comparison_plot(df_plot2, paper_vals, "1987-1999", plot_path2, art_path2)
    
    # 3. 1987 to 2014 (full period)
    plot_path3 = os.path.join(output_dir, "freshwater_flux_comparison_1987_2014.png")
    art_path3 = "/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63/freshwater_flux_comparison_1987_2014.png"
    save_comparison_plot(df, paper_vals, "1987-2014", plot_path3, art_path3)
    
    print("\nGenerated and saved all three plots:")
    print(f" - Plot 1 (2000-2014): {plot_path1}")
    print(f" - Plot 2 (1987-1999): {plot_path2}")
    print(f" - Plot 3 (1987-2014): {plot_path3}")
    
    # Print Mean Statistics comparisons
    print("\n=========================================")
    print("freshwater fluxes multi-period observed means (Sv)")
    print("=========================================")
    print(f"{'Basin':<10} | {'Mean 1987-1999':<16} | {'Mean 2000-2014':<16} | {'Mean 1987-2014 (Full)':<22} | {'Paper Constant':<15}")
    print("-" * 90)
    for b in ['F_NA', 'F_SA', 'F_SO', 'F_PO', 'F_IO']:
        m1 = df_plot2[b].mean()
        m2 = df_plot1[b].mean()
        m3 = df[b].mean()
        pap = paper_vals[b]
        print(f"{b:<10} | {m1:<16.4f} | {m2:<16.4f} | {m3:<22.4f} | {pap:<15.4f}")

if __name__ == "__main__":
    process_fluxes()
