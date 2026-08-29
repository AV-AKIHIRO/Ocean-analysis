import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def visualize():
    data_dir = "/mnt/c/Users/Areen Vaghasiya/OneDrive - iiit-b/Courses/Areen SEM9/Ocean analysis/data/precipitation_evaporation_CM_SAF_2000_TO_2014"
    sample_file = os.path.join(data_dir, "EMPmm20000101000000213SCPOS01GL.nc")
    
    ds = xr.open_dataset(sample_file)
    budg = ds['budg'].isel(time=0)
    
    lat = budg.lat.values
    lon = budg.lon.values
    ocean_mask = ~np.isnan(budg.values)
    
    # 2D coordinates
    lat_2d = np.tile(lat[:, np.newaxis], (1, len(lon)))
    lon_2d = np.tile(lon[np.newaxis, :], (len(lat), 1))
    
    # Initialize mask array
    # 0: Land/Ice, 1: NA, 2: SA, 3: SO, 4: PO, 5: IO
    masks = np.zeros_like(budg.values, dtype=int)
    
    # Southern Ocean (SO)
    masks[(lat_2d <= -60.0) & ocean_mask] = 3
    
    # South Atlantic (SA)
    masks[(lat_2d > -60.0) & (lat_2d <= 0.0) & (lon_2d >= -70.0) & (lon_2d < 20.0) & ocean_mask] = 2
    
    # Indian Ocean (IO)
    mask_IO_cond = (lat_2d > -60.0) & (lat_2d <= 30.0) & (
        ((lat_2d <= 0.0) & (lon_2d >= 20.0) & (lon_2d < 140.0)) |
        ((lat_2d > 0.0) & (lon_2d >= 20.0) & (lon_2d < 100.0))
    ) & ocean_mask
    masks[mask_IO_cond] = 5
    
    # North Atlantic (NA)
    cond_NA_tropics1 = (lat_2d > 0.0) & (lat_2d <= 10.0) & (lon_2d >= -80.0) & (lon_2d < 20.0)
    cond_NA_tropics2 = (lat_2d > 10.0) & (lat_2d <= 20.0) & (lon_2d >= -90.0) & (lon_2d < 20.0)
    cond_NA_mid = (lat_2d > 20.0) & (lat_2d <= 60.0) & (lon_2d >= -100.0) & (lon_2d < 20.0)
    cond_NA_high = (lat_2d > 60.0) & (((lon_2d >= -180.0) & (lon_2d < -169.0)) | (lon_2d >= 20.0) | ((lon_2d >= -100.0) & (lon_2d < 20.0)))
    mask_NA = (cond_NA_tropics1 | cond_NA_tropics2 | cond_NA_mid | cond_NA_high) & ocean_mask
    masks[mask_NA] = 1
    
    # Pacific Ocean (PO)
    mask_PO_cond = (lat_2d > -60.0) & (masks == 0) & ocean_mask
    masks[mask_PO_cond] = 4
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Define color mappings for the colormap
    from matplotlib.colors import ListedColormap
    colors_list = [
        '#1f77b4', # 0: Land/Ice (muted blue)
        '#2ca02c', # 1: NA (green)
        '#9467bd', # 2: SA (purple)
        '#e377c2', # 3: SO (pink)
        '#bcbd22', # 4: PO (yellow-green)
        '#17becf'  # 5: IO (cyan)
    ]
    custom_cmap = ListedColormap(colors_list)
    
    # Force discrete coloring using vmin and vmax matching custom_cmap bins
    im = plt.imshow(masks, extent=[lon.min(), lon.max(), lat.min(), lat.max()], 
                    cmap=custom_cmap, origin='upper', aspect='auto', vmin=-0.5, vmax=5.5)
    
    # Colorbar with ticks aligned at integer centers [0, 1, 2, 3, 4, 5]
    cbar = plt.colorbar(im, shrink=0.7)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(['Land/Ice', 'North Atlantic (NA)', 'South Atlantic (SA)', 'Southern Ocean (SO)', 'Pacific Ocean (PO)', 'Indian Ocean (IO)'])
    
    plt.title('Ocean Basin Masks matching 5-Box Model (Refined)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    artifact_path = "/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63/ocean_masks.png"
    plt.savefig(artifact_path, bbox_inches='tight', dpi=150)
    print(f"Mask visualization saved to: {artifact_path}")

if __name__ == "__main__":
    visualize()
