import os
import json5
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable # Required for individual colorbars
import warnings

warnings.filterwarnings('ignore')

def plot_BLprecip_3panels(ds_mcs, ds_deep, ds_other, year_str):
    
    colors2 = plt.cm.jet(np.linspace(0.3, 0.9, 8))
    colors1 = plt.cm.binary(np.linspace(0.6, 0.4, 2))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    # Using a slightly wider figure to accommodate three colorbars
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    datasets = [ds_mcs, ds_deep, ds_other]
    titles = [f'MCS ({year_str})', f'Non-MCS, Deep ({year_str})', f'Other ({year_str})']
    
    precip_levels = np.arange(0, 10, 0.5)
    
    for i, (ds, title) in enumerate(zip(datasets, titles)):
        ax = axes[i]
        
        # 1. Aggregate across years
        samples = ds.samples.sum(dim='year')
        # Calculate the fraction of reliable samples (must be samples_reliable / samples)
        prec_sum = ds.prec_sum.sum(dim='year')

        # Calculate conditional precip and PDF
        cond_prec = prec_sum / samples
        pdf = np.log10((samples / samples.sum(dim=(['bins_cape', 'bins_subsat']))))

        # Masking based on sample size
        cond_prec = cond_prec.where(samples > 100)
        pdf = pdf.where(samples > 100)

        # --- Plotting ---

        # 1. Plot the Conditional Precipitation (Color Fill)
        cp = ax.contourf(ds.bins_subsat + 0.25, ds.bins_cape + 0.25, 
                        cond_prec, levels=precip_levels, cmap=mymap, extend='max')

        # 2. Add Precip Contours (Black lines)
        ax.contour(ds.bins_subsat + 0.25, ds.bins_cape + 0.25, 
                cond_prec, levels=precip_levels[::2], colors=['k'], linewidths=1)

        # 3. PDF Contours (White lines)
        ax.contour(ds.bins_subsat, ds.bins_cape, -pdf, 
                levels=np.arange(1, 3.5, 0.5), colors=['w'], linewidths=1.5)
        
        # Add buoyancy lines (Diagonal dashed lines)
        x = np.arange(-10, 15, 1)
        for offset in [0, -2.5, -5]:
            if offset == 0:
                ax.plot(x, x + offset, ls='-', color='m', linewidth=1.5, alpha=0.6)
            else:
                ax.plot(x, x + offset, ls='--', color='m', linewidth=1.5, alpha=0.6)
                
        # Add sample min. denotion 
        ax.text(x=0, y=5.5, s='min. sample=100',fontsize=8.5)
       
        # Grid and Axes settings
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim([-0.5, 10])
        ax.set_ylim([-5, 6.5])
        ax.set_xlabel('B$_{L, SUBSAT}$ (K)', fontsize=12)
        ax.set_ylabel('B$_{L, CAPE}$ (K)', fontsize=12)
        ax.set_title(f'{title}', loc='left', x=0.05, y=1, fontsize=10, fontweight='bold')
        
        # --- Create Individual Colorbar ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(cp, cax=cax)
        cbar.set_label('Precipitation (mm/hr)', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    return fig

def load_combined_data(category, surface_type):
    # Find all files matching the category across the years
    pattern = f'BLprec_2Dhist_{category}_combined.*.{regrid_res:.1f}deg.nc'
    files = sorted(list(stats_dir.glob(pattern)))
    
    if not files:
        raise IOError(f"No files found for {category}")
        
    print(f"Loading {len(files)} files for {category}...")
    # open_mfdataset handles the concatenation along a new 'year' dimension
    return xr.open_mfdataset(files, concat_dim='year', combine='nested').sel(surface_type=surface_type)

if __name__ == "__main__":

    main_dir = Path(os.getenv('CODE_ROOT', '.'))
    pod_dir = main_dir / 'diagnostics/MCS_precip_buoy_stats'
    settings_path = pod_dir / 'settings.jsonc'
    with open(settings_path, 'r') as f:
        runjob_settings = json5.load(f)
    opts = runjob_settings['pod_options']
    # read regridding resolution from settings.jsonc, default is using 1-degree 
    regrid_res = opts.get('regrid_res', 1) 

    work_dir = Path(os.environ["WORK_DIR"])
    #work_dir = Path('/scratch/wmtsai/mdtf_miniforge/wkdir/MDTF_output/MCS_precip_buoy_stats')
    parent_dir = work_dir.parent
   # Get case name and model resolution info 
    df = pd.read_csv(parent_dir / "MDTF_postprocessed_data.csv")
    model_id = df['source_id'].iloc[0]
    ds_file = df["path"].iloc[0]
    lon_res = abs(np.diff(xr.open_dataset(ds_file).lon)[0])
    lat_res = abs(np.diff(xr.open_dataset(ds_file).lat)[0])
    model_info = (model_id, np.round(lon_res,2), np.round(lat_res,2))

    mcsmask_dir = work_dir / 'model/netCDF/MCS_identifiers'
    stats_dir = work_dir / 'model/netCDF/stats'
    fig_dir = work_dir / 'fig'
    os.makedirs(str(fig_dir), exist_ok=True)
   
    # Define the range of years
    year_list = sorted([int(d.name) for d in mcsmask_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    start_year = year_list[0]
    end_year = year_list[-1]
    years_str = f'{start_year}-{end_year}'
    
    for surface_type in ['ocean','land']:
        # Load and combine categories
        ds_mcs_all   = load_combined_data('mcs', surface_type)
        ds_deep_all  = load_combined_data('deep', surface_type)
        ds_other_all = load_combined_data('other', surface_type)

        # Plotting 3 panels in one figure
        fig = plot_BLprecip_3panels(ds_mcs_all, ds_deep_all, ds_other_all, years_str)
        fig.suptitle(rf'{model_info[0]} ($\Delta$x:{model_info[1]}$^o$, $\Delta$y:{model_info[2]}$^o$)'+rf', Regrid: {regrid_res}$^o$, {surface_type}',
                    y=1.05, fontsize=14, fontweight='bold')    
        # Save the merged figure
        out_name = f'BLprecip_capesubsat_merged.{surface_type}.png'
        fig.savefig(fig_dir / out_name, dpi=1000, bbox_inches='tight')
        
        print(f'{out_name} ...completed!')
        plt.close()