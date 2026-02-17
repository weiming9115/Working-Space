import os
import sys
import subprocess
import time
import warnings
from pathlib import Path

import json5
import xarray as xr
import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

def print_banner(message):
    print('\n' + '-'*50)
    print(f"Step: {message}")
    print('-'*50)

if __name__ == "__main__":
    start_time_execute = time.time()
    
    # --- STEP 0: Path Configuration ---
    # Attempt to get CODE_ROOT from environment; fallback to current directory
    main_dir = Path(os.getenv('CODE_ROOT', '.'))
    if main_dir.exists():
        print(f"MDTF home directory: {main_dir}")

    pod_dir = main_dir / 'diagnostics/MCS_precip_buoy_stats'
    utils_dir = pod_dir / 'mcs_utils'
    work_dir = Path(os.environ["WORK_DIR"]) # /wkdir/MDTF_output.v.../MCS_precip_buoy_stats
    
    # Add utils to path for direct function imports
    sys.path.append(str(utils_dir))
    try:
        from process_PyFLEXTRKR_MCSmask_writeout_parallel import dask_write_cloudid_PyFLEXTRKR
        from process_layer_thetae_writeout import process_thetae_layers
    except ImportError as e:
        print(f"Critical Error: Could not import utility functions: {e}")
        sys.exit(1)

    # --- STEP 1: Load Settings ---
    settings_path = pod_dir / 'settings.jsonc'
    with open(settings_path, 'r') as f:
        runjob_settings = json5.load(f)
    
    opts = runjob_settings['pod_options']
    
    # Flags (Ensuring they are Booleans even if passed as strings)
    def to_bool(val): return str(val).lower() == 'true'

    run_mcs_id      = to_bool(opts.get('run_mcs_identification', False))
    run_precip_st   = to_bool(opts.get('run_precip_statistics', False))
    plot_precip_st  = to_bool(opts.get('plot_precip_statistics', False))
    run_buoy_calc   = to_bool(opts.get('run_buoyancy_calculation', False))
    run_buoy_st     = to_bool(opts.get('run_buoyancy_statistics', False))
    plot_buoy_st    = to_bool(opts.get('plot_buoyancy_statistics', False))

    lat_bounds = slice(opts['latitude_min'], opts['latitude_max'])
    
    # --- STEP 2: MCS Identification ---
    if run_mcs_id:
        print_banner("Running snapshot-based MCS identification")
        ds_var2d = xr.merge([
            xr.open_dataset(os.environ['PR_FILE']), 
            xr.open_dataset(os.environ['RLUT_FILE'])
        ]).sel(lat=lat_bounds)

        start_year = int(str(opts['start_time'])[:4])
        end_year = int(str(opts['end_time'])[:4])
        
        for year in range(start_year, end_year + 1):
            dask_write_cloudid_PyFLEXTRKR(ds_var2d, year)
    else:
        # Check for existing data if we are skipping identification
        mcsmask_dir = work_dir / 'model/netCDF/MCS_identifiers/'
        if not (mcsmask_dir.exists() and any(mcsmask_dir.iterdir())):
            raise FileNotFoundError(
                f"No MCS mask files found in {mcsmask_dir}. Check settings.jsonc."
            )
        print("Existing MCS masks found. Skipping identification.")

    # --- STEP 3: Precipitation Statistics ---
    if run_precip_st:
        print_banner("Start MCS-precipitation diagnostics")
        try:
            subprocess.run(['python', str(utils_dir / 'process_prmcs_stats_writeout.py')], check=True)
        except subprocess.CalledProcessError:
            print("Error: process_prmcs_stats_writeout.py failed.")
            sys.exit(1)
            
    if plot_precip_st:
        subprocess.run(['python', str(utils_dir / 'plot_precip_stats.py')], check=True)

    # --- STEP 4: Buoyancy Calculations ---
    if run_buoy_calc:
        print_banner("Calculate low-tropospheric buoyancy")
        # Only open 3D variables if we are actually calculating buoyancy
        ds_var3d = xr.merge([
            xr.open_dataset(os.environ["TA_FILE"]), 
            xr.open_dataset(os.environ["HUS_FILE"])
        ]).sel(lat=lat_bounds)
        
        thetae_dir = work_dir / 'model/netCDF/layer_averaged_thetae/'
        thetae_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_thetae_layers(ds_var3d, thetae_dir, ntime=None, num_workers=8)
        except Exception as e:
            print(f"Error during buoyancy calculation: {e}")
            sys.exit(1)

    # --- STEP 5: Buoyancy Statistics & Plotting ---
    if run_buoy_st:
        print_banner("Calculate histogram statistics")
        try:
            subprocess.run(['python', str(utils_dir / 'process_BLcapesubsat_regions_unified.multiprocess.py')], check=True)
        except subprocess.CalledProcessError:
            print("Error: process_BLcapesubsat_regions_unified.multiprocess.py failed.")
            sys.exit(1)
        
    if plot_buoy_st:
        print_banner("Plot 2-D buoyancy-precip statistics")
        try:
            subprocess.run(['python', str(utils_dir / 'plot_BLprecip_relations_2D.py')], check=True)
        except subprocess.CalledProcessError:
            print("Error: plot_BLprecip_relations_2D.py failed.")
            sys.exit(1)
            
    # --- Finalize ---
    execution_time = time.time() - start_time_execute
    print(f"\n--- Total Execution Time: {execution_time:.2f} seconds ---")
    print("POD execution finished successfully!")
    sys.exit(0)