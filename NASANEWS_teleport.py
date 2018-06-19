#!/usr/bin/env python
# coding=utf-8
"""
"""

__author__ = 'Brian Matilla, Brian Mapes'
__version__= '1.0.0'
__maintainer__= 'Brian Matilla'
__email__= 'bmatilla@rsmas.miami.edu'


import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from datetime import datetime

#First, build the parser for the dates, times, and lat-lon, as well as the case name, output directory and RAMADDA publish

def get_args():
    '''This function is the one that obtains and parses the arguments passed through the command
    line'''
    
    if __name__ == '__main__':
        
        parser = argparse.ArgumentParser(description='Script to create a 16-panel plot of precipitation estimates from several observational and reanalysis data products as well as the ensemble mean and RMS. A user can "teleport" a multipanel plot to a specific time and location based on date, time, and lat-lon bounds.',
                                        epilog= "Example use case on the command line: python NASANEWS_teleport.py "
                                        "-ds date_start " 
                                         "-de date_end "
                                         "-bbox S,N,W,E "
                                         "-case foo ")
                                     
        parser.add_argument('-ds' , '--date_start', type=str,
                        help= 'Starting date for case study, expressed as YYYY-mm-dd.',
                        required= True)
    
        parser.add_argument('-de', '--date_end', type=str,
                        help= 'Ending date for case study, expressed as YYYY-mm-dd.',
                        required= True) #Not required for reason of interests in single-day cases.
    
        parser.add_argument('-bbox', '--boundingbox', nargs=4, type=int,
                        help='Set the bounding box of the plot with boundaries: '
                             'south, north, west, east.',
                        metavar=("south", "north", "west", "east"), required=True)
    
        parser.add_argument('-case', '--nameofcase', type=str, nargs='+',
                        help='Case name to prefix the plot.',
                        required=True)
    
        parser.add_argument('-outdir', '--output_directory', type=os.path.isdir,
                        help='Set the output path to place the output;'
                             'default is current directory from where the script is run',
                        required=False) #Doesn't quite work yet...
        
        args= parser.parse_args()
        
        date_start= args.date_start
        date_end= args.date_end
        bbox= args.boundingbox
        nameofcase= args.nameofcase
        outdir= args.output_directory
        
        if args.boundingbox:
            south= bbox[0]
            north= bbox[1]
            west= bbox[2]
            east= bbox[3]
        
        if args.output_directory:
            output_directory= args.output_directory
        
        if args.nameofcase:
            nameofcase= os.path.join(args.nameofcase[0]) #Will add output_directory soon.
            
        return date_start, date_end, bbox, nameofcase, outdir
    
date_start, date_end, bbox, nameofcase, outdir= get_args()

#Below are the calls to load all of the datasets, create lists of the names and links,
#and convert them to a callable, iterable 2D array.

gldas ='http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuZ2xkYXMuZGFpbHlfYWdnLm5jbWw=/entry.das'
cpcu = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuY3BjdS5kYWlseV9hZ2cubmNtbA==/entry.das'
chirps = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuY2hpcnBzLmRhaWx5X2FnZy5uY21s/entry.das'
trmm3b42 = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAudHJtbTNiNDIuZGFpbHlfYWdnLm5jbWw=/entry.das'
persiann = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAucGVyc2lhbm4uZGFpbHlfYWdnLm5jbWw=/entry.das'
mswep = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAubXN3ZXAuZGFpbHlfYWdnLm5jbWw=/entry.das'
merra = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAubWVycmEuZGFpbHlfYWdnLm5jbWw=/entry.das'
merrav2 = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAubWVycmEyLmRhaWx5X2FnZy5uY21s/entry.das'
jra55 = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuanJhNTUuZGFpbHlfYWdnLm5jbWw=/entry.das'
gsmaprnl = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuZ3NtYXBybmwuZGFpbHlfYWdnLm5jbWw=/entry.das'
gpcp1dd = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuZ3BjcDFkZC5kYWlseV9hZ2cubmNtbA==/entry.das'
ecint = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuZWNpbnQuZGFpbHlfYWdnLm5jbWw=/entry.das'
cmorph = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuY21vcnBoLmRhaWx5X2FnZy5uY21s/entry.das'
cfsr = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuY2Zzci5kYWlseV9hZ2cubmNtbA==/entry.das'
ens_mean = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuZW5zbWVhbi5kYWlseV9hZ2cubmNtbA==/entry.das'
ens_rms = 'http://weather.rsmas.miami.edu/repository/opendap/synth:d68cf65c-8cdd-4886-a61f-d03e877fea67:L2FnZ3JlZ2F0aW9ucy9kYWlseS9wcmVjaXAuZW5zcm1zLmRhaWx5X2FnZy5uY21s/entry.das'

#Group the names. 
dset_names= ['gldas', 'cpcu', 'chirps', 'trmm3b42', 'persiann', 'mswep', 
             'merra', 'merrav2', 'jra55', 'gsmaprnl', 'gpcp1dd', 'ecint',
            'cmorph', 'cfsr', 'ens_mean', 'ens_rms']

#and then group the links
dset_links=[gldas, cpcu, chirps, trmm3b42, persiann, mswep, merra, merrav2, 
           jra55, gsmaprnl, gpcp1dd, ecint, cmorph, cfsr, ens_mean, ens_rms]

data_map= []
for names in dset_names:
    for links in dset_links:
        data_map= np.column_stack(tuple([dset_names, dset_links]))    
        
#Plot error prechecks:
#--------
#Check for date mismatch by computing difference between date_start and date_end:
from datetime import datetime
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

#Date check
days_between(date_start, date_end)
if date_start > date_end:
    raise ValueError('Your end date is before your start date. Check your dates and try again.')
    
if days_between(date_start, date_end) >= 10:
    print('Be patient... this is quite a large time period so the data retrieval may take some time.')

#Longitudes
if bbox[3] < bbox[2]:
    raise ValueError('Your east longitude is less than your west longitude. Check your longitudes and try again.')

#Latitudes
if bbox[1] < bbox[0]:
    raise ValueError('Your north latitude is less than your south latitude. Check your latitudes and try again.')

#Check complete with no errors
else:
    print('No initial errors. Starting plot sequence...')

#--------START PLOT SEQUENCE---------#

print('Plot sequence running... Please wait...')

# Create the figure instance

fig = plt.figure(figsize=(25, 25)) #Need a size large enough to avoid cramping of figure quality. 
gs = gridspec.GridSpec(5, 5, height_ratios=[1, 1, 1, 1, 0.05], width_ratios=[1,1,1,1,0.05], bottom=.05, top=.95, wspace=0.2, hspace=0.2)

cols= 4 #Since we have 16 datasets, 4 columns will make plot evenly distributed.
rows = int(len(data_map) / cols) # 16 datasets/ 4 datasets/column= 4 rows

#Define the plot background here. It speeds up the plotting process to not embed it in the forthcoming loop.
def plot_background(ax):
    ax.set_extent([datalon.min(), datalon.max(), datalat.min(), datalat.max()], crs= ccrs.PlateCarree())
    ax.coastlines('10m', edgecolor='black', linewidth=0.5)
    ax.set_xticks(np.linspace(datalon.min(), datalon.max(), num= 3, endpoint= True), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(datalat.min(), datalat.max(), num= 3, endpoint= True), crs=ccrs.PlateCarree())
    ax.tick_params(labelsize=11)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.RIVERS)
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray')
    return ax

#Retrieve the data from the data sources and perform calculations.

# 1st loop obtains global data max.
pcp_global_max = 0
for i in range(cols):
    for j in range(rows):
        try:
            idx = (cols*i)+j #Obtain each individual data URL as index.
            dnum= data_map[idx][0] #Retrieve datanumber
        
            data = xr.open_dataset(data_map[idx][1], decode_times=True)
            datatitle= data.product_id
        
            datalat= data.lat[(bbox[0])+90:(bbox[1])+90] #Because the latitude units are strictly in degrees north, we must add 90 to account for southern hemisphere. Otherwise, we will end up with index values which would yield incorrect latitudes.
            datalon= data.lon[bbox[2]:bbox[3]]
            event = data.precip.sel(time=slice(date_start, date_end),lat=slice(datalat.min(), datalat.max()),lon=slice(datalon.min(), datalon.max()))
            pcp = np.sum(event, axis=0)*86400
        
            if pcp.max() > pcp_global_max:
                pcp_global_max = pcp.max()
        except IndexError:
            pcp= pcp * 0

# plots data for all datasets TODO: reduce to single loop and store all values:
for i in range(cols):
    for j in range(rows):
    
        idx = (cols*i)+j 
        dnum= data_map[idx][0] 
        
        data = xr.open_dataset(data_map[idx][1], decode_times=True)
        datatitle= data.product_id
        if idx == 14:
            datatitle= 'ens mean'
        elif idx == 15:
            datatitle= 'ens rms'
        try:
            datalat= data.lat[(bbox[0])+90:(bbox[1])+90]
            datalon= data.lon[bbox[2]:bbox[3]]
            event = data.precip.sel(time=slice(date_start, date_end),lat=slice(datalat.min(), datalat.max()),lon=slice(datalon.min(), datalon.max()))
            pcp = np.sum(event, axis=0)*86400
        except IndexError:
            pcp= pcp * 0
            
        #Define cartopy grid:
        crs = ccrs.PlateCarree(central_longitude=((datalon.max()+datalon.min())/2))

        #Assign the 2D lon and lat variables to construct plot.
        
        lon_2d= np.linspace(datalon.min(), datalon.max(), num= ((datalon.max()- datalon.min())+1), endpoint= True)
        lat_2d= np.linspace(datalat.min(), datalat.max(), num= ((datalat.max()- datalat.min())+1), endpoint= True)
                
        extent= ([datalon.min(), datalon.max(), datalat.min(), datalat.max()])

        #Now for the evenly-spaced contours, separated into 4 categories of precip accumulation:
        if pcp.max() or pcp_global_max <= 500:
            levels= np.arange(0, pcp_global_max, 50)
        elif 500 < pcp.max() <= 1000:
            levels= np.arange(0, pcp_global_max, 75)
        elif pcp.max() > 1000:
            levels= np.arange(0, pcp_global_max, 100)
        
        if idx < 15:
                bounds_pcp_global= np.linspace(0,np.around(pcp_global_max, decimals=-1), num= 11, endpoint=True)      
                norm_global = mpl.colors.BoundaryNorm(boundaries=bounds_pcp_global, ncolors=256)
        elif idx == 15:
            bounds_pcp= np.linspace(0,np.around(pcp.max(), decimals=-1), num= 11, endpoint=True)        
            norm = mpl.colors.BoundaryNorm(boundaries=bounds_pcp, ncolors=256)
        
        # plot the data accordingly
        ax1 = plt.subplot(gs[i, j], projection=crs)
        plot_background(ax1)
        
        cf1= plt.pcolormesh((lon_2d-0.5), (lat_2d-0.5), pcp, cmap='Greens', norm=norm_global, transform=ccrs.PlateCarree())

        if idx == 15: #Ensemble RMS plot
            cf1= plt.pcolormesh((lon_2d-0.5), (lat_2d-0.5), pcp, cmap='Blues', transform=ccrs.PlateCarree())
        c1 = plt.contour(pcp, colors='red', levels=levels, linewidths=2, norm=norm_global, extent=extent)
        
        ax1.clabel(c1, fontsize=15, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
        
        ax1.set_title('({}) {} '.format(chr(97+idx), datatitle), fontsize=22)

side_bar_ax = plt.subplot(gs[4, :-1])
pcp_global_bounds = np.linspace(0,np.around(pcp_global_max, decimals=-1), num= 11, endpoint=True)
pcp_bounds= np.linspace(0,np.around(pcp.max(), decimals=-1), num=11, endpoint=True)
cb = mpl.colorbar.ColorbarBase(
                side_bar_ax, 
                cmap='Greens',
                norm=norm_global,
                extend='max',
                ticks=pcp_global_bounds,
                spacing='uniform',
                orientation='horizontal')
cb.set_label('Precipitation (mm/day)', fontsize=22)
cb.ax.tick_params(labelsize=18)

side_bar_ax = plt.subplot(gs[3, 4])
cb = mpl.colorbar.ColorbarBase(side_bar_ax, cmap='Blues',norm=norm, extend='max',
                               ticks=bounds_pcp, spacing='uniform', orientation='vertical')

cb.ax.tick_params(labelsize=18)
cb.set_label('Precipitation (mm/day)', fontsize=22)

if date_start == date_end:   
    fig.suptitle('rainfall accumulation for case on: ' + date_start, fontsize=28)
elif date_start != date_end:
    fig.suptitle('rainfall accumulation for case from: ' + date_start + ' to ' +date_end, fontsize=28)

print('Plot sequence complete. Rendering...')
plt.savefig(nameofcase + '.png', bbox_inches='tight')

print('SUCCESS! Your plot is complete...')
print('Plot saved as ' + nameofcase + '.png')

plt.show()
