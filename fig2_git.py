"""
Generate Figure 2 for the manuscript.

This script loads day indices with dynamical changes, converts those indices
to actual dates, and then reads ERA5 temperature anomaly data for four
counterfactual/factual scenario combinations. It computes JJA mean anomalies,
assesses significance with a Wilcoxon test and false discovery rate control,
and plots the results on a 2x2 figure grid with hatching over significant
regions.

Data sources:
    - `extra_files/Indices_Emma.csv`: indices for dynamical change days
    - `extra_files/cdo_anomaly_detrended_fldmean.nc`: time coordinate data
    - `climate_data/era5-land_europe_daily_tmean_*_<scenario>Z500_anomaly.nc`: ERA5 anomaly fields

Output:
    - `figures/fig2_claude.png`
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import pandas
import scipy
import geopandas as gpd
import scipy.stats
import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs


BASE_DIR = Path("path")
SHAPE_DIR = Path("path")
OUTPUT_PATH = BASE_DIR / "figures" / "fig2_claude.png"
OUTPUT_PATH2 = BASE_DIR / "figures" / "fig_z500_appendix.png"
DATA_DIR = BASE_DIR / "mortality_data"

# Base directories used for data loading, shapefiles, and output figure saving.

# Load indices for days with dynamical changes (from Faranda et al. 2023 https://doi.org/10.1073/pnas.2214525120)
myIndices = pandas.read_table(BASE_DIR / "extra_files" / "Indices.csv", header = None, sep = ";", usecols= range(1,916))
myTime = xr.open_dataset(BASE_DIR / "extra_files" / "cdo_anomaly_detrended_fldmean.nc").time
# convert indices to actual times
indices1 = myTime[myIndices.iloc[0,:].to_numpy(dtype = np.int32)-1] # subtract one because matlab
indices2 = myTime[myIndices.loc[1,~np.isnan(myIndices.iloc[1,:])].to_numpy(dtype = np.int32)-1]
indices3 = myTime[myIndices.loc[2,~np.isnan(myIndices.iloc[2,:])].to_numpy(dtype = np.int32)-1]
indices4 = myTime[myIndices.loc[3,~np.isnan(myIndices.iloc[3,:])].to_numpy(dtype = np.int32)-1]

#Fig 2
myMax = 3
myMin = -3
cmap = plt.cm.bwr
bins = np.arange(myMin,myMax+1,1)
norm = mpl.colors.BoundaryNorm(bins, cmap.N )

# Define colormap and discrete color boundaries for the anomaly map.


fig, ax = plt.subplots(ncols = 2,nrows = 2, figsize=(20,15))

# Create a 2x2 grid of axes for the four comparison panels.

compName = ['counterfactual_positive_', 'factual_positive_', 'counterfactual_negative_','factual_negative_']
my_subtitle = ['a) 1950 counterfactual; \n positive occurrence trends', 'b) Trended counterfactual; \n positive occurrence trends', 'c) 1950 counterfactual;\n negative occurrence trends','d) Trended counterfactual;\n negative occurrence trends', 'e) Counterfactual both', 'f) Factual both']

for ii in list(np.arange(0,4)):      
    # Load multiple NetCDF files for the given comparison dataset and concatenate along time
    z500CompData = xr.open_mfdataset("/climate_data/" + "era5-land_europe_daily_tmean_*_" + compName[ii] + "Z500_anomaly.nc",combine = "nested",concat_dim = "time")
    z500Comp = z500CompData.sel(time =(z500CompData.time.dt.season == 'JJA')).tmean.mean(dim = "time")
    # Compute seasonally averaged temperature anomaly fields for JJA
    is_sig = scipy.stats.wilcoxon(z500CompData.sel(time =(z500CompData.time.dt.season == 'JJA')).tmean)[1]
    # Perform Wilcoxon test across the JJA time series and extract p-values
    is_sig_2 = scipy.stats.false_discovery_control(np.nan_to_num(is_sig,nan = 1))<0.05
    # Apply false discovery rate control and mask significant grid cells at p<0.05
    z500Comp = z500Comp.rolling(longitude=3, latitude = 3).mean()
    # Smooth the spatial field with a 3x3 rolling mean over longitude and latitude
    # Plot the smoothed mean anomaly field on the appropriate subplot.
    mm = ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].pcolormesh(z500Comp.longitude,z500Comp.latitude,z500Comp,cmap = cmap,norm = norm)
    ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].contourf(z500Comp.longitude, z500Comp.latitude, is_sig_2,levels=[-1,0,1], colors='none',hatches=[None,'..'] )
    # Overlay hatching on statistically significant regions
    ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].set_title(my_subtitle[ii], fontsize=26)
cbar = fig.colorbar(sm, location = "bottom", cmap = cmap,ax=ax.ravel().tolist(), fraction = 0.05)
cbar.set_label('anomaly in average 2m temperature ($^\circ$C)', fontsize=22)
cbar.ax.tick_params(size = 22,labelsize=22)

fig.subplots_adjust(left=0.175,
                    bottom=0.2, 
                    right=0.875, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.2)
plt.savefig(OUTPUT_PATH)

### Second figure for Z500 fields

cmap = plt.cm.bwr
myMax = 6000
myMin = 5400
bins = np.arange(myMin,myMax+100,100)
norm = mpl.colors.BoundaryNorm(bins, cmap.N )

myMax = 80
myMin = -80
bins2 = np.arange(myMin,myMax+20,20)
norm2 = mpl.colors.BoundaryNorm(bins2, cmap.N )



fig, ax = plt.subplots(ncols = 2,nrows = 2, figsize=(28,12),subplot_kw=dict(projection=ccrs.PlateCarree()))

compName = ['z500_positive_merged.nc', 'z500_positive_merged.nc', 'z500_negative_merged.nc','z500_negative_merged.nc']
my_subtitle = ['a) Z500 absolute; \n positive occurrence trends', 'b) Z500 anomaly; \n positive occurrence trends', 'c) Z500 absolute;\n negative occurrence trends','d) Z500 anomaly;\n negative occurrence trends']
clim = xr.open_dataset('path/smoothed_z500_clim_paper_4.nc')
for ii in list(np.arange(0,4)):     
    z500CompData = xr.open_dataset('path/Z500_from_Rich/' + compName[ii]).isel(level = 0).drop_vars('level')
    z500Comp = z500CompData.z.mean(dim = "time")
    if np.mod(ii,2)==1:
        z500Comp = (z500CompData.groupby("time.dayofyear") - clim)
        is_sig = scipy.stats.wilcoxon(z500Comp.z)[1]
        is_sig_2 = scipy.stats.false_discovery_control(np.nan_to_num(is_sig,nan = 1))<0.05
        mm = ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].pcolormesh(z500Comp.lon,z500Comp.lat,z500Comp.z.mean(dim = "time")/9.8,transform=ccrs.PlateCarree(),cmap = cmap,norm = norm2)
        ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].contourf(z500Comp.lon, z500Comp.lat, is_sig_2,levels=[-1,0,1], colors='none',hatches=[None,'..'] )
        ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].coastlines()
    else:
        mm = ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].pcolormesh(z500Comp.lon,z500Comp.lat,z500Comp/9.8,transform=ccrs.PlateCarree(),cmap = cmap,norm = norm)
        ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].coastlines()
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    ax[int(np.floor((ii-2)/2))+1,np.mod(ii,2)].set_title(my_subtitle[ii], fontsize=26)
cbar = fig.colorbar(sm, location = "left", cmap = cmap,ax=ax.ravel().tolist(), fraction = 0.05)
cbar.set_label('Geopotential height (m)', fontsize=22)
cbar.ax.tick_params(size = 26,labelsize=26)
cbar2 = fig.colorbar(sm2, location = "right", cmap = cmap,ax=ax.ravel().tolist(), fraction = 0.05)
cbar2.set_label('Geopotential height anomaly (m)', fontsize=22)
cbar2.ax.tick_params(size = 26,labelsize=26)

fig.subplots_adjust(left=0.2,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.2)
plt.savefig(OUTPUT_PATH2)
