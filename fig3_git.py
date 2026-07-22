"""
Generate Figure 3 for the manuscript.

This script loads mortality and exposure datasets for counterfactual and
factual heat/cold scenarios, loads administrative shapefiles, and builds a
4-panel time series figure of attributable fraction (AF) differences.

Processing steps:
    - read scenario NetCDF results and observed reference data
    - assign weekly datetime coordinates
    - compute AF differences and apply a 5-year rolling mean
    - plot four panels for positive/negative occurrence trend scenarios
    - save output to `figures/fig3.png`

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import pandas
import scipy
import geopandas 
import scipy.stats
import matplotlib as mpl


BASE_DIR = Path("path")
SHAPE_DIR = Path("path")
OUTPUT_PATH = BASE_DIR / "figures" / "fig3.png"
DATA_DIR = BASE_DIR / "mortality_data"
myRegions = pandas.read_csv('path/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_both_Z500_METATABLE.csv')

# Load mortality results metadata for the region filtering and plotting logic.

#Load and assign shape file regions
myShp = geopandas.read_file('path/europe_allnuts_gisco_2003_2021_nuts3_shap.geojson')
myShp2 = geopandas.read_file('path/CNTR_RG_10M_2020_3035.geojson')
# Separate NUTS region identifiers and geometries by administrative level.
myNuts0 = myShp[myShp['LEVL_CODE']==0.0]["NUTS_ID"].to_numpy()
myNuts1 = myShp[myShp['LEVL_CODE']==1.0]["NUTS_ID"].to_numpy()
myNuts2 = myShp[myShp['LEVL_CODE']==2.0]["NUTS_ID"].to_numpy()
myNuts3 = myShp[myShp['LEVL_CODE']==3.0]["NUTS_ID"].to_numpy()
myGeom0 = myShp[myShp['LEVL_CODE']==0.0]["geometry"].to_numpy()
myGeom1 = myShp[myShp['LEVL_CODE']==1.0]["geometry"].to_numpy()
myGeom2 = myShp[myShp['LEVL_CODE']==2.0]["geometry"].to_numpy()
myGeom3 = myShp[myShp['LEVL_CODE']==3.0]["geometry"].to_numpy()

myData_cf_pos = xr.open_dataset('path/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_positive_Z500_AF_TOT.nc')
myData_cf_neg = xr.open_dataset('path/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_negative_Z500_AF_TOT.nc')
myData_cf_both = xr.open_dataset('path/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_both_Z500_AF_TOT.nc')
myData_f_pos = xr.open_dataset('path/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_positive_Z500_AF_TOT.nc')
myData_f_neg = xr.open_dataset('path/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_negative_Z500_AF_TOT.nc')
myData_f_both = xr.open_dataset('path/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_both_Z500_AF_TOT.nc')

# Load NetCDF scenario results for the six case combinations.


true_factual = pandas.read_excel('path/TABLE_WEEKS_AF_totalheat_allsex_allage_iSENSITIVITY.0_sANALOGUE.true_factual_COU.xlsx', engine='openpyxl')
euro_time_series = true_factual.iloc[0, 2:].values 
myData = [myData_cf_pos,myData_f_pos,myData_cf_neg,myData_f_neg,myData_cf_both,myData_f_both]

# Collect the loaded scenario datasets into a list for the plotting loop.

# Extract the reference AF time series from the observed table.

# whole period, total cold and heat, point estimate
fig, ax = plt.subplots(ncols = 2,nrows = 2, figsize=(30,30))
myMax = 10
myMin = 0 
import matplotlib as mpl
cmap2 = plt.cm.PuRd
bins = np.arange(myMin,myMax+3,3)
norm2 = mpl.colors.BoundaryNorm(bins, cmap2.N )
# Configure the 4-panel plot and color mapping for the time series.
myRegions_obs = [(np.isin(myRegions['location'].values,myNuts0)),(np.isin(myRegions['location'].values,myNuts1)),(np.isin(myRegions['location'].values,myNuts2)),(np.isin(myRegions['location'].values,myNuts3))]
# Build boolean masks for regions at different NUTS administrative levels.
my_subtitle = ['a) Factual - 1950 counterfactual;\n positive occurrence trends', 'b) Factual - trended counterfactual;\n positive occurrence trends', 'c) Factual - 1950 counterfactual;\n negative occurrence trends','d) Factual - trended counterfactual;\n negative occurrence trends']
pandasResult = pandas.DataFrame({
    "week": pandas.date_range(start="1950-01-01", end="2021-12-02", freq="W")
})

# Build a weekly date index matching the AF data coordinate.

for ii in list(np.arange(0,4)):   
    # Select the relevant data slice for this scenario and collapse unwanted dimensions.
    my_selected_data = myData[ii].isel(dim1 = 0, dim3=2, dim4 = 0)
    # Subtract the model-based series from the reference observed time series.
    my_selected_data = euro_time_series-my_selected_data.isel(dim2 = slice(0,-1))
    # Assign the weekly datetime coordinate to the time dimension.
    my_selected_data['dim2'] = pandasResult['week'].values
    # Smooth the resulting AF difference series with a centered 5-year rolling mean.
    my_selected_data = my_selected_data.var1_1.rolling(dim2 = 52*5, center = True).mean()
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].scatter(my_selected_data.dim2.values,my_selected_data.values,color = 'red', alpha = 0.9, linewidth = 5)
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].set_title(my_subtitle[ii], fontsize=34)
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].set_ylim(-0.05, 0.15)    
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].set_xlabel('time (year)', fontsize=28)
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].set_ylabel('AF (%)', fontsize=28)
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].tick_params(size = 28, labelsize=24)
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].axhline(0, alpha = 0.9, linewidth = 5, color = 'black')
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].axhline(my_selected_data.mean(), alpha = 0.9, linewidth = 5, color = 'black')


fig.subplots_adjust(left=0.1,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.2)



plt.savefig(BASE_DIR / "figures" / "fig3.png")
# Save the completed figure to the manuscript figures directory.
