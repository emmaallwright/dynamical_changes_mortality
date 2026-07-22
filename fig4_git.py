"""Generate Figure 4 for the manuscript.

This script creates a 4-panel regional map figure showing attributable fraction
(AF) differences at NUTS administrative levels. It loads regional AF data for
counterfactual and factual scenarios, filters to JJA season, computes statistics
with significance testing, and plots choropleth maps with insets for specific
regions (Cyprus, Canary Islands, Madeira) and non-significant areas masked.

Data sources:
  - `mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.*_Z500_AF_REG.nc`: regional AF
  - `extra_files/Indices.csv`: dynamical change indices
  - `Observation_data/europe_allnuts_gisco_2003_2021_nuts3_shap.geojson`: NUTS geometries

Output:
  - `figures/fig4.png`
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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a colormap to a specified range.
    
    Args:
        cmap: Colormap to truncate.
        minval: Minimum value in range [0, 1].
        maxval: Maximum value in range [0, 1].
        n: Number of color levels in output colormap.
    
    Returns:
        Truncated colormap.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


BASE_DIR = Path("path")
SHAPE_DIR = Path("path")
OUTPUT_PATH = BASE_DIR / "figures" / "fig4.png"
DATA_DIR = BASE_DIR / "mortality_data"
myRegions = pandas.read_csv('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_both_Z500_METATABLE.csv')

# Load mortality results metadata for the region filtering and plotting logic.

#Load and assign shape file regions
myShp = geopandas.read_file('path/Observation_data/europe_allnuts_gisco_2003_2021_nuts3_shap.geojson')
myShp2 = geopandas.read_file('path/Observation_data/CNTR_RG_10M_2020_3035.geojson')
# Separate NUTS region identifiers and geometries by administrative level.
myNuts0 = myShp[myShp['LEVL_CODE']==0.0]["NUTS_ID"].to_numpy()
myNuts1 = myShp[myShp['LEVL_CODE']==1.0]["NUTS_ID"].to_numpy()
myNuts2 = myShp[myShp['LEVL_CODE']==2.0]["NUTS_ID"].to_numpy()
myNuts3 = myShp[myShp['LEVL_CODE']==3.0]["NUTS_ID"].to_numpy()
myGeom0 = myShp[myShp['LEVL_CODE']==0.0]["geometry"].to_numpy()
myGeom1 = myShp[myShp['LEVL_CODE']==1.0]["geometry"].to_numpy()
myGeom2 = myShp[myShp['LEVL_CODE']==2.0]["geometry"].to_numpy()
myGeom3 = myShp[myShp['LEVL_CODE']==3.0]["geometry"].to_numpy()


# Load NetCDF regional AF datasets for the six scenario combinations.
myData_cf_pos = xr.open_dataset('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_positive_Z500_AF_REG.nc')
myData_cf_neg = xr.open_dataset('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_negative_Z500_AF_REG.nc')
myData_cf_both = xr.open_dataset('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_both_Z500_AF_REG.nc')
myData_f_pos = xr.open_dataset('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_positive_Z500_AF_REG.nc')
myData_f_neg = xr.open_dataset('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_negative_Z500_AF_REG.nc')
myData_f_both = xr.open_dataset('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_both_Z500_AF_REG.nc')
true_factual = pandas.read_excel('path/mortality_data/TABLE_WEEKS_AF_totalheat_allsex_allage_iSENSITIVITY.0_sANALOGUE.true_factual_REG.xlsx', engine='openpyxl')

# Load indices for days with dynamical changes (from Faranda et al. 2023 https://doi.org/10.1073/pnas.2214525120)
myIndices = pandas.read_table(BASE_DIR / "extra_files" / "Indices.csv", header = None, sep = ";", usecols= range(1,916))
myTime = xr.open_dataset(BASE_DIR / "extra_files" / "cdo_anomaly_detrended_fldmean.nc").time
# Convert MATLAB-style indices to actual date coordinates (subtract 1 for zero-based indexing).
indices1 = myTime[myIndices.iloc[0,:].to_numpy(dtype = np.int32)-1] # subtract one because matlab
indices2 = myTime[myIndices.loc[1,~np.isnan(myIndices.iloc[1,:])].to_numpy(dtype = np.int32)-1]
indices3 = myTime[myIndices.loc[2,~np.isnan(myIndices.iloc[2,:])].to_numpy(dtype = np.int32)-1]
indices4 = myTime[myIndices.loc[3,~np.isnan(myIndices.iloc[3,:])].to_numpy(dtype = np.int32)-1]

# ,"location","name_off","name_eng","level","nuts0","nuts1","nuts2","nuts3","year_shp","year_ini","year_end"
myRegions = pandas.read_csv('path/mortality_data/figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_both_Z500_METATABLE.csv')
myData_cf_pos['dim1'] = myRegions['location'].values
myData_cf_neg['dim1'] = myRegions['location'].values
myData_cf_both['dim1'] = myRegions['location'].values
myData_f_pos['dim1'] = myRegions['location'].values
myData_f_neg['dim1'] = myRegions['location'].values
myData_f_both['dim1'] = myRegions['location'].values

# Load the observed reference AF data by region for computing differences.
region_time_series = true_factual.iloc[:, 2:].values 
region_codes = true_factual.iloc[:, 1].values 
myData = [myData_cf_pos,myData_f_pos,myData_cf_neg,myData_f_neg]

# Setup figure layout and colormap for regional AF difference maps.
fig, ax = plt.subplots(ncols = 2,nrows = 2, figsize=(30,25))
myMax = 0.25
myMin = -0.025
myColStep = 0.025

import matplotlib as mpl
import matplotlib.colors as colors
# Create a truncated blue-white-red colormap emphasizing positive values.
cmap2 = plt.get_cmap('bwr')
cmap2 = truncate_colormap(cmap2, 0.45, 1.0)

bins = np.arange(myMin,myMax+np.abs(myMin),myColStep)
norm2 = mpl.colors.BoundaryNorm(bins, cmap2.N)
myRegions_obs = [(np.isin(myRegions['location'].values,myNuts0)),(np.isin(myRegions['location'].values,myNuts1)),(np.isin(myRegions['location'].values,myNuts2)),(np.isin(myRegions['location'].values,myNuts3))]
my_subtitle = ['a) 1950 counterfactual;\n positive occurrence trends', 'b) Trended counterfactual;\n positive occurrence trends', 'c) 1950 counterfactual;\n negative occurrence trends','d) Trended counterfactual;\n negative occurrence trends']
pandasResult = pandas.DataFrame({
    "week": pandas.date_range(start="1950-01-01", end="2021-12-02", freq="W")
})
for ii in list(np.arange(0,4)):   
    my_selected_data = region_time_series.T-myData[ii].isel(dim3= 2, dim4 = 0, dim2 = slice(0,-1))
    if ii<2:
        my_selected_data = my_selected_data#*946/275
    else:
        my_selected_data = my_selected_data#*946/15
    #myAFRA = myData.isel(dim2 = -1,dim3= 0, dim4 = 0).sel(dim1 = myRegions_obs[0])
    #myDF = geopandas.GeoDataFrame(myAFRA.sel(dim1 = myNuts0[np.isin(myNuts0,myAFRA.dim1.values)])).set_geometry(myGeom0[np.isin(myNuts0,myAFRA.dim1.values)])
    #mm = myDF.plot(column=0, cmap=cmap2, k=5, linewidth=0.3,edgecolor='0.1', ax = ax[2,ii], norm = norm2 )
    # Assign weekly datetime coordinates.\n    my_selected_data['dim2'] = pandasResult['week'].values
    # Filter to JJA season only.
    my_selected_data['dim2'] = pandasResult['week'].values
    my_selected_data = my_selected_data.sel(dim2 =(my_selected_data.dim2.dt.season == 'JJA'))
    # Perform Wilcoxon test and extract p-values.
    is_sig = scipy.stats.wilcoxon(my_selected_data.var1_1.where(my_selected_data.var1_1!=0), axis = 0,nan_policy = 'omit')[1]
    # Apply false discovery rate control and create boolean mask for significance.
    is_sig_2 = scipy.stats.false_discovery_control(np.nan_to_num(is_sig,nan = 1))<0.05
    is_sig_2 = xr.DataArray(data=is_sig_2,dims=["dim1"],coords=dict(dim1=(["dim1"], my_selected_data.dim1.data)))
    myAFRA = my_selected_data.sel(dim1 = myRegions_obs[1]).var1_1
    mySig = is_sig_2.sel(dim1 = myRegions_obs[1])
    myDF = geopandas.GeoDataFrame(myAFRA.sel(dim1 = myNuts1[np.isin(myNuts1,myAFRA.dim1.values)]).mean(dim = 'dim2')).set_geometry(myGeom1[np.isin(myNuts1,myAFRA.dim1.values)])
    mm = myDF.plot(column=0, cmap=cmap2, k=5, linewidth=0.3,edgecolor='0.1', ax = ax[int(np.floor(ii/2)),int(np.mod(ii,2))], norm = norm2 )
    myDF_sig = geopandas.GeoDataFrame(mySig[~mySig]).set_geometry(myGeom1[np.isin(myNuts1,myAFRA.dim1.values)][~mySig])
    mm10 = myDF_sig.plot(color='lightgrey', k=5, linewidth=0.3,edgecolor='0.1', ax = ax[int(np.floor(ii/2)),int(np.mod(ii,2))])    
    myAFRA = my_selected_data.sel(dim1 = myRegions_obs[2]).var1_1
    mySig = is_sig_2.sel(dim1 = myRegions_obs[2])
    myDF = geopandas.GeoDataFrame(myAFRA.sel(dim1 = myNuts2[np.isin(myNuts2,myAFRA.dim1.values)]).mean(dim = 'dim2')).set_geometry(myGeom2[np.isin(myNuts2,myAFRA.dim1.values)])
    mm = myDF.plot(column=0, cmap=cmap2, k=5, linewidth=0.3,edgecolor='0.1', ax = ax[int(np.floor(ii/2)),int(np.mod(ii,2))], norm = norm2)
    myDF_sig = geopandas.GeoDataFrame(mySig[~mySig]).set_geometry(myGeom2[np.isin(myNuts2,myAFRA.dim1.values)][~mySig])
    mm10 = myDF_sig.plot(color='lightgrey', k=5, linewidth=0.3,edgecolor='0.1', ax = ax[int(np.floor(ii/2)),int(np.mod(ii,2))])    
    myAFRA = my_selected_data.sel(dim1 = myRegions_obs[3]).var1_1
    mySig = is_sig_2.sel(dim1 = myRegions_obs[3])
    myDF = geopandas.GeoDataFrame(myAFRA.sel(dim1 = myNuts3[np.isin(myNuts3,myAFRA.dim1.values)]).mean(dim = 'dim2')).set_geometry(myGeom3[np.isin(myNuts3,myAFRA.dim1.values)])
    mm = myDF.plot(column=0, cmap=cmap2, k=5, linewidth=0.3,edgecolor='0.1', ax = ax[int(np.floor(ii/2)),int(np.mod(ii,2))], norm = norm2 )
    myDF_sig = geopandas.GeoDataFrame(mySig[~mySig]).set_geometry(myGeom3[np.isin(myNuts3,myAFRA.dim1.values)][~mySig])
    mm10 = myDF_sig.plot(color='lightgrey', k=5, linewidth=0.3,edgecolor='0.1', ax = ax[int(np.floor(ii/2)),int(np.mod(ii,2))])    
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].axis('off')
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].set_title(my_subtitle[ii], fontsize=30)
    axins = ax[int(np.floor(ii/2)),int(np.mod(ii,2))].inset_axes([0.29, 0.82, 0.1, 0.1], xticklabels=[], yticklabels=[])
    #axins.imshow(Z2, extent=extent, origin="lower")
    axins.set_yticks([])
    axins.set_xticks([], minor=True)
    axins.set_xticks([])
    myAFRA = my_selected_data.sel(dim1 = 'CY000')  
    mySig = is_sig_2.sel(dim1 = 'CY000')
    myDF = geopandas.GeoDataFrame(myAFRA.mean(dim = 'dim2').to_pandas()).set_geometry(myGeom3[np.isin(myNuts3, 'CY000')])
    mm = myDF.plot(column=0, cmap=cmap2, k=5, linewidth=0.3,edgecolor='0.1', ax = axins, norm = norm2 )
    if (~mySig):
        myDF_sig = geopandas.GeoDataFrame(mySig).set_geometry(myGeom3[np.isin(myNuts3, 'CY000')])
        mm10 = myDF_sig.plot(color='lightgrey', k=5, linewidth=0.3,edgecolor='0.1', ax =axins)
    axins2 = ax[int(np.floor(ii/2)),int(np.mod(ii,2))].inset_axes([0.0, 0.59, 0.15, 0.15], xticklabels=[], yticklabels=[])
    axins2.set_yticks([])
    axins2.set_xticks([], minor=True)
    axins2.set_xticks([])
    myAFRA = my_selected_data.sel(dim1= ['ES703','ES704','ES705','ES706','ES707','ES708','ES709'])
    mySig = is_sig_2.sel(dim1 = ['ES703','ES704','ES705','ES706','ES707','ES708','ES709'])
    myDF = geopandas.GeoDataFrame(myAFRA.mean(dim = 'dim2').var1_1).set_geometry(myGeom3[np.isin(myNuts3, ['ES703','ES704','ES705','ES706','ES707','ES708','ES709'] )])
    mm = myDF.plot(column=0, cmap=cmap2, k=5, linewidth=0.3,edgecolor='0.1', ax = axins2, norm = norm2 )
    myDF_sig = geopandas.GeoDataFrame(mySig[~mySig]).set_geometry(myGeom3[np.isin(myNuts3, ['ES703','ES704','ES705','ES706','ES707','ES708','ES709'] )][~mySig])
    mm10 = myDF_sig.plot(color='lightgrey', k=5, linewidth=0.3,edgecolor='0.1', ax = axins2)    
    axins4 = ax[int(np.floor(ii/2)),int(np.mod(ii,2))].inset_axes([0.0, 0.372, 0.15, 0.15], xticklabels=[], yticklabels=[])
    axins4.set_yticks([])
    axins4.set_xticks([], minor=True)
    axins4.set_xticks([])
    myAFRA = my_selected_data.sel(dim1 = 'PT300')
    mySig = is_sig_2.sel(dim1 = 'PT300')
    myDF = geopandas.GeoDataFrame(myAFRA.mean(dim = 'dim2').to_pandas()).set_geometry(myGeom3[np.isin(myNuts3, 'PT300')])
    mm = myDF.plot(column=0, cmap=cmap2, k=5, linewidth=0.3,edgecolor='0.1', ax = axins4, norm = norm2 )
    if (~mySig):
        myDF_sig = geopandas.GeoDataFrame(mySig).set_geometry(myGeom3[np.isin(myNuts3, 'PT300')])
        mm10 = myDF_sig.plot(color='lightgrey', k=5, linewidth=0.3,edgecolor='0.1', ax = axins4)    
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].annotate('Canary Islands \n(Spain)', ((2300000),(140000+0.76*5500000)), fontsize = 22)  
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].annotate('Cyprus', ((2300000 +0.18*6100000),(140000+0.921*5500000)), fontsize = 22)    
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].annotate('Madeira \n(Portugal)', ((2300000),(140000+0.622*5500000)), fontsize = 22)    
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].set_xlim(2300000, 6100000)
    ax[int(np.floor(ii/2)),int(np.mod(ii,2))].set_ylim(1400000, 5500000)    
    myAFRA = [0]
    myDF = geopandas.GeoDataFrame(myAFRA).set_geometry(myShp2[np.isin(myShp2['CNTR_ID'],['RU'])]['geometry'].values)
    mm = myDF.plot(color="white", k=5, linewidth=0.3,edgecolor='0.1', ax = ax[int(np.floor(ii/2)),int(np.mod(ii,2))])

fig.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.7, 
                    top=0.9, 
                    wspace=0.0, 
                    hspace=0.05)
cbar = fig.colorbar(sm2,location = 'right', cmap = cmap2, norm = norm2 ,ax=ax.ravel(), fraction = 0.03)
cbar.ax.tick_params(size = 24,labelsize=20)
cbar.set_label(r'attributable fraction (AF) (%)', fontsize=30)
# Save the completed figure to the manuscript figures directory.
plt.savefig(OUTPUT_PATH)
