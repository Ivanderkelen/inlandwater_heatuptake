
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : June 2019

Main subroutine to do plotting for input variables: 
- natural lake area fraction from hydroLAKES
- reservoir area fraction from GRanD 
- lake depth from GLDB v3 

"""

# import packages

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy as ctp
import mplotutils as mpu 
import xarray as xr 
import numpy as np
# Import libraries
from functions import map_global

# do settings

mpl.rc('axes',edgecolor='grey')
mpl.rc('axes',labelcolor='dimgrey')
mpl.rc('xtick',color='dimgrey')
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',color='dimgrey')
mpl.rc('ytick',labelsize=12)
mpl.rc('axes',titlesize=14)
mpl.rc('axes',labelsize=12)
mpl.rc('legend',fontsize='large')
mpl.rc('text',color='dimgrey')



def map_lakepct(var,lon,lat,levels,title_str,clrmap,fig_name,alone=True,ax=False):


    # figure creation
    if alone:
        fig, ax = plt.subplots(1,1,figsize=(13,8), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
    ax.coastlines(color='darkgrey')

    #ax.coastlines(color="grey")

    LON, LAT = mpu.infer_interval_breaks(lon, lat)

    # add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    cmap, norm = mpu.from_levels_and_cmap(levels, clrmap)

    # do not show zeros
    var[var==0] = np.nan
    cmap.set_over(cmap(0.99))
    h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=norm)
    # set the extent of the cartopy geoAxes to "global"
    ax.set_global()
    ax.set_title(title_str, pad=10, loc='right')

    # remove the frame
    ax.outline_patch.set_visible(False)

    # plot the colorbar
    cbar = mpu.colorbar(h, ax, orientation='horizontal', pad = 0.1, extend='max')
    cbar.ax.set_xlabel('Natural lake area fraction [%]', size=16)
    cbar.set_ticks(np.arange(0,110,10))  # horizontal colorbar
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    # if so, save the figure
    if fig_name != 0 and alone:
        plt.savefig(fig_name+'.jpeg',dpi=1000, bbox_inches='tight')


def map_respct(var,lon,lat,levels,title_str,clrmap,fig_name,alone=True,ax=False):


    # figure creation
    if alone:
        fig, ax = plt.subplots(1,1,figsize=(13,8), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
    ax.coastlines(color='darkgrey')

    #ax.coastlines(color="grey")

    LON, LAT = mpu.infer_interval_breaks(lon, lat)

    # add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    cmap, norm = mpu.from_levels_and_cmap(levels, clrmap)

    # do not show zeros
    var[var==0] = np.nan
    cmap.set_over(cmap(0.99))
    h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=norm)
    # set the extent of the cartopy geoAxes to "global"
    ax.set_global()
    ax.set_title(title_str, pad=10, loc='right')

    # remove the frame
    ax.outline_patch.set_visible(False)

    # plot the colorbar
    cbar = mpu.colorbar(h, ax, orientation='horizontal', pad = 0.1, extend='max')
    cbar.ax.set_xlabel('Reservoir area fraction [%]', size=16)
    cbar.set_ticks(np.arange(0,110,10))  # horizontal colorbar
    # if so, save the figure
    if fig_name != 0 and alone:
        plt.savefig(fig_name+'.jpeg',dpi=1000, bbox_inches='tight')

# lakedepth
def map_global_lakedepth(var,lon,lat,clr_map,clb_label,title_str,fig_name,alone=True,ax=False):
    """
    Funtion to map a variable over the global domain and save into .png file
    """   

    # define the projection
    projection = ccrs.PlateCarree()

    # initiate the figure 
    if alone:
        fig, ax = plt.subplots(1,1,figsize=(13,8), subplot_kw={'projection': ccrs.PlateCarree()})

    # add the coastlines to the plot
    ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
    ax.coastlines(color='darkgrey')
      
    # the original cesm lon and lat assumes center of gridcel. pcolormesh assumes border. 
    LON = lon
    LAT = lat
    
    # do not show zeros
    var[var<=0] = np.nan
    N = 13

   # define the colormap
    cmap =plt.cm.get_cmap(clr_map,N)
   
    # add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=mpl.colors.SymLogNorm(linthresh=2.8, linscale=1.0,vmin=0, vmax=1000))
    
    # set the extent of the cartopy geoAxes to "global"
    ax.set_global()

    # plot the colorbar
    cbar = plt.colorbar(h,ax=ax, label=clb_label, orientation='horizontal', ticks=[0,1, 10, 10**2, 10**3],pad = 0.05,extend='max')
    cbar.ax.set_xticklabels(['0','1', '10', '100', '1000'])
    cbar.set_label(label=clb_label, size=16)
    #cbar.ax.tick_params(labelsize=14) 

    # remove the frame
    ax.outline_patch.set_visible(False)
    plt.title(title_str, fontsize=20, pad=10, loc='right')
    
    # save the figure, adjusting the resolution 
    if fig_name != 0 and alone:
        plt.savefig(fig_name+'.jpeg',dpi=1000, bbox_inches='tight')


#%% big function
def do_plotting_globalmaps(indir_lakedata, plotdir, years_grand,start_year,end_year):

    # GLDB lake depths (GLDB lake depths, hydrolakes lake area)
    lakedepth_path        = indir_lakedata + 'dlake_1km_ll_remapped_0.5x0.5.nc'
    lakepct_path          = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191203.nc'

    # load variables
    gldb_lakedepth      = xr.open_dataset(lakedepth_path)
    hydrolakes_lakepct  = xr.open_dataset(lakepct_path)

    lake_depth         = gldb_lakedepth.dl.values[0,:,:]
    lake_pct           = hydrolakes_lakepct.PCT_LAKE.values# to have then in fraction
    landmask           = hydrolakes_lakepct.LANDMASK.values

    lon = hydrolakes_lakepct.PCT_LAKE.lon
    lat = hydrolakes_lakepct.PCT_LAKE.lat

    lake_depth = lake_depth * landmask
    end_year = 2017
    # take analysis years of lake_pct
    lake_pct  = lake_pct[years_grand.index(start_year):years_grand.index(end_year), :, :]

    natlake_pct  = lake_pct[0, :, :]

    # select all reservois appeared 
    res_pct = lake_pct[-1,:,:] - lake_pct[0,:,:]

    # %% Make seperate maps
    # map lake pct
    map_lakepct(natlake_pct,lon,lat,np.arange(0,105,5),'Natural lake area fraction, based on HydroLAKES', 'Blues',plotdir+'map_lake_fraction')

    # map reservoir pct 
    map_respct(res_pct,lon,lat,np.arange(0,110,10),'Reservoir area fraction in 2017, based on GRanD', 'Blues',plotdir+'map_res_fraction')

    # map global lake depth
    map_global_lakedepth(lake_depth,lon,lat,'Blues','Lake depth [m]','Lake depth based on GLDB v3',plotdir+'map_lakedepth')

# %% Make one figure - not yet working. 

# fig, (ax1,ax2,ax3) = plt.subplots(3,1,subplot_kw={'projection': ccrs.PlateCarree()})

# # map global lake depth
# map_global_lakedepth(lake_depth,lon,lat,'Blues','Lake depth [m]','Lake depth based on GLDB v3',plotdir+'map_lakedepth',  False,ax1)
# #ax1.text(0.03, 0.92, '(a)', transform=ax1.transAxes, fontsize=14)

# # map lake pct
# map_lakepct(natlake_pct,lon,lat,np.arange(0,105,5),'Natural lake area fraction, based on HydroLAKES', 'Blues',plotdir+'map_lake_fraction', False, ax2)

# # map reservoir pct 
# map_respct(res_pct,lon,lat,np.arange(0,110,10),'Reservoir area fraction in 2017, based on GRanD', 'Blues',plotdir+'map_res_fraction', False, ax3)
# plt.tight_layout()

# plt.savefig(plotdir+'maps_input'+'.png')




#%%
