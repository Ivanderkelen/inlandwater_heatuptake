"""

Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : September 2019

Test script to fill temperatures with nearest neigbours
contains great lake region plotting area
"""

#%%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patheffects import Stroke
import mplotutils as mpu 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy as ctp
import geopandas as gpd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely.geometry as sgeom
from shapely.geometry import box
from dict_functions import *


mpl.rc('axes',edgecolor='grey')
mpl.rc('axes',labelcolor='dimgrey')
mpl.rc('xtick',color='dimgrey')
mpl.rc('xtick',labelsize=16)
mpl.rc('ytick',color='dimgrey')
mpl.rc('ytick',labelsize=16)
mpl.rc('axes',titlesize=20)
mpl.rc('text',color='dimgrey')


# functions

def extract_from_hydrolakes(extend,extend_fname):
    """Script to extract the great polygon files from hydrolakes and GranD dataset
        input= extend (array), extend_name (string including path)
        filters lakes based on an area threshold """
    area_threshold = 100 #in km²
    from shapely.geometry import box
    if not os.path.isfile(extend_fname):
        hydrolakes_dams_path = '/home/inne/documents/phd/data/processed/polygon_to_cellareafraction/Hydrolakes_dams.shp'
        print('Loading HydroLAKES ...')
        lakes = gpd.read_file(hydrolakes_dams_path)
        boundingbox = box(extend[0], extend[1], extend[2], extend[3])
        print('Extracting lakes ...')
        lakes_selected = lakes[lakes['Lake_area']>=area_threshold]
        lakes_extracted = lakes_selected[lakes.geometry.intersects(boundingbox)]
        lakes_extracted.to_file(extend_fname)
    else:
        print('Already extracted '+extend_fname)
#%%
# Settings



#%% Plotting function


    
# read heat content data
# load lake heat (to be removed)
lakeheat = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()

# load lon lat of lakeheat data
lon,lat = get_lonlat(indir_lakedata)

# calculate timeseries maps of mean of all models and forcings 
lakeheat_anom = calc_anomalies(lakeheat, flag_ref)
var = ens_spmean_ensmean(lakeheat_anom) # output np array (time,lat,lon)
var = var[-1,:,:]
# 
clb_label='J'
title_str=name_str+ ' heat content anomaly'
fig_name='Heat_content_'+name
cmap = 'YlOrBr'

# actual figure creation
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13,8), subplot_kw={'projection': ccrs.PlateCarree()})

def plot_region_hc_map(ax, extent, continent_extent, name, name_str,path_lakes,ax_location,levels):

    ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
    ax.coastlines(color="grey")

    LON, LAT = mpu.infer_interval_breaks(lon,lat)

    # add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    cmap, norm = mpu.from_levels_and_cmap(levels, cmap, extend='max')

    h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=norm)
    # load the lake shapefile
    lakes = gpd.read_file(path_lakes)
    
    lakes.plot(ax=ax,edgecolor='gray',facecolor='none')

    extent[1], extent [2] = extent[2], extent[1]
    ax.set_extent(extent)

    # create effect for map borders: 
    effect = Stroke(linewidth=1.5, foreground='darkgray')
    # set effect for main ax
    ax.outline_patch.set_path_effects([effect])


    # Create an inset GeoAxes showing the location of the lakes region
    #x0 y0 width height
    sub_ax = fig.add_axes(ax_location,
                            projection=ccrs.PlateCarree())
    sub_ax.set_extent(continent_extent)
    #lakes.plot(ax=sub_ax)

    # Make a nice border around the inset axes.
    #effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)

    sub_ax.outline_patch.set_path_effects([effect])
    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='none',
                            edgecolor='red', linewidth=2)


    # Add the land, coastlines and the extent of the inset axis
    sub_ax.add_feature(cfeature.LAND, edgecolor='gray')
    sub_ax.coastlines(color='gray')
    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='none',
                            edgecolor='black', linewidth=2)


    # plot the colorbar
    cbar = mpu.colorbar(h, ax, extend='max', orientation='horizontal', pad = 0.05)
    cbar.ax.set_xlabel(clb_label, size=16)
    ax.set_title(title_str, pad=10)


    plotdir='/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/'
    plt.savefig(plotdir+fig_name+'.png')

#%%Settings

lakes_path = '/home/inne/documents/phd/data/processed/lakes_shp/'

# Laurentian Great Lakes
extent_LGL = [-92.5,41,-75.5,49.5]
continent_extent_NA = [-128.7,-61.4,6.5,62.2] # continent_extent for inset
ax_location_NA = [0.7, 0.61, 0.25, 0.2]
name_LGL   = 'LaurentianGreatLakes'
name_str_LGL = 'Laurentian Great Lakes'
path_LGL   = lakes_path+name_LGL+'.shp'
levels_LGL = np.arange(np.nanmin(var),np.nanmax(var),(np.nanmax(var)-np.nanmin(var))/5)



# African Great Lakes
extent_AGL = [28,-10,35.5,2.5]
continent_extent_AF = [-18.5,51,-34.5,37] 
ax_location_AF = [0.3, 0.7, 0.6, 0.3]
name_AGL   = 'AfricanGreatLakes'
name_str_AGL = 'African Great Lakes'
lakes_path = '/home/inne/documents/phd/data/processed/lakes_shp/'
path_AGL   = lakes_path+name_AGL+'.shp'


plot_region_hc_map(extent_LGL, continent_extent_NA, name_LGL, name_str_LGL ,path_LGL,ax_location_NA,levels_LGL)

plot_region_hc_map(extent_AGL, continent_extent_AF, name_AGL, name_str_AGL ,path_AGL,ax_location_AGL,levels_AGL)




# general settings
name_str = 'African Great Lakes'
name = name_AGL
extent = extent_AGL
# switch xmin and xmax
continent_extent = continent_extent_AF
ax_location = [0.55, 0.125, 0.1, 0.2]
# do the same for the African Great Lakes
levels = np.arange(0,2.2e21,0.2e21)
