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
import sys
sys.path.append(r'/home/inne/documents/phd/scripts/python/calc_lakeheat_isimip/lakeheat_isimip')
from dict_functions import *


# paths from main
basepath = '/home/inne/documents/phd/'
project_name = 'isimip_lakeheat/'
indir_lakedata   = basepath + 'data/isimip_laketemp/' # directory where lake fraction and depth are located

outdir = basepath + 'data/processed/'+ project_name
flag_ref = 'pre-industrial'

start_year = 1900
end_year = 2017
years_analysis         = range(start_year,end_year,1)


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
    fname = '/home/inne/documents/phd/data/processed/lakes_shp/'+extend_fname+'.shp'

    """Script to extract the great polygon files from hydrolakes and GranD dataset
        input= extend (array), extend_name (string including path)
        filters lakes based on an area threshold """
    area_threshold = 1000 #in km²
    from shapely.geometry import box
    if not os.path.isfile(extend_fname):
        hydrolakes_dams_path = '/home/inne/documents/phd/data/HydroLAKES_polys_v10_shp/Hydrolakes_light.shp'
        print('Loading HydroLAKES ...')
        lakes = gpd.read_file(hydrolakes_dams_path)

        boundingbox = box(extend[0], extend[1], extend[2], extend[3])
        print('Extracting lakes ...')
        lakes_selected = lakes[lakes['Lake_area']>=area_threshold]
        lakes_extracted = lakes_selected[lakes.geometry.intersects(boundingbox)]
        lakes_extracted.to_file(fname)
    else:
        print('Already extracted '+fname)




#%% Plotting function


def plot_region_hc_map(var, region_props, lakes_path, indir_lakedata):

    # get region specific info from dictionary
    extent            = region_props['extent']
    continent_extent  = region_props['continent_extent']
    name              = region_props['name']
    name_str          = region_props['name_str']
    ax_location       = region_props['ax_location']
    levels            = region_props['levels']
    fig_size           = region_props['fig_size']
    cb_orientation    = region_props['cb_orientation']

    path_lakes = lakes_path+name+'.shp'

    # settings
    clb_label='Joule'
    title_str=name_str+ ' heat content anomaly'
    fig_name='Heat_content_'+name
    cmap = 'YlOrBr'

    cmap, norm = mpu.from_levels_and_cmap(levels, cmap, extend='max')
    lon,lat = get_lonlat(indir_lakedata)
    LON, LAT = mpu.infer_interval_breaks(lon,lat)
    lakes = gpd.read_file(path_lakes)

    # plotting
    fig, ax = plt.subplots(1,1,figsize=fig_size, subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
    ax.coastlines(color="grey")
    # add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=norm)
    # load the lake shapefile

    lakes.plot(ax=ax,edgecolor='gray',facecolor='none')

    # set grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gainsboro', alpha=0.5)
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_bottom = None
    gl.ylabels_right = None

    # set extent (in right way)
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
    cbar = mpu.colorbar(h, ax, extend='max', orientation=cb_orientation, pad = 0.05)
    if cb_orientation =='vertical':
        cbar.ax.set_ylabel(clb_label, size=16)
    elif cb_orientation == 'horizontal':
        cbar.ax.set_xlabel(clb_label, size=16)

    #ax.set_title(title_str, pad=10)

    plotdir='/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/'
    plt.savefig(plotdir+fig_name+'.png')


def plot_region_hc_rivers_map(var, region_props, indir_lakedata):

    # get region specific info from dictionary
    extent            = region_props['extent']
    continent_extent  = region_props['continent_extent']
    name              = region_props['name']
    name_str          = region_props['name_str']
    ax_location       = region_props['ax_location']
    levels            = region_props['levels']
    fig_size           = region_props['fig_size']
    cb_orientation    = region_props['cb_orientation']

    # settings
    clb_label='Joule'
    title_str=name_str+ ' heat content anomaly'
    fig_name='Heat_content_'+name
    cmap = 'YlOrBr'

    cmap, norm = mpu.from_levels_and_cmap(levels, cmap, extend='max')
    lon,lat = get_lonlat(indir_lakedata)
    LON, LAT = mpu.infer_interval_breaks(lon,lat)
 
    # plotting
    fig, ax = plt.subplots(1,1,figsize=fig_size, subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
    ax.coastlines(color="grey")
    # add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=norm)

    # set grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gainsboro', alpha=0.5)
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_bottom = None
    gl.ylabels_right = None

    # set extent (in right way)
    extent[1], extent [2] = extent[2], extent[1]
    ax.set_extent(extent)

    # create effect for map borders: 
    effect = Stroke(linewidth=1.5, foreground='darkgray')
    # set effect for main ax
    ax.outline_patch.set_path_effects([effect])
    ax1.text(0.03, 0.92, label, transform=ax1.transAxes, fontsize=14)


    # Create an inset GeoAxes showing the location of the lakes region
    #x0 y0 width height
    sub_ax = fig.add_axes(ax_location,
                            projection=ccrs.PlateCarree())
    sub_ax.set_extent(continent_extent)

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
    cbar = mpu.colorbar(h, ax, extend='max', orientation=cb_orientation, pad = 0.05)
    if cb_orientation =='vertical':
        cbar.ax.set_ylabel(clb_label, size=16)
    elif cb_orientation == 'horizontal':
        cbar.ax.set_xlabel(clb_label, size=16)

    #ax.set_title(title_str, pad=10)

    plotdir='/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/'
    plt.savefig(plotdir+fig_name+'.png')




def plot_global_hc_map(name_str, var, lakes_path, indir_lakedata):

    # get region specific info from dictionary
    if name_str == 'global_absolute':
        levels            = np.arange(-1e17,1.1e17,0.1e17)
    elif name_str == 'global':
        levels            = np.arange(-1e19,1.1e19,0.1e19)

    cb_orientation    = 'horizontal'
    path_lakes = lakes_path+name_str+'.shp'

    # settings
    clb_label='Joule'
    title_str=name_str+ ' heat content anomaly'
    fig_name='Heat_content_'+name_str
    cmap = 'RdBu_r'#, 'YlOrBr'

    cmap, norm = mpu.from_levels_and_cmap(levels, cmap, extend='max')
    lon,lat = get_lonlat(indir_lakedata)
    LON, LAT = mpu.infer_interval_breaks(lon,lat)

    # plotting
    fig, ax = plt.subplots(1,1,figsize=(13,8), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
    ax.coastlines(color="grey")
    # add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=norm)

    # set grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gainsboro', alpha=0.5)
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_bottom = None
    gl.ylabels_right = None


    effect = Stroke(linewidth=1.5, foreground='darkgray')
    # set effect for main ax
    ax.outline_patch.set_path_effects([effect])

    # plot the colorbar
    cbar = mpu.colorbar(h, ax, extend='max', orientation=cb_orientation, pad = 0.05)
    if cb_orientation =='vertical':
        cbar.ax.set_ylabel(clb_label, size=16)
    elif cb_orientation == 'horizontal':
        cbar.ax.set_xlabel(clb_label, size=16)

    #ax.set_title(title_str, pad=10)

    plotdir='/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/'
    plt.savefig(plotdir+fig_name+'.png')
#%%Settings

lakes_path = '/home/inne/documents/phd/data/processed/lakes_shp/'


# read heat content data
# load lake heat (to be removed)
lakeheat_climate = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()


# load lon lat of lakeheat data

# calculate timeseries maps of mean of all models and forcings 
lakeheat_ensmean = ens_spmean_ensmean(lakeheat_climate) # output np array (time,lat,lon)

lakeheat_pi = np.nanmean(lakeheat_ensmean[0:30,:,:],axis=0)
lakeheat_pres = np.nanmean(lakeheat_ensmean[-10:-1,:,:],axis=0)

lakeheat_anom_spmean = lakeheat_pres-lakeheat_pi
#np.nanmean(lakeheat_anom[years_analysis.index(2006):-1,:,:], axis=0)


# river heat: 
riverheat_anom_spmean = np.load(outdir+'riverheat/riverheat_anom_spmean.npy') 


#%%# make dictionaries with lake region properties
# extends need to be 0.25 ending

region_LGL = {
    'extent'          : [-92.5,41,-75.5,49.5], # original extent 
    'calc_extent'     : [-92.75,41.25,-75.75,49.75], # original extent [-92.5,41,-75.5,49.5]
    'continent_extent': [-128.7,-61.4,6.5   ,62.2 ],     # continent_extent for inset
    'ax_location'     : [0.705 , 0.61,0.25  , 0.2 ],
    'name'            : 'LaurentianGreatLakes',          # replace by 'LaurentianGreatLakes' to define shapefile etc 
    'name_str'        : 'Laurentian Great Lakes', 
    'levels'          :  np.arange(0,10e17,1e17),#np.arange(0,6.6e20,0.6e20),
    'fig_size'         : (13,8),
    'cb_orientation'  : 'horizontal'
}

# Laurentian Great Lakes
# extent_LGL = [-92.5,41,-75.5,49.5]
# extent_LGL = [-92.75,41.25,-75.75,49.75] # for calculation
# continent_extent_NA = [-128.7,-61.4,6.5,62.2] # continent_extent for inset
# ax_location_NA = [0.705, 0.61, 0.25, 0.2]
# name_LGL   = 'LaurentianGreatLakes'
# name_LGL= 'great_lakes_only'
# name_str_LGL = 'Laurentian Great Lakes'
# path_LGL   = lakes_path+name_LGL+'.shp'
# levels_LGL = np.arange(0,6.6e20,0.6e20)
# figsize_LGL = (13,8)
# cb_orientation_LGL = 'horizontal'

region_AGL = {
    'extent'          : [27.5,-9,36,2.5], # original extent [27.5,-9,36,2.5]
    'calc_extent'     : [27.75,-9.25,36.25,2.75],
    'continent_extent': [-18.5,51,-34.5,37] ,     # continent_extent for inset
    'ax_location'     : [0.640, 0.116, 0.15, 0.2],
    'name'            : 'AfricanGreatLakes',          # replace by 'LaurentianGreatLakes' to define shapefile etc 
    'name_str'        : 'African Great Lakes', 
    'levels'          : np.arange(0,32.2e17,2e17),
    'fig_size'         : (8,8),
    'cb_orientation'  : 'vertical'
}

# Great European Lake region
region_GEL = {
    'extent'          : [25,58,37,64], # original extent [27.5,-9,36,2.5]
    'calc_extent'     : [24.75,59.25,36.75,64.25],
    'continent_extent': [2.5,53,44,72],     # continent_extent for inset
    'ax_location'     : [0.75, 0.157, 0.15, 0.2],
    'name'            : 'GreatEuropeanLakes',       
    'name_str'        : 'Great European Lakes', 
    'levels'          : np.arange(0,2.2e17,0.2e17),
    'fig_size'         : (13,8),
    'cb_orientation'  : 'horizontal'
}

#%%

#%%
# Amazon region
region_AM = {
    'extent'          : [-78,-10,-48,3.5], # original extent [27.5,-9,36,2.5]
    'calc_extent'     : [-78.25,-10.25,-48.25,3.75],
    'continent_extent': [-84,-33,-55,13],     # continent_extent for inset
    'ax_location'     : [0.6545, 0.22, 0.4, 0.2],
    'name'            : 'Amazon',       
    'name_str'        : 'Amazon river basin', 
    'levels'          : np.arange(0,8.5e17,0.5e17),
    'fig_size'         : (13,8),
    'cb_orientation'  : 'horizontal'
}


#%% actual figure creation

plot_region_hc_map(lakeheat_anom_spmean, region_LGL, lakes_path, indir_lakedata)

plot_region_hc_map(lakeheat_anom_spmean, region_AGL, lakes_path, indir_lakedata)

plot_region_hc_map(lakeheat_anom_spmean, region_GEL, lakes_path, indir_lakedata)

plot_region_hc_rivers_map(riverheat_anom_spmean, region_AM, indir_lakedata)

# plot global
plot_global_hc_map('global',lakeheat_anom_spmean, lakes_path, indir_lakedata)


#%% Functions for time series plotting


def calc_region_hc_ts(lakeheat, lakes_path, region_props, indir_lakedata, flag_ref, years_analysis):
    """ Calculate the timeseries of the regions heat content, weighted by lake pct of shapefile
    input: lakeheat dictionary """
 
    extent            = region_props['calc_extent']
    name              = region_props['name']

    # initiate
    resolution=0.5
    path_lakes   = lakes_path+name+'.shp'
    outdir_lakepct= lakes_path+'pct_lake/'
    outfilename  = name+'_lake_pct'
    # do calculation

    lake_pct_region = calc_areafrac_shp2rst_region(path_lakes,outdir_lakepct,outfilename,resolution,extent)
    # extract region lake heat from dictionary and apply weights 
    lakeheat_wgt_region = extract_region(indir_lakedata,lakeheat,extent)
    
    # calculate anomaly for extracted lake region
    lakeheat_wgt_anom =  calc_anomalies(lakeheat_wgt_region, flag_ref,years_analysis)

    return lakeheat_wgt_anom



# -----------------------------------------------------
#plot the figure  timeseries of heat uptake by individual lake

def plot_region_hc_ts(ax1,flag_uncertainty,region_props,lakeheat_wgt_anom, label, colors, years_analysis):

    # get region specific properties from dictionary
    name_str          = region_props['name_str']

    lakeheat_wgt_region_ensmean_ts  = moving_average(ensmean_ts(lakeheat_wgt_anom))
    lakeheat_wgt_region_ensmin_ts   = moving_average(ensmin_ts(lakeheat_wgt_anom))
    lakeheat_wgt_region_ensmax_ts   = moving_average(ensmax_ts(lakeheat_wgt_anom))
    lakeheat_wgt_region_std_ts      = moving_average(ens_std_ts(lakeheat_wgt_anom))

    x_values = moving_average(np.asarray(years_analysis))

    # subplot 1: natural lakes heat uptake
    line_zero = ax1.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
    line1, = ax1.plot(x_values,lakeheat_wgt_region_ensmean_ts, color=colors[0])

    # uncertainty based on choice
    if flag_uncertainty == 'envelope':
        # full envelope
        area2 = ax1.fill_between(x_values,lakeheat_wgt_region_ensmin_ts,lakeheat_wgt_region_ensmax_ts, color=colors[1],alpha=0.5)
    elif flag_uncertainty =='2std':
    # 2x std error
        under_2std = lakeheat_wgt_region_ensmean_ts - 2*lakeheat_wgt_region_std_ts
        upper_2std = lakeheat_wgt_region_ensmean_ts + 2*lakeheat_wgt_region_std_ts
        area2 = ax1.fill_between(x_values,under_2std,upper_2std, color=colors[1],alpha=0.5)

    ax1.set_xlim(x_values[0],x_values[-1])
    ax1.set_xticks(ticks= np.array([1902,1920,1940,1960,1980,2000,2014]))
    ax1.set_xticklabels([1900,1920,1940,1960,1980,2000,2015] )
    #ax1.set_ylim(-0.4e20,1e20)
    ax1.set_ylabel('Energy [J]')
    ax1.set_title(name_str, loc='right')
    ax1.text(0.03, 0.92, label, transform=ax1.transAxes, fontsize=14)


#%%
# do the plotting

mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
mpl.rc('axes',titlesize=14)
mpl.rc('axes',labelsize=12)


flag_uncertainty = '2std' # or '2std' or 'envelope'

# Laurentian Great Lakes
f,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(6,14))

# calculate lake heat anomaly for shapefiles of region, weighted with lake pct
lakeheat_wgt_anom = calc_region_hc_ts(lakeheat_climate, lakes_path, region_LGL, indir_lakedata, flag_ref,years_analysis)
label = '(b)'
colors = ('coral','sandybrown')
plot_region_hc_ts(ax1,flag_uncertainty,region_LGL,lakeheat_wgt_anom, label, colors, years_analysis)

# African Great Lkaes
lakeheat_wgt_anom = calc_region_hc_ts(lakeheat_climate, lakes_path, region_AGL, indir_lakedata, flag_ref,years_analysis)
label = '(d)'
colors = ('coral','sandybrown')
plot_region_hc_ts(ax2,flag_uncertainty,region_AGL,lakeheat_wgt_anom, label, colors, years_analysis)


lakeheat_wgt_anom = calc_region_hc_ts(lakeheat_climate, lakes_path, region_GEL, indir_lakedata, flag_ref,years_analysis)
label = '(f)'
colors = ('coral','sandybrown')
plot_region_hc_ts(ax3,flag_uncertainty,region_GEL,lakeheat_wgt_anom, label, colors, years_analysis)

riverheat_region_anom = np.load(outdir+'riverheat/riverheat_amazon_anom.npy').item() 
label = '(h)'
colors = ('coral','sandybrown')
plot_region_hc_ts(ax4,flag_uncertainty,region_AM,riverheat_region_anom, label, colors, years_analysis)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
plotdir='/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/'
plt.savefig(plotdir+'regions_hc_ts.png')




#%%
