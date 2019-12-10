"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : October 2019

Scripts for isimip postprocessing with CDO

notes: 
reservoir construction enabled only works until 2000: necessary to update lake_pct input data to newly GRanD files
' climate '

"""
#%%
import os 

# settings for windows or linux machine (for paths)
if os.name == 'nt': # working on windows
    sys.path.append(r'E:/scripts/python/utils')
    sys.path.append(r'E:/scripts/python/calc_lakeheat_isimip/lakeheat_isimip')
    basepath = 'E:/'
else:
    basepath = '/home/inne/documents/phd/'
    sys.path.append(r'/home/inne/documents/phd/scripts/python/calc_lakeheat_isimip/lakeheat_isimip')

    from cdo import Cdo
    cdo = Cdo()

import xarray as xr
import numpy as np
import geopandas as gpd
from calc_grid_area    import calc_grid_area
from scipy.interpolate import griddata

# import my own functions
from dict_functions import *

# -----------------------------------------------------------
# Flags

flag_preprocess = False

flag_interpolate_CLM45 = False # make interpolation of CLM temperature fields. (takes time)


# flag to set which scenario is used for heat calculation
flag_scenario = 'climate'  # 'climate'    : only climate change (lake cover constant at 2005 level)
                              # 'reservoirs' : only reservoir construction (temperature constant at 1900 level)
                              # 'both'       : reservoir construction and climate

# Reference to which period/year anomalies are calculated
flag_ref = 'pre-industrial' # 'pre-industrial': first 30 years (1900-1929 for start_year =1900)
                             # 1971 or any integer: year as a reference 

# whether or not to save calculated lake heat 
flag_savelakeheat = True

# -----------------------------------------------------------
# initialise



indir  = basepath + 'data/ISIMIP/OutputData/lakes_global/'
outdir = basepath + 'data/processed/isimip_lakeheat/'
plotdir= basepath + 'data/processed/isimip_lakeheat/plots/'

indir_lakedata   = basepath + 'data/isimip_laketemp/'

models      = ['CLM45']#,'ALBM','SIMSTRAT-UoG','VIC-LAKE','LAKE']
forcings    = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']
experiments = ['historical','future']

# experiment used for future simulations (needed to differentiate between filenames)
future_experiment = 'rcp60' 

variables   = ['watertemp']


start_year = 1900
end_year = 2017

years_isimip           = range(1861,2099,1)
years_grand            = range(1900,2010,1)
years_analysis         = range(start_year,end_year,1)
years_pi               = range(1861,1891,1)

# define constants
resolution = 0.5 # degrees

# constants values to check
cp_liq = 4.188e3   #[J/kg K] heat capacity liquid water
cp_ice = 2.11727e3 #[J/kg K] heat capacity ice
cp_salt= 3.993e3  #[J/kg K] heat capacity salt ocean water (not used)

rho_liq = 1000     #[kg/m³] density liquid water
rho_ice = 0.917e3  #[kg/m³] density ice

# %%
# -------------------------------
# pre-process lake surface temperatures from raw ISIMIP data

if flag_preprocess: 
    for model in models:
        
        for forcing in forcings:

            for variable in variables:

                for experiment in experiments:
                # differentiate for future experiments filenames
                    if experiment == 'future': 
                        experiment_fn = future_experiment 
                        period = '2006_2099'
                    elif experiment == 'historical': 
                        experiment_fn = experiment
                        period = '1861_2005'  

                    path = indir+model+'/'+forcing+'/'+experiment+'/'
                    infile = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'monthly'+'_'+period+'.nc4'
                    outfile_assembled = model.lower()+'_'+forcing+'_historical_'+future_experiment+'_'+variable+'_'+'1861_2099'+'_'+'annual'+'.nc4'

                    # if simulation is available 
                    if os.path.isfile(path+infile): 

                        # make output directory per model if not done yet
                        outdir_model = outdir+variable+'/'+model+'/'
                        if not os.path.isdir(outdir_model):
                            os.system('mkdir '+outdir_model)
                    
                        # calculate annual means per model for each forcing (if not done so)
                        outfile_annual = model.lower()+'_'+forcing+'_'+experiment_fn+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                        if (not os.path.isfile(outdir_model+outfile_assembled)):
                            print('calculating annual means of '+infile)
                            cdo.yearmean(input=path+infile,output=outdir_model+outfile_annual)


                # assemble historical and future simulation
                infile_hist = model.lower() +'_'+forcing+'_historical_'           +variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                infile_fut  =  model.lower()+'_'+forcing+'_'+future_experiment+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                
                if (not os.path.isfile(outdir_model+outfile_assembled)):
                    print('concatenating historical and '+future_experiment+' simulations of '+model+' '+forcing)
                    cdo.mergetime(input=outdir_model+infile_hist+' '+outdir_model+infile_fut,output=outdir_model+outfile_assembled )
                    # clean up 
                    os.system('rm '+outdir_model+infile_hist +' '+outdir_model+infile_fut)

                # calculate ensemble mean forcing for each model (if not done so)
                outfile_ensmean = model.lower()+'_historical_'+future_experiment+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'_'+'ensmean'+'.nc4'
                if (not os.path.isfile(outdir_model+outfile_ensmean)):
                    print('calculating ensemble means of '+model)
                    cdo.ensmean(input=outdir_model+outfile_assembled,output=outdir_model+outfile_ensmean)
                else:
                    print(model+' '+forcing+' is already preprocessed.')


#%% 
# ----------------------------------------------------
# Lake volumes 


# GLDB lake depths (GLDB lake depths, hydrolakes lake area)
lakedepth_path        = indir_lakedata + 'dlake_1km_ll_remapped_0.5x0.5.nc'
lakepct1900_path      = indir_lakedata + 'pct_lake_1900_0.5x0.5.nc'
lakepct2000_path      = indir_lakedata + 'pct_lake_2000_0.5x0.5.nc'
lakepct_path          = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1900-2000_c20190826.nc'

# load variables
gldb_lakedepth      = xr.open_dataset(lakedepth_path)
hydrolakes_lakepct  = xr.open_dataset(lakepct_path)

lake_depth         = gldb_lakedepth.dl.values[0,:,:]
lake_pct           = hydrolakes_lakepct.PCT_LAKE.values/100 # to have then in fraction
landmask           = hydrolakes_lakepct.LANDMASK.values

lon = hydrolakes_lakepct.PCT_LAKE.lon
lat = hydrolakes_lakepct.PCT_LAKE.lat

# this should be removed when updating new reservoir file
end_year_res = 2000

# take analysis years of lake_pct
lake_pct  = lake_pct[years_grand.index(start_year):years_grand.index(end_year_res), :, :]

# this should be removed when updating new reservoir file
# extend lake_pct data set with values further than 2000: 
lake_const=lake_pct[years_grand.index(end_year_res-1),:,:]
lake_const = lake_const[np.newaxis,:,:]
for ind,year in enumerate(np.arange(end_year_res,end_year)):
    lake_pct= np.append(lake_pct,lake_const,axis=0)



# calculate lake area per grid cell (m²)
grid_area      = calc_grid_area(resolution)
lake_area      = lake_pct * grid_area  

# calculate lake volume (assumption of bathtub -> other assumptions?)
lake_volume_tot = lake_area * lake_depth  # m³

# Define different lake layer thicknesses. 
layer_thickness_clm = np.array([0.1, 1, 2, 3, 4, 5, 7, 7, 10.45, 10.45]) # m
layer_thickness_rel_clm = layer_thickness_clm/np.sum(layer_thickness_clm)


# calculate volumes per layer for different timesteps (area)
# based on relative layer depths and depth of GLDB
def calc_volume_per_layer(layer_thickness_rel):

    # take lake area constant at 1900 level.
    if flag_scenario == 'climate' : # in this scenario, volume per layer has only 3 dimensions


        lake_depth_expanded = np.expand_dims(lake_depth,axis=0)
        layer_thickness_rel = np.expand_dims(layer_thickness_rel,axis=1)
        layer_thickness_rel = np.expand_dims(layer_thickness_rel,axis=2)

        depth_per_layer = layer_thickness_rel * lake_depth_expanded
        lake_area_endyear = lake_area[0,:,:]
        volume_per_layer = depth_per_layer * lake_area_endyear

    else:
        lake_depth_expanded = np.expand_dims(lake_depth,axis=0)
        layer_thickness_rel = np.expand_dims(layer_thickness_rel,axis=1)
        layer_thickness_rel = np.expand_dims(layer_thickness_rel,axis=2)

        depth_per_layer = layer_thickness_rel * lake_depth_expanded
        depth_per_layer = np.expand_dims(depth_per_layer,axis=0)
        lake_area_expanded = np.expand_dims(lake_area,axis=1)
        volume_per_layer = depth_per_layer * lake_area_expanded

    return volume_per_layer

volume_per_layer_clm = calc_volume_per_layer(layer_thickness_rel_clm)




#%% Interpolate CLM45 temperatures and store them again as a netcdf

# see interp_clmtemps.py script        
if flag_interpolate_CLM45: 

    # interpolation function
    def interpolate_nn(a,mask):
        """ Function to interpolate a 3D numpy array with nearest neighbour method """
        # get mask of nan values (where both mask is True and a has nan)
        mask_nan = np.where(np.isnan(a) & mask,np.nan,0)

        x, y, z = np.indices(a.shape)
        interp = np.array(a)

        interp[np.isnan(mask_nan)] = griddata(
            (x[~np.isnan(a)], y[~np.isnan(a)], z[~np.isnan(a)]), # points we know
            a[~np.isnan(a)], # values we know
            (x[np.isnan(mask_nan)], y[np.isnan(mask_nan)], z[np.isnan(mask_nan)]) # values we want to know
            ,method='nearest')     
        interp = np.where(interp==0,np.nan,interp)  
 
        return interp

    # make mask based on where there are lakes
    lakepct_path          = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1900-2000_c20190826.nc'
    hydrolakes_lakepct    = xr.open_dataset(lakepct_path)
    lake_pct              = hydrolakes_lakepct.PCT_LAKE.values # to have then in fraction
    mask = np.where(lake_pct[-1,:,:]>0,True,False)


    for forcing in forcings:
        print('Interpolating '+forcing)
        
        outdir_clm45 = outdir+'/watertemp/CLM45/'
        outfile_clm45= 'clm45_'+forcing+'_'+'historical_'+future_experiment+'_watertemp_'+'1861_2099'+'_'+'annual'+'.nc4'
        outfile_clm45_interp= 'clm45_'+forcing+'_'+'historical_'+future_experiment+'_watertemp_interp_'+'1861_2099'+'_'+'annual'+'.nc4'


        if not os.path.isfile(outdir_clm45+outfile_clm45_interp): 

            # load the clm lake temperature 
            ds_laketemp = xr.open_dataset(outdir_clm45+outfile_clm45,decode_times=False)
            laketemp = ds_laketemp.watertemp.values

            # do interpolation

            for i in range(0,laketemp.shape[0]):
                #print('timestep '+str(i))
                # apply mask 
                # interpolate
                laketemp[i,:,:,:] = interpolate_nn(laketemp[i,:,:,:],mask)
            
            # save interpolation back again in netcdf

            lat    = ds_laketemp['lat'].values
            lon    = ds_laketemp['lon'].values
            levlak = ds_laketemp['levlak'].values
            time   = ds_laketemp['time'].values

            laketemp_interp =  xr.DataArray(laketemp, coords={'lat': lat, 'lon': lon, 
                                    'levlak': levlak, 'time':time},
                dims=['time','levlak', 'lat', 'lon'])

            ds_laketemp['watertemp']=laketemp_interp
            ds_laketemp.to_netcdf(outdir_clm45 + outfile_clm45_interp)

            # clean up
            del ds_laketemp, laketemp_interp, laketemp




#%% 
# ----------------------------------------------------
# HEAT CONTENT CALCULATION
# ----------------------------------------------------
# 
# saves output in dictionary per model, per forcing in lake_heat

# define experiment
experiment= 'historical_'+future_experiment # can also be set to 'historical' or 'rcp60', but start_year will in this case have to be within year range

# CLM45 (loop can be expanded) overwritten with other dictionary
# for the different models
lakeheat = {}

for model in models:
    lakeheat_model={} # sub directory for each model

    for forcing in forcings:

        # define directory and filename
        variable = 'watertemp'                        
        outdir_model = outdir+variable+'/'+model+'/'

        if model == 'CLM45': # load interpolated temperatures
            outfile_annual = model.lower()+'_'+forcing+'_'+experiment+'_'+variable+'_interp_1861_2099'+'_'+'annual'+'.nc4'
        else: 
            outfile_annual = model.lower()+'_'+forcing+'_'+experiment+'_'+variable+'_1861_2099'+'_'+'annual'+'.nc4'

        # if simulation is available 
        if os.path.isfile(outdir_model+outfile_annual): 
            print('Calculating lake heat of '+ model + ' ' + forcing)
            ds_laketemp = xr.open_dataset(outdir_model+outfile_annual,decode_times=False)
            laketemp = ds_laketemp.watertemp.values
    
            if flag_scenario == 'reservoirs': 
                # use lake temperature from first year of analysis
                laketemp = laketemp[years_isimip.index(start_year),:,:,:]
            else: 
                # extract years of analysis
                laketemp = laketemp[years_isimip.index(start_year):years_isimip.index(end_year),:,:,:]

            lakeheat_layered =  rho_liq  * volume_per_layer_clm * cp_liq * laketemp
        
            # sum up for total layer (less memory)
            lakeheat_forcing = lakeheat_layered.sum(axis=1)

            # clean up
            del laketemp, ds_laketemp

            # save lakeheat in directory structure per forcing
        if not lakeheat_model:
            lakeheat_model = {forcing:lakeheat_forcing}
        else: 
            lakeheat_model.update({forcing:lakeheat_forcing})

    # save lakeheat of forcings in directory structure per model
    if not lakeheat:
        lakeheat = {model:lakeheat_model}
    else:
        lakeheat.update({model:lakeheat_model})    

# save calculated lake heat (this needs to be cleaned up before continuing working on code.)

# Save according to scenario flag
if flag_savelakeheat:
    lakeheat_filename = 'lakeheat_'+flag_scenario+'.npy'
    np.save(outdir+lakeheat_filename, lakeheat) 




# %%
# ------------------------------------------------------------------------
# DATA AGGREGATION
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Absolute heat content calculations

# calculate global timeseries of total lake heat of ensemble members

lakeheat_ts = timeseries(lakeheat)
lakeheat_ens = ensmean(lakeheat)
lakeheat_ensmean_ts = ensmean_ts(lakeheat)
lakeheat_ensmin_ts = ensmin_ts(lakeheat)
lakeheat_ensmax_ts = ensmax_ts(lakeheat)

lakeheat_ens_spmean = ens_spmean(lakeheat)

# laketemp_averaged = np.average(laketemp, axis=1, weights = layer_thickness_clm_rel)

#%%
# ---------------------------------------------------------------------------
# Anomaly heat content calculations

# calculate anomalies

def calc_anomalies(lakeheat):
    lakeheat_anom = {}

    for model in lakeheat:
        lakeheat_anom_model = {}

        for forcing in lakeheat[model]: 

            # determine reference
            if flag_ref == 'pre-industrial':   # period of first 30 years of simulation (1900-1929)
                lakeheat_ref_forcing = np.nanmean(lakeheat[model][forcing][0:30,:,:])
            elif isinstance(flag_ref,int):
                lakeheat_ref_forcing = lakeheat[model][forcing][years_analysis.index(flag_ref),:,:]

            # subtract reference to calculate anomaly 
            lakeheat_anom_model[forcing] = lakeheat[model][forcing] - lakeheat_ref_forcing
        lakeheat_anom[model] = lakeheat_anom_model
        return lakeheat_anom

lakeheat_anom = calc_anomalies(lakeheat)
# -----------------------------------------------------------------------------
# Aggregate - calculate timeseries and averages

lakeheat_anom_ts = timeseries(lakeheat_anom)

lakeheat_anom_ens = ensmean(lakeheat_anom)

lakeheat_anom_ensmean_ts = moving_average(ensmean_ts(lakeheat_anom))
lakeheat_anom_ensmin_ts  = moving_average(ensmin_ts(lakeheat_anom))
lakeheat_anom_ensmax_ts  = moving_average(ensmax_ts(lakeheat_anom))

lakeheat_anom_ens_spmean = ens_spmean(lakeheat_anom)

lakeheat_anom_spcumsum = ensmean_spcumsum(lakeheat_anom)

lake_anom = [lakeheat_anom_ts['CLM45']['gfdl-esm2m'][-1], lakeheat_anom_ts['CLM45']['ipsl-cm5a-lr'][-1], lakeheat_anom_ts['CLM45']['hadgem2-es'][-1],lakeheat_anom_ts['CLM45']['miroc5'][-1] ]
np.std(lake_anom)
lakeheat_anom_ts['CLM45']['ipsl-cm5a-lr'][-1]



# calculate trend. 

#%%
# get values

# total lake heat added over the 20th century (based on m):
start_year_values = 1900
total_heatcontent_increase = lakeheat_ensmean_ts['CLM45'][-1]-lakeheat_ensmean_ts['CLM45'][years_analysis.index(start_year_values)]
print('Increase in natural lake heat content from '+str(start_year_values)+' to 2017:  '+str(total_heatcontent_increase)+' J')
total_heatcontent_trend = total_heatcontent_increase/len(range(start_year_values,end_year))
print('Trend    in natural lake heat content from '+str(start_year_values)+' to 2017: '+str(total_heatcontent_trend) +' J/yr')
 
start_year_values = 1971
total_heatcontent_increase = lakeheat_ensmean_ts['CLM45'][-1]-lakeheat_ensmean_ts['CLM45'][years_analysis.index(start_year_values)]
print('Increase in lake heat content from '+str(start_year_values)+' to 2017:  '+str(total_heatcontent_increase)+' J')
total_heatcontent_trend = total_heatcontent_increase/len(range(start_year_values,end_year))
print('Trend    in lake heat content from '+str(start_year_values)+' to 2017: '+str(total_heatcontent_trend) +' J/yr')

#%%
# plotting

# settings


import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy as ctp
import mplotutils as mpu 

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


#%%
# ---------------------------------------------------------------------------------------
# lineplots of one model anomaly for all forcings 

for model in models:
        
    f,ax = plt.subplots(2,2, figsize=(8,7))
    x_values = np.asarray(years_analysis)

    ax = ax.ravel()

    for nplot,forcing in enumerate(forcings):

        line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
        line1 = ax[nplot].plot(x_values,lakeheat_anom_ts[model][forcing], color='coral')
        ax[nplot].set_xlim(x_values[0],x_values[-1])
        #ax[nplot].set_ylim(-6e20,8e20)
        ax[nplot].set_ylabel('Energy [J]')
        ax[nplot].set_title(forcing, pad=15)

    f.suptitle(model+'Lake heat anomalies (reference 1900-1929)', fontsize=16)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plotdir+model+'heat_acc_per_forcing'+'.png',dpi=300)


#%%
# ---------------------------------------------------------------------------------------
# lineplot of one model anomaly- all forcings 5-year moving average
f,ax = plt.subplots(figsize=(7,4))

x_values = moving_average(np.asarray(years_analysis))

line_zero = ax.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

line2 = ax.plot(x_values,lakeheat_anom_ensmax_ts[model], color='sandybrown')
line3 = ax.plot(x_values,lakeheat_anom_ensmin_ts[model], color='sandybrown')

area1 = ax.fill_between(x_values,lakeheat_anom_ensmin_ts[model],lakeheat_anom_ensmax_ts[model], color='sandybrown')
line1 = ax.plot(x_values,lakeheat_anom_ensmean_ts[model], color='coral')

# zeroline
# line2 = ax.plot(np.asarray(years_analysis), color='steelblue')
# area2 = ax.fill_between(np.asarray(years_analysis),hc_anom_climate_1900,hc_anom_resexp_1900+hc_anom_climate_1900, color='skyblue')


ax.set_xlim(x_values[0],x_values[-1])
#ax.set_ylim(-6e20,8e20)
#ax.grid(color='lightgrey')
ax.set_ylabel('Energy [J]')
ax.set_title('Lake heat content anomalies (reference 1900-1929, 5-year moving average)', pad=15)
#ax.legend((area1,area2),['climate change', 'reservoir expansion'],frameon=False,loc='upper left')

plt.savefig(plotdir+'heat_anom1900_5yav'+'.png',dpi=300)


#%%
# ---------------------------------------------------------------------------------------
# Accumulation lineplot of one model accumulation - all forcings 5-year moving average

f,ax = plt.subplots(figsize=(7,4))

x_values = moving_average(np.asarray(years_analysis))


line_zero = ax.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')


area1 = ax.fill_between(x_values,np.cumsum(lakeheat_anom_ensmin_ts[model]),np.cumsum(lakeheat_anom_ensmax_ts[model]), color='sandybrown')
line1 = ax.plot(x_values,np.cumsum(lakeheat_anom_ensmean_ts[model]), color='coral')


line2 = ax.plot(x_values, np.cumsum(lakeheat_anom_ensmax_ts[model]), color='sandybrown')
line3 = ax.plot(x_values,np.cumsum(lakeheat_anom_ensmin_ts[model]), color='sandybrown')



# zeroline
# line2 = ax.plot(np.asarray(years_analysis), color='steelblue')
# area2 = ax.fill_between(np.asarray(years_analysis),hc_anom_climate_1900,hc_anom_resexp_1900+hc_anom_climate_1900, color='skyblue')


ax.set_xlim(x_values[0],x_values[-1])
#ax.set_ylim(-4e19,10e19)
#ax.grid(color='lightgrey')
ax.set_ylabel('Energy [J]')
ax.set_title('Natural lake heat accumulation, 5-year moving mean (reference 1900-1929)', pad=15)
#ax.legend((area1,area2),['climate change', 'reservoir expansion'],frameon=False,loc='upper left')

plt.savefig(plotdir+'heat_acc_1900_5yav'+'.png',dpi=300)


#%%
# plot MAP of cumulative heat content. 


# user settings
var = lakeheat_anom_ens_spmean['CLM45']

clb_label='Joule'
title_str='Natural lake heat anomaly compared to 1900-1929 '+ model

cmap = 'YlOrBr'
levels = np.arange(-1*10**20,100*10**20,10*10**20)

# figure creation
fig, ax = plt.subplots(1,1,figsize=(13,8), subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(ctp.feature.OCEAN, color='gainsboro')
ax.coastlines(color="grey")

LON, LAT = mpu.infer_interval_breaks(lon,lat)
 #add the data to the map (more info on colormaps: https://matplotlib.org/users/colormaps.html)
cmap, norm = mpu.from_levels_and_cmap(levels, cmap, extend='max')

h = ax.pcolormesh(LON,LAT,var, cmap=cmap, norm=norm)

# set the extent of the cartopy geoAxes to "global"
ax.set_global()

# plot the colorbar
#cbar = mpu.colorbar(h, ax, extend='max', orientation='horizontal', pad = 0.05)
#cbar.ax.set_xlabel(clb_label, size=16)
ax.set_title(title_str, pad=10)

# save the figure, adjusting the resolution 
plt.savefig(plotdir+'map_heat_acc.png')

#%%
# Aggregate several scenarios

# load lake heat calculated for several scenarios
# Load climate only (natural lakes)

lakeheat_climate = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()
lakeheat_climate_anom = calc_anomalies(lakeheat_climate)
lakeheat_climate_anom_ensmean_ts = moving_average(ensmean_ts(lakeheat_climate_anom))
lakeheat_climate_anom_ensmin_ts  = moving_average(ensmin_ts(lakeheat_climate_anom))
lakeheat_climate_anom_ensmax_ts  = moving_average(ensmax_ts(lakeheat_climate_anom))

lakeheat_res = np.load(outdir+'lakeheat_reservoirs.npy',allow_pickle='TRUE').item()
lakeheat_res_anom = calc_anomalies(lakeheat_res)
lakeheat_res_anom_ensmean_ts = moving_average(ensmean_ts(lakeheat_res_anom))
lakeheat_res_anom_ensmin_ts  = moving_average(ensmin_ts(lakeheat_res_anom))
lakeheat_res_anom_ensmax_ts  = moving_average(ensmax_ts(lakeheat_res_anom))

lakeheat_both = np.load(outdir+'lakeheat_both.npy',allow_pickle='TRUE').item()
lakeheat_both_anom = calc_anomalies(lakeheat_both)
lakeheat_both_anom_ensmean_ts = moving_average(ensmean_ts(lakeheat_both_anom))
lakeheat_both_anom_ensmin_ts  = moving_average(ensmin_ts(lakeheat_both_anom))
lakeheat_both_anom_ensmax_ts  = moving_average(ensmax_ts(lakeheat_both_anom))


lakeheat_onlyresclimate_anom_ensmean_ts = lakeheat_both_anom_ensmean_ts[model] - (lakeheat_res_anom_ensmean_ts[model]+lakeheat_climate_anom_ensmean_ts[model])

# get some values out
start_year_values = 1900
# only reservoir Climate change
total_heatcontent_increase = lakeheat_onlyresclimate_anom_ensmean_ts[-1]-lakeheat_onlyresclimate_anom_ensmean_ts[years_analysis.index(start_year_values)]
print('Increase in lake heat content '+str(start_year_values)+'-2017, reservoir climate change:  ')
print(str(total_heatcontent_increase)+' J')
total_heatcontent_trend = total_heatcontent_increase/len(range(start_year_values,end_year))
print('Trend    in lake heat content '+str(start_year_values)+'-2017, reservoir climate change:: ')
print(str(total_heatcontent_trend) +' J/yr')

# only natural lakes
total_heatcontent_increase_clim = lakeheat_climate_anom_ensmean_ts['CLM45'][-1]-lakeheat_climate_anom_ensmean_ts['CLM45'][years_analysis.index(start_year_values)]
print('Increase in lake heat content '+str(start_year_values)+'-2017, natural lakes:  ')
print(str(total_heatcontent_increase_clim)+' J')
total_heatcontent_trend = total_heatcontent_increase_clim/len(range(start_year_values,end_year))
print('Trend    in lake heat content '+str(start_year_values)+'-2017, natural lakes: ')
print(str(total_heatcontent_trend) +' J/yr')

# only reservoir expansion
total_heatcontent_increase_res = lakeheat_res_anom_ensmean_ts['CLM45'][-1]-lakeheat_res_anom_ensmean_ts['CLM45'][years_analysis.index(start_year_values)]
print('Increase in lake heat content '+str(start_year_values)+'-2017, reservoir expansion:  ')
print(str(total_heatcontent_increase_res)+' J')
total_heatcontent_trend = total_heatcontent_increase_res/len(range(start_year_values,end_year))
print('Trend    in lake heat content '+str(start_year_values)+'-2017, reservoir expansion: ')
print(str(total_heatcontent_trend) +' J/yr')

# both
total_heatcontent_increase_both = lakeheat_both_anom_ensmean_ts['CLM45'][-1]-lakeheat_both_anom_ensmean_ts['CLM45'][years_analysis.index(start_year_values)]
print('Increase in lake heat content '+str(start_year_values)+'-2017, both reservoir expansion and climate change:  ')
print(str(total_heatcontent_increase_both)+' J')
total_heatcontent_trend = total_heatcontent_increase_both/len(range(start_year_values,end_year))
print('Trend    in lake heat content '+str(start_year_values)+'-2017, both reservoir expansion and climate change: ')
print(str(total_heatcontent_trend) +' J/yr')

# relative values
# rel_res = total_heatcontent_increase_res/total_heatcontent_increase_both *100
# rel_clim =  total_heatcontent_increase_clim/total_heatcontent_increase_both *100
print('Portion of reservoir expansion in total heating: '+ str(rel_res) +' %')
print('Portion of climate change in total heating: '+ str(rel_clim) +' %')

#%% 
# Do the actual plotting

model='CLM45'
# plot them on one graph. 
f,ax = plt.subplots(figsize=(7,4))
x_values = moving_average(np.asarray(years_analysis))

line_zero = ax.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

#area1 = ax.fill_between(x_values,np.cumsum(lakeheat_anom_ensmin_ts[model]),np.cumsum(lakeheat_anom_ensmax_ts[model]), color='sandybrown')
line1, = ax.plot(x_values,lakeheat_climate_anom_ensmean_ts[model], color='coral')
area2 = ax.fill_between(x_values,lakeheat_climate_anom_ensmin_ts[model],lakeheat_climate_anom_ensmax_ts[model], color='sandybrown',alpha=0.5)


line2, = ax.plot(x_values,lakeheat_res_anom_ensmean_ts[model], color='yellowgreen')
area2 = ax.fill_between(x_values,lakeheat_res_anom_ensmin_ts[model],lakeheat_res_anom_ensmax_ts[model], color='yellowgreen',alpha=0.5)

#line2, = ax.plot(x_values,lakeheat_climate_anom_ensmean_ts[model]+lakeheat_res_anom_ensmean_ts[model], color='yellowgreen')
line3, = ax.plot(x_values,lakeheat_both_anom_ensmean_ts[model], color='steelblue')
area3 = ax.fill_between(x_values,lakeheat_both_anom_ensmin_ts[model],lakeheat_both_anom_ensmax_ts[model], color='lightskyblue',alpha=0.5)

# zeroline
# line2 = ax.plot(np.asarray(years_analysis), color='steelblue')
#area1 = ax.fill_between(np.asarray(years_analysis),hc_anom_climate_1900,hc_anom_resexp_1900+hc_anom_climate_1900, color='skyblue')


ax.set_xlim(x_values[0],x_values[-1])
#ax.set_ylim(-4e19,10e19)
#ax.grid(color='lightgrey')
ax.set_ylabel('Energy [J]')
ax.set_title('Lake heat anomalies, 5-year moving mean (reference 1900-1929)', pad=15)
ax.legend((line1,line2,line3),['climate change', 'reservoir expansion', 'both'],frameon=False,loc='upper left')

plt.savefig(plotdir+'res+clim+both_heat_acc_1900_5yav'+'.png',dpi=300)


#%% 
# Plot like IPCC figure

model='CLM45'
# plot them on one graph. 
f,ax = plt.subplots(figsize=(7,4))
x_values = moving_average(np.asarray(years_analysis))

line_zero = ax.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

line1, = ax.plot(x_values,lakeheat_climate_anom_ensmean_ts[model], color='coral')
line2, = ax.plot(x_values,lakeheat_climate_anom_ensmean_ts[model]+lakeheat_res_anom_ensmean_ts[model], color='steelblue')
area2 = ax.fill_between(x_values,lakeheat_climate_anom_ensmean_ts[model],lakeheat_climate_anom_ensmean_ts[model]+lakeheat_res_anom_ensmean_ts[model], color='skyblue')
area1 = ax.fill_between(x_values,lakeheat_climate_anom_ensmean_ts[model], color='sandybrown')

ax.set_xlim(x_values[0],x_values[-1])
#ax.set_ylim(-4e19,10e19)
#ax.grid(color='lightgrey')
ax.set_ylabel('Energy [J]')
ax.set_title('Lake heat anomalies, 5-year moving mean (reference 1900-1929)', pad=15)
ax.legend((area1,area2),['climate change', 'reservoir expansion'],frameon=False,loc='upper left')

plt.savefig(plotdir+'res+clim_IPCC_heat_acc_1900_5yav'+'.png',dpi=300)
#%%
# plot only reservoir + climate (only warming of human constructed reservoirs)
f,ax = plt.subplots(figsize=(7,4))
x_values = moving_average(np.asarray(years_analysis))

line_zero = ax.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

#area1 = ax.fill_between(x_values,np.cumsum(lakeheat_anom_ensmin_ts[model]),np.cumsum(lakeheat_anom_ensmax_ts[model]), color='sandybrown')
line1 = ax.plot(x_values,lakeheat_onlyresclimate_anom_ensmean_ts, color='coral')

# zeroline
# line2 = ax.plot(np.asarray(years_analysis), color='steelblue')
# area2 = ax.fill_between(np.asarray(years_analysis),hc_anom_climate_1900,hc_anom_resexp_1900+hc_anom_climate_1900, color='skyblue')


ax.set_xlim(x_values[0],x_values[-1])
#ax.set_ylim(-4e19,10e19)
#ax.grid(color='lightgrey')
ax.set_ylabel('Energy [J]')
ax.set_title('Constructed reservoir heat anomalies (reference 1900-1929)', pad=15)
#ax.legend((line1,line2,line3),['climate change', 'reservoir expansion', 'both'],frameon=False,loc='upper left')

plt.savefig(plotdir+'res_heat_acc_1900_5yav'+'.png',dpi=300)




#%%
# 
#  Check lake heat calculations for individual lakes
indir_shp = basepath + 'data/processed/lakes_shp/'


# extract lake heat based on coordinates

# Load shapefiles of lakes 
great_slave_lake    = gpd.read_file(indir_shp+'great_slave_lake.shp')
lake_nasser         = gpd.read_file(indir_shp+'lake_nasser.shp')
lake_valencia       = gpd.read_file(indir_shp+'lake_valencia.shp')
lake_aegeri         = gpd.read_file(indir_shp+'lake_aegeri.shp')
lake_aegeri['Lake_name'] = 'Aegeri'
lake_superior       = gpd.read_file(indir_shp+'lake_superior.shp')
lake_eaugalle       = gpd.read_file(indir_shp+'lake_eaugalle.shp')
lake_eaugalle['Lake_name'] = 'Eau Galle'
lake_michigan       = gpd.read_file(indir_shp+'lake_michigan.shp')
lake_huron          = gpd.read_file(indir_shp+'lake_huron.shp')
lake_ontario        = gpd.read_file(indir_shp+'lake_ontario.shp')
lake_erie           = gpd.read_file(indir_shp+'lake_erie.shp')


gpd_data = lake_aegeri
heat_field = lakeheat_ensmean_sp['CLM45']


def get_individual_lakeheat(gpd_data,heat_field):


    heat_lake = []
    # check whether lake is smaller then 1 grid cell (only 1 grid cell lake heat to use)
    # take grid cell corresponding to centroid. 
    if grid_cell_area > lake_area[0]: 
    
        # from here starting to write function to retrieve lake heat. 
        def getXY(pt):
            return (pt.x, pt.y)
        centroidseries = gpd_data['geometry'].centroid
        lon_centr,lat_centr = [list(t) for t in zip(*map(getXY, centroidseries))]

        id_lat = (np.abs(lat.values - lat_centr)).argmin()
        id_lon = (np.abs(lon.values - lon_centr)).argmin()

        heat_lake_gridcell = heat_field[id_lat,id_lon]
        lake_area = gpd_data['Lake_area'].to_numpy()*10**6 # in m²
        grid_cell_area = grid_area[id_lat,id_lon]
        
        # rescale heat to lake area
        heat_lake = heat_lake_gridcell/grid_cell_area * lake_area
        print(gpd_data['Lake_name'].values[0]+': '+str(round(heat_lake[0],2))+' J')
    else: 
        # work with mask? 
        print(gpd_data['Lake_name'].values[0]+' is too big ...')

    return heat_lake


heat_lake_aegeri      = get_individual_lakeheat(lake_aegeri     ,lakeheat_ensmean_sp['CLM45'])
heat_great_slave_lake = get_individual_lakeheat(great_slave_lake,lakeheat_ensmean_sp['CLM45'])
heat_lake_nasser      = get_individual_lakeheat(lake_nasser     ,lakeheat_ensmean_sp['CLM45'])
heat_lake_valencia    = get_individual_lakeheat(lake_valencia   ,lakeheat_ensmean_sp['CLM45']) 
heat_lake_superior    = get_individual_lakeheat(lake_superior   ,lakeheat_ensmean_sp['CLM45'])
heat_lake_michigan    = get_individual_lakeheat(lake_michigan   ,lakeheat_ensmean_sp['CLM45'])
heat_lake_ontario     = get_individual_lakeheat(lake_ontario    ,lakeheat_ensmean_sp['CLM45'])
heat_lake_erie        = get_individual_lakeheat(lake_erie       ,lakeheat_ensmean_sp['CLM45'])
heat_lake_huron       = get_individual_lakeheat(lake_huron      ,lakeheat_ensmean_sp['CLM45'])
heat_lake_eaugalle    = get_individual_lakeheat(lake_eaugalle   ,lakeheat_ensmean_sp['CLM45'])


#%%



#%%
