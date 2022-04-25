"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : March 2022

Main script for heat calculation and plotting
Update for land heat inventory analysis. 

"""

#%%
# -------------------------------------------------------------------------
# PYTHON PACKAGES
# -------------------------------------------------------------------------

import os 
import sys

sys.path.append(os.getcwd())

#from cdo import Cdo
#cdo = Cdo()

import xarray as xr
import numpy as np
import pandas as pd
from dict_functions import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

# -----------------------------------------------------------
# FLAGS

# ------------------------------
# turn on/off parts of script

flag_preprocess = False # this is done on the cluster, using the same scripts

flag_interpolate_watertemp = False # make interpolation of CLM temperature fields. (takes time)

flag_calcheat  = False # if false use saved lake heat (otherwise use saved lake heat), for ALBM done on the cluster. 

# whether or not to save calculated lake heat (can only be true if flag_calcheat is true)
flag_savelakeheat = False

flag_get_values = True

flag_plotting_forcings = False

flag_plotting_paper = True

flag_plotting_input_maps = False

flag_save_plots = True

flag_save_variables = True # save variables of plotting. 

flag_do_evaluation = False


# -----------------------------
# scenarios

# flag to set which scenario is used for heat calculation
flag_scenario = 'climate'  # 'climate'    : only climate change (lake cover constant at 2005 level)
                              # 'reservoirs' : only reservoir construction (temperature constant at 1900 level)
                              # 'both'       : reservoir construction and climate

# Reference to which period/year anomalies are calculated
flag_ref = 'pre-industrial'  #'pre-industrial'  # 'pre-industrial': first 30 years (1900-1929 for start_year =1900) 

#flag_ref =  1971  # 1971 or any integer: year as a reference 

flag_volume = 'truncated_cone_cst' #,'truncated_cone_cst' # 'cylindrical' for original calculation with cylindrical lakes
                            # 'truncated_cone_cst': use constant median Vd calculated from GLOBathy
                            # 


# -----------------------------------------------------------
# PATHS

basepath = os.getcwd()

indir  = basepath + '/data/ISIMIP/OutputData/lakes_global/'
outdir = basepath + '/data/processed/'
plotdir= basepath + '/data/processed/plots/'
indir_lakedata   = basepath + '/data/auxiliary_data/' # directory where lake fraction and depth are located

# paths on hydra (where preprocessing is done)
#project_name = 'isimip_lakeheat/'

#indir  = '/gpfs/projects/climate/data/dataset/isimip/isimip2b/OutputData/lakes_global/'
#outdir = '/scratch/brussel/100/vsc10055/'+ project_name
#plotdir= '/scratch/brussel/100/vsc10055/'+ project_name + '/plots/'


# -----------------------------------------------------------
# MODELS & FORCINGS

models      = [ 'CLM45','SIMSTRAT-UoG','GOTM','ALBM']#,'GOTM']#'ALBM','GOTM']#,'VIC-LAKE','LAKE']
forcings    = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']
experiments = ['historical','future']



# experiment used for future simulations (needed to differentiate between filenames)
future_experiment = 'rcp60' 

variables   = ['watertemp']


# -----------------------------------------------------------
# PERIODS
start_year = 1896
end_year = 2025 #2026

years_grand            = range(1850,2018,1)
years_analysis         = range(start_year,end_year,1)
years_pi               = range(1861,1891,1)

# depending on model 
years_isimip = {}
years_isimip['CLM45'] = range(1891,2030,1)
years_isimip['SIMSTRAT-UoG'] = range(1891,2030,1)
years_isimip['ALBM'] = range(1891,2030,1)
years_isimip['GOTM'] = range(1891,2030,1)




# -----------------------------------------------------------
# CONSTANTS

resolution = 0.5 # degrees

# constants values to check
cp_liq = 4.188e3   # [J/kg K] heat capacity liquid water
cp_ice = 2.11727e3 # [J/kg K] heat capacity ice
cp_salt= 3.993e3   # [J/kg K] heat capacity salt ocean water (not used)
l_fus = 3.337e5    # [J/kg]  latent heat of future


rho_liq = 1000     # [kg/m2] density liquid water
rho_ice = 0.917e3  # [kg/m2] density ice


# -----------------------------------------------------------
# VOLUME CALCULATION
# flag to set volume calculation
from calc_volumes import *


if flag_volume == 'truncated_cone_cst':
    Vd = calc_median_Vd(indir_lakedata)
    flag_volume = Vd


#%%
# -------------------------------------------------------------------------
# PREPROCESS raw ISIMIP variables
# Save them into annual timeseries for wanted period and store in correct folder
# -------------------------------------------------------------------------

if flag_preprocess: 
    from preprocess_isimip import *
    preprocess_isimip(models, forcings, variables, experiments, future_experiment, indir, outdir)
    
    
    #from preprocess_iceheat import *
    #preprocess_iceheat()



#%%
# -------------------------------------------------------------------------
# INTERPOLATE lake temperatures of CLM45 
# based on lakepct mask and saves interpolated watertemps into netcdf 
# -------------------------------------------------------------------------

if flag_interpolate_watertemp:
    from interp_watertemp import *
    for model in models:
        interp_watertemp(indir_lakedata,outdir,forcings,future_experiment,model)  


#%%
# -------------------------------------------------------------------------
# CALCULATE VOLUMES and LAKEHEAT  
# loads hydrolakes + GLDB data to calculate lake volume per layer 
# -------------------------------------------------------------------------

if flag_calcheat: 
    #from calc_volumes  import *
    from calc_lakeheat import *

    #lakeheat = calc_lakeheat(models,forcings,future_experiment, indir_lakedata, years_grand, resolution,outdir, years_isimip,start_year, end_year, flag_scenario, flag_savelakeheat, rho_liq, cp_liq, rho_ice, cp_ice)
    lakeheat = calc_lakeheat_with_volume(models,forcings,future_experiment, indir_lakedata, years_grand, resolution,outdir, years_isimip,start_year, end_year, flag_scenario, flag_savelakeheat, flag_volume, rho_liq, cp_liq, rho_ice, cp_ice)
else: 

    from load_lakeheat_albm import *

    # load from file based on scenario: (ALBM separate as these are calculated on HPC)
    if flag_scenario == 'climate':
        lakeheat = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()
        lakeheat_albm = load_lakeheat_albm(outdir,flag_scenario,years_analysis)
       
    #    lakeheat_albm = load_lakeheat_albm(outdir,flag_scenario,years_analysis,forcings)
    elif flag_scenario == 'reservoirs':
        lakeheat = np.load(outdir+'lakeheat_reservoirs.npy',allow_pickle='TRUE').item()
        lakeheat_albm = load_lakeheat_albm(outdir,flag_scenario,years_analysis)
        
    elif flag_scenario == 'both':
        lakeheat = np.load(outdir+'lakeheat_both.npy',allow_pickle='TRUE').item()
        lakeheat_albm = load_lakeheat_albm(outdir,flag_scenario,years_analysis)

    # add ALBM dictionary to lakeheat dict. 
    lakeheat.update(lakeheat_albm)


#%%
# ------------------------------------------------------------------
# Calculate onlyres climate
# -------------------------------------------------------------------------

# calculate reservoir warming (difference total and (climate+reservoir expansion))
calc_reservoir_warming(outdir,years_analysis)

#%%
# -------------------------------------------------------------------------
# SAVE global mean HEAT CONTENT timeseries
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
from dict_functions import *

ind_1960 = years_analysis.index(1960)
ind_1900 = years_analysis.index(1900)
ind_year = ind_1960
flag_ref = 1960
# natural lakes, reservoirs and both
for scenario in ['climate','onlyresclimate','lake_and_reservoir']:

    if scenario == 'lake_and_reservoir': 
        #(lakeheat_ensmean, lakeheat_std) = calc_ensmean_std_lakes_and_reservoirs_heatuptake(outdir,flag_ref, years_analysis)
        lakeheat_ensmean_climate, lakeheat_std_climate = load_lakeheat_no_movingmean('climate',outdir,flag_ref, years_analysis)
        lakeheat_ensmean_onlyresclimate, lakeheat_std_onlyresclimate = load_lakeheat_no_movingmean('onlyresclimate',outdir,flag_ref, years_analysis)
        lakeheat_ensmean = lakeheat_ensmean_climate + lakeheat_ensmean_onlyresclimate
        lakeheat_std = lakeheat_std_onlyresclimate + lakeheat_std_climate
    else: 
        """
        lakeheat= np.load(outdir+'lakeheat_'+scenario+'.npy',allow_pickle='TRUE').item()
        
        if scenario =='climate':
            lakeheat_albm = load_lakeheat_albm(outdir,scenario,years_analysis)
            lakeheat.update(lakeheat_albm)
            del lakeheat_albm

        lakeheat_anom = calc_anomalies(lakeheat, flag_ref, years_analysis)

        lakeheat_ensmean = ensmean_ts(lakeheat_anom)
        lakeheat_std     = ens_std_ts(lakeheat_anom)
        del lakeheat_anom, lakeheat
        """
        lakeheat_ensmean, lakeheat_std = load_lakeheat_no_movingmean(scenario,outdir,flag_ref, years_analysis)
   
   
    if scenario == 'lake_and_reservoir': 
        fn = 'lake_and_reservoir'
    elif scenario == 'onlyresclimate': 
        fn = 'reservoir'
    elif scenario == 'climate': 
        fn = 'natural_lake'

    data = np.stack([lakeheat_ensmean, lakeheat_std],axis=1)
    data = data[ind_year:,:]
    years = years_analysis[ind_year:]

    df = pd.DataFrame(data =data, index = years,columns=['Global mean heat storage [J]','Standard deviation heat storage [J]'] )
        #del lakeheat_anom, lakeheat

        #fn = 'inlandwater_'
    #df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatstorage_1900-2021_'+fn+'.dat')
    df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatstorage_1960ref_'+fn+'.dat')

    #return (anom_ensmean, anom_ensmin, anom_ensmax, anom_std)

# river heat
riverheat_ensmean = np.load(outdir+'riverheat/riverheat_ts_ensmean.npy',allow_pickle='TRUE')
riverheat_std = np.load(outdir+'riverheat/riverheat_ts_std.npy',allow_pickle='TRUE')

data = np.stack([riverheat_ensmean, riverheat_std],axis=1)
data = data[ind_year:,:]
years = years_analysis[ind_year:]

df = pd.DataFrame(data =data, index = years,columns=['Global mean heat storage [J]','Standard deviation heat storage [J]'] )
    #del lakeheat_anom, lakeheat

fn = 'river'

    #fn = 'inlandwater_'
#df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatstorage_1900-2021_'+fn+'.dat')
df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatstorage_1960ref_'+fn+'.dat')


#%% 
# -------------------------------------------------------------------------
# SAVE global mean annual HEAT FLUX timeseries
# -------------------------------------------------------------------------


# get lake area and calculate reservoir area
lake_area    = np.load(indir_lakedata+'lake_area.npy')
lake_area_ts = np.sum(lake_area, axis=(1,2))
res_area_ts = lake_area_ts - lake_area_ts[0]

ind_1900 = years_analysis.index(1900)
ind_year = ind_1900

for scenario in ['climate','onlyresclimate','lake_and_reservoir']:

    if scenario == 'climate': 
        area = lake_area_ts[0]
    elif scenario == 'onlyresclimate': 
        area = lake_area_ts - lake_area_ts[0]
        area = area[:-1]
    elif scenario == 'lake_and_reservoir': 
        area = lake_area_ts[:-1]

    lakeheat= np.load(outdir+'lakeheat_'+scenario+'.npy',allow_pickle='TRUE').item()

    if scenario =='climate':
        lakeheat_albm = load_lakeheat_albm(outdir,scenario,years_analysis)
        lakeheat.update(lakeheat_albm)
        del lakeheat_albm

    if scenario == 'lake_and_reservoir':
        #(lakeheat_ensmean, lakeheat_std) = calc_ensmean_std_lakes_and_reservoirs_heatuptake(outdir,flag_ref, years_analysis)
        lakeheat_climate = np.load(outdir+'lakeheat_'+'climate'+'.npy',allow_pickle='TRUE').item()
        lakeheat_albm = load_lakeheat_albm(outdir,'climate',years_analysis)
        lakeheat_climate.update(lakeheat_albm)
        del lakeheat_albm

        lakeheat_reservoir =  np.load(outdir+'lakeheat_'+'onlyresclimate'+'.npy',allow_pickle='TRUE').item() 
        
        # adding the values with common key
        lakeheat = {}
        for key in lakeheat_climate:
            if key in lakeheat_reservoir:
                temp = {}
                for key2 in lakeheat_climate[key]:
                    temp[key2] = lakeheat_climate[key][key2] + lakeheat_reservoir[key][key2]
                lakeheat[key] = temp
            else: 
                 lakeheat[key] = lakeheat_climate[key]
        #lakeheat = lakeheat_climate + lakeheat_reservoir

    #calculate ensemble mean heat flux
    heatflux_ensmean = calc_ensmean_heatflux(lakeheat,area,years_analysis)    

    # calculate standard deviation heat flux
    heatflux_std = ens_std_heatflux(lakeheat,area,years_analysis)

    data = np.stack([heatflux_ensmean, heatflux_std],axis=1)
    data = data[ind_year:,:]
    years = years_analysis[ind_year:-1]
    df = pd.DataFrame(data =data, index = years,columns=['Global mean annual heat flux [W/m²]','Standard deviation annual heat flux [W/m²]'] )
    
    if scenario == 'lake_and_reservoir': 
        fn = 'lake_and_reservoir'
    elif scenario == 'onlyresclimate': 
        fn = 'reservoir'
    elif scenario == 'climate': 
        fn = 'natural_lake'

        #fn = 'inlandwater_'
    df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatflux_1900-2021_'+fn+'.dat')



# river heat flux 
riverheat_heatflux_ensmean = np.load(outdir+'riverheat/riverheat_heatflux_ensmean.npy',allow_pickle='TRUE')
riverheat_heatflux_std = np.load(outdir+'riverheat/riverheat_heatflux_std.npy',allow_pickle='TRUE')

data = np.stack([riverheat_heatflux_ensmean, riverheat_heatflux_std],axis=1)
data = data[ind_year:,:]
years = years_analysis[ind_year:-1]

df = pd.DataFrame(data =data, index = years,columns=['Global mean annual heat flux [W/m²]','Standard deviation annual heat flux [W/m²]'] )
    #del lakeheat_anom, lakeheat

fn = 'river'

    #fn = 'inlandwater_'
df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatflux_1900-2021_'+fn+'.dat')


#%%
# -------------------------------------------------------------------------
# SAVE global mean per area HEAT CONTENT timeseries for lakes and reservoirs! 
# -------------------------------------------------------------------------
ind_year = ind_1960
flag_ref = 1960

# get lake area and calculate reservoir area
lake_area    = np.load(indir_lakedata+'lake_area.npy')
lake_area_ts = np.sum(lake_area, axis=(1,2))

# natural lakes, reservoirs and both
scenario = 'lake_and_reservoir'

area = lake_area_ts #[:-1]

lakeheat_ensmean_climate, lakeheat_std_climate = load_lakeheat_no_movingmean('climate',outdir,flag_ref, years_analysis)
lakeheat_ensmean_onlyresclimate, lakeheat_std_onlyresclimate = load_lakeheat_no_movingmean('onlyresclimate',outdir,flag_ref, years_analysis)
lakeheat_ensmean = lakeheat_ensmean_climate + lakeheat_ensmean_onlyresclimate
lakeheat_std = lakeheat_std_onlyresclimate + lakeheat_std_climate

lakeheat_ensmean_perarea = lakeheat_ensmean/area
lakeheat_std_perarea = lakeheat_std/area

fn = 'lake_and_reservoir'

data = np.stack([lakeheat_ensmean_perarea, lakeheat_std_perarea],axis=1)
data = data[ind_year:,:]
years = years_analysis[ind_year:]

df = pd.DataFrame(data =data, index = years,columns=['Global mean heat storage per area [J/m2]','Standard deviation heat storage [J/m2]'] )
    #del lakeheat_anom, lakeheat

    #fn = 'inlandwater_'
#df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatstorage_perarea_1900-2021_'+fn+'.dat')
df.to_csv(outdir+'inlandwater_heatuptake_timeseries/heatstorage_perarea_1960ref_'+fn+'.dat')

#return (anom_ensmean, anom_ensmin, anom_ensmax, anom_std)

#%% 
# -------------------------------------------------------------------------
# SAVE global mean annual HEAT MAP 
#  -------------------------------------------------------------------------
from datetime import date
# load example netcdf file
# add today (for saving to netCDF later)
today = date.today()
date = today.strftime("%c")

# year to calculate heat uptake for
year = 2021

# get lake area and calculate reservoir area
lake_area    = np.load(indir_lakedata+'lake_area.npy')

ind_year = years_analysis.index(year)

# natural lakes, reservoirs and both
fn = 'lake_and_reservoir'

lakeheat_climate = np.load(outdir+'lakeheat_'+'climate'+'.npy',allow_pickle='TRUE').item()
lakeheat_albm = load_lakeheat_albm(outdir,'climate',years_analysis)
lakeheat_climate.update(lakeheat_albm)
del lakeheat_albm

lakeheat_reservoir =  np.load(outdir+'lakeheat_'+'onlyresclimate'+'.npy',allow_pickle='TRUE').item() 

# adding the values with common key
lakeheat = {}
for key in lakeheat_climate:
    if key in lakeheat_reservoir:
        temp = {}
        for key2 in lakeheat_climate[key]:
            temp[key2] = lakeheat_climate[key][key2] + lakeheat_reservoir[key][key2]
        lakeheat[key] = temp
    else: 
        lakeheat[key] = lakeheat_climate[key]


# calculate timeseries maps of mean of all models and forcings 
lakeheat_ensmean = ens_spmean_ensmean(lakeheat) # output np array (time,lat,lon)

lakeheat_pi = np.nanmean(lakeheat_ensmean[0:30,:,:],axis=0)
lakeheat_pres = lakeheat_ensmean[ind_year,:,:]

lakeheat_anom_spmean = lakeheat_pres -lakeheat_pi

#lakeheat_anom = calc_anomalies(lakeheat, flag_ref, years_analysis)

#lakeheat_anom_ensmean = ens_spmean_ensmean(lakeheat_anom)    

# select year to save into netcdf and calculate per area
lakeheat_anom_ensmean_year_perarea = lakeheat_anom_spmean / lake_area[ind_year,:,:]

## write NETCDF file
filename_netcdf = outdir+'inlandwater_heatuptake_timeseries/heatstorage_map_'+str(year)+'_'+fn+'.nc'
# attributes

variable_name = 'heatstorage'
values = lakeheat_anom_ensmean_year_perarea
# variable attributes
attrs_variable = {'units': 'J/m2', 'long_name' : 'Lake and reservoir heat storage for '+str(year)+' compared to 1900-1929 mean'}

# global attributes
attrs_global = {'creation_date': date,
                        'title': 'Inland water heat storage',
                        'contact' : 'Inne Vanderkelen - VUB (inne.vanderkelen@vub.be)'}

# writing

resolution = 0.5
lons= np.arange(-180+resolution/2,180+resolution/2,resolution)
lats= np.arange(-90+resolution/2,90+resolution/2,resolution)

lon_da = xr.DataArray(lons, 
                        coords = {'lon':lons}, 
                        dims='lon', 
                        attrs={'units':'degrees_east', 'axis':"X"})

lat_da = xr.DataArray(lats,
                        coords = {'lat':lats}, 
                        dims='lat', 
                        attrs={'units':'degrees_north', 'axis':"X"})


values_da = xr.DataArray(values, 
                        coords = {'lon':lons,'lat':lats},
                        dims=('lat','lon'),
                        attrs = attrs_variable)


ds = xr.Dataset(data_vars={ 'lon' : lon_da,   
                            'lat' : lat_da,
                            variable_name : values_da},
                            attrs=attrs_global)
                            
ds.to_netcdf(filename_netcdf, format='NETCDF4_CLASSIC',mode='w')



#%%
# -------------------------------------------------------------------------
# GET VALUES for paper
# -------------------------------------------------------------------------

if flag_get_values: 
    from get_values_lakeheat import * 
    get_values(outdir,flag_ref, years_analysis, indir_lakedata, resolution)

#%%
# -------------------------------------------------------------------------
# PLOTTING
# Do the plotting - works with internal flags
# data aggregation is done from within functions 
# -------------------------------------------------------------------------

if flag_plotting_forcings: 
    from plotting_lakeheat import * 
    plot_forcings(flag_save_plots, plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir)

if flag_plotting_paper: 
    from plotting_lakeheat import * 
    #from plotting_casestudies import *
    
    do_plotting(flag_save_plots, flag_save_variables, plotdir, flag_ref, years_analysis,outdir)
    plot_forcings_allmodels(flag_save_plots,  plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir)
    plot_forcings_allmodels_Franciscostyle(flag_save_plots, plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir)
    #plot_casestudies(basepath,indir_lakedata,outdir,flag_ref,years_analysis)

if flag_plotting_input_maps: # plotting of lake/reservoir area fraction and lake depth
    from plotting_globalmaps import *
    do_plotting_globalmaps(indir_lakedata, plotdir, years_grand,start_year,end_year)



# %%

filenames = ['river','lake_and_reservoir','reservoir','natural_lake']


for fn in filenames:     
    plot_heatstorage_ts(fn,1900,end_year,outdir)
    plot_heatflux_ts(fn,1900,end_year,outdir) 
    
plot_heatstorage_ts_perarea('lake_and_reservoir',1900,end_year,outdir)

# %%
plot_heatflux_ts('lake_and_reservoir',1900,end_year,outdir) 

# %%
