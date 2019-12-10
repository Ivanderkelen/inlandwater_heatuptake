"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : June 2019

Main script for lake heat calculation and plotting

Notes: SCRIPT IN PROGRESS


"""

# TO DO in the future: 
# - handle heat content per lake layer 
# - handle resampling issues (lake pct)
# - correction for lake shape? 
# - gldb lake depth no good assumption for reservoirs
# -reservoir construction enabled only works until 2000: necessary to update lake_pct input data to newly GRanD files' climate 


#%%
# -------------------------------------------------------------------------
# PYTHON PACKAGES
# -------------------------------------------------------------------------

import os 
import sys

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




# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

# -----------------------------------------------------------
# FLAGS

# ------------------------------
# turn on/off parts of script

flag_preprocess = False

flag_interpolate_CLM45 = False # make interpolation of CLM temperature fields. (takes time)

flag_calcheat  = True # whether or not to calculate lake heat (otherwise use saved lake heat)

# whether or not to save calculated lake heat (can only be true if flag_calcheat is true)
flag_savelakeheat = True

flag_get_values = True

flag_plotting = True

flag_plotting_input_maps = False

flag_save_plots = False

# -----------------------------
# scenarios

# flag to set which scenario is used for heat calculation
flag_scenario = 'reservoirs' # 'climate'    : only climate change (lake cover constant at 2005 level)
                              # 'reservoirs' : only reservoir construction (temperature constant at 1900 level)
                              # 'both'       : reservoir construction and climate

# Reference to which period/year anomalies are calculated
flag_ref = 'pre-industrial'  # 'pre-industrial': first 30 years (1900-1929 for start_year =1900)
                             # 1971 or any integer: year as a reference 




# -----------------------------------------------------------
# PATHS

project_name = 'isimip_lakeheat/'

indir  = basepath + 'data/ISIMIP/OutputData/lakes_global/'
outdir = basepath + 'data/processed/'+ project_name
plotdir= basepath + 'data/processed/'+ project_name+ '/plots/'
indir_lakedata   = basepath + 'data/isimip_laketemp/' # directory where lake fraction and depth are located



# -----------------------------------------------------------
# MODELS & FORCINGS

models      = ['CLM45']#['SIMSTRAT-UoG']#'CLM45','ALBM','VIC-LAKE','LAKE']
forcings    = ['gfdl-esm2m','hadgem2-es']#,'ipsl-cm5a-lr','miroc5']
experiments = ['historical','future']

# experiment used for future simulations (needed to differentiate between filenames)
future_experiment = 'rcp60' 

variables   = ['watertemp']


# -----------------------------------------------------------
# PERIODS
start_year = 1900
end_year = 2017

years_isimip           = range(1861,2099,1)
years_grand            = range(1900,2018,1)
years_analysis         = range(start_year,end_year,1)
years_pi               = range(1861,1891,1)


# -----------------------------------------------------------
# CONSTANTS

resolution = 0.5 # degrees

# constants values to check
cp_liq = 4.188e3   # [J/kg K] heat capacity liquid water
cp_ice = 2.11727e3 # [J/kg K] heat capacity ice
cp_salt= 3.993e3   #[J/kg K] heat capacity salt ocean water (not used)

rho_liq = 1000     # [kg/m²] density liquid water
rho_ice = 0.917e3  # [kg/m²] density ice


#%%
# -------------------------------------------------------------------------
# PREPROCESS raw ISIMIP variables
# Save them into annual timeseries for wanted period and store in correct folder
# -------------------------------------------------------------------------

if flag_preprocess: 
    from preprocess_isimip import *
    preprocess_isimip(models, forcings, variables, experiments, future_experiment, indir, outdir)

# possible to add here preprocessing for ice


#%%
# -------------------------------------------------------------------------
# INTERPOLATE lake temperatures of CLM45 
# based on lakepct mask and saves interpolated watertemps into netcdf 
# -------------------------------------------------------------------------

if flag_interpolate_CLM45:
    from interp_clm_watertemp import *
    interp_clm_watertemp(indir_lakedata,outdir,forcings,future_experiment)  


#%%
# -------------------------------------------------------------------------
# CALCULATE VOLUMES and LAKEHEAT  
# loads hydrolakes + GLDB data to calculate lake volume per layer 
# -------------------------------------------------------------------------

if flag_calcheat: 
    from calc_volumes import *
    from calc_lakeheat import *

    volume_per_layer = calc_volume_per_layer(flag_scenario, indir_lakedata, years_grand, start_year,end_year, resolution, models,outdir)
    lakeheat = calc_lakeheat(models,forcings,future_experiment,volume_per_layer, outdir, years_isimip,start_year, end_year, flag_scenario, flag_savelakeheat, rho_liq, cp_liq)
else: 
    # load from file based on scenario: 
    if flag_scenario == 'climate':
        lakeheat = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()
    elif flag_scenario == 'reservoirs':
        lakeheat = np.load(outdir+'lakeheat_reservoirs.npy',allow_pickle='TRUE').item()
    elif flag_scenario == 'both':
        lakeheat = np.load(outdir+'lakeheat_both.npy',allow_pickle='TRUE').item()

#%%
# -------------------------------------------------------------------------
# GET VALUES for paper
# -------------------------------------------------------------------------

if flag_get_values: 
    from get_values_lakeheat import * 
    #get_values()

#%%
# -------------------------------------------------------------------------
# PLOTTING
# Do the plotting - works with internal flags
# data aggregation is done from within scripts. 
# -------------------------------------------------------------------------

if flag_plotting: 
    from plotting_lakeheat import * 
    do_plotting(flag_save_plots, plotdir, models , forcings, lakeheat, flag_ref, years_analysis,outdir)

if flag_plotting_input_maps: # plotting of lake/reservoir area fraction and lake depth
    from plotting_globalmaps import *
    do_plotting_globalmaps(indir_lakedata, plotdir, years_grand,start_year,end_year)


#%%
