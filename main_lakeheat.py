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
import geopandas as gpd
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

flag_save_plots = False

flag_do_evaluation = False

# -----------------------------
# scenarios

# flag to set which scenario is used for heat calculation
flag_scenario = 'climate'  # 'climate'    : only climate change (lake cover constant at 2005 level)
                              # 'reservoirs' : only reservoir construction (temperature constant at 1900 level)
                              # 'both'       : reservoir construction and climate

# Reference to which period/year anomalies are calculated
flag_ref = 'pre-industrial'  # 'pre-industrial': first 30 years (1900-1929 for start_year =1900) 

#flag_ref =  1971  # 1971 or any integer: year as a reference 




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

models      = ['CLM45','SIMSTRAT-UoG', 'ALBM']#,'GOTM']#,'VIC-LAKE','LAKE']
forcings    = ['gfdl-esm2m','ipsl-cm5a-lr','hadgem2-es','miroc5'] #,'miroc5']
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


#%%
# -------------------------------------------------------------------------
# PREPROCESS raw ISIMIP variables
# Save them into annual timeseries for wanted period and store in correct folder
# -------------------------------------------------------------------------

if flag_preprocess: 
    from preprocess_isimip import *
    preprocess_isimip(models, forcings, variables, experiments, future_experiment, indir, outdir)
    
    
    from preprocess_iceheat import *
    preprocess_iceheat()



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

    #volume_per_layer = calc_volume_per_layer(flag_scenario, indir_lakedata, years_grand, start_year,end_year, resolution, models,outdir)
    lakeheat = calc_lakeheat(models,forcings,future_experiment, indir_lakedata, years_grand, resolution,outdir, years_isimip,start_year, end_year, flag_scenario, flag_savelakeheat, rho_liq, cp_liq, rho_ice, cp_ice)

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
    
    do_plotting(flag_save_plots, plotdir, models , forcings, lakeheat, flag_ref, years_analysis,outdir)
    plot_forcings_allmodels(flag_save_plots, plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir)

    #plot_casestudies(basepath,indir_lakedata,outdir,flag_ref,years_analysis)

if flag_plotting_input_maps: # plotting of lake/reservoir area fraction and lake depth
    from plotting_globalmaps import *
    do_plotting_globalmaps(indir_lakedata, plotdir, years_grand,start_year,end_year)


#%%
# -------------------------------------------------------------------------
# EVALUATION
# 
# Do spot evaluations 
# -------------------------------------------------------------------------

if flag_do_evaluation: 
    from preprocess_obs import * 
    from do_evaluation import *
    preprocess_obs(basepath)
    do_evaluation()

# %%

# %%
