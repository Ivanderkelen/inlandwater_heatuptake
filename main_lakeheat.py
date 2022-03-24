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

flag_get_values = False

flag_plotting_forcings = True

flag_plotting_paper = False

flag_plotting_input_maps = False

flag_save_plots = True

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

flag_volume = 'cylindrical' #,'truncated_cone_cst' # 'cylindrical' for original calculation with cylindrical lakes
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

models      = ['CLM45','SIMSTRAT-UoG','GOTM'] # 'ALBM']#,'VIC-LAKE','LAKE']
forcings    = ['gfdl-esm2m']#,'ipsl-cm5a-lr','hadgem2-es','miroc5'] #,'miroc5']

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
# SENSITIVITY STUDY ON LAKE HEAT
# -------------------------------------------------------------------------

from calc_lakeheat import *

# flag to set volume calculation
vol_develoment_params = np.arange(0.3,1.35,0.05) # truncated_cone_cst
                    # "cylindrical"
if flag_calcheat:                               # if to use constant Vd, give number of Vd. e.g. 0.8
    lakeheat_sensitivity = {}
    for n,vd in enumerate(vol_develoment_params):
        print('calculating '+str(n+1)+ ' from '+str(len(vol_develoment_params))+' parameters')
        flag_volume = vd
        lakeheat_sensitivity[vd] = calc_lakeheat_with_volume(models,forcings,future_experiment, indir_lakedata, years_grand, resolution,outdir, years_isimip,start_year, end_year, flag_scenario, flag_savelakeheat, flag_volume, rho_liq, cp_liq, rho_ice, cp_ice)

    # Save according to scenario flag
    if flag_savelakeheat:
        lakeheat_filename = 'lakeheat_'+flag_scenario+'_vd_sensitivity.npy'
        np.save(outdir+lakeheat_filename, lakeheat_sensitivity) 

else: 

    lakeheat_sensitivity = np.load(outdir+'lakeheat_climate_vd_sensitivity.npy',allow_pickle='TRUE').item()
    lakeheat_cylindrical = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()
    lakeheat_cstVd = np.load(outdir+'lakeheat_climate_truncated_cstVd.npy',allow_pickle='TRUE').item()



#%%
# -------------------------------------------------------------------------
# PLOT SENSITIVITY STUDY ON LAKE HEAT
# -------------------------------------------------------------------------
from plotting_lakeheat import * 
import matplotlib as mpl

xticks = np.array([1900,1920,1940,1960,1980,2000,2021])

# calculate anomalies
lakeheat_anom = {}
lakeheat_anom_ts = {}

# colors

# extract all colors from the .jet map

for vd in vol_develoment_params: 
    lakeheat_anom[vd] = calc_anomalies(lakeheat_sensitivity[vd], flag_ref,years_analysis)
    # Calculate timeseries of lake heat anomaly
    lakeheat_anom_ts[vd] = timeseries(lakeheat_anom[vd])

lakeheat_anom_cylindrical = calc_anomalies(lakeheat_cylindrical, flag_ref,years_analysis)
lakeheat_anom_ts_cylindrical = timeseries(lakeheat_anom_cylindrical)

lakeheat_anom_cstVd = calc_anomalies(lakeheat_cstVd, flag_ref,years_analysis)
lakeheat_anom_ts_cstVd = timeseries(lakeheat_anom_cstVd)

#%%

cmap = mpl.cm.viridis(np.linspace(0, 1, len(vol_develoment_params)))
cmap = np.flip(cmap,axis=0)

nplot=0
f,ax = plt.subplots(3,1, figsize=(6,12))
x_values = np.asarray(years_analysis)

ax = ax.ravel()

# 4x4 individual forcing plot per model plot 
for model in models:
        

    forcing = forcings[0]

    line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.8,color='darkgray')
    for n,vd in enumerate(vol_develoment_params):
        line1 = ax[nplot].plot(x_values,lakeheat_anom_ts[vd][model][forcing],color=cmap[n],label=None, alpha=0.5)
    
    line2 = ax[nplot].plot(x_values,lakeheat_anom_ts_cstVd[model][forcing],color='darkorange',label='Vd = 1.19')
    line3 = ax[nplot].plot(x_values,lakeheat_anom_ts_cylindrical[model][forcing],color='tab:red',label='cylindrical')
    if nplot == 0: 
        ax[nplot].legend(frameon=False)
    ax[nplot].set_xlim(1900,2021)
    ax[nplot].set_xticks(ticks=xticks)
    #ax[nplot].set_ylim(-0.5e20,1.5e20)
    ax[nplot].set_ylabel('Energy [J]')
    ax[nplot].set_title(model + ' '+forcing, pad=15, loc='right')
    nplot = nplot+1

f.suptitle('Lake heat anomalies sensitivity to volume calculation \n Range for Vd from 0.3 to 1.3', fontsize=16)
f.tight_layout(rect=[0, 0.03, 1, 0.95])

if flag_save_plots:
    plt.savefig(outdir+'plots/sensitivity_volume_'+forcing+'.png',dpi=300)



# %%
