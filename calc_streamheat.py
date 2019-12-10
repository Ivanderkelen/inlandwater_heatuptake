"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Scripts for calculating river heat content based on river temperatures and river storage 

"""
# %%

from cdo import Cdo
cdo = Cdo()
import os 
import xarray as xr
import numpy as np

# for windows, delete afterwards
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
from dict_functions import *


# %%
# load river storages

flag_preprocess=False

# Reference to which period/year anomalies are calculated
flag_ref = 'pre-industrial' # 'pre-industrial': first 30 years (1900-1929 for start_year =1900)


# -----------------------------------------------------------
# initialise


indir  =  basepath + 'data/ISIMIP/OutputData/water_global'
outdir =  basepath + 'data/processed/isimip_lakeheat/riverheat/'
plotdir=  basepath + '/data/processed/isimip_lakeheat/plots/'

indir_lakedata = '/home/inne/documents/phd/data/isimip_laketemp/'


models      = ['WaterGAP2']#,  'MATSIRO'    ]
scenarios   = ['histsoc_co2', '2005soc_co2'] # order need to be corresponding to models

forcings    = ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']

experiments = ['historical','future']
future_experiment = 'rcp60' 


variables   = ['riverstor']


start_year = 1900
end_year = 2017

years_isimip           = range(1861,2099,1)
years_grand            = range(1900,2010,1)
years_analysis         = range(start_year,end_year,1)
years_pi               = range(1861,1891,1)

# define constants
resolution = 0.5   # degrees

# constants values to check
cp_liq = 4.188e3   #[J/kg K] heat capacity liquid water
cp_ice = 2.11727e3 #[J/kg K] heat capacity ice

rho_liq = 1000     #[kg/m³] density liquid water
rho_ice = 0.917e3  #[kg/m³] density ice

#%%
# preprocess river storage files

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

                    for scenario in scenarios:
                        path = indir+'/'+model+'/'+forcing+'/'+experiment+'/'
                        infile = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+scenario+'_'+variable+'_'+'global'+'_'+'monthly'+'_'+period+'.nc4'
                        outfile_assembled = model.lower()+'_'+forcing+'_historical_'+future_experiment+'_'+variable+'_'+'1861_2099'+'_'+'annual'+'.nc4'

                        # make output directory per model if not done yet
                        outdir_model = outdir+'/'+variable+'/'+model+'/'
                        if not os.path.isdir(outdir_model):
                            os.system('mkdir '+outdir_model)
                    
                        # if simulation is available 
                        if os.path.isfile(path+infile): 

                            # calculate annual means per model for each forcing (if not done so)
                            outfile_annual = model.lower()+'_'+forcing+'_'+experiment_fn+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                            if (not os.path.isfile(outdir_model+outfile_assembled)):
                                print('calculating annual means of '+infile)
                                cdo.yearmean(input=path+infile,output=outdir_model+outfile_annual)


                # assemble historical and future simulation
                infile_hist = model.lower() +'_'+forcing+'_historical_'           +variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                infile_fut  = model.lower()+'_'+forcing+'_'+future_experiment+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                
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
# Calculate river heat content


# based on river temperatures and river storage. (load both)
# see lake heat storage as an example. 

# define experiment
experiment= 'historical_'+future_experiment # can also be set to 'historical' or 'rcp60', but start_year will in this case have to be within year range

# for the different models
riverheat = {}
rivertemp = {}
rivermass = {}

# define names and paths for stream temperatures
var_streamtemp = 'streamtemp'                        
forcings_streamtemp = ['GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
outdir_streamtemp = outdir+'/'+var_streamtemp+'/'


# load continent area 
continentalarea_fn = indir_lakedata + 'watergap_22d_static_continentalarea.nc4' 
ds_ca = xr.open_dataset(continentalarea_fn,decode_times=False)

# use continental area in model (also takes coastal values into account)
continentalarea = ds_ca.continentalarea.values * 10**(9) # in m²

for model in models:
    riverheat_model={} # sub directory for each model
    rivertemp_model={}
    rivermass_model={}
    for ind_forcing,forcing in enumerate(forcings):

        # define directory and filename
        variable = 'riverstor'                        
        outdir_model = outdir+variable+'/'+model+'/'
        outfile_annual = model.lower()+'_'+forcing+'_'+experiment+'_'+variable+'_'+'1861_2099'+'_'+'annual'+'.nc4'

        outfile_streamtemp =  var_streamtemp+'_'+forcings_streamtemp[ind_forcing]+'_1861_2099_annual.nc4'

        # if simulation is available 
        if os.path.isfile(outdir_model+outfile_annual): 
            print('Calculating river heat of '+ model + ' ' + forcing)
            
            ds_rivertemp = xr.open_dataset(outdir_streamtemp+outfile_streamtemp,decode_times=False)
            ds_riverstor = xr.open_dataset(outdir_model+outfile_annual         ,decode_times=False)

            riverstor = ds_riverstor.riverstor.values
            rivertemp_values = ds_rivertemp.streamtemp.values

            # select years? 
            rivertemp_forcing = rivertemp_values[years_isimip.index(start_year):years_isimip.index(end_year),:,:]
            riverstor = riverstor[years_isimip.index(start_year):years_isimip.index(end_year),:,:] # constant at first year

            # convert riverstorage from kg/m² to kg
            rivermass_forcing = riverstor * continentalarea.squeeze()

            riverheat_forcing =  cp_liq  *  rivermass_forcing * rivertemp_forcing 

            # save riverheat in directory structure per forcing
            if not riverheat_model:
                riverheat_model = {forcing:riverheat_forcing}
                rivertemp_model = {forcing:rivertemp_forcing}
                rivermass_model = {forcing:rivermass_forcing}
            else: 
                riverheat_model.update({forcing:riverheat_forcing})
                rivertemp_model.update({forcing:rivertemp_forcing})
                rivermass_model.update({forcing:rivermass_forcing})
    # save riverheat of forcings in directory structure per model
    if not riverheat:

        riverheat = {model:riverheat_model}
        rivertemp = {model:rivertemp_model}
        rivermass = {model:rivermass_model}

    else:

        riverheat.update({model:riverheat_model})  
        rivertemp.update({model:rivertemp_model})
        rivermass.update({model:rivermass_model})



#%%
# ------------------------------------------------------------------------
# Absolute heat content calculations



riverheat_ts = timeseries(riverheat)
riverheat_ens = ensmean(riverheat)
riverheat_ensmean_ts = ensmean_ts(riverheat)
riverheat_ensmin_ts = ensmin_ts(riverheat)
riverheat_ensmax_ts = ensmax_ts(riverheat)

riverheat_ens_spmean = ens_spmean(riverheat)

#%%
# ---------------------------------------------------------------------------
# Anomaly heat content calculations

# define reference 
riverheat_anom = {}

for model in riverheat:
    riverheat_anom_model = {}

    for forcing in riverheat[model]: 

        # determine reference
        if flag_ref == 'pre-industrial':   # period of first 30 years of simulation (1900-1929)
            riverheat_ref_forcing = np.nanmean(riverheat[model][forcing][0:30,:,:])
        elif isinstance(flag_ref,int):
            riverheat_ref_forcing = riverheat[model][forcing][years_analysis.index(flag_ref),:,:]

        # subtract reference to calculate anomaly 
        riverheat_anom_model[forcing] = riverheat[model][forcing] - riverheat_ref_forcing
    riverheat_anom[model] = riverheat_anom_model

# River temperature
# define reference 
rivertemp_anom = {}

for model in rivertemp:
    rivertemp_anom_model = {}

    for forcing in rivertemp[model]: 

        # determine reference
        if flag_ref == 'pre-industrial':   # period of first 30 years of simulation (1900-1929)
            rivertemp_ref_forcing = np.nanmean(rivertemp[model][forcing][0:30,:,:])
        elif isinstance(flag_ref,int):
            rivertemp_ref_forcing = rivertemp[model][forcing][years_analysis.index(flag_ref),:,:]

        # subtract reference to calculate anomaly 
        rivertemp_anom_model[forcing] = rivertemp[model][forcing] - rivertemp_ref_forcing
    rivertemp_anom[model] = rivertemp_anom_model


# river mass
# define reference 
rivermass_anom = {}

for model in rivermass:
    rivermass_anom_model = {}

    for forcing in rivermass[model]: 

        # determine reference
        if flag_ref == 'pre-industrial':   # period of first 30 years of simulation (1900-1929)
            rivermass_ref_forcing = np.nanmean(rivermass[model][forcing][0:30,:,:])
        elif isinstance(flag_ref,int):
            rivermass_ref_forcing = rivermass[model][forcing][years_analysis.index(flag_ref),:,:]

        # subtract reference to calculate anomaly 
        rivermass_anom_model[forcing] = rivermass[model][forcing] - rivermass_ref_forcing
    rivermass_anom[model] = rivermass_anom_model

# -----------------------------------------------------------------------------
# Aggregate - calculate timeseries and averages

riverheat_anom_ts = timeseries(riverheat_anom)

riverheat_anom_ens = ensmean(riverheat_anom)

riverheat_anom_ensmean_ts = moving_average(ensmean_ts(riverheat_anom))
riverheat_anom_ensmin_ts  = moving_average(ensmin_ts(riverheat_anom))
riverheat_anom_ensmax_ts  = moving_average(ensmax_ts(riverheat_anom))

riverheat_anom_ensmean_ts = moving_average(ensmean_ts(riverheat_anom))
riverheat_anom_ensmin_ts  = moving_average(ensmin_ts(riverheat_anom))
riverheat_anom_ensmax_ts  = moving_average(ensmax_ts(riverheat_anom))

riverheat_anom_ens_spmean = ens_spmean(riverheat_anom)

# rivertemp and river mass
rivertemp_anom_ts = timeseries_mean(rivertemp_anom)
rivermass_anom_ts = timeseries(rivermass_anom)



#%%
# ---------------------------------------------------------------------------------------
# Plotting
# 

# settings


import matplotlib.pyplot as plt
import matplotlib as mpl

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

plotdir='/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/'
plotdir= 'E:/data/processed/isimip_lakeheat/plots/'
#%%
# lineplot of one model anomaly- all forcings 5-year moving average
colors_primary = ['deepskyblue','coral']
colors_secondary = ['lightskyblue', 'sandybrown']


f,ax = plt.subplots(figsize=(7,4))

x_values = moving_average(np.asarray(years_analysis))

line_zero = ax.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

for ind_mod, model in enumerate(models):
    print
    line2 = ax.plot(x_values,riverheat_anom_ensmax_ts[model], color=colors_secondary[ind_mod])
    line3 = ax.plot(x_values,riverheat_anom_ensmin_ts[model], color=colors_secondary[ind_mod])
    area1 = ax.fill_between(x_values,riverheat_anom_ensmin_ts[model],riverheat_anom_ensmax_ts[model], color=colors_secondary[ind_mod])

    ax.plot(x_values,riverheat_anom_ensmean_ts[model], color=colors_primary[ind_mod])

ax.set_xlim(x_values[0],x_values[-1])
#ax.set_ylim(-4e19,10e19)
#ax.grid(color='lightgrey')
ax.set_ylabel('Energy [J]')
ax.set_title('River heat content anomalies (reference 1900-1929)', pad=15)

plt.savefig(plotdir+'riverheat_anom1900_5yav'+'.png',dpi=300)


#%%
# ---------------------------------------------------------------------------------------
# lineplots of one model anomaly for all forcings 
# river heat 
for model in models:
        
    f,ax = plt.subplots(2,2, figsize=(8,7))
    x_values = np.asarray(years_analysis)

    ax = ax.ravel()

    for nplot,forcing in enumerate(forcings):

        line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
        line1 = ax[nplot].plot(x_values,riverheat_anom_ts[model][forcing], color='deepskyblue')
        ax[nplot].set_xlim(x_values[0],x_values[-1])
        #ax[nplot].set_ylim(-4e19,10e19)
        ax[nplot].set_ylabel('Energy [J]')
        ax[nplot].set_title(forcing, pad=15)

    f.suptitle(model+' river heat anomalies (reference 1900-1929)', fontsize=16)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plotdir+model+'riverheat_acc_per_forcing'+'.png',dpi=300)

# river temperature
for model in models:
        
    f,ax = plt.subplots(2,2, figsize=(8,7))
    x_values = np.asarray(years_analysis)

    ax = ax.ravel()

    for nplot,forcing in enumerate(forcings):

        line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
        line1 = ax[nplot].plot(x_values,rivertemp_anom_ts[model][forcing], color='brown')
        ax[nplot].set_xlim(x_values[0],x_values[-1])
        ax[nplot].set_ylim(-0.2,0.8)
        ax[nplot].set_ylabel('Temperature [K]')
        ax[nplot].set_title(forcing, pad=15)

    f.suptitle(model+' river temperature anomalies (reference 1900-1929)', fontsize=16)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plotdir+model+'rivertemp_per_forcing'+'.png',dpi=300)

# river mass
for model in models:
        
    f,ax = plt.subplots(2,2, figsize=(8,7))
    x_values = np.asarray(years_analysis)

    ax = ax.ravel()

    for nplot,forcing in enumerate(forcings):

        line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
        line1 = ax[nplot].plot(x_values,rivermass_anom_ts[model][forcing], color='deepskyblue')
        ax[nplot].set_xlim(x_values[0],x_values[-1])
        ax[nplot].set_ylim(-5e14,10e14)
        ax[nplot].set_ylabel('Mass [kg]')
        ax[nplot].set_title(forcing, pad=15)

    f.suptitle(model+' river mass anomalies (reference 1900-1929)', fontsize=16)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plotdir+model+'rivermass_acc_per_forcing'+'.png',dpi=300)




#%%
