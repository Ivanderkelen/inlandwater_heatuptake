"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Script for calculating river heat content based on river temperatures and river storage 

"""
# %%
import os 
import sys
from cdo import Cdo
cdo = Cdo()
from calc_grid_area   import calc_grid_area
from dict_functions import *

sys.path.append(os.getcwd())


# %%
# load river storages

flag_preprocess = True 

flag_saveriverheat = True
flag_saveriverheat_forAmazon = False
# Reference to which period/year anomalies are calculated
flag_ref = 'pre-industrial' # 'pre-industrial': first 30 years (1900-1929 for start_year =1900)


# -----------------------------------------------------------
# initialise


indir  =  basepath + '/data/ISIMIP/OutputData/water_global'
outdir =  basepath + '/data/processed/riverheat/'
plotdir=  basepath + '/data/processed/plots/'



models      = ['WaterGAP2',  'MATSIRO'    ]
scenarios   = ['histsoc_co2', '2005soc_co2'] # order need to be corresponding to models

forcings    = ['hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']#

experiments = ['historical','future']
future_experiment = 'rcp60' 


variables   = ['riverstor']


start_year = 1896
end_year = 2025

years_isimip           = range(1861,2099,1)
years_grand            = range(1900,2010,1)
years_analysis         = range(start_year,end_year,1)

# define constants
resolution = 0.5   # degrees

# constants values to check
cp_liq = 4.188e3   #[J/kg K] heat capacity liquid water
cp_ice = 2.11727e3 #[J/kg K] heat capacity ice

rho_liq = 1000     #[kg/m³] density liquid water
rho_ice = 0.917e3  #[kg/m³] density ice

#%%
# preprocess river storage files
# see also preprocess_rivertemp.py file. 

if flag_preprocess: 
    print('Run calc_rivertemperatures.py manually (done on cluster)')

#%%
# Calculate river heat content


# based on river temperatures and river storage. (load both)
# see lake heat storage as an example. 

# define experiment
experiment= 'historical_'+future_experiment # can also be set to 'historical' or 'rcp60', but start_year will in this case have to be within year range

# for the different models
riverheat = {}
rivertemp = {}
riverstor = {}

# define names and paths for stream temperatures
var_streamtemp = 'streamtemp'                        
forcings_streamtemp = ['GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
outdir_streamtemp = outdir+'/'+var_streamtemp+'/'

# calculate grid area (for riverstorage conversion)
grid_area      = calc_grid_area(resolution)

for model in models:
    riverheat_model={} # sub directory for each model
    rivertemp_model={}
    riverstor_model={}
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
            rivermass_forcing = riverstor * grid_area 

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
# Save river heat for case study maps 

def calc_region_riverhc_ts(riverheat, region_props, indir_lakedata, flag_ref,years_analysis):
    """ Calculate the timeseries of the regions heat content """
 
    extent            = region_props['calc_extent']
    name              = region_props['name']


  # extract region lake heat from dictionary and apply weights 
    riverheat_region = extract_region(indir_lakedata,riverheat,extent)
    riverheat_region_anom =  calc_anomalies(riverheat_region, flag_ref,years_analysis)

    return riverheat_region_anom



if flag_saveriverheat_forAmazon: 

    riverheat_ensmean = ens_spmean_ensmean(riverheat)

    riverheat_pi = np.nanmean(riverheat_ensmean[0:30,:,:],axis=0)
    riverheat_pres = np.nanmean(riverheat_ensmean[-10:-1,:,:],axis=0)

    riverheat_anom_spmean = riverheat_pres - riverheat_pi

    np.save(outdir+'riverheat_anom_spmean.npy', riverheat_anom_spmean) 

    # extract area for timeseries
     
    # load necessary variables
    # Amazon region
    region_AM = {
        'extent'          : [-78,-10,-48,3.5], # original extent [27.5,-9,36,2.5]
        'calc_extent'     : [-78.25,-10.25,-48.25,3.75],
        'continent_extent': [-84,-33,-55,13],     # continent_extent for inset
        'ax_location'     : [0.6545, 0.22, 0.4, 0.2],
        'name'            : 'Amazon',       
        'name_str'        : 'Amazon river', 
        'levels'          : np.arange(0,8.5e17,0.5e17),
        'fig_size'         : (13,8),
        'cb_orientation'  : 'horizontal'
    }
    indir_lakedata   = basepath + 'data/isimip_laketemp/' # directory where lake fraction and depth are located

    riverheat_amazon_anom = calc_region_riverhc_ts(riverheat, region_AM, indir_lakedata, flag_ref,years_analysis)
    np.save(outdir+'riverheat_amazon_anom.npy', riverheat_amazon_anom) 




# riverheat_ts = timeseries(riverheat)
# riverheat_ens = ensmean(riverheat)
# riverheat_ensmean_ts = ensmean_ts(riverheat)
# riverheat_ensmin_ts = ensmin_ts(riverheat)
# riverheat_ensmax_ts = ensmax_ts(riverheat)

# riverheat_ens_spmean = ens_spmean(riverheat)

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
del riverheat
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


# rivertemp and river mass
rivertemp_anom_ts = timeseries_mean(rivertemp_anom)
rivermass_anom_ts = timeseries(rivermass_anom)


if flag_saveriverheat: 
    riverheat_anom_ensmean_ts = moving_average(ensmean_ts(riverheat_anom))
    riverheat_anom_ensmin_ts  = moving_average(ensmin_ts(riverheat_anom))
    riverheat_anom_ensmax_ts  = moving_average(ensmax_ts(riverheat_anom))
    riverheat_anom_std_ts     = moving_average(ensmax_ts(riverheat_anom))

    np.save(outdir+'riverheat_ensmean.npy', riverheat_anom_ensmean_ts) 
    np.save(outdir+'riverheat_ensmin.npy',riverheat_anom_ensmin_ts) 
    np.save(outdir+'riverheat_ensmax.npy',riverheat_anom_ensmax_ts) 
    np.save(outdir+'riverheat_std.npy', riverheat_anom_std_ts ) 
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




#%% Make supplementary figure 

var3 = riverheat_anom_ts
ylim3  = (-0.7e21,1.6e21)
ylabel3 = 'Energy [J]'
clr3 = 'coral'
figname = 'riverheat_per_forcing_2mods'

f,ax = plt.subplots(2,4, figsize=(13,6))
x_values = np.asarray(years_analysis)
labels = ['(m)','(n)','(o)','(p)','(q)','(r)','(s)','(t)']


ax = ax.ravel()

nplot = 0

for model in models: 

    for forcing in forcings:

        line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
        line1 = ax[nplot].plot(x_values,var3[model][forcing], color=clr3)
        ax[nplot].set_xlim(x_values[0],x_values[-1])
        ax[nplot].set_ylim(ylim3)
        ax[nplot].text(0.02, 0.90, labels[nplot], transform=ax[nplot].transAxes, fontsize=12)
        

        # only plot ylabel in first column 
        if (nplot/4).is_integer(): 
            ax[nplot].set_ylabel(ylabel3)

        # plot forcings only at the top row
        if nplot < 4: 
            ax[nplot].set_title(forcing, loc='right')
        nplot=nplot+1


    #f.suptitle(model+' river mass anomalies (reference 1900-1929)', fontsize=16)
f.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.text(-0.04, 0.91, models[0], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')       
plt.text(-0.04, 0.45, models[1], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')            

plt.savefig(plotdir+figname+'.png')


# %%
# plot river mass and heat 

# river mass
var = rivermass_anom_ts
ylim  = (-5.5e14,13e14)
ylabel = 'Mass [kg]'
clr = 'deepskyblue'

# river temperature
var2 = rivertemp_anom_ts[models[0]]
ylabel2 = 'Temperature [K]'
ylim2  = (-0.3,1)
clr2 = 'brown'


figname = 'rivermass_and_temp_per_forcing_2mods'


f,ax = plt.subplots(3,4, figsize=(12,8))
x_values = np.asarray(years_analysis)
labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)', '(l)']
ax = ax.ravel()

nplot = 0


# plot row of temperature
for forcing in forcings:
    line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
    line1 = ax[nplot].plot(x_values,var2[forcing], color=clr2)
    ax[nplot].set_xlim(x_values[0],x_values[-1])
    ax[nplot].set_ylim(ylim2)
    ax[nplot].text(0.02, 0.90, labels[nplot], transform=ax[nplot].transAxes, fontsize=12)
    ax[nplot].set_title(forcing, loc='right')
    if (nplot/4).is_integer(): 
        ax[nplot].set_ylabel(ylabel2)
    nplot = nplot+1


# plot two rows of river mass 
for model in models: 

    for forcing in forcings:

        line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
        line1 = ax[nplot].plot(x_values,var[model][forcing], color=clr)
        ax[nplot].set_xlim(x_values[0],x_values[-1])
        ax[nplot].set_ylim(ylim)
        ax[nplot].text(0.02, 0.90, labels[nplot], transform=ax[nplot].transAxes, fontsize=12)
        

        # only plot ylabel in first column 
        if (nplot/4).is_integer(): 
            ax[nplot].set_ylabel(ylabel)

        nplot=nplot+1


    #f.suptitle(model+' river mass anomalies (reference 1900-1929)', fontsize=16)
f.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.text(-0.065, 0.92, 'Punzet et al. (2012)', fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')            
plt.text(-0.065, 0.62, models[0], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')            
plt.text(-0.065, 0.31, models[1], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')       

plt.savefig(plotdir+figname+'.png')


#%%
