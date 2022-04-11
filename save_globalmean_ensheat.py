#%%
# -------------------------------------------------------------------------
# SAVE global mean HEAT CONTENT timeseries
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
from dict_functions import *


# natural lakes, reservoirs and both
for scenario in ['climate','reservoirs','both']:
    lakeheat= np.load(outdir+'lakeheat_'+scenario+'.npy',allow_pickle='TRUE').item()
    


    if not scenario =='onlyresclimate':
        lakeheat_albm = load_lakeheat_albm(outdir,scenario,years_analysis)
        lakeheat.update(lakeheat_albm)
        del lakeheat_albm


    lakeheat_ensmean,stacked_per_model = ensmean_ts(lakeheat)
    lakeheat_std     = ens_std_ts(lakeheat)

    # manually correct missing values
    lakeheat['GOTM']['miroc5'][45:54,:,:] = np.nan
    for forcing in forcings: 
        lakeheat['ALBM'][forcing] = np.where(lakeheat['ALBM'][forcing] ==0,np.nan,lakeheat['ALBM'][forcing])


    data = np.stack([lakeheat_ensmean, lakeheat_std],axis=1)

    df = pd.DataFrame(data =data, index = years_analysis,columns=['Global mean heat storage [J]','Standard deviation heat storage [J]'] )
        #del lakeheat_anom, lakeheat

    if scenario == 'both': 
        fn = 'lake_and_reservoir_'
    elif scenario == 'reservoirs': 
        fn = 'reservoir_'
    elif scenario == 'climate': 
        fn = 'natural_lake_'

        #fn = 'inlandwater_'
    df.to_csv(outdir+'inlandwater_heatuptake_timeseries/'+fn+'heatstorage.dat')
    #return (anom_ensmean, anom_ensmin, anom_ensmax, anom_std)

# river heat
riverheat_ts_ensmean = np.load(outdir+'riverheat/riverheat_ts_ensmean.npy',allow_pickle='TRUE')
riverheat_ts_std = np.load(outdir+'riverheat/riverheat_ts_std.npy',allow_pickle='TRUE')

data = np.stack([riverheat_ts_ensmean, riverheat_ts_std],axis=1)

df = pd.DataFrame(data =data, index = years_analysis,columns=['Global mean heat storage [J]','Standard deviation heat storage [J]'] )
    #del lakeheat_anom, lakeheat

fn = 'river_'

    #fn = 'inlandwater_'
df.to_csv(outdir+'inlandwater_heatuptake_timeseries/'+fn+'heatstorage.dat')


#%% 
# -------------------------------------------------------------------------
# SAVE global mean annual HEAT FLUX timeseries
# -------------------------------------------------------------------------


# get lake area and calculate reservoir area
lake_area      = np.load(indir_lakedata+'lake_area.npy')
lake_area_ts = np.sum(lake_area, axis=(1,2))
res_area_ts = lake_area_ts - lake_area_ts[0]



for scenario in ['climate','reservoirs','both']:

    if scenario == 'climate': 
        area = lake_area_ts[0]
    elif scenario == 'reservoirs': 
        area = res_area_ts[:-1]
    elif scenario == 'both': 
        area = lake_area_ts[:-1]

    lakeheat= np.load(outdir+'lakeheat_'+scenario+'.npy',allow_pickle='TRUE').item()


    if not scenario =='onlyresclimate':
        lakeheat_albm = load_lakeheat_albm(outdir,scenario,years_analysis)
        lakeheat.update(lakeheat_albm)
        del lakeheat_albm

    #calculate ensemble mean heat flux
    heatflux_ensmean = calc_ensmean_heatflux(lakeheat,area,years_analysis)    

    # calculate standard deviation heat flux
    heatflux_std = ens_std_heatflux(lakeheat,area,years_analysis)

    data = np.stack([heatflux_ensmean, heatflux_std],axis=1)

    df = pd.DataFrame(data =data, index = years_analysis[:-1],columns=['Global mean annual heat flux [W/m²]','Standard deviation annual heat flux [W/m²]'] )
    
    if scenario == 'both': 
        fn = 'lake_and_reservoir_'
    elif scenario == 'reservoirs': 
        fn = 'reservoir_'
    elif scenario == 'climate': 
        fn = 'natural_lake_'

        #fn = 'inlandwater_'
    df.to_csv(outdir+'inlandwater_heatuptake_timeseries/'+fn+'heatflux.dat')


# river heat flux 
riverheat_heatflux_ensmean = np.load(outdir+'riverheat/riverheat_heatflux_ensmean.npy',allow_pickle='TRUE')
riverheat_heatflux_std = np.load(outdir+'riverheat/riverheat_heatflux_std.npy',allow_pickle='TRUE')

data = np.stack([riverheat_heatflux_ensmean, riverheat_heatflux_std],axis=1)

df = pd.DataFrame(data =data, index = years_analysis[:-1],columns=['Global mean annual heat flux [W/m²]','Standard deviation annual heat flux [W/m²]'] )
    #del lakeheat_anom, lakeheat

fn = 'river_'

    #fn = 'inlandwater_'
df.to_csv(outdir+'inlandwater_heatuptake_timeseries/'+fn+'heatflux.dat')


# %%
# -------------------------------------------------------------------------
# Check saved Heat flux and heat storage values
# -------------------------------------------------------------------------# %%

# heat storage
