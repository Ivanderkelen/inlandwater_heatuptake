
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : April 2020

Subroutine to calculate latent heat of fusion
    - load ice thickness
    - calculate latent heat of fusion per day
    - take annual average
    - calculate the multi model mean spatially.
"""


# information from main

import os
import sys
import xarray as xr
import numpy as np
import time

# flag to set which scenario is used for heat calculation
flag_scenario = 'climate'     # 'climate'    : only climate change (lake cover constant at 2005 level)
                              # 'reservoirs' : only reservoir construction (temperature constant at 1900 level)
                              # 'both'       : reservoir construction and climate

# Reference to which period/year anomalies are calculated
flag_ref = 'pre-industrial'  # 'pre-industrial': first 30 years (1900-1929 for start_year =1900)
                             # 1971 or any integer: year as a reference

# PATHS

indir  = '/gpfs/projects/climate/data/dataset/isimip/isimip2b/OutputData/lakes_global/'

outdir = '/scratch/brussel/100/vsc10055/isimip_lakeheat/'



# MODELS & FORCINGS
models      = ['CLM45'] #,'ALBM']#['CLM45','SIMSTRAT-UoG', 'ALBM']#,'VIC-LAKE','LAKE']
forcings    = ['miroc5'] # ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']
experiments = ['historical','future']#['historical','future']

# PERIODS
start_year = 1896
end_year = 2025
years_analysis         = range(start_year,end_year,1)

# CONSTANTS

resolution = 0.5 # degrees

# constants values to check
cp_liq = 4.188e3   # [J/kg K] heat capacity liquid water
cp_ice = 2.11727e3 # [J/kg K] heat capacity ice
cp_salt= 3.993e3   # [J/kg K] heat capacity salt ocean water (not used)
l_fus = 3.337e5    # [J/kg]  latent heat of future

rho_liq = 1000     # [kg/m3] density liquid water
rho_ice = 0.917e3  # [kg/m3] density ice

lake_area = np.load('lake_area.npy')

# take all lakes
lake_area = lake_area[-1,:,:]

print('Calculating time series of annual heat of fusion ...')
for model in models:
    print('Processing '+model)

    # make output directory per model if not done yet
    outdir_model = outdir+'iceheat/'+model+'/'
    if not os.path.isdir(outdir_model):  os.system('mkdir '+outdir_model)

    for forcing in forcings:
        start_time = time.time()

        # initiate empty list for iceheat dataset.
        iceheat_list = []

        for experiment in experiments:
        # differentiate for future experiments filenames
            if experiment == 'future':
                experiment_fn = 'rcp60'
                period = '2006_2099'
                period_daily = ['2006_2010','2011_2020']

            elif experiment == 'historical':
                experiment_fn = experiment
                period = '1861_2005'
                period_daily = ['1891_1900','1901_1910','1911_1920','1921_1930','1931_1940','1941_1950',
                                        '1951_1960','1961_1970','1971_1980','1981_1990','1991_2000','2001_2005']


            path = indir+model+'/'+forcing+'/'+experiment+'/'

            for p in period_daily:
                print('Processing years '+p+' for '+forcing)

                # define input filename for the different models
                if model == 'CLM45':
                    infile_icefrac = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_lakeicefrac_'+'global'+'_'+'daily_'+p+'.nc4'

                    # open icefrac file
                    ds_icefrac = xr.open_dataset(path+infile_icefrac)
                    ds_icefrac_ymean = ds_icefrac.groupby('time.year').mean(dim='time')
                    icefrac = ds_icefrac_ymean['lakeicefrac']
                    del ds_icefrac, ds_icefrac_ymean

                    layer_thickness_clm= np.array([0.1, 1, 2, 3, 4, 5, 7, 7, 10.45, 10.45]) # m
                    layer_thickness = layer_thickness_clm[np.newaxis,:,np.newaxis,np.newaxis]
                    # calculate resulting icethickness taking ice fraction into account
                    icethick_per_layer = icefrac * layer_thickness
                    icethick = icethick_per_layer.sum(dim = 'levlak') # specify dimension over which to sum.

                    del icethick_per_layer

                    # calculate ice heat [J] = [m] * [J/kg] * [m^2] * [kg/m^3]
                    iceheat = icethick * l_fus * np.flip(lake_area,axis=0) * rho_ice
                    iceheat_ymean = iceheat.mean(dim=('lon','lat'))
                    
                    		    
                    # clean up
                    del icethick


                elif model == 'SIMSTRAT-UoG':

                    infile_icethick  = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_icethick_'+'global'+'_'+'daily_'+p+'.nc4'
                    infile_icefrac   = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_lakeicefrac_'+'global'+'_'+'daily_'+p+'.nc4'

                    # open icethickness file
                    ds_icethick = xr.open_dataset(path+infile_icethick)
                    icethick_only = ds_icethick['icethick']

                    # open icefrac file
                    ds_icefrac = xr.open_dataset(path+infile_icefrac)
                    icefrac = ds_icefrac['lakeicefrac']

                    # calculate resulting icethickness taking ice fraction into account
                    icethick = icefrac*icethick_only

                    del icefrac, icethick_only, ds_icefrac, ds_icethick

                    # calculate ice heat [J] = [m] * [J/kg] * [m^2] * [kg/m^3]
                    iceheat = icethick * l_fus * lake_area * rho_ice

                    # clean up
                    del icethick

                    # calculate annual mean ice heat
                    iceheat_ymean = iceheat.groupby('time.year').mean(dim=('time','lon','lat'))

                elif model == 'ALBM':
                    infile_icethick = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_icethick_'+'global'+'_'+'daily_'+p+'.nc4'

                    # open icethickness file
                    ds_icethick = xr.open_dataset(path+infile_icethick)
                    icethick = ds_icethick['icethick']

                    # calculate ice heat [J] = [m] * [J/kg] * [m^2] * [kg/m^3]
                    iceheat = icethick * l_fus * lake_area * rho_ice

                    # clean up
                    del icethick, ds_icethick

                    # calculate annual mean ice heat
                    iceheat_ymean = iceheat.groupby('time.year').mean(dim=('time','lon','lat'))

                    del iceheat
                # turn annual means into dataset and append to list
                iceheat_ds = iceheat_ymean.to_dataset(name='iceheat')
                iceheat_list.append(iceheat_ds)

        # concatenate all years
        iceheat_concat = xr.concat(iceheat_list,dim="year")

        # save ice heat in nc file per forcing.
        outfile = model.lower()+'_'+forcing+'_historical_'+experiment_fn+'_iceheat_'+'1891_2020'+'_'+'annual.nc4'
        iceheat_concat.to_netcdf(outdir+'iceheat/'+model+'/'+outfile, 'w')

        # print time
        print("--- %s minutes---" %(time.time() - start_time))


