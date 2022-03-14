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

project_name = 'isimip_lakeheat/'
basepath = '/home/inne/documents/phd/'

indir  = basepath + 'data/ISIMIP/OutputData/lakes_global/'

outdir = basepath + 'data/processed/'+ project_name
plotdir= basepath + 'data/processed/'+ project_name+ '/plots/'
indir_lakedata   = basepath + 'data/isimip_laketemp/' # directory where lake fraction and depth are located



# MODELS & FORCINGS

models      = ['SIMSTRAT-UoG']#['CLM45','SIMSTRAT-UoG', 'ALBM']#,'VIC-LAKE','LAKE']
forcings    = ['gfdl-esm2m']#['hadgem2-es','ipsl-cm5a-lr','miroc5']
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
l_fus = 3.337e5    # [J/kg]  latent heat of fusion

rho_liq = 1000     # [kg/m3] density liquid water
rho_ice = 0.917e3  # [kg/m3] density ice

lake_area_all = np.load('/home/inne/documents/phd/data/isimip_laketemp/lake_area.npy')

# take all lakes
lake_area = lake_area_all[-1,:,:]

del lake_area_all

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

                    del ds_icefrac, icethick_per_layer

                    # calculate ice heat [J] = [m] * [J/kg] * [m^2] * [kg/m^3]
                    iceheat_ymean = icethick * l_fus * lake_area * rho_ice

                    # clean up 
                    del icethick


                elif model == 'SIMSTRAT-UoG':
         
                    infile_icethick  = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_icethick_'+'global'+'_'+'daily_'+p+'.nc4'
                    infile_icefrac   = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_lakeicefrac_'+'global'+'_'+'daily_'+p+'.nc4'
                   
                    # open icethickness file
                    ds_icethick = xr.open_dataset(path+infile_icethick)
                    icethick = ds_icethick['icethick']

                    del ds_icethick

                    # open icefrac file
                    ds_icefrac = xr.open_dataset(path+infile_icefrac)
                    icefrac = ds_icefrac['lakeicefrac']
                    del ds_icefrac

                    icefrac_ymean = icefrac.groupby('time.year').mean(dim=('time'))

                    icethick_ymean = icethick.groupby('time.year').mean(dim=('time'))

                    # calculate difference with previous timestep (positive = ice melted)
                    delta_icethick = icethick_ymean.diff('year')

                    del icethick, icefrac

                    # calculate ice heat [J] = [m] * [J/kg] * [m^2] * [kg/m^3]
                    iceheat = icethick_ymean * icefrac_ymean * l_fus * lake_area * rho_ice

                    # calculate annual mean ice heat 
                    iceheat_spmean = iceheat.mean(dim=('lon','lat'))
                    del iceheat
                
                elif model == 'ALBM':
                    infile_icethick = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_icethick_'+'global'+'_'+'daily_'+p+'.nc4'
                   
                    # open icethickness file
                    ds_icethick = xr.open_dataset(path+infile_icethick)
                    icethick = ds_icethick['icethick']
                    icethick_ymean = icethick.groupby('time.year').mean(dim=('time'))
                    
                    del icethick, ds_icethick

                    # calculate difference with previous timestep (positive = ice melted)
                    delta_icethick = icethick_ymean.diff('year')

                    # calculate ice heat [J] = [m] * [J/kg] * [m^2] * [kg/m^3]
                    iceheat = icethick_ymean * l_fus * lake_area * rho_ice

                    # calculate annual mean ice heat 
                    iceheat_ymean = iceheat.groupby('time.year').mean(dim=('time','lon','lat'))
                
                    del iceheat        


                # turn annual means into dataset and append to list
                iceheat_ds = iceheat_spmean.to_dataset(name='iceheat')
                iceheat_list.append(iceheat_ds)

        
        # concatenate all years
        iceheat_concat = xr.concat(iceheat_list,dim="year")
        icethick_concat = xr.concat(icethick_list,dim="year")
        # save ice heat in nc file per forcing. 
        outfile = model.lower()+'_'+forcing+'_historical_'+experiment_fn+'_iceheat_'+'1891_2020'+'_'+'annual.nc4'
        iceheat_concat.to_netcdf(outdir+'iceheat/'+outfile, 'w')        
                
        # print time
        print("--- %s seconds---" %(time.time() - start_time))
        
