"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : April 2020

Subroutine to merge latent heat of fusion and calculate numbers 

"""


# information from main 

import os 
import sys
import xarray as xr
import numpy as np
import time 
import pandas as pd
import csv as csv
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
outdir = basepath + 'data/processed/'+ project_name

indir = basepath + 'data/processed/isimip_lakeheat/iceheat/'

# MODELS & FORCINGS

models      = ['SIMSTRAT-UoG','ALBM', 'CLM45']#['CLM45','SIMSTRAT-UoG', 'ALBM']#,'VIC-LAKE','LAKE']
forcings    = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']

# PERIODS
start_year = 1896
end_year = 2025
years_analysis         = range(start_year,end_year,1)


# --------------------------------------
# load file per model and save in dict
iceheat = {}
tot_list = []
comp_list = []
model_list = []

for model in models:
    
    iceheat_model = {}

    for forcing in forcings:

        filename = model.lower()+'_'+forcing+'_historical_rcp60_iceheat_'+'1891_2020'+'_'+'annual.nc4'
        iceheat_forcing_ds = xr.open_dataset(indir+model+'/'+filename) 
        iceheat_forcing = iceheat_forcing_ds['iceheat'].values

        tot_iceheat = iceheat_forcing[0] - iceheat_forcing[-1]

        comp_iceheat = iceheat_forcing[0:29].mean() - iceheat_forcing[-20:-1].mean()

        tot_list.append(tot_iceheat)
        comp_list.append(comp_iceheat)
        model_list.append(model+' '+forcing)

        if not iceheat_model:
            iceheat_model = {forcing:iceheat_forcing}
        else: 
            iceheat_model.update({forcing:iceheat_forcing})

    # save lakeheat of forcings in directory structure per model
    if not iceheat:
        iceheat = {model:iceheat_model}
    else:
        iceheat.update({model:iceheat_model})    


# ------------------
# get values


# multimodel mean and std

tot = np.array(tot_list) 
tot_mean = tot.mean()
tot_std = tot.std()


comp = np.array(comp_list)
comp_mean = comp.mean()
comp_std = comp.std()
print('mean ')

# create pandas dataframe with results
results_dict = {'Ice heat over whole period [J]':tot_mean,'Std': tot_std, 'Comparable to period [J]': comp_mean, 'Std_comp':comp_std}
a_file  = open(outdir+'iceheat.csv','w')

writer = csv.writer(a_file)
for key,value in results_dict.items():
    writer.writerow([key,value])
a_file.close()


natlakes = 2.9e20+0.06e20

frac = tot_mean/natlakes