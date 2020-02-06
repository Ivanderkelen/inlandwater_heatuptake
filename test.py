# test script

# test sensitivity on remapping method for lake area pct. 

import xarray as xr 
import numpy as np 
from calc_grid_area import calc_grid_area
import matplotlib.pyplot as plt

# %% 
# import 


f1 = {'a':np.array([[[1,1],[1,1]],[[1,1],[1,1]]]),'b':np.array([[[2,2],[2,2]],[[1,1],[1,1]]])}
f2 = {'a':np.array([[[10,10],[10,10]],[[20,20],[20,20]]]),'b':np.array([[[10,10],[10,10]],[[10,10],[10,10]]])}

models = {'X':f1,'Y':f2}
models = {'X':f1}

res = ensmean_ts(models)
res = ensmean_ts(f1)

indict = f1

def ensmean_ts(indict):
    # ensemble mean timeseries
    ens_summed = {}
    tempdict = {}
for f in indict:
    tempdict[f] = np.nansum(indict[f],axis=(1,2))
stacked = np.stack(tempdict.values())
ens_summed[k] = np.nanmean(stacked,axis=0)

    stacked_per_model = np.stack(ens_summed.values())
    ensmean_allmodels = np.nanmean(stacked_per_model,axis=0)


def ensmean_ts(indict):
    # ensemble mean timeseries
    ens_summed = {}
    tempdict = {}
    for k in indict: 
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
        stacked = np.stack(tempdict.values())
        ens_summed[k] = np.nanmean(stacked,axis=0)

    stacked_per_model = np.stack(ens_summed.values())
    ensmean_allmodels = np.nanmean(stacked_per_model,axis=0)
    return ensmean_allmodels




lakearea = np.load('lakearea_bil.npy')

lakeheat_perarea =  np.load('lakeheat.npy')
lakeheat = np.load('/home/inne/documents/phd/scripts/python/calc_lakeheat_isimip/lakeheat.npy')
filename = '/home/inne/documents/phd/data/isimip_laketemp/'+ 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191220.nc'

ds_laketemp = xr.open_dataset(filename,decode_times=False)
xr.Variable(lat,lon,time,lakeheat)
                laketemp = ds_laketemp.watertemp.values
test = xr.Dataset({'lakeheat':(('time','lat','lon'),lakeheat)})
test.to_netcdf(path='test.nc')

test.lakeheat.isel(time=1).plot()


#%%
PATH='/home/inne/documents/phd/data/processed/isimip_lakeheat/watertemp/SIMSTRAT-UoG/'
FN = 'simstrat-uog_gfdl-esm2m_historical_rcp60_watertemp_interp_1861_2099_annual.nc4'

ds = xr.open_dataset(PATH+FN)

watertemp_1 = ds.watertemp.isel(time=1,levlak=1)
watertemp_1.plot() 

#%%
