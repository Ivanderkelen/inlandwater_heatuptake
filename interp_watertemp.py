"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : October 2019

Scripts for interpolating CLM45 lake temperatures

"""
#%%
from cdo import Cdo
cdo = Cdo()
import os 
import xarray as xr
import numpy as np
from calc_grid_area   import calc_grid_area
from scipy.interpolate import griddata



def interp_watertemp(indir_lakedata,outdir,forcings,future_experiment,model): 

    # interpolation function
    def interpolate_nn(a,mask):
        """ Function to interpolate a 3D numpy array with nearest neighbour method """
        # get mask of nan values (where both mask is True and a has nan)
        mask_nan = np.where(np.isnan(a) & mask,np.nan,0)

        x, y, z = np.indices(a.shape)
        interp = np.array(a)

        interp[np.isnan(mask_nan)] = griddata(
            (x[~np.isnan(a)], y[~np.isnan(a)], z[~np.isnan(a)]), # points we know
            a[~np.isnan(a)], # values we know
            (x[np.isnan(mask_nan)], y[np.isnan(mask_nan)], z[np.isnan(mask_nan)]) # values we want to know
            ,method='nearest')     
        interp = np.where(interp==0,np.nan,interp)  
 
        return interp

    # make mask based on where there are lakes
    lakepct_path        = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191220.nc'
    hydrolakes_lakepct  = xr.open_dataset(lakepct_path)
    lake_pct            = hydrolakes_lakepct.PCT_LAKE.values # to have then in fraction
    mask = np.flipud(np.where(lake_pct[-1,:,:]>0,True,False))


    for forcing in forcings:
        
        outdir_mod = outdir+'/watertemp/'+model+'/'
        outfile=  model.lower()+'_'+forcing+'_'+'historical_'+future_experiment+'_watertemp_'+'1861_2099'+'_'+'annual'+'.nc4'
        outfile_interp= model.lower()+'_'+forcing+'_'+'historical_'+future_experiment+'_watertemp_interp_'+'1861_2099'+'_'+'annual'+'.nc4'


        if not os.path.isfile(outdir_mod+outfile_interp): 
            print('Interpolating '+model +' ' + forcing)

            # load the clm lake temperature 
            ds_laketemp = xr.open_dataset(outdir_mod+outfile,decode_times=False)
            laketemp = ds_laketemp.watertemp.values

            # do interpolation

            for i in range(0,laketemp.shape[0]):
                print('timestep '+str(i))
                # apply mask 
                # interpolate
                laketemp[i,:,:,:] = interpolate_nn(laketemp[i,:,:,:],mask)
            
            # save interpolation back again in netcdf

            lat    = ds_laketemp['lat'].values
            lon    = ds_laketemp['lon'].values
            levlak = ds_laketemp['levlak'].values
            time   = ds_laketemp['time'].values

            laketemp_interp =  xr.DataArray(laketemp, coords={'lat': lat, 'lon': lon, 
                                    'levlak': levlak, 'time':time},
                dims=['time','levlak', 'lat', 'lon'])

            ds_laketemp['watertemp']=laketemp_interp
            ds_laketemp.to_netcdf(outdir_mod+outfile_interp)

            # clean up
            del ds_laketemp, laketemp_interp, laketemp
        else: 
            print(model +' ' + forcing+' is already interpolated. ')


