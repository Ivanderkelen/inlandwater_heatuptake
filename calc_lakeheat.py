
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Subroutine to calculate lake heat content per grid cell 
    - calculates lake heat per layer, and make weighted sum
    - saves output in dictionary per model, per forcing in lake_heat
    - is saved in variable when flag is on 

"""

import os 
import xarray as xr
import numpy as np
from calc_volumes import * 


def calc_lakeheat(models,forcings,future_experiment, indir_lakedata, years_grand, resolution,outdir, years_isimip,start_year, end_year, flag_scenario, flag_savelakeheat, rho_liq, cp_liq, rho_ice, cp_ice):

    # define experiment
    experiment= 'historical_'+future_experiment # can also be set to 'historical' or 'rcp60', but start_year will in this case have to be within year range

    lakeheat = {}

    for model in models:
        lakeheat_model={} # sub directory for each model
        
        # calculate depth per layer 
        depth_per_layer = calc_depth_per_layer(flag_scenario, indir_lakedata, years_grand, start_year,end_year, resolution, model,outdir)

        for forcing in forcings:

            # define directory and filename
            variables = ['watertemp']#, 'lakeicefrac', 'icetemp']
            outdir_model = {}
            outfile_annual = {}

            for variable in variables:
                variable_fn = variable+'_interp' if variable == 'watertemp' else variable   
                if not outdir_model: 
                    outdir_model = {variable:outdir+variable+'/'+model+'/'}
                else:
                    outdir_model.update({variable:outdir+variable+'/'+model+'/'})      
                
                if not outfile_annual:
                    outfile_annual = {variable:model.lower()+'_'+forcing+'_'+experiment+'_'+variable_fn+'_1861_2099'+'_'+'annual'+'.nc4'} 
                else: 
                    outfile_annual.update({variable:model.lower()+'_'+forcing+'_'+experiment+'_'+variable_fn+'_1861_2099'+'_'+'annual'+'.nc4'})

            # if simulation is available
            print(outdir_model['watertemp']+outfile_annual['watertemp'])
            if os.path.isfile(outdir_model['watertemp']+outfile_annual['watertemp']): 
                print('Calculating lake heat of '+ model + ' ' + forcing)
                
                # open lake heat variable 
                ds_laketemp = xr.open_dataset(outdir_model['watertemp']+outfile_annual['watertemp'],decode_times=False)
                laketemp = ds_laketemp.watertemp.values
                print('Lake temps opened')
                # open ice fraction variable
               # ds_icefrac = xr.open_dataset(outdir_model['lakeicefrac']+outfile_annual['lakeicefrac'],decode_times=False)
               # icefrac = ds_icefrac.lakeicefrac.values

                # insert here icetemp with designated flag (or based on model)

                if flag_scenario == 'reservoirs': 
                    # use lake temperature from first year of analysis
                    laketemp = laketemp[years_isimip[model].index(start_year),:,:,:]
                   
                   # icefrac = icefrac[years_isimip[model].index(start_year),:,:,:]

                else: 
                    # extract years of analysis
                    laketemp = laketemp[years_isimip[model].index(start_year):years_isimip[model].index(end_year),:,:,:]
                    #icefrac = icefrac[years_isimip[model].index(start_year):years_isimip[model].index(end_year),:,:,:]

                #lakeheat_layered =  ((rho_liq * cp_liq * (1-icefrac)) + (rho_ice * cp_ice * icefrac) )* depth_per_layer* laketemp
                lakeheat_layered =  rho_liq * cp_liq * depth_per_layer * laketemp

                # add manual time dimension for reservoir scenario. 
                if flag_scenario == 'reservoirs': lakeheat_layered = np.expand_dims(lakeheat_layered,axis=0)

                # create a 2D mask to only give 0 to grid cells where there are or will be lakes (saving memory) 
                mask_nan = np.isnan(lakeheat_layered[-1,:,:,:]).sum(axis=0)!=13
                # sum up for total layer (less memory)
                lakeheat_perarea = np.empty([lakeheat_layered.shape[0],lakeheat_layered.shape[2],lakeheat_layered.shape[3]])
                for i in range(lakeheat_layered.shape[2]):
                    for j in range(lakeheat_layered.shape[3]):
                        if mask_nan[i,j]: 
                            lakeheat_perarea[:,i,j] = np.nansum(lakeheat_layered[:,:,i,j],axis=1)
                        else: 
                            lakeheat_perarea[:,i,j] = np.nan

                #np.save('lakeheat_perarea.npy', np.flip(lakeheat_perarea,axis=1))

                lakeheat_forcing = calc_lakeheat_area(resolution, indir_lakedata, flag_scenario,  np.flip(lakeheat_perarea,axis=1), years_grand,start_year,end_year)
               
                # clean up
                del laketemp, ds_laketemp, lakeheat_layered, lakeheat_perarea, icefrac, ds_icefrac

                # save lakeheat in directory structure per forcing
            if not lakeheat_model:
                lakeheat_model = {forcing:lakeheat_forcing}
            else: 
                lakeheat_model.update({forcing:lakeheat_forcing})

            del outdir_model
            del outfile_annual 

        # save lakeheat of forcings in directory structure per model
        if not lakeheat:
            lakeheat = {model:lakeheat_model}
        else:
            lakeheat.update({model:lakeheat_model})    
        
        # clean up   
        del lakeheat_model, lakeheat_forcing
    # save calculated lake heat (this needs to be cleaned up before continuing working on code.)

    # Save according to scenario flag
    if flag_savelakeheat:
        lakeheat_filename = 'lakeheat_'+flag_scenario+'.npy'
        np.save(outdir+lakeheat_filename, lakeheat) 

    return lakeheat
