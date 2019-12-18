"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Subroutine to calculate lake volumes for every lake layer
    - load lake depth and lake fraction
    - calculate volume per layer
    - to be extended for ice
"""


from calc_grid_area    import calc_grid_area
import numpy as np
import xarray as xr


# calculate volumes per layer for different timesteps (area)
# based on relative layer depths and depth of GLDB
def calc_volume_per_layer(flag_scenario, indir_lakedata, years_grand, start_year,end_year, resolution, models,outdir):

    # GLDB lake depths (GLDB lake depths, hydrolakes lake area)
    lakedepth_path        = indir_lakedata + 'dlake_1km_ll_remapped_0.5x0.5.nc'
    lakepct_path          = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191203.nc'

    # load variables
    gldb_lakedepth      = xr.open_dataset(lakedepth_path)
    hydrolakes_lakepct  = xr.open_dataset(lakepct_path)

    lake_depth         = gldb_lakedepth.dl.values[0,:,:]
    lake_pct           = hydrolakes_lakepct.PCT_LAKE.values/100 # to have then in fraction
    landmask           = hydrolakes_lakepct.LANDMASK.values

    lon = hydrolakes_lakepct.PCT_LAKE.lon
    lat = hydrolakes_lakepct.PCT_LAKE.lat

    # this should be removed when updating new reservoir file
    end_year_res = 2000

    # take analysis years of lake_pct
    lake_pct  = lake_pct[years_grand.index(start_year):years_grand.index(end_year_res), :, :]

    # this should be removed when updating new reservoir file
    # extend lake_pct data set with values further than 2000: 
    lake_const=lake_pct[years_grand.index(end_year_res-1),:,:]
    lake_const = lake_const[np.newaxis,:,:]
    for ind,year in enumerate(np.arange(end_year_res,end_year)):
        lake_pct= np.append(lake_pct,lake_const,axis=0)


    # calculate lake area per grid cell (m²)
    grid_area      = calc_grid_area(resolution)
    lake_area      = lake_pct * grid_area  

    volume_per_layer = {}
    
    # Define different lake layer thicknesses (here the other lake models will be added )
    for model in models:
        if model == 'CLM45':
            layer_thickness_clm = np.array([0.1, 1, 2, 3, 4, 5, 7, 7, 10.45, 10.45]) # m
            layer_depth_clm = np.array([0.05, 0.6, 2.1, 4.6, 8.1, 12.6, 18.6 , 25.6, 34.325, 44.775]) #m

            # field of layer thickness of clm
            layer_thickness_clm_sp = np.empty((len(layer_thickness_clm),np.size(lake_depth,0),np.size(lake_depth,1)))
            # cut off layer depth if larger then GLDB given depth
            for ind,layerdepth in enumerate(layer_depth_clm):
                layer_thickness_clm_sp[ind,:,:] = np.where(lake_depth>=layerdepth,layerdepth,np.nan)

            layer_thickness_rel = layer_thickness_clm/np.sum(layer_thickness_clm)
             
            layer_thickness_rel = np.expand_dims(layer_thickness_rel,axis=1)
            layer_thickness_rel = np.expand_dims(layer_thickness_rel,axis=2)


        # add other models in here
        elif model == 'SIMSTRAT-UoG':
            # load just one annual watertemp file with lake levels. 
            variable = 'watertemp'                        
            outdir_model = outdir+variable+'/'+model+'/'
            outfile_annual = model.lower()+'_hadgem2-es_historical_rcp60_'+variable+'_1861_2099_annual.nc4'
            ds_lakelev = xr.open_dataset(outdir_model+outfile_annual,decode_times=False)
            lakelevdepth = ds_lakelev.depth.values
            # assume layer depth is at middle of layer
            layer_thickness = np.empty((np.size(lakelevdepth,0),np.size(lake_depth,0),np.size(lake_depth,1)))
            for lev in range(1,np.size(lakelevdepth,0)):
                if lev == 0: 
                    layer_thickness[lev,:,:] = lakelevdepth[lev,:,:]*2
                else: 
                    layer_thickness[lev,:,:] = (lakelevdepth[lev,:,:]-lakelevdepth[lev-1,:,:])*2

            # convert lake level depth to lake layer thickness
            layer_thickness_rel = layer_thickness/np.nansum(layer_thickness, axis=0)

        lake_depth_expanded = np.expand_dims(lake_depth,axis=0)


        # take lake area constant at 1900 level.
        if flag_scenario == 'climate' : # in this scenario, volume per layer has only 3 dimensions

            depth_per_layer  = layer_thickness_rel * lake_depth_expanded
            lake_area_endyear = lake_area[0,:,:]
            volume_per_layer[model] = depth_per_layer * lake_area_endyear

        else:

            depth_per_layer = layer_thickness_rel * lake_depth_expanded
            depth_per_layer = np.expand_dims(depth_per_layer,axis=0)
            lake_area_expanded = np.expand_dims(lake_area,axis=1)
            volume_per_layer[model] = depth_per_layer * lake_area_expanded

    return volume_per_layer




