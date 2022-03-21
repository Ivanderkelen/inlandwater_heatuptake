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


def calc_depth_per_layer(flag_scenario, indir_lakedata, years_grand, start_year,end_year, resolution, model,outdir):

    """Calculate the depth per lake layer (depending on model used. ) """

    # GLDB lake depths (GLDB lake depths, hydrolakes lake area)
    lakedepth_path        = indir_lakedata + 'dlake_1km_ll_remapped_0.5x0.5.nc'
    gldb_lakedepth        = xr.open_dataset(lakedepth_path)
    lake_depth            = gldb_lakedepth.dl.values[0,:,:]

    # Define different lake layer thicknesses (here the other lake models will be added )
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
    else:
        # load just one annual watertemp file with lake levels. 
        variable = 'watertemp'                        
        outdir_model = outdir+variable+'/'+model+'/'
        if model == 'GOTM': 
            outfile_annual = model.lower()+'_gfdl-esm2m_historical_rcp60_'+variable+'_1861_2099_annual.nc4'

        else: 
            outfile_annual = model.lower()+'_hadgem2-es_historical_rcp60_'+variable+'_1861_2099_annual.nc4'

        ds_lakelev = xr.open_dataset(outdir_model+outfile_annual,decode_times=False)
        lakelevdepth = ds_lakelev.depth.values
        # not necessary to flip here. 
        #lakelevdepth_nonflipped = ds_lakelev.depth.values
        #lakelevdepth = np.flip(lakelevdepth_nonflipped,axis=1)

        # assume layer depth is at middle of layer
        layer_thickness = np.empty((np.size(lakelevdepth,0),np.size(lake_depth,0),np.size(lake_depth,1)))
        for lev in range(1,np.size(lakelevdepth,0)):
            if lev == 0: 
                layer_thickness[lev,:,:] = lakelevdepth[lev,:,:]*2 # why is this multiplied by 2????
            else: 
                layer_thickness[lev,:,:] = (lakelevdepth[lev,:,:]-lakelevdepth[lev-1,:,:])*2

        # convert lake level depth to lake layer thickness
        layer_thickness_rel = layer_thickness/np.nansum(layer_thickness, axis=0)

    # expand lake depth dataset to also account for lake layers
    depth_per_layer     = layer_thickness_rel * lake_depth
    
    return depth_per_layer, layer_thickness_rel



## OLD function
def calc_lakeheat_area(resolution, indir_lakedata, flag_scenario, lakeheat_perarea,years_grand, start_year,end_year):   
    
    # see script test_resarea_sensitivity for sensitivity tests on different input datasets. 
    # old file


    # lakepct_path          = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1900-2000_c20190826.nc'
    lakepct_path       = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191220.nc'
    hydrolakes_lakepct = xr.open_dataset(lakepct_path)
    lake_pct           = hydrolakes_lakepct.PCT_LAKE.values/100 # to have then in fraction

    # Extract period of lake pct file 
    end_year_res = 2017

    # take analysis years of lake_pct
    lake_pct  = lake_pct[years_grand.index(start_year):years_grand.index(end_year_res), :, :]


    lake_const=lake_pct[-1,:,:]
    lake_const = lake_const[np.newaxis,:,:]
    for ind,year in enumerate(np.arange(end_year_res,end_year)):
        lake_pct= np.append(lake_pct,lake_const,axis=0)

    # calculate lake area per grid cell (m²)
    grid_area      = calc_grid_area(resolution)
    lake_area      = lake_pct * grid_area  
    # np.save('lake_area.npy',lake_area)
    

    # improved volume estimate comes here
    # assumption of lake volume as reversed wedding pie
    # area_per_layer =  calc_area_per_layer(layer_thickness_rel,lake_area,volume_development)


    # lake_area = np.load('lake_area.npy')
    # take lake area constant at 1900 level.
    if flag_scenario == 'climate': # in this scenario, volume per layer has only 3 dimensions

        lake_area_endyear = lake_area[0,:,:]
        lakeheat_total = lakeheat_perarea * lake_area_endyear

    else:
        #lakeheat_total= lakeheat_perarea * lake_area
        lakeheat_total = lakeheat_perarea * lake_area

    return lakeheat_total


# calculate area per layer (corresponding to depth)
# see paper of Johannson et al., 2007 
def calc_area_per_layer(layer_thickness_rel,lake_area,flag_volume):

    # extend lake area to also include depth dimension. 
    if np.ndim(lake_area) == 3: # with time dimension
        lake_area_3d = np.repeat(lake_area[:,:, :, np.newaxis], len(layer_thickness_rel), axis=3)
        lake_area_3d = np.moveaxis(lake_area_3d,3,1)

    else: #without time dimension
        lake_area_3d = np.repeat(lake_area[:, :, np.newaxis], len(layer_thickness_rel), axis=2)
        lake_area_3d = np.moveaxis(lake_area_3d,2,0)

    area_per_layer = np.empty_like(lake_area_3d)
    if flag_volume == 'cylindrical': # this means, the area is the same for every layer
        area_per_layer = lake_area_3d

    # here the other options with truncated cone can be inserted, as well as calculations on Vd values. 
    elif not isinstance(flag_volume,str): # flag_volume == Vd 
        Vd = flag_volume
        f_Vd = 1.7*Vd**(-1)+2.5-2.4*Vd+0.23*Vd**3

        for i,single_thickness in enumerate(layer_thickness_rel):
            print(i)
            if np.ndim(lake_area) == 3:
                area_per_layer[:,i,:,:] = lake_area* ((1-single_thickness)*(1+single_thickness*np.sin(np.sqrt(single_thickness))))**(f_Vd)
            else: 
                area_per_layer[i,:,:] = lake_area* ((1-single_thickness)*(1+single_thickness*np.sin(np.sqrt(single_thickness))))**(f_Vd)

    return area_per_layer


def load_lakearea(resolution, indir_lakedata, years_grand, start_year,end_year,flag_scenario): 
    
    # load the lake area from hydrolakes

    # lakepct_path          = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1900-2000_c20190826.nc'
    lakepct_path       = indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191220.nc'
    hydrolakes_lakepct = xr.open_dataset(lakepct_path)
    lake_pct           = hydrolakes_lakepct.PCT_LAKE.values/100 # to have then in fraction

    # Extract period of lake pct file 
    end_year_res = 2017

    # take analysis years of lake_pct
    lake_pct  = lake_pct[years_grand.index(start_year):years_grand.index(end_year_res), :, :]


    lake_const=lake_pct[-1,:,:]
    lake_const = lake_const[np.newaxis,:,:]
    for ind,year in enumerate(np.arange(end_year_res,end_year)):
        lake_pct= np.append(lake_pct,lake_const,axis=0)

    # calculate lake area per grid cell (m²)
    grid_area      = calc_grid_area(resolution)
    lake_area      = lake_pct * grid_area 

    if flag_scenario == 'climate': # in this scenario, lake area has no time dimension

        lake_area = lake_area[0,:,:] # lake area in end year


    return lake_area

def calc_volume_per_layer(flag_scenario, resolution, indir_lakedata, years_grand, start_year,end_year, model,outdir, flag_volume): 

    # load depth per layer and relative depth per layer
    depth_per_layer, layer_thickness_rel = calc_depth_per_layer(flag_scenario, indir_lakedata, years_grand, start_year,end_year, resolution, model,outdir)

    # load lake area 
    lake_area = load_lakearea(resolution, indir_lakedata, years_grand, start_year,end_year,flag_scenario)

    area_per_layer = calc_area_per_layer(layer_thickness_rel, lake_area, flag_volume)


    # for now, assume cylindrical volume per layer. 
    volume_per_layer = area_per_layer * depth_per_layer 

    return volume_per_layer