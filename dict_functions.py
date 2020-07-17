"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Scripts for dictionary isismip models 

"""
import numpy as np 
import xarray as xr

# for cellarea functions
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt
import pandas as pd
import os
from osgeo import gdal
from load_lakeheat_albm import  *

# ------------------------------------------------------------------------
# Data Agggregation functions

def timeseries(indict):
    # calculate global timeseries of total lake heat of ensemble members (dict in dict)
    outdict = {}
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
            tempdict[f][tempdict[f] == 0.0] = np.nan
        outdict[k] = tempdict
    return outdict

def timeseries_mean(indict):
    # calculate global timeseries of total lake heat of ensemble members (dict in dict)
    outdict = {}
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nanmean(indict[k][f],axis=(1,2))
        outdict[k] = tempdict
    return outdict


def ensmean(indict):
    # calculate ensemble mean of dictionary in dictionary
    outdict = {}
    for k in indict: 
        stacked = np.stack(indict[k].values())
        outdict[k] = np.nanmean(stacked,axis=0)
    return outdict

def ensmean_ts_per_model(indict):
    # ensemble mean timeseries (mean from all forcings, per model)
    outdict = {}
    tempdict = {}
    for k in indict: 
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
        stacked = np.stack(tempdict.values())
        outdict[k] = np.nanmean(stacked,axis=0)
    return outdict

def ensmean_ts(indict):
    # ensemble mean timeseries
    ens_summed = {}
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
            tempdict = cor_for_albm(tempdict,k,f)
        stacked = np.stack(tempdict.values())
        ens_summed[k] = np.nanmean(stacked,axis=0)

    stacked_per_model = np.stack(ens_summed.values())
    ensmean_allmodels = np.nanmean(stacked_per_model,axis=0)
    return ensmean_allmodels

def ens_std_ts(indict):
    # calculate standard deviation of timeseries
    concat_stacked = np.array([])
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
            tempdict = cor_for_albm(tempdict,k,f)
        stacked = np.stack(tempdict.values())
        # put all forcings of models together. 
        if concat_stacked.size == 0:
            concat_stacked = stacked
        else:
            concat_stacked = np.concatenate((concat_stacked,stacked),axis=0)
    # calculate average over ensemble members
    ens_std = np.nanstd(concat_stacked,axis=0)

    return ens_std


def ensmin_ts_per_model(indict):
    # ensemble minimum timeseries
    outdict = {}
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
        stacked = np.stack(tempdict.values())
        outdict[k] = np.nanmin(stacked,axis=0)
    return outdict


def ensmin_ts(indict):
    # ensemble minimum timeseries
    ens_summed = {}
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
            tempdict = cor_for_albm(tempdict,k,f)
        stacked = np.stack(tempdict.values())
        ens_summed[k] = np.nanmin(stacked,axis=0)

    stacked_per_model = np.stack(ens_summed.values())
    ensmean_allmodels = np.nanmin(stacked_per_model,axis=0)
    return ensmean_allmodels

def ensmax_ts_per_model(indict):
    # ensemble maximum timeseries
    outdict = {}
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
            tempdict = cor_for_albm(tempdict,k,f)
        stacked = np.stack(tempdict.values())
        outdict[k] = np.nanmax(stacked,axis=0)
    return outdict


def ensmax_ts(indict):
    # ensemble minimum timeseries
    ens_summed = {}
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = np.nansum(indict[k][f],axis=(1,2))
            tempdict = cor_for_albm(tempdict,k,f)

        stacked = np.stack(tempdict.values())
        ens_summed[k] = np.nanmax(stacked,axis=0)

    stacked_per_model = np.stack(ens_summed.values())
    ensmean_allmodels = np.nanmax(stacked_per_model,axis=0)
    return ensmean_allmodels


def ens_spmean(indict):
# calculate timeseries maps of spatial mean of all forcings, per model
# output: dict per model
    outdict = {}
    for k in indict: 
        stacked = np.stack(indict[k].values())
        ensmean = np.nanmean(stacked,axis=0)
        print(ensmean.shape)
        outdict[k] = np.nansum(ensmean,axis=0)
    return outdict

def ens_spmean_ensmean(indict):
# calculate timeseries maps of spatial mean of all forcings and all models
# output: np array of (timestep,lon,lat)
    ensmean_per_model = {}
    for k in indict: 
        stacked = np.stack(indict[k].values())
        ensmean = np.nanmean(stacked,axis=0)
        ensmean_per_model[k] = ensmean
    
    stacked_per_model = np.stack(ensmean_per_model.values())
    ensmean_allmodels = np.nanmean(stacked_per_model,axis=0)

    return ensmean_allmodels


def ens_spmean_ensmean2(indict):
# calculate timeseries maps of spatial mean of all forcings and all models
# output: np array of (timestep,lon,lat)
    ensmean_per_model = {}
    for k in indict: 
        stacked = np.stack(indict[k].values())
        ensmean = np.nanmean(stacked,axis=0)
        ensmean_per_model[k] = ensmean
    
    stacked_per_model = np.stack(ensmean_per_model.values())
    ensmean_allmodels = np.nanmean(stacked_per_model,axis=0)

    return ensmean_allmodels



def ensmean_spcumsum(indict):
    # calculate ensemble mean an acumulates spatially (returns lon lat field)
    outdict = {}
    for k in indict: 
        stacked = np.stack(indict[k].values())
        ensemblemean = np.nanmean(stacked,axis=0)
        outdict[k] = np.cumsum(ensemblemean,axis=0)
    return outdict

def moving_average(indict, n=10):
    # calculate 5-day moving average of dictionary of nps or dictorionary in dictionary of nps 

    # movign average op np array
    def moving_average_np(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # if indict is not a dictionary
    if type(indict) is not dict:
        outdict = moving_average_np(indict,n)

    else:    
        outdict = {}
        for k in indict:
            tempdict = {}

            # if indict is dictionary of dictionary
            if type(indict[k]) is dict:
                for f in indict[k]:
                    tempdict[f] = moving_average_np(indict[k][f],n)
                outdict[k] = tempdict
            # if input is dictionary of np arrays
            else:
                outdict[k] = moving_average_np(indict[k],n)

    return outdict 


# to continue making - not finished yet!!!
def ensmean_ts_allmodels(indict):
    # ensemble mean timeseries 
    outdict = {}
    for k in indict: 
        stacked = np.stack(indict[k].values())
        ensmean = np.nanmean(stacked,axis=0)
        ensmean_permodel[k] = np.nansum(ensmean,axis=(1,2))

    return array


# calculate anomalies for dict structure
def calc_anomalies(lakeheat, flag_ref, years_analysis):

   # if indict is not a dictionary
    if type(lakeheat) is not dict:
        if flag_ref == 'pre-industrial':   # period of first 30 years of simulation (1900-1929)
            lakeheat_ref_forcing = np.nanmean(lakeheat[0:30,:,:])
        elif isinstance(flag_ref,int):
            lakeheat_ref_forcing = lakeheat[years_analysis.index(flag_ref),:,:]
    # subtract reference to calculate anomaly 
        lakeheat_anom = lakeheat - lakeheat_ref_forcing

    else:    

        lakeheat_anom = {}

        for model in lakeheat:
            lakeheat_anom_model = {}

            for forcing in lakeheat[model]: 

                # determine reference
                if flag_ref == 'pre-industrial':   # period of first 30 years of simulation (1900-1929)
                    lakeheat_ref_forcing = np.nanmean(lakeheat[model][forcing][0:30,:,:])
                elif isinstance(flag_ref,int):
                    lakeheat_ref_forcing = lakeheat[model][forcing][years_analysis.index(flag_ref),:,:]

                # subtract reference to calculate anomaly 
                lakeheat_anom_model[forcing] = lakeheat[model][forcing] - lakeheat_ref_forcing
            lakeheat_anom[model] = lakeheat_anom_model
    return lakeheat_anom


def get_lonlat(indir_lakedata):
    """"
    Opens file and reads lon and lat (necessary for plotting purposes)
    """
    lakedepth_path        = indir_lakedata + 'dlake_1km_ll_remapped_0.5x0.5.nc'

    # load variables
    gldb_lakedepth      = xr.open_dataset(lakedepth_path)


    lon = gldb_lakedepth.dl.lon
    lat = gldb_lakedepth.dl.lat


    return lon,lat




def rasterize(feature_name,lon_min,lon_max,lat_min,lat_max,resolution,outdir,filename):
        
    """
    This function rasterizes a .shp file and saves it as a .tiff in the same directory
    Only for global extent

    input:      feature_name: Fieldname of shapefile to be burned in raster
                resolution: horizontal resolution in degrees  
                filename: input and output filename
    """

            # check whether pct_grid shapefile is already existing
    if os.path.isfile(outdir+filename+".tiff"): 
        print(' ') #print(filename+'.tiff already exists')
    else:
        # define command
        command = 'gdal_rasterize -a '+ feature_name\
        + ' -ot Float32 -of GTiff -te '+ str(lon_min)+' '+str(lat_min)+' '+str(lon_max)+' '+str(lat_max)+' -tr ' + str(resolution) +' '+ str(resolution)\
        + ' -co COMPRESS=DEFLATE -co PREDICTOR=1 -co ZLEVEL=6 -l '+ filename\
        + ' ' + outdir+filename+'.shp ' + outdir+filename +'.tiff'

        os.system(command)    

# cell area functions
def read_raster(filename):
    """
    Function to read raster file
    input: file name of raster (ends in .tiff)
    output: 2D numpy array
    """
    raster = gdal.Open(filename)
    myarray = np.array(raster.GetRasterBand(1).ReadAsArray())
    myarray = np.flipud(myarray)
    return myarray

def make_grid(xmin,xmax,ymin,ymax,resolution):
        """
        Function to make a regular polygon grid
        spanning over xmin, xmax, ymin, ymax 
        and with a given resolution

        output: geoDataFrame of grid
        """

        nx = np.arange(xmin, xmax,resolution)
        ny = np.arange(ymin, ymax,resolution)

        # create polygon grid
        polygons = []
        for x in nx:
                for y in ny:
                        poly  = Polygon([(x,y), (x+resolution, y), (x+resolution, y-resolution), (x, y-resolution)])
                        # account for precision (necessary to create grid at exact location)
                        poly = wkt.loads(wkt.dumps(poly, rounding_precision=2))
                        polygons.append(poly)
                
        # store polygons in geodataframe
        grid = gpd.GeoDataFrame({'geometry':polygons})
        return grid

def calc_pctarea(polygons,grid,feature_name):
    """
    This function calculates the percentage of polygons in a grid cell
    input: poygons (geopandas geodataframe)
           grid (geopandas geodatframe) 
           feature_name name of new feature created containing percent coverage
    output: pct (geodataframe with extent of grid and feature representing 
            percentage coverage of grid cell
    """

    # calculate area per grid cell. (more save than taking one value per cell, if grid is projected)
    grid['gridcell_area'] = grid.area
    grid['grid_index'] = grid.index

    # check if lakes are present
    if not polygons.empty:

        # calculate intersection between lakes and grid (with overlay in geopandas)
        intersected = gpd.overlay(grid,polygons,how='intersection')
        intersected['intersect_area'] = intersected.area
        intersected[feature_name] = intersected['intersect_area']/intersected['gridcell_area']*100
        # make exception for when polygon is just touching grid, but not lies within it
        if intersected.empty:
            grid_pct = gpd.GeoDataFrame()
        else: 
            intersected = intersected.dissolve(by='grid_index', aggfunc='sum') 
            grid_pct = grid.merge(intersected[[feature_name]], on='grid_index', copy='False')

    else:
        grid_pct=gpd.GeoDataFrame() 

    return grid_pct

def calc_areafrac_shp2rst_region(shp_path,outdir,outfilename,resolution,coord):
    """
    This is the main function to be called in a script
    """

    import numpy as np
    

    # define sections and resolution of section at which processed (all in degrees)
    # function works global by default. 
    # coord =  [lon_min,lon_max,lat_min,lat_max]
    lon_min = coord[0]
    lon_max = coord[2]
    lat_min = coord[1]
    lat_max = coord[3]
    res_processed=1 # degrees

    
    # check whether pct_grid shapefile is already existing
    if os.path.isfile(outdir+outfilename+".shp"): 
        print(' ')   #print(outfilename+'.shp already exists')
    else:
            # read shapefile
            shp_data=gpd.read_file(shp_path)


            # define lon lat bounds. 
            # lon_max, lat_max both +1 to account also for last defined boundary (inherent to python)
            # both lats: +resolution (to really start at 0, artefact of grid making method)
            lon_bounds = np.arange(lon_min,lon_max+1,res_processed)
            lat_bounds = np.arange(lat_min+resolution,lat_max+resolution+1,res_processed)

            # initialise counter 
            count = 0
            # create empty geodataframe to store results
            grid_pct = gpd.GeoDataFrame()


            # loop over different sections
            for indx, xmin in enumerate(lon_bounds[:-1]):
                    for indy, ymin in enumerate(lat_bounds[:-1]):
                    
                            # counter
                            count = count+1
                            # print('Processing gridcell '+ str(count) +' of '+ str(lon_bounds[:-1].size*lat_bounds[:-1].size))
                            
                            # define xmax, ymax
                            xmax = lon_bounds[indx+1]
                            ymax = lat_bounds[indy+1]

                            # create grid
                            grid = make_grid(xmin,xmax,ymin,ymax,resolution)

                            # clip lakes for grid area
                            clip_area = grid.geometry.unary_union
                            shp_clipped = shp_data[shp_data.geometry.intersects(clip_area)]

                            # calculate percent area of clipped zone
                            grid_pct_clipped=calc_pctarea(shp_clipped,grid,'PCT_area')

                            # concatenate the different shapefiles
                            grid_pct = pd.concat([grid_pct,grid_pct_clipped], sort=False)
                    

            # save to shape file
            grid_pct.to_file(outdir+outfilename+".shp")

    # rasterize
    rasterize('PCT_area',lon_min,lon_max,lat_min,lat_max,resolution,outdir,outfilename)
    out_pct_raster = read_raster(outdir+outfilename+'.tiff')
    
    return out_pct_raster


    # functions 

def extract_region(indir_lakedata,indict,extent):
    """ Extract lake region based on extent, input can be up to 2 level dictionary"""
    # cut out corresponding region of lakeheat 
    lon,lat = get_lonlat(indir_lakedata)
    # if indict is not a dictionary
    if type(indict) is not dict:
        temp = indict[:,np.where(lat.values == extent[3])[0].item():np.where(lat.values == extent[1])[0].item(),np.where(lon.values == extent[0])[0].item():np.where(lon.values == extent[2])[0].item()]
        outdict = temp #* lake_pct_region

    else:    
        outdict = {}
        for k in indict:
            tempdict = {}

            # if indict is dictionary of dictionary
            if type(indict[k]) is dict:
                for f in indict[k]:
                    tempdict[f] = indict[k][f][:,np.where(lat.values == extent[3])[0].item():np.where(lat.values == extent[1])[0].item(),np.where(lon.values == extent[0])[0].item():np.where(lon.values == extent[2])[0].item()] #* lake_pct_region 
                outdict[k] = tempdict
            # if input is dictionary of np arrays
            else:
                outdict[k] =  indict[k][:,np.where(lat.values == extent[3])[0].item():np.where(lat.values == extent[1])[0].item(),np.where(lon.values == extent[0])[0].item():np.where(lon.values == extent[2])[0].item()] #* lake_pct_region 

    return outdict 


# functions to load calculated lakeheat according to different scenarios and calculate anomalies

def load_lakeheat(scenario,outdir,flag_ref, years_analysis):
    lakeheat= np.load(outdir+'lakeheat_'+scenario+'.npy',allow_pickle='TRUE').item()
    
    if not scenario =='onlyresclimate':
        lakeheat_albm = load_lakeheat_albm(outdir,scenario,years_analysis)
        lakeheat.update(lakeheat_albm)
        del lakeheat_albm

    lakeheat_anom = calc_anomalies(lakeheat, flag_ref, years_analysis)

    anom_ensmean = moving_average(ensmean_ts(lakeheat_anom))
    anom_ensmin  = moving_average(ensmin_ts(lakeheat_anom))
    anom_ensmax  = moving_average(ensmax_ts(lakeheat_anom))
    anom_std     = moving_average(ens_std_ts(lakeheat_anom))
    del lakeheat_anom, lakeheat

    return (anom_ensmean, anom_ensmin, anom_ensmax, anom_std)


def load_riverheat(outdir):
    anom_ensmean = np.load(outdir+'riverheat/riverheat_ensmean.npy',allow_pickle='TRUE')
    anom_ensmin = np.load(outdir+'riverheat/riverheat_ensmin.npy',allow_pickle='TRUE')
    anom_ensmax = np.load(outdir+'riverheat/riverheat_ensmax.npy',allow_pickle='TRUE')
    anom_std = np.load(outdir+'riverheat/riverheat_std.npy',allow_pickle='TRUE')
    return (anom_ensmean, anom_ensmin, anom_ensmax, anom_std)


def load_lakeheat_totalclimate(outdir,flag_ref, years_analysis):
    lakeheat_climate = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()
    scenario = 'climate'
    lakeheat_albm = load_lakeheat_albm(outdir,scenario,years_analysis)
    lakeheat_climate.update(lakeheat_albm)    
    del lakeheat_albm
    
    lakeheat_climate_anom = calc_anomalies(lakeheat_climate, flag_ref,years_analysis)
    climate_anom_ensmean = moving_average(ensmean_ts(lakeheat_climate_anom))
    climate_anom_std     = moving_average(ens_std_ts(lakeheat_climate_anom))

    lakeheat_onlyresclimate = np.load(outdir+'lakeheat_onlyresclimate.npy',allow_pickle='TRUE').item()
    lakeheat_onlyresclimate_anom = calc_anomalies(lakeheat_onlyresclimate, flag_ref,years_analysis)
    onlyresclimate_anom_ensmean = moving_average(ensmean_ts(lakeheat_onlyresclimate_anom))
    onlyresclimate_anom_std     = moving_average(ens_std_ts(lakeheat_onlyresclimate_anom))

    riverheat_anom_ensmean = np.load(outdir+'riverheat/riverheat_ensmean.npy',allow_pickle='TRUE')
    riverheat_anom_std = np.load(outdir+'riverheat/riverheat_std.npy',allow_pickle='TRUE')

    totheat_climate = climate_anom_ensmean + onlyresclimate_anom_ensmean + riverheat_anom_ensmean
    totheat_climate_std = climate_anom_std + onlyresclimate_anom_std + riverheat_anom_std

    return (totheat_climate, totheat_climate_std)


def calc_reservoir_warming(outdir):

    """  Calculate reservoir warming (difference total and (climate+reservoir expansion)
         and save to file
    """

    lakeheat_climate = np.load(outdir+'lakeheat_climate.npy',allow_pickle='TRUE').item()
    lakeheat_res = np.load(outdir+'lakeheat_reservoirs.npy',allow_pickle='TRUE').item()
    lakeheat_both = np.load(outdir+'lakeheat_both.npy',allow_pickle='TRUE').item()

    lakeheat_onlyresclimate = {}
    indict = {}
    for k, v in lakeheat_both.items():
        lakeheat_onlyresclimate.update({k:indict})
        for f , values in lakeheat_both[k].items():
            lakeheat_onlyresclimate[k][f] = values - (lakeheat_climate[k].get(f, np.nan) + lakeheat_res[k].get(f, 0)) # returns value if k exists in d2, otherwise 0
    
    del lakeheat_both, lakeheat_res, lakeheat_climate

    lakeheat_filename = 'lakeheat_onlyresclimate.npy'
    np.save(outdir+lakeheat_filename, lakeheat_onlyresclimate) 


# create grid (1cel longitude, all latitudes)
def make_grid(xmin,xmax,ymin,ymax,resolution):
        """
        Function to make a regular polygon grid
        spanning over xmin, xmax, ymin, ymax 
        and with a given resolution

        output: geoDataFrame of grid
        """

        nx = np.arange(xmin, xmax,resolution)
        ny = np.arange(ymin, ymax,resolution)

        # create polygon grid
        polygons = []
        for x in nx:
                for y in ny:
                        poly  = Polygon([(x,y), (x+resolution, y), (x+resolution, y-resolution), (x, y-resolution)])
                        # account for precision (necessary to create grid at exact location)
                        poly = wkt.loads(wkt.dumps(poly, rounding_precision=2))
                        polygons.append(poly)
                
        # store polygons in geodataframe
        grid = gpd.GeoDataFrame({'geometry':polygons})
        return grid


def calc_grid_area(res): 

    """
    Function to calculate the area of each grid cell for a global grid
    given the resolution
    Returns a numpy array with the size of the grid containing areas for each grid cell
    """
    xmin=0
    xmax=xmin+res
    ymin= -90+res
    ymax= 90+res

    grid_1d = make_grid(xmin,xmax,ymin,ymax,res)
    grid_1d.crs = {'init':'epsg:4326'}

    # reproject grid to cilindrical equal-area projection
    grid_1d = grid_1d.to_crs({'init':'epsg:6933'})

    # calculate area per polygon of projected grid
    grid_1d["area"]=grid_1d.area

    # retrieve areas as a np array
    areas_1d = grid_1d["area"].values

    # concatenate areas to make global grid
    areas_global = np.empty([int(180/res),int(360/res)])

    ncol = int(360/res)

    for i in range(ncol):
        areas_global[:,i]=areas_1d

    return areas_global