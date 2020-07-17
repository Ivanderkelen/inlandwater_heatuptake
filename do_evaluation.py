"""
Code for reading in evaluation data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import os 
import xarray as xr
from sklearn.linear_model import LinearRegression


# plotting settings
mpl.rc('axes',edgecolor='grey')
mpl.rc('axes',labelcolor='dimgrey')
mpl.rc('xtick',color='dimgrey')
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',color='dimgrey')
mpl.rc('ytick',labelsize=12)
mpl.rc('axes',titlesize=14)
mpl.rc('axes',labelsize=12)
mpl.rc('legend',fontsize='large')
mpl.rc('text',color='dimgrey')



# functions
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

def ens_spmean_ensstd(indict):
# calculate timeseries maps of spatial mean of all forcings and all models
# output: np array of (timestep,lon,lat)
    ensstd_per_model = {}
    for k in indict: 
        stacked = np.stack(indict[k].values())
        ensstd = np.nanstd(stacked,axis=0)
        ensstd_per_model[k] = ensstd
    
    stacked_per_model = np.stack(ensstd_per_model.values())
    ensmean_allmodels = np.nanmean(stacked_per_model,axis=0)

    return ensmean_allmodels

def ens_std_map(indict):
    # calculate standard deviation of timeseries
    concat_stacked = np.array([])
    for k in indict: 
        tempdict = {}
        for f in indict[k]:
            tempdict[f] = indict[k][f]
        stacked = np.stack(tempdict.values())
        # put all forcings of models together. 
        if concat_stacked.size == 0:
            concat_stacked = stacked
        else:
            concat_stacked = np.concatenate((concat_stacked,stacked),axis=0)
    # calculate average over ensemble members
    ens_std = np.nanstd(concat_stacked,axis=0)

    return ens_std



from shapely.geometry import Polygon
from shapely import wkt
import pandas as pd
import geopandas as gpd
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




def do_evaluation(): 

    # define over how many years delta is calculates
    nyears = 10

    # define directory where obs data is located
    indir_lakedata = '/home/inne/documents/phd/data/isimip_laketemp/'
    obsdir = '/home/inne/documents/phd/data/isimip_laketemp/evaluation/' 
    moddir = '/home/inne/documents/phd/data/processed/isimip_lakeheat/'
    resolution = 0.5

    start_year = 1896
    end_year = 2025
    years_analysis         = range(start_year,end_year,1)

    # import model data - multimodel mean
    lon_mod,lat_mod = get_lonlat(indir_lakedata)
    lakeheat_allmod = np.load(moddir+'lakeheat_climate.npy',allow_pickle='TRUE').item()

    # calculate ensemble mean and delete dict
    lakeheat_mod_abs = ens_spmean_ensmean(lakeheat_allmod)# output np array (time,lat,lon)
    lakeheat_std = ens_std_map(lakeheat_allmod)
    del lakeheat_allmod

    # extract lake area and calculate lake area per grid cell (m^2)
    hydrolakes_lakepct = xr.open_dataset(indir_lakedata + 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191220.nc')
    lake_pct           = hydrolakes_lakepct.PCT_LAKE.values/100 # to have then in fraction
    lake_area      = lake_pct[-1,:,:] * calc_grid_area(resolution)

    # calculate modelled lake heat per m² 
    lakeheat_mod = lakeheat_mod_abs/lake_area
    lakeheat_std_mod = lakeheat_std/lake_area

    # flip lats to be able to properly extract 
    lakeheat_mod = np.flip(lakeheat_mod, axis=1)
    lakeheat_std_mod = np.flip(lakeheat_std_mod, axis=1)

    # load obs data
    obsdict = np.load(obsdir+'lakeheat_observations.npy',allow_pickle='TRUE').item()
    #lakenames = ['Allequash','BigMuskellunge', 'Crystal','Mendota', 'Monona','Sparkling','Toolik', 'Trout', 'Wingra' ]
    lakenames = obsdict['lakenames']

    # initialise linear regression model 
    linregmodel = LinearRegression() # normalize=True? 


    # initialise empty lists
    delta_mod_list = []
    delta_obs_list = []
    std_list = []
    lakenames_long = []
    years_mod = []


    # loop over different lakes 
    for lakename in lakenames: 

        lakedict = obsdict[lakename]

        df_obs = lakedict['lakeheat']
        
        # calculate annual means 
        df_ymean = df_obs.groupby(df_obs.index.year).mean()

        # select last 10 years of dataset (and print last ten years)
        if len(df_ymean) < 10: nyears = len(df_ymean)
        else: nyears = 10
        #nyears = len(df_ymean)

        
        start_year = df_ymean.index[0]
        end_year = df_ymean.index[-1]

        # turn end and start year into datetime objects and calculate difference in seconds
        endyear_datetime = datetime.strptime(str(end_year), '%Y')
        startyear_datetime = datetime.strptime(str(start_year), '%Y')
        nsec = (endyear_datetime-startyear_datetime).total_seconds()

        years_obs = str(start_year) + '-' + str(end_year)
        # calculate delta heat for observations

        # calculate trend with linear regression. 
        x = np.arange(1,nyears).reshape(-1,1)
        linregmodel.fit(x,df_ymean.iloc[-nyears:-1])
        obs_trend    = linregmodel.coef_
        obs_trend_Wm = obs_trend /(60*60*254*365)

        delta_obs = (df_ymean.iloc[-1] - df_ymean.iloc[-nyears]) /np.mean(df_ymean.iloc[-nyears])
        delta_obs_Wm = delta_obs  / nsec
        #print(lakedict['name_long']+" %d to %d "% (start_year,end_year))
        #print('numer of years '+str(len(df_ymean)))
        #print('Heat uptake is %2.2e J/m²year'%obs_trend)
        #print('Heat uptake is %.4f W/m²'%obs_trend_Wm)
        #print('')

        # get model data for specific lake (extract grid cell based on lon lat )
        id_lon = (np.abs(lon_mod.values-lakedict['lon'])).argmin()
        id_lat = (np.abs(lat_mod.values-lakedict['lat'])).argmin()

        # extract begin and end year as defined by obs
        mod_gridcell = lakeheat_mod[years_analysis.index(start_year):years_analysis.index(end_year)+1,id_lat,id_lon]
        std_gridcell = lakeheat_std_mod[years_analysis.index(start_year):years_analysis.index(end_year)+1,id_lat,id_lon]

        # calculate obs per model 
        linregmodel.fit(x,mod_gridcell[-nyears:-1])
        mod_trend    = linregmodel.coef_
        mod_trend_Wm = mod_trend /(60*60*254*365)

        linregmodel.fit(x,std_gridcell[-nyears:-1])
        std_trend    = linregmodel.coef_
        std_trend_Wm = std_trend /(60*60*254*365)



        # calculate delta heat for model 
        delta_mod = (mod_gridcell[-1] - mod_gridcell[-nyears]) / np.mean(mod_gridcell[-nyears])
        delta_mod_Wm = delta_mod  / nsec

        #print('Modelled heat uptake is %2.2e J/m²year'%mod_trend)
        #print('Heat uptake is %.4f W/m²'%mod_trend_Wm)
        #print('')

        # save delta's in array per lake 
        delta_mod_list.append(mod_trend_Wm[0])
        std_list.append(std_trend_Wm[0])
        delta_obs_list.append(obs_trend_Wm[0][0])
        lakenames_long.append(lakedict['name_long'])
        years_mod.append(years_obs) 

    #%%
    # make scatterplot
    colors = cm.tab20(np.linspace(0,1,len(lakenames)))

    fig,ax = plt.subplots(figsize=(8,5))
    for i,name_long in enumerate(lakenames_long):
        ax.scatter(delta_obs_list[i],delta_mod_list[i],color=colors[i], label=name_long)

    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width * 0.8, box.height])


    ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5), fontsize=11)
    ax.grid(True)
    ax.set_xlabel('Observed trend [W/m²]', fontsize=14)
    ax.set_ylabel('Modelled trend [W/m²]',fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    #lims = [-0.2,0.7]
    ax.set_xlim([-0.2,0.7])
    ax.set_ylim([-0,0.01])

    plt.savefig(obsdir+'scatterplot.png',dpi=1000, bbox_inches='tight')


    #%%
    # create pandas dataframe with results
    results_dict = {'Lake Name': lakenames_long,'years': years_mod, 'Observed W/m²': delta_obs_list, 'Modeled W/m² ': delta_mod_list,'Std dev W/m2':std_list}
    results = pd.DataFrame(data = results_dict)
    results.to_csv(obsdir+'evaluation_Wm2.csv')

    #%%
    # create pandas dataframe with results
    results_dict = {'Lake Name': lakenames_long,'years': years_mod, 'Observed J/m²year': delta_obs_list, 'Modeled J/m²year ': delta_mod_list}
    results = pd.DataFrame(data = results_dict)

    #results.to_csv(obsdir+'evaluation_Jm2year.csv')