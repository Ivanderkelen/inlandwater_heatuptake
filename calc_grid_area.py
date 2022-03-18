"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : August 2019

Script to calcualte area per grid cell of a global regular grid
"""
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt
import pandas as pd
import matplotlib.pyplot as plt

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