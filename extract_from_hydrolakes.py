"""

Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Test script to extract shapefiles from Hydrolakes
"""
import os
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
import shapely.geometry as sgeom
from shapely.geometry import box

sys.path.append(r'/home/inne/documents/phd/scripts/python/calc_lakeheat_isimip/lakeheat_isimip')
from dict_functions import *


def extract_from_hydrolakes(extend,extend_fname):
    """Script to extract the great polygon files from hydrolakes and GranD dataset
        input= extend (array), extend_name (string including path)
        filters lakes based on an area threshold """
    area_threshold = 100 #in km^2
    from shapely.geometry import box
    if not os.path.isfile(extend_fname):
        hydrolakes_dams_path = '/home/inne/documents/phd/data/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp'
        print('Loading HydroLAKES ...')
        lakes = gpd.read_file(hydrolakes_dams_path)
        boundingbox = box(extend[0], extend[1], extend[2], extend[3])
        print('Extracting lakes ...')
        lakes_selected = lakes[lakes['Lake_area']>=area_threshold]
        lakes_extracted = lakes_selected[lakes.geometry.intersects(boundingbox)]
        lakes_extracted.to_file(extend_fname)
    else:
        print('Already extracted '+extend_fname)
#%%
# Settings

# Laurentian Great Lakes
extent_LGL = [-92.5,41,-75.5,49.5]
name_LGL   = 'LaurentianGreatLakes'
lakes_path = '/home/inne/documents/phd/data/processed/lakes_shp/'
path_LGL   = lakes_path+name_LGL+'.shp'
extract_from_hydrolakes(extent_LGL,path_LGL)

# African Great Lakes
extent_AGL = [28,-10,35.5,2.5]
name_AGL   = 'AfricanGreatLakes'
lakes_path = '/home/inne/documents/phd/data/processed/lakes_shp/'
path_AGL   = lakes_path+name_AGL+'.shp'
extract_from_hydrolakes(extent_AGL,path_AGL)
