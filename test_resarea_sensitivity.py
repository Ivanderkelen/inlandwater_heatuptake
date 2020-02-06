# test script

# test sensitivity on remapping method for lake area pct. 

import xarray as xr 
import numpy as np 
from calc_grid_area import calc_grid_area
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rc('axes',edgecolor='grey')
mpl.rc('axes',labelcolor='dimgrey')
mpl.rc('xtick',color='dimgrey')
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',color='dimgrey')
mpl.rc('ytick',labelsize=12)
mpl.rc('axes',titlesize=12)
mpl.rc('axes',labelsize=12)
mpl.rc('legend',fontsize='large')

mpl.rc('text',color='dimgrey')



def import_lakepct(name):
    """ Import lake pct data set and calculate anomaly timeseries of global sum """
    # import 
    filename = 'mksurf_lake_0.5x0.5_'+name+'_hist_clm5_hydrolakes_1850-2017.nc'
    if name == 'dir': filename = 'mksurf_lake_0.5x0.5_hist_clm5_hydrolakes_1850-2017_c20191220.nc'

    ds = xr.open_dataset('/home/inne/documents/phd/data/isimip_laketemp/'+filename)

    pct_lake = ds.PCT_LAKE[:,:,:]

    # calculate area and take anomaly sum 
    tot = pct_lake/100*calc_grid_area(0.5)
    end_pct_lake = tot[0,:,:]
    sum_pct_lake = tot.sum(dim=('lat','lon')).values - end_pct_lake.sum(dim=('lat','lon')).values
    return sum_pct_lake


sum_bil = import_lakepct('bil')
sum_nn = import_lakepct('nn')
sum_con2 = import_lakepct('con2')
sum_bic = import_lakepct('bic')

sum_dir = import_lakepct('dir')

# import grand dataset
ts_grand = np.load('/home/inne/documents/phd/data/processed/isimip_lakeheat/grand_areas.npy')
sum_grand =( ts_grand- ts_grand[0])

# plot 
years = range(1850,2018,1) 
f,ax = plt.subplots()
ax.plot(years,sum_grand,label='GRanD polygons',color='k')
ax.plot(years,sum_con2,label='conservative 2nd order')
ax.plot(years,sum_bil,label='bilinear')
ax.plot(years,sum_nn,label='nearest neighbour')
ax.plot(years,sum_bic,label='bicubic')
ax.plot(years,sum_dir,label='directly mapped')
ax.set_xlim(1850,2017)
ax.legend()
ax.set_title('Reservoir expansion for different input datasets',loc='right')
ax.set_ylabel('Reservoir area [m²]')

plt.savefig('/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/'+'res_area_input_sensitivity'+'.png',dpi=300)