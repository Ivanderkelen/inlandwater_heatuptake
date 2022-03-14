"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Scripts for isimip postprocessing with CDO 

"""
# %%

from cdo import Cdo

cdo = Cdo()
import os 
import xarray as xr
import numpy as np


flag_streamtemps=True
# -----------------------------------------------------------
# initialise

indir  = '/gpfs/projects/climate/data/dataset/isimip/isimip2b/InputData/GCM_atmosphere/biascorrected/global/'
outdir = '/gpfs/work/vsc10055/inlandwater_heat/riverheat/streamtemp/'



forcings    = ['GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']

experiments = ['historical','rcp60']

variable   = 'tas'

# periods
period_hist =['18610101-18701231', '18710101-18801231', '18810101-18901231',\
              '18910101-19001231', '19010101-19101231', '19110101-19201231',\
              '19210101-19301231', '19310101-19401231', '19410101-19501231',\
              '19510101-19601231', '19610101-19701231', '19710101-19801231',\
              '19810101-19901231', '19910101-20001231', '20010101-20051231' ]

period_fut  =['20060101-20101231', '20110101-20201231', '20210101-20301231',\
              '20310101-20401231', '20410101-20501231', '20510101-20601231',\
              '20610101-20701231', '20710101-20801231', '20810101-20901231',\
              '20910101-20991231']



# %%
# -------------------------------
# Functions

def calc_streamtemp(tas):
    """ Global standard regression equation from Punzet et al. (2012)
        Calculates grid cell stream temperature based on air temperature 
        Both input and output temperature are in K"""
    
    # global constants, taken from Punzet et al., 2012
    c0 = 32; c1 = -0.13; c2 = 1.94

    tas_C = tas - 273.15
    streamtemp_C = c0/(1+np.exp(c1*tas_C+c2))
    streamtemp = streamtemp_C + 273.15
    return streamtemp

# %%
# -------------------------------
# calculate stream temperatures from air temperatures and format in right period

if flag_streamtemps:
    
    for forcing in forcings:

        outfile = 'streamtemp_'+forcing+'_1861_2099_annual.nc4'

        for experiment in experiments:


            # choose right forcing periods
            if experiment == 'historical':  periods = period_hist
            if experiment == 'rcp60':       periods = period_fut

            # define paths
            indir_exp = indir+experiment+'/'+forcing+'/'   
            allfiles = variable+'_day_'+forcing+'_*_r1i1p1_EWEMBI_monmean_*.nc4'
        
            # calculate montly mean per period
            for period in periods:
                raw_file     = variable+'_day_'+forcing+'_'+experiment+'_r1i1p1_EWEMBI_'+period+'.nc4'
                monmean_file = variable+'_day_'+forcing+'_'+experiment+'_r1i1p1_EWEMBI_monmean_'+period+'.nc4'
               
                if (not os.path.isfile(outdir+monmean_file)):
                    print('Calculating montly means of '+forcing +' '+ experiment+'...')
                    print(period)
                    cdo.monmean(input=indir_exp+raw_file,output=outdir+monmean_file)

        if (not os.path.isfile(outdir+outfile)):
            print('Calculating stream temperatures for '+forcing+'...')

            # merge files and save as xarray 
            ds = xr.open_dataset(cdo.select("name="+variable,input=outdir+allfiles, returnXarray=True))
            
            # clean up
            os.system('rm '+outdir+allfiles)  

            # calculate stream temperatures for every grid cell
            ds['streamtemp'] = calc_streamtemp(ds['tas'])
            ds.streamtemp.attrs['standard_name'] = 'stream_temperature' 
            ds.streamtemp.attrs['long_name']     = 'Stream Temperature, derived from Near-Surface Air Temperature using statistical regression by Punzet et al (2012)' 
            ds.streamtemp.attrs['units']         = 'K' 
        
            # save as a netcdf
            ds = ds.groupby('time.year').mean('time')
            ds.to_netcdf(outdir+outfile, mode='w')
        else: 
            print('Stream temperatures are already calculated for '+forcing)
