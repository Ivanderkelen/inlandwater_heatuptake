"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Function to load lake heat of ALBM and fill nan on years with issues

"""
import numpy as np


def load_lakeheat_albm(outdir,flag_scenario,years_analysis): 

    # load lakeheat from file
    lakeheat_albm = np.load(outdir+'lakeheat_'+flag_scenario+'_ALBM.npy',allow_pickle='TRUE').item()
    forcings    = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']

    # issues are only present for 'climate' and 'both' as they are in the lake temperature profiles
    if not flag_scenario == 'reservoirs':

        # change the data in these periods to nan
        for forcing in forcings:
            lakeheat_forcing = lakeheat_albm['ALBM'][forcing]

            # 3. miroc5: bad value for year 1997. replace with nan
            ind_start = years_analysis.index(2006)
            ind_end = years_analysis.index(2013)
            lakeheat_forcing[ind_start:ind_end,:,:] = np.nan

            if forcing == 'miroc5':

                # 3. miroc5: bad value for year 1997. replace with nan
                ind_1996 = years_analysis.index(1996)
                ind_1997 = years_analysis.index(1998)
                lakeheat_forcing[ind_1996:ind_1997,:,:] = np.nan

        
        lakeheat_albm['ALBM'][forcing] = lakeheat_forcing
        
    return lakeheat_albm


def cor_for_albm(dict_forcing,model,forcing): 
    # function to correct for missing values of albm
    years_analysis         = range(1896,2025,1)

    if model == 'ALBM': 
        ind_start = years_analysis.index(2006)
        ind_end = years_analysis.index(2013)            
        dict_forcing[forcing][ind_start:-1] = np.nan

        #ind_1996 = years_analysis.index(2013)
        #dict_forcing[forcing][ind_1996:-1] = np.nan
    
        if forcing == 'miroc5': 
            ind_1996 = years_analysis.index(1996)
            ind_1997 = years_analysis.index(1998)
            dict_forcing[forcing][ind_1996:ind_1997] = np.nan
      
    return dict_forcing