
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Main subroutine to do get values for lake heat calculations

"""
import csv
import os 
from dict_functions import *


def get_values(outdir, flag_ref, years_analysis):

    scenarios = 'climate', 'onlyresclimate', 'rivers','total climate change', 'reservoirs'
    # remove old file
    os.system('rm '+outdir+'lakeheat_values.csv')

    # define over how many years back you want to calculate the uptake:
    years_increase = 10

    # define number of years over which to calculate the trend 
    years_trend = 30


    # load values 
    for i,scenario in enumerate(scenarios):

        if scenario == 'rivers': 

            (heat_anom_ensmean_ts, 
                heat_anom_ensmin_ts, 
                heat_anom_ensmax_ts,
                heat_anom_std_ts ) = load_riverheat(outdir)

        elif scenario == 'total climate change': 
             (heat_anom_ensmean_ts, heat_anom_std_ts) = load_lakeheat_totalclimate(outdir,flag_ref, years_analysis)

        else: 

            # load and calculate timeseries 
            (heat_anom_ensmean_ts, 
                heat_anom_ensmin_ts, 
                heat_anom_ensmax_ts,
                heat_anom_std_ts ) = load_lakeheat(scenario,outdir, flag_ref, years_analysis)


        # calculate total heat content increase
        print(np.mean(heat_anom_ensmean_ts))
        print(heat_anom_ensmean_ts.size)
        total_heatcontent_increase = np.mean(heat_anom_ensmean_ts[-years_increase:-1])
        total_heatcontent_std      = np.mean(heat_anom_std_ts[-years_increase:-1])
        total_heatcontent_trend    = (heat_anom_ensmean_ts[-1]-heat_anom_ensmean_ts[-years_trend])/years_trend
        
        # save the calculations in a dict
        dict_values = {'Scenario':scenario, 
                        'Lake heat increase [J]': total_heatcontent_increase, 
                        'Standard deviation [J]':total_heatcontent_std, 
                        'Trend [J/yr] over the last '+str(years_trend)+' years': total_heatcontent_trend}

        with open(outdir+'lakeheat_values.csv', mode='a') as csv_file:
            fieldnames = dict_values.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if i == 0: 
                writer.writeheader()
            writer.writerow(dict_values)
