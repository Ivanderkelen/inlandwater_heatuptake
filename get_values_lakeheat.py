
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Main subroutine to do get values for lake heat calculations

"""
import csv
import os 
from dict_functions import *
from datetime import datetime

from sklearn.linear_model import LinearRegression
import xarray as xr

def get_values(outdir, flag_ref, years_analysis, indir_lakedata, resolution):

    scenarios = 'climate', 'onlyresclimate', 'rivers','total climate change', 'reservoirs'

    # remove old file
    os.system('rm '+outdir+'lakeheat_values.csv')

    # define over how many years back you want to calculate the uptake:
    years_increase = 10

    # define number of years over which to calculate the trend 
    years_trend = 31

    # get lake area and calculate reservoir area
    lake_area      = np.load(indir_lakedata+'lake_area.npy')
    lake_area_ts = np.sum(lake_area, axis=(1,2))
    res_area_ts = lake_area_ts - lake_area_ts[0]


    # river area in m^2 from Allen et al., 2018
    river_area = 773000000000  

    os.system("rm "+outdir+'lakeheat_values.csv')


    # load values 
    for i,scenario in enumerate(scenarios):

        if scenario == 'rivers': 

            (heat_anom_ensmean_ts, 
                heat_anom_ensmin_ts, 
                heat_anom_ensmax_ts,
                heat_anom_std_ts ) = load_riverheat(outdir)
        
            area = river_area

        elif scenario == 'total climate change': 
            (heat_anom_ensmean_ts, heat_anom_std_ts) = load_lakeheat_totalclimate(outdir,flag_ref, years_analysis)

            area = lake_area_ts + river_area
            np.save(outdir+'timeseries.npy', heat_anom_ensmean_ts) 
            np.save(outdir+'timeseries_std.npy', heat_anom_std_ts) 

        else: 

            # load and calculate timeseries 
            (heat_anom_ensmean_ts, 
                heat_anom_ensmin_ts, 
                heat_anom_ensmax_ts,
                heat_anom_std_ts ) = load_lakeheat(scenario,outdir, flag_ref, years_analysis)

            if scenario == 'climate': area = lake_area_ts[0]
            if scenario == 'onlyresclimate' or scenario == 'reservoirs': area = res_area_ts

        # calculate total heat content increase
        #print(np.mean(heat_anom_ensmean_ts))
        #print(heat_anom_ensmean_ts.size)

        total_heatcontent_increase = np.mean(heat_anom_ensmean_ts[-years_increase:-1])
        total_heatcontent_std      = np.mean(heat_anom_std_ts[-years_increase:-1])

        # heat flux over whole period
        # 100 is # years between 1915 and 2015, the middle period in which is calculated
        endyear_datetime = datetime.strptime(str(2015), '%Y')
        startyear_datetime = datetime.strptime(str(1915), '%Y')
        nsecs = (endyear_datetime-startyear_datetime).total_seconds() 

        if not isinstance(area,(float,int)):
            heat_flux_fullperiod = np.mean(heat_anom_ensmean_ts[-years_increase:-1]/ area[-years_increase:-1])/nsecs
            heat_flux_fullperiod_std = np.mean(heat_anom_std_ts[-years_increase:-1]/ area[-years_increase:-1])/nsecs
        else: 
            heat_flux_fullperiod = np.mean(heat_anom_ensmean_ts[-years_increase:-1]/ area)/nsecs
            heat_flux_fullperiod_std = np.mean(heat_anom_std_ts[-years_increase:-1]/ area)/nsecs


        # heat flux over last nyears
        startyear_datetime = datetime.strptime(str(years_analysis[-years_trend]), '%Y')
        endyear_datetime = datetime.strptime(str(years_analysis[-1]), '%Y')
        nsecs = (endyear_datetime-startyear_datetime).total_seconds() 

        if not isinstance(area,(float,int)):
            heat_diff_30y = (heat_anom_ensmean_ts[-1]/area[-1] - heat_anom_ensmean_ts[-years_trend]/area[-years_trend])/nsecs
            heat_diff_std = (heat_anom_std_ts[-1]/area[-1] - heat_anom_std_ts[-years_trend]/area[-years_trend] )/nsecs

        else: 
            heat_diff_30y = (heat_anom_ensmean_ts[-1]/area - heat_anom_ensmean_ts[-years_trend]/area)/nsecs
            heat_diff_std = (heat_anom_std_ts[-1]/area - heat_anom_std_ts[-years_trend]/area)/nsecs


        # calculate heat flux over same period as Gentine et al., 2020. 
        # 2004 - 2014
        if scenario == 'total climate change':
            year_start = 2006
            year_end = 2020
            ind_start = years_analysis.index(year_start)
            ind_end = years_analysis.index(year_end)

            startyear_datetime = datetime.strptime(str(year_start), '%Y')
            endyear_datetime = datetime.strptime(str(year_end), '%Y')
            nsecs = (endyear_datetime-startyear_datetime).total_seconds() 
            if not isinstance(area,(float,int)):
                heat_diff = heat_anom_ensmean_ts[-1]/area[ind_end] - heat_anom_ensmean_ts[ind_start]/area[ind_start] 
            else: 
                heat_diff = heat_anom_ensmean_ts[-1]/area - heat_anom_ensmean_ts[ind_start]/area

            heat_flux_PG = heat_diff/nsecs
            print("Inland water heat uptake in "+str(year_start) +'-'+str(year_end)+'is: '+str(heat_flux_PG))
            land_uptake = 0.24 #W/m2
            print("Compared to land uptake: " + str(heat_flux_PG/land_uptake)+"%")

            # # 1999-2014

            # year_start = 1999
            # year_end = 2020
            # ind_start = years_analysis.index(year_start)
            # ind_end = years_analysis.index(year_end)

            # startyear_datetime = datetime.strptime(str(year_start), '%Y')
            # endyear_datetime = datetime.strptime(str(year_end), '%Y')
            # nsecs = (endyear_datetime-startyear_datetime).total_seconds() 
            # if not isinstance(area,(float,int)):
            #     heat_diff = heat_anom_ensmean_ts[ind_end]/area[ind_end] - heat_anom_ensmean_ts[ind_start]/area[ind_start] 
            # else: 
            #     heat_diff = heat_anom_ensmean_ts[ind_end]/area - heat_anom_ensmean_ts[ind_start]/area

            # heat_flux_PG = heat_diff/nsecs
            # print("Inland water heat uptake in "+str(year_start) +'-'+str(year_end)+'is: '+str(heat_flux_PG))
            # land_uptake = 0.24 #W/m2
            # print("Compared to land uptake: " + str(heat_flux_PG/land_uptake)+"%")

        
        # calculate trend - last 30 years. 
        linregmodel = LinearRegression() # normalize=True? 
        x = np.arange(1,years_trend).reshape(-1,1)
        linregmodel.fit(x,heat_anom_ensmean_ts[-years_trend:-1])

        total_heatcontent_trend    = linregmodel.coef_


        # 1900-1930. 

        # save the calculations in a dict
        dict_values = {'Scenario':scenario, 
                        'Lake heat increase [J]': total_heatcontent_increase, 
                        'Standard deviation [J]':total_heatcontent_std, 
                        'Heat flux [W/m²]': heat_flux_fullperiod,                         
                        'stdev [W/m²]': heat_flux_fullperiod_std,                         
                        'Trend [J/yr] over the last '+str(years_trend)+' years': total_heatcontent_trend[0],
                        'Heat flux [W/m²] over lasts'+str(years_trend): heat_diff_30y,                  
                        'stdev last years [W/m²]': heat_diff_std                      
                        }

        with open(outdir+'lakeheat_values.csv', mode='a') as csv_file:
            fieldnames = dict_values.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if i == 0: 
                writer.writeheader()
            writer.writerow(dict_values)

    years = np.arange(1900,2021,1)



        # with open(outdir+'inlandwater_heat_timeseries.csv', mode='a') as csv_file:
        #     fieldnames = dict_ts.keys()
        #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        #     if i == 0: 
        #         writer.writeheader()
        #     writer.writerow(dict_ts)