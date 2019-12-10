
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Main subroutine to do get values for lake heat calculations

"""
import csv

def get_values():

    scenarios = 'climate', 'onlyresclimate','reservoirs', 'both'
    # remove old file
    os.system('rm '+outdir+'lakeheat_values.csv')
    # define over how many years back you want to calculate the increase:
    years_increase = 10
    # load values 
    for i,scenario in enumerate(scenarios):

        # load and calculate timeseries 
        (lakeheat_anom_ensmean_ts, 
            lakeheat_anom_ensmin_ts, 
            lakeheat_anom_ensmax_ts,
            lakeheat_anom_std_ts ) = load_lakeheat(scenario,outdir, flag_ref, years_analysis)

        # calculate total heat content increase
        total_heatcontent_increase = np.mean(lakeheat_anom_ensmean_ts[-years_increase:-1])
        total_heatcontent_std      = np.mean(lakeheat_anom_std_ts[-years_increase:-1])
        #total_heatcontent_trend = total_heatcontent_increase/len(range(start_year_values,end_year))

        # save the calculations in a dict
        dict_values = {'Scenario':scenario, 
                        'Lake heat increase [J]': total_heatcontent_increase, 
                        'Standard deviation [J]':total_heatcontent_std}

        with open(outdir+'lakeheat_values.csv', mode='a') as csv_file:
            fieldnames = dict_values.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if i == 0: 
                writer.writeheader()
            writer.writerow(dict_values)
