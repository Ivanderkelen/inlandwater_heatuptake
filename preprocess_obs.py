"""
Code for reading in evaluation data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os 


def preprocess_obs():
    # define idrectory where data is located
    evaldir = '/home/inne/documents/phd/data/isimip_laketemp/evaluation' 
    datadir = '/LakeData/'

    # read lakenames and metadata
    metadata = pd.read_csv(evaldir+datadir+'lakes_isimip_metadata.csv')

    # define own lakenames
    #lakenames = ['Allequash', 'Biel', 'BigMuskellunge', 'BurleyGriffin',
    #        'Crystal', 'Delavan', 'Dickie','Erken', 'Fish',
    #       'Kivu', 'LowerZurich', 'Mendota', 'Monona', 'Neuchatel',
    #       'Okauchee', 'Rotorua', 'Sparkling', 'Stechlin',
    #       'Sunapee',  'Toolik', 'Trout','TwoSisters',
    #        'Wingra']
    
    # lter lakes https://lter.limnology.wisc.edu/about/lakes
    lakenames = ['Allequash','BigMuskellunge', 'Crystal','Mendota', 'Monona','Sparkling','Toolik', 'Trout', 'Wingra' ]
    lakes_lessthan_10y = ['Vendyurskoe', 'Paajarvi','Tahoe']
    # initialise dictionary to save lake heat data for all lakes
    obsdict = {'lakenames' : lakenames}

    lakenames = lakenames#_opendata

    # process data for every lake
    for lakename in lakenames: 
        
        filename = lakename+'_temp_daily.csv'
            
        if not os.path.isfile(evaldir+datadir+lakename+'/'+filename):
            print(lakename+' daily file does not exists')
        
        else:    

            print('Processing '+lakename)
            # load lake data 
            lakedata = pd.read_csv(evaldir+datadir+lakename+'/'+filename)
        

            # load metadata of lake 
            metadata_lake = metadata[metadata['Lake Short Name']==lakename]
            mean_depth = metadata_lake['mean depth (m)'].values[0]


            # convert temperature from C to K
            lakedata['WTEMP'] = lakedata['WTEMP'].apply(lambda x: x+273.16)

            # convert timestamp to datetime object
            lakedata['TIMESTAMP'] = lakedata['TIMESTAMP'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

            # only continue if more than 10 years 
            lakedata['TIMESTAMP'].iloc[-1] - lakedata['TIMESTAMP'].iloc[0]

            # calculate lake layer thickness
            lakedata['LAYER_THICKNESS'] = np.zeros_like(lakedata['DEPTH'])

            # loop over unique days
            for day in lakedata['TIMESTAMP'].unique(): 

                depths = lakedata[lakedata['TIMESTAMP']==day]['DEPTH'].values
                layer_thickness = np.zeros_like(depths)

                # loop over lake depths per day and calculate thickness
                for i,d in enumerate(depths): 
                    
                    if i == 0: 
                        layer_thickness[i] = depths[i]

                    # if last layer is less deep than mean lake depth, extend last layer untill mean lake depth
                    elif i == len(depths)-1 and depths[i] < mean_depth:  
                        last_layer = depths[i] - depths[i-1]
                        until_bottom = mean_depth - depths[i]
                        layer_thickness[i] = last_layer + until_bottom

                    else : 
                        layer_thickness[i] = depths[i] - depths[i-1]
                    

                lakedata['LAYER_THICKNESS'][lakedata['TIMESTAMP']== day] = layer_thickness


            # calculate lake heat J per layer 
            cp_liq = 4.188e3   # [J/kg K] heat capacity liquid water
            rho_liq = 1000     # [kg/m2] density liquid water

            lakedata['LAKEHEAT'] = rho_liq * cp_liq * lakedata['WTEMP']* lakedata['LAYER_THICKNESS']
            # sum lake heat J per layer per day and save lake heat in panda dataframe
            df_sum = lakedata.groupby('TIMESTAMP')['LAKEHEAT'].agg('sum')
            df_lakeheat = df_sum.to_frame()



            # make and save plot to inspect data
            plot_title = lakename + ': '+lakedata['TIMESTAMP'].iloc[0].strftime("%d-%m-%Y")+' until '+lakedata['TIMESTAMP'].iloc[-1].strftime("%d-%m-%Y")
            df_lakeheat.plot(title=plot_title)
            plt.savefig(evaldir+'/plots/'+lakename+'.png')
            plt.close()

            # save all necessary data in csv per variable
            lakedict = {
                'name'      : lakename,
                'name_long' : metadata_lake['Lake Name'].values[0],
                'lakeheat'  : df_lakeheat,
                'timestamp' : lakedata['TIMESTAMP'].unique(),
                'mean_depth': mean_depth,
                'lat'       : metadata_lake['latitude (dec deg)'].values[0],
                'lon'       : metadata_lake['longitude (dec deg)'].values[0]
            }

                
            obsdict.update({lakename : lakedict})

    # save lake heat values per lake 
    os.system('rm '+evaldir+'/lakeheat_observations.npy')
    np.save(evaldir+'/lakeheat_observations.npy', obsdict)


