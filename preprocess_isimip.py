
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

preprocess_laketemp.py

 This scritpt:
 - calculates the annual means
 - appends the historical and rcp period (which is chosen in main script)
 - saves the resulting series in netcdf files for each model-forcing combination
 - calculates and saves the ensemble mean in netcdf files

Settings and necesarry modules are loaded in main_lakeheat.py

"""
import os
# needs also cdo (loaded in main program because depending on system)

from cdo import Cdo
cdo = Cdo()

def preprocess_isimip(models, forcings, variables, experiments, future_experiment, indir, outdir):

    print('Preprocessing ISIMIP variable...')
    for model in models:

        for forcing in forcings:

            for variable in variables:

                for experiment in experiments:
                    print('Processing '+experiment)
                # differentiate for future experiments filenames
                    if experiment == 'future':
                        experiment_fn = future_experiment
                        period = '2006_2099'
                        period_daily = ['2006_2010', '2011_2020', '2021_2030']#, '2031_2040',
                                        #'2041_2050', '2051_2060', '2061_2070', '2071_2080',
                                        #'2081_2090', '2091_2099']

                    elif experiment == 'historical':
                        experiment_fn = experiment
                        period = '1861_2005'
                        period_daily = ['1891_1900','1901_1910','1911_1920','1921_1930','1931_1940','1941_1950',
                                        '1951_1960','1961_1970','1971_1980','1981_1990','1991_2000','2001_2005']


                    path = indir+model+'/'+forcing+'/'+experiment+'/'
                    outfile_assembled = model.lower()+'_'+forcing+'_historical_'+future_experiment+'_'+variable+'_'+'1861_2099'+'_'+'annual'+'.nc4'
                    outfile_annual = model.lower()+'_'+forcing+'_'+experiment_fn+'_'+variable+'_'+'annual'+'.nc4'

                    # make output directory per model if not done yet
                    outdir_model = outdir+variable+'/'+model+'/'
                    if not os.path.isdir(outdir_model):
                        os.system('mkdir '+outdir_model)

                    # define input filename for the different models
                    if model == 'CLM45':
                        infile_annual_all = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'annual'+'_*'
                        if not os.path.isfile(outdir_model+outfile_annual):
                            for p in period_daily:
                                infile_daily  = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'daily'+'_'+p+'.nc4'
                                infile_annual = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'annual'+'_'+p+'.nc4'

                                if not os.path.isfile(outdir_model+infile_annual):
                                    if os.path.isfile(path+infile_daily): # if file is available
                                        print('Calculating annual means from daily files for ' + forcing + ', period '+ p)
                                        cdo.yearmean(input=path+infile_daily, output=outdir_model+infile_annual)

                         # merge all annual files
                            print('Merging ...')
                            os.system('cdo mergetime '+outdir_model+infile_annual_all+' '+outdir_model+outfile_annual)
                            cdo.mergetime(input=outdir_model+infile_annual_all, output=outdir_model+outfile_annual)

                    elif model == 'SIMSTRAT-UoG': # daily temperatures to deal with

                        infile_annual_all = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_'+variable+'_'+'global'+'_'+'annual'+'_*'
                        if not os.path.isfile(outdir_model+outfile_annual):
                            for p in period_daily:
                                infile_daily  = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_'+variable+'_'+'global'+'_'+'daily'+'_'+p+'.nc4'
                                infile_annual = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_'+variable+'_'+'global'+'_'+'annual'+'_'+p+'.nc4'

                                if not os.path.isfile(outdir_model+infile_annual):
                                    if os.path.isfile(path+infile_daily): # if file is available
                                        print('Calculating annual means from daily files for ' + forcing + ', period '+ p)
                                        cdo.yearmean(input=path+infile_daily, output=outdir_model+infile_annual)

                            # merge all annual files
                            print('Merging ...')
                            os.system('cdo mergetime '+outdir_model+infile_annual_all+' '+outdir_model+outfile_annual)
                            cdo.mergetime(input=outdir_model+infile_annual_all, output=outdir_model+outfile_annual)

                            # clean up annual files
                            #os.system('rm '+outdir_model+infile_annual_all)

                    elif model == 'ALBM': # daily temperatures to deal with

                        infile_annual_all = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'annual'+'_*'
                        print(outdir_model+outfile_annual)
                        if not os.path.isfile(outdir_model+outfile_annual):
                            for p in period_daily:
                                infile_monthly  = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'monthly'+'_'+p+'.nc4'
                                infile_annual = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'annual'+'_'+p+'.nc4'


                                if not os.path.isfile(outdir_model+infile_annual):
                                    if os.path.isfile(path+infile_monthly): # if file is available
                                        print('Calculating annual means from monthly files for ' + forcing + ', period '+ p)
                                        cdo.yearmean(input=path+infile_monthly, output=outdir_model+infile_annual)

                        # merge all annual files
                        print('Merging ...')
                        os.system('cdo mergetime '+outdir_model+infile_annual_all+' '+outdir_model+outfile_annual)
                        cdo.mergetime(input=outdir_model+infile_annual_all, output=outdir_model+outfile_annual)

                                        # clean up annual files
                                        #os.system('rm '+outdir_model+infile_annual_all)

                # assemble historical and future simulation
                infile_hist =  model.lower() +'_'+forcing+'_historical_'          +variable+'_'+'annual'+'.nc4'
                infile_fut  =  model.lower()+'_'+forcing+'_'+future_experiment+'_'+variable+'_'+'annual'+'.nc4'

                if (not os.path.isfile(outdir_model+outfile_assembled)):
                    print('concatenating historical and '+future_experiment+' simulations of '+model+' '+forcing)
                    cdo.mergetime(input=outdir_model+infile_hist+' '+outdir_model+infile_fut,output=outdir_model+outfile_assembled )
                    # clean up
                    os.system('rm '+outdir_model+infile_hist +' '+outdir_model+infile_fut)


                   # account for levlak same variable name - manually
                    #if model == 'SIMSTRAT-UoG':
                        #os.system('cdo chname,levlak,depth '+outdir_model+outfile_assembled +' '+outdir_model+'temp.nc')
                        #os.system('mv '+outdir_model+'temp.nc '+ outdir_model+outfile_assembled)
                # calculate ensemble mean forcing for each model (if not done so)

                # convert °C to K! manually (only for SIMSTRAT)

                outfile_ensmean = model.lower()+'_historical_'+future_experiment+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'_'+'ensmean'+'.nc4'
                if (not os.path.isfile(outdir_model+outfile_ensmean)):
                    print('calculating ensemble means of '+model)
                    cdo.ensmean(input=outdir_model+outfile_assembled,output=outdir_model+outfile_ensmean)
                else:
                    print(model+' '+forcing+' is already preprocessed.')


