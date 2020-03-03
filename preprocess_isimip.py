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

    print('Preprocessing ISIMIP lake temperature...')
    for model in models:
        
        for forcing in forcings:

            for variable in variables:

                for experiment in experiments:
                # differentiate for future experiments filenames
                    if experiment == 'future': 
                        experiment_fn = future_experiment 
                        period = '2006_2099'
                    elif experiment == 'historical': 
                        experiment_fn = experiment
                        period = '1861_2005'  

                    path = indir+model+'/'+forcing+'/'+experiment+'/'
                    outfile_assembled = model.lower()+'_'+forcing+'_historical_'+future_experiment+'_'+variable+'_'+'1861_2099'+'_'+'annual'+'.nc4'

                    # define input filename for the different models
                    if model == 'CLM45': 
                        infile = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'2005soc_co2'+'_'+variable+'_'+'global'+'_'+'monthly'+'_'+period+'.nc4'
                    elif model == 'SIMSTRAT-UoG':
                        infiles_daily = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_'+variable+'_'+'global'+'_'+'daily'+'_*.nc4'
                        infile = model.lower()+'_'+forcing+'_'+'ewembi'+'_'+experiment_fn+'_'+'nosoc_co2'+'_'+variable+'_'+'global'+'_'+'monthly'+'_'+period+'.nc4'
                        cdo.monmean(input= '- mergetime '+path+infiles_daily,output=path+infile)

                    
                    # if simulation is available 
                    if os.path.isfile(path+infile): 

                        # make output directory per model if not done yet
                        outdir_model = outdir+variable+'/'+model+'/'
                        if not os.path.isdir(outdir_model):
                            os.system('mkdir '+outdir_model)
                    
                        # calculate annual means per model for each forcing (if not done so)
                        outfile_annual = model.lower()+'_'+forcing+'_'+experiment_fn+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                        if (not os.path.isfile(outdir_model+outfile_assembled)):
                            print('calculating annual means of '+infile)
                            cdo.yearmean(input=path+infile,output=outdir_model+outfile_annual)


                # assemble historical and future simulation
                infile_hist =  model.lower() +'_'+forcing+'_historical_'           +variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                infile_fut  =  model.lower()+'_'+forcing+'_'+future_experiment+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'.nc4'
                
                if (not os.path.isfile(outdir_model+outfile_assembled)):
                    print('concatenating historical and '+future_experiment+' simulations of '+model+' '+forcing)
                    cdo.mergetime(input=outdir_model+infile_hist+' '+outdir_model+infile_fut,output=outdir_model+outfile_assembled )
                    # clean up 
                    os.system('rm '+outdir_model+infile_hist +' '+outdir_model+infile_fut)
                   
                   # account for levlak same variable name
                    if model == 'SIMSTRAT-UoG':
                        cdo.chname('input=levlak,depth '+outdir_model+outfile_assembled, output=outdir_model+outfile_assembled+'1')
                        os.system('mv '+outdir_model+outfile_assembled+'1' +' '+outdir_model+outfile_assembled)

                # calculate ensemble mean forcing for each model (if not done so)
                outfile_ensmean = model.lower()+'_historical_'+future_experiment+'_'+variable+'_'+'1861_2005'+'_'+'annual'+'_'+'ensmean'+'.nc4'
                if (not os.path.isfile(outdir_model+outfile_ensmean)):
                    print('calculating ensemble means of '+model)
                    cdo.ensmean(input=outdir_model+outfile_assembled,output=outdir_model+outfile_ensmean)
                else:
                    print(model+' '+forcing+' is already preprocessed.')

