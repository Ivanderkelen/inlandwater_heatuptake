
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : June 2019

Main subroutine to do plotting for lake heat

"""

# import packages

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy as ctp
import mplotutils as mpu 
from dict_functions import *

# do settings

mpl.rc('axes',edgecolor='grey')
mpl.rc('axes',labelcolor='dimgrey')
mpl.rc('xtick',color='dimgrey')
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',color='dimgrey')
mpl.rc('ytick',labelsize=12)
mpl.rc('axes',titlesize=14)
mpl.rc('axes',labelsize=12)
mpl.rc('legend',fontsize='large')
mpl.rc('text',color='dimgrey')



def do_plotting(flag_save_plots, plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir):


    #%% Plot raw heat uptake per model forcing 
    # calculate anomalies
    lakeheat_anom = calc_anomalies(lakeheat, flag_ref,years_analysis)
    # Calculate timeseries of lake heat anomaly
    lakeheat_anom_ts = timeseries(lakeheat_anom)

    # Plotting functions 
    # 4x4 individual forcing plot per model plot 
    for model in models:
            
        f,ax = plt.subplots(2,2, figsize=(8,7))
        x_values = np.asarray(years_analysis)

        ax = ax.ravel()

        for nplot,forcing in enumerate(forcings):

            line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
            line1 = ax[nplot].plot(x_values,lakeheat_anom_ts[model][forcing], color='coral')
            ax[nplot].set_xlim(x_values[0],x_values[-1])
            ax[nplot].set_xticks(ticks= np.array([1900,1920,1940,1960,1980,2000,2015]))
            ax[nplot].set_xticklabels([1900,1920,1940,1960,1980,2000,2015] )
            #ax[nplot].set_ylim(-6e20,8e20)
            ax[nplot].set_ylabel('Energy [J]')
            ax[nplot].set_title(forcing, pad=15)

        f.suptitle(model+' lake heat anomalies (reference 1900-1929)', fontsize=16)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

        if flag_save_plots:
            plt.savefig(plotdir+model+'heat_acc_per_forcing'+'.png',dpi=300)




    # ------------------------------------------------
    # load saved heat storages for different scenarios

    # only climate
    (lakeheat_climate_anom_ensmean_ts, 
        lakeheat_climate_anom_ensmin_ts, 
        lakeheat_climate_anom_ensmax_ts,
        lakeheat_climate_anom_std_ts ) = load_lakeheat('climate',outdir, flag_ref, years_analysis)

    # only reservoir expansion
    (lakeheat_res_anom_ensmean_ts, 
        lakeheat_res_anom_ensmin_ts, 
        lakeheat_res_anom_ensmax_ts,
        lakeheat_res_anom_std_ts ) = load_lakeheat('reservoirs',outdir, flag_ref, years_analysis)

    # both expansion and warming
    (lakeheat_both_anom_ensmean_ts, 
        lakeheat_both_anom_ensmin_ts, 
        lakeheat_both_anom_ensmax_ts,
        lakeheat_both_anom_std_ts ) = load_lakeheat('both',outdir, flag_ref, years_analysis)


    # calculate reservoir warming (difference total and (climate+reservoir expansion))
    calc_reservoir_warming(outdir)

    (lakeheat_onlyresclimate_anom_ensmean_ts, 
        lakeheat_onlyresclimate_anom_ensmin_ts, 
        lakeheat_onlyresclimate_anom_ensmax_ts,
        lakeheat_onlyresclimate_anom_std_ts ) = load_lakeheat('onlyresclimate',outdir, flag_ref, years_analysis)


    # %% Figure 1
    # ------------------------------------------------
    # create plot lake heat uptake due to climate change (global)
    # each figure own y-axis
    # plot them on one graph. 
    flag_uncertainty = 'envelope' # or '2std' or 'envelope'

    f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    x_values = moving_average(np.asarray(years_analysis))

    # -------------------------------------
    # subplot 1: natural lakes heat uptake
    line_zero = ax1.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
    line1, = ax1.plot(x_values,lakeheat_climate_anom_ensmean_ts, color='coral')

    # uncertainty based on choice
    if flag_uncertainty == 'envelope':
        # full envelope
        area2 = ax1.fill_between(x_values,lakeheat_climate_anom_ensmin_ts,lakeheat_climate_anom_ensmax_ts, color='sandybrown',alpha=0.5)
    elif flag_uncertainty =='2std':
    # 2x std error
        under_2std = lakeheat_climate_anom_ensmean_ts - 2*lakeheat_climate_anom_std_ts
        upper_2std = lakeheat_climate_anom_ensmean_ts + 2*lakeheat_climate_anom_std_ts
        area2 = ax1.fill_between(x_values,under_2std,upper_2std, color='sandybrown',alpha=0.5)

    ax1.set_xlim(x_values[0],x_values[-1])
    ax1.set_xticks(ticks= np.array([1902,1920,1940,1960,1980,2000,2014]))
    ax1.set_xticklabels([1900,1920,1940,1960,1980,2000,2015] )
    #ax1.set_ylim(-0.4e20,1e20)
    ax1.set_ylabel('Energy [J]')
    ax1.set_title('Natural lake heat uptake', loc='right')
    ax1.text(0.03, 0.92, '(a)', transform=ax1.transAxes, fontsize=14)


    # -------------------------------------
    # subplot 2: reservoir heat uptake
    line_zero = ax2.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

    line1, = ax2.plot(x_values,lakeheat_onlyresclimate_anom_ensmean_ts, color='mediumvioletred')

    # uncertainty based on choice
    if flag_uncertainty == 'envelope':
        # full envelope
        area2 = ax2.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmin_ts,lakeheat_onlyresclimate_anom_ensmax_ts, color='plum',alpha=0.5)
    elif flag_uncertainty =='2std':
    # 2x std error
        under_2std = lakeheat_onlyresclimate_anom_ensmean_ts - 2*lakeheat_onlyresclimate_anom_std_ts
        upper_2std = lakeheat_onlyresclimate_anom_ensmean_ts + 2*lakeheat_onlyresclimate_anom_std_ts
        area2 = ax2.fill_between(x_values,under_2std,upper_2std, color='plum',alpha=0.5)

    ax2.set_xlim(x_values[0],x_values[-1])
    ax2.set_xticks(ticks= np.array([1902,1920,1940,1960,1980,2000,2014]))
    ax2.set_xticklabels([1900,1920,1940,1960,1980,2000,2015] )
    #ax1.set_ylim(-0.4e20,1e20)
    ax2.set_ylabel('Energy [J]')
    ax2.set_title('Reservoir heat uptake', loc='right')
    ax2.text(0.03, 0.92, '(b)', transform=ax2.transAxes, fontsize=14)
    #f.suptitle('Reference period 1900-1929, 5 year moving average')
    plt.tight_layout()
    if flag_save_plots:
        plt.savefig(plotdir+'fig1_heat_uptake_CC'+'.png',dpi=300)


    # %% Figure 2 Heat Budgets
    # ------------------------------------------------


    # functions 
    def add_arrow_value(ax,y1,y2,ar_color,ar_text, y1_textpos_toadd =0.05e21 ):
        y_half = (y2-y1)/2
        x_pos = 1.03
        y1_textpos = y1- y1_textpos_toadd
        if y2-y1 > 0.1e21:
            ax.annotate('', xy=(x_pos, y1), xycoords=("axes fraction","data"), xytext=(x_pos, y2), 
                    arrowprops=dict(arrowstyle="<->", color=ar_color,linewidth=1.5))
            ax.annotate(ar_text+'%', xy=(x_pos+0.02, y1), xycoords=("axes fraction","data"), xytext=(x_pos+0.02, y_half), fontsize=12)
        else:
            ax.annotate('', xy=(x_pos-0.02, y1), xycoords=("axes fraction","data"), xytext=(x_pos+0.02, y1), 
                    arrowprops=dict(arrowstyle="<-", color=ar_color,linewidth=1.5))
            ax.annotate(ar_text+'%', xy=(x_pos+0.02, y1_textpos), xycoords=("axes fraction","data"), xytext=(x_pos+0.02, y1_textpos), fontsize=12)





    # plotting
    flag_uncertainty = '2std' # or '2std' or 'envelope'

    f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    x_values = moving_average(np.asarray(years_analysis))

    # -------------------------------------
    # subplot 1: climate change heat accumulation
    line_zero = ax1.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')


    # reservoirs climate change
    line1, = ax1.plot(x_values,lakeheat_onlyresclimate_anom_ensmean_ts, color='mediumvioletred')
    area1 = ax1.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmean_ts, color='mediumvioletred')

    # natural lakes
    line2, = ax1.plot(x_values,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts, color='coral')
    area2  = ax1.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmean_ts,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts, color='sandybrown')


    # Add joint uncertainty? 
    #if flag_uncertainty == 'envelope':
        # full envelope
        #area2 = ax1.fill_between(x_values,lakeheat_climate_anom_ensmin_ts,lakeheat_climate_anom_ensmax_ts, color='sandybrown',alpha=0.5)
    #elif flag_uncertainty =='2std':
    # 2x std error
        #under_2std = lakeheat_climate_anom_ensmean_ts - 2*lakeheat_climate_anom_std_ts
        #upper_2std = lakeheat_climate_anom_ensmean_ts + 2*lakeheat_climate_anom_std_ts
        #area2 = ax1.fill_between(x_values,under_2std,upper_2std, color='sandybrown',alpha=0.5)

    ax1.set_xlim(x_values[0],x_values[-1])
    ax1.set_xticks(ticks= np.array([1902,1920,1940,1960,1980,2000,2014]))
    ax1.set_xticklabels([1900,1920,1940,1960,1980,2000,2015] )
    #ax1.set_ylim(-0.4e20,1e20)
    ax1.set_ylabel('Energy [J]')
    ax1.set_title('Heat accumulation from climate change', loc='right')
    ax1.legend((area1,area2),['reservoir heat uptake','natural lake heat uptake'],frameon=False,loc='upper left', bbox_to_anchor = (0.01,0.92))

    ax1.text(0.03, 0.92, '(a)', transform=ax1.transAxes, fontsize=14)

    # add arrows

    total = (lakeheat_climate_anom_ensmean_ts+lakeheat_onlyresclimate_anom_ensmean_ts)
    clim_frac = np.round(lakeheat_climate_anom_ensmean_ts[-1]/total[-1] *100,1)
    resclim_frac = np.round(lakeheat_onlyresclimate_anom_ensmean_ts[-1]/total[-1] *100,1)

    # clim only
    y1 = lakeheat_onlyresclimate_anom_ensmean_ts[-1]
    y2 = y1+lakeheat_climate_anom_ensmean_ts[-1]
    add_arrow_value(ax1,y1,y2,'coral',str(clim_frac))

    # resclim
    y1 = 0
    y2 = lakeheat_onlyresclimate_anom_ensmean_ts[-1]
    add_arrow_value(ax1,y1,y2,'mediumvioletred',str(resclim_frac),0.01e21)


    # -------------------------------------
    # subplot 2: reservoir heat uptake
    line_zero = ax2.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

    # reservoirs climate change
    line1, = ax2.plot(x_values,lakeheat_onlyresclimate_anom_ensmean_ts, color='mediumvioletred')
    line2, = ax2.plot(x_values,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts, color='coral')
    line3, = ax2.plot(x_values,lakeheat_climate_anom_ensmean_ts+lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_res_anom_ensmean_ts, color='steelblue')

    area3 = ax2.fill_between(x_values,lakeheat_climate_anom_ensmean_ts+lakeheat_onlyresclimate_anom_ensmean_ts,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts+lakeheat_res_anom_ensmean_ts, color='skyblue')
    area2  = ax2.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmean_ts,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts, color='sandybrown')
    area1 = ax1.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmean_ts, color='mediumvioletred')

    # natural lakes

    # reservoirs redistribution

    # See whether possible to add in joint uncertainty bands? 
    #if flag_uncertainty == 'envelope':
        # full envelope
        #area2 = ax2.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmin_ts,lakeheat_onlyresclimate_anom_ensmax_ts, color='plum',alpha=0.5)
    #elif flag_uncertainty =='2std':
    # 2x std error
        #under_2std = lakeheat_onlyresclimate_anom_ensmean_ts - 2*lakeheat_onlyresclimate_anom_std_ts
        #upper_2std = lakeheat_onlyresclimate_anom_ensmean_ts + 2*lakeheat_onlyresclimate_anom_std_ts
        #area2 = ax2.fill_between(x_values,under_2std,upper_2std, color='plum',alpha=0.5)

    ax2.set_xlim(x_values[0],x_values[-1])
    ax2.set_xticks(ticks= np.array([1902,1920,1940,1960,1980,2000,2014]))
    ax2.set_xticklabels([1900,1920,1940,1960,1980,2000,2015] )
    #ax1.set_ylim(-0.4e20,1e20)
    ax2.set_ylabel('Energy [J]')
    ax2.set_title('Heat accumulation with redistribution', loc='right')
    ax2.text(0.03, 0.92, '(b)', transform=ax2.transAxes, fontsize=14)
    ax2.legend((area1,area2,area3),['natural lake heat uptake','reservoir heat uptake', 'reservoir expansion'],frameon=False,loc='upper left', bbox_to_anchor = (0.01,0.92))

    total = (lakeheat_climate_anom_ensmean_ts+lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_res_anom_ensmean_ts)
    res_frac = np.round(lakeheat_res_anom_ensmean_ts[-1]/total[-1] *100,1)
    clim_frac = np.round(lakeheat_climate_anom_ensmean_ts[-1]/total[-1] *100,1)
    resclim_frac = np.round(lakeheat_onlyresclimate_anom_ensmean_ts[-1]/total[-1] *100,1)


    # res only 
    y1 = lakeheat_climate_anom_ensmean_ts[-1]+lakeheat_onlyresclimate_anom_ensmean_ts[-1]
    y2 = y1+lakeheat_res_anom_ensmean_ts[-1]
    add_arrow_value(ax2,y1,y2,'steelblue',str(res_frac))

    # clim only
    y1 = lakeheat_onlyresclimate_anom_ensmean_ts[-1]
    y2 = y1+lakeheat_climate_anom_ensmean_ts[-1]
    add_arrow_value(ax2,y1,y2,'coral',str(clim_frac))

    # resclim
    y1 = 0
    y2 = lakeheat_onlyresclimate_anom_ensmean_ts[-1]
    add_arrow_value(ax2,y1,y2,'mediumvioletred',str(resclim_frac))

    #f.suptitle('Reference period 1900-1929, 5 year moving average')
    plt.tight_layout()
    if flag_save_plots:
        plt.savefig(plotdir+'fig2_heat_acc'+'.png',dpi=300)