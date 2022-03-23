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
#import mplotutils as mpu 
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



def plot_forcings(flag_save_plots, plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir):
    
    
    xticks = np.array([1900,1920,1940,1960,1980,2000,2021])

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
            ax[nplot].set_xlim(1900,2021)
            ax[nplot].set_xticks(ticks=xticks)
            #ax[nplot].set_ylim(-0.5e20,1.5e20)
            ax[nplot].set_ylabel('Energy [J]')
            ax[nplot].set_title(forcing, pad=15)

        f.suptitle(model+' lake heat anomalies (reference 1900-1929)', fontsize=16)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

        if flag_save_plots:
            plt.savefig(plotdir+model+'heat_acc_per_forcing'+'.png',dpi=300)


def plot_forcings_allmodels(flag_save_plots, plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir):

    xticks = np.array([1900,1940,1980,2021])
    xlims=(1900,2021)


        #%% Plot raw heat uptake per model forcing 
    # calculate anomalies
    lakeheat_anom = calc_anomalies(lakeheat, flag_ref,years_analysis)
    # Calculate timeseries of lake heat anomaly
    lakeheat_anom_ts = timeseries(lakeheat_anom)

    # Plotting functions 
    # all forcings in a row per list of models 
    nmodels = len(lakeheat)
    nforcings = len(lakeheat[list(lakeheat.keys())[0]])
    f,ax = plt.subplots(nmodels,nforcings, figsize=(14,10))
    x_values = np.asarray(years_analysis)
    labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)', '(l)','(m)','(n)','(o)','(p)','(q)','(r)','(s)']

    nplot=0
    ax = ax.ravel()
    for model in models:
        
        for forcing in forcings:

            line_zero = ax[nplot].plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')
            line1 = ax[nplot].plot(x_values,lakeheat_anom_ts[model][forcing], color='coral')
            ax[nplot].set_xlim(xlims)
            #ax[nplot].set_ylim(-0.22e19,0.82e19)
            ax[nplot].set_xticks(ticks=xticks)
            if model == 'CLM45': ax[nplot].set_ylim(-7e20,10e20)
            if model == 'SIMSTRAT-UoG': ax[nplot].set_ylim(-0.5e20,2e20)
            if model == 'ALBM': ax[nplot].set_ylim(-0.5e20,1.5e20)
            if model == 'GOTM': ax[nplot].set_ylim(-0.4e20,1.4e20)

            if nplot == 0 or nplot ==4 or nplot==8 or nplot==12 : ax[nplot].set_ylabel('Energy [J]')
            if nplot < 4: ax[nplot].set_title(forcing, loc='right')
            ax[nplot].text(0.02, 0.90, labels[nplot], transform=ax[nplot].transAxes, fontsize=12)

            nplot = nplot+1

    f.tight_layout()#rect=[0, 0.03, 1, 0.95])
    plt.text(-0.03, 0.96, models[0], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')       
    plt.text(-0.09, 0.72, models[1], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')            
    plt.text(-0.03, 0.46, models[2], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold')            
    plt.text(-0.03, 0.23, models[3], fontsize=14, transform=plt.gcf().transFigure, fontweight = 'bold');            

    if flag_save_plots:
       plt.savefig(plotdir+'heat_acc_per_forcing'+'.jpeg',dpi=1000, bbox_inches='tight')







def do_plotting(flag_save_plots, plotdir, models,forcings, lakeheat, flag_ref, years_analysis,outdir):
    # ------------------------------------------------
    # load saved heat storages for different scenarios

    print('Loading variables for plotting ...')
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

    (riverheat_anom_ensmean_ts, 
        riverheat_anom_ensmin_ts, 
        riverheat_anom_ensmax_ts,
        riverheat_anom_std_ts ) = load_riverheat(outdir)

    # load river heat variables

    # general plotting settings: 
    xticks = np.array([1900,1920,1940,1960,1980,2000,2021])
    xlims=(1900,2021)

    # %% Figure 1
    # ------------------------------------------------
    # create plot lake heat uptake due to climate change (global)
    # each figure own y-axis
    # plot them on one graph. 
    flag_uncertainty = '2std' # or '2std' or 'envelope'

    f,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(6,10))
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
        under_2std = lakeheat_climate_anom_ensmean_ts - lakeheat_climate_anom_std_ts
        upper_2std = lakeheat_climate_anom_ensmean_ts + lakeheat_climate_anom_std_ts
        area2 = ax1.fill_between(x_values,under_2std,upper_2std, color='sandybrown',alpha=0.5)

    ax1.set_xlim(xlims)
    ax1.set_xticks(ticks= xticks)
    #ax1.set_ylim(-0.4e20,1e20)
    ax1.set_ylabel('Energy [J]')
    ax1.set_title('Natural lake heat uptake', loc='right')
    ax1.text(0.02, 0.92, '(a)', transform=ax1.transAxes, fontsize=14)


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
        under_2std = lakeheat_onlyresclimate_anom_ensmean_ts - lakeheat_onlyresclimate_anom_std_ts
        upper_2std = lakeheat_onlyresclimate_anom_ensmean_ts + lakeheat_onlyresclimate_anom_std_ts
        area2 = ax2.fill_between(x_values,under_2std,upper_2std, color='plum',alpha=0.5)

    ax2.set_xlim(xlims)
    ax2.set_xticks(ticks= xticks)
    #ax1.set_ylim(-0.4e20,1e20)
    ax2.set_ylabel('Energy [J]')
    ax2.set_title('Reservoir heat uptake', loc='right')
    ax2.text(0.02, 0.92, '(b)', transform=ax2.transAxes, fontsize=14)

    # -------------------------------------
    # subplot 3: river heat uptake
    line_zero = ax3.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

    line1, = ax3.plot(x_values,riverheat_anom_ensmean_ts, color='darkslateblue')
    flag_uncertainty = 'envelope'
    # uncertainty based on choice
    if flag_uncertainty == 'envelope':
        # full envelope
        area3 = ax3.fill_between(x_values,riverheat_anom_ensmin_ts,riverheat_anom_ensmax_ts, color='lightsteelblue')
    elif flag_uncertainty =='2std':
    # 2x std error
        under_2std = riverheat_anom_ensmean_ts - 2*riverheat_anom_std_ts
        upper_2std = riverheat_anom_ensmean_ts + 2*riverheat_anom_std_ts
        area3 = ax3.fill_between(x_values,under_2std,upper_2std, color='lightsteelblue')

    ax3.set_xlim(xlims)
    ax3.set_xticks(ticks=xticks)
    #ax1.set_ylim(-0.4e20,1e20)
    ax3.set_ylabel('Energy [J]')
    ax3.set_title('River heat uptake', loc='right')
    ax3.text(0.02, 0.92, '(c)', transform=ax3.transAxes, fontsize=14)
   
    #f.suptitle('Reference period 1900-1929, 5 year moving average')
    plt.tight_layout(h_pad=1.4)

    if flag_save_plots:
        plt.savefig(plotdir+'fig1_heat_uptake_CC'+'.jpeg',dpi=1000, bbox_inches='tight')


# %% Figure 2 Heat Budgets
    # ------------------------------------------------

    colors_res = ['mediumvioletred','mediumvioletred']
    colors_rivers = ['lightseagreen','mediumturquoise']
    colors_natlak = ['coral','sandybrown']
    colors_onlyres = ['steelblue','skyblue']
    colors_total='darkslateblue'

    # functions 
    def add_arrow_value(ax,y1,y2,ar_color,ar_text,  flag_onlyres=False, y1_textpos_toadd =0.05e21,flag_gobelow=False):
        y_half = (abs(y2)-abs(y1))/2
        x_pos = 1.04
        x_pos_tex = 1.04

        y1_textpos = y1- y1_textpos_toadd
        
        if flag_gobelow:
            ax.annotate('', xy=(x_pos-0.02,-0.05e21), xycoords=("axes fraction","data"), xytext=(x_pos+0.02, -0.25e21), 
                    arrowprops=dict(arrowstyle="<-", color=ar_color,linewidth=1.5))
            ax.annotate(ar_text+'%', xy=(x_pos_tex+0.02, y1_textpos+0.2), xycoords=("axes fraction","data"), xytext=(x_pos_tex+0.02,-0.25e21), fontsize=12)
        
        elif flag_onlyres==1: #only reservoirs fig 1 
            x_pos = 1.03

            ax.annotate('', xy=(x_pos-0.02, -0.5), xycoords=("axes fraction","data"), xytext=(x_pos+0.02, y1), 
                    arrowprops=dict(arrowstyle="<-", color=ar_color,linewidth=1.5))
            ax.annotate(ar_text+'%', xy=(x_pos_tex+0.02, y1_textpos), xycoords=("axes fraction","data"), xytext=(x_pos_tex+0.02, y1_textpos+0.2), fontsize=12)
        elif flag_onlyres==2: #only reservoirs fig 2 
            x_pos = 1.03
            print(ar_text)
            ax.annotate('', xy=(x_pos-0.02, -0.5), xycoords=("axes fraction","data"), xytext=(x_pos+0.02, y1), 
                    arrowprops=dict(arrowstyle="<-", color=ar_color,linewidth=1.5))
            ax.annotate(ar_text+'%', xy=(x_pos_tex+0.02, y1_textpos-0.1e21), xycoords=("axes fraction","data"), xytext=(x_pos_tex+0.02, y1_textpos-0.03e21), fontsize=12)
        
       
        elif y2-y1 > 0.01e21:
            x_pos = 1.04
            ax.annotate('', xy=(x_pos, y1), xycoords=("axes fraction","data"), xytext=(x_pos, y2), 
                    arrowprops=dict(arrowstyle="<->", color=ar_color,linewidth=1.5))
            ax.annotate(ar_text+'%', xy=(x_pos_tex+0.02, y1), xycoords=("axes fraction","data"), xytext=(x_pos_tex+0.02, y_half), fontsize=12)
        elif y2-y1 < -0.1e20: # rivers in fig 1
            x_pos = 1.02
            ax.annotate('', xy=(x_pos, y1), xycoords=("axes fraction","data"), xytext=(x_pos, y2), 
                    arrowprops=dict(arrowstyle="<->", color=ar_color,linewidth=1.5))
            ax.annotate(ar_text+'%', xy=(x_pos_tex+0.02, y1-0.1e20), xycoords=("axes fraction","data"), xytext=(x_pos_tex+0.02, -y_half-0.1e20), fontsize=12)
 

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

    a1 = lakeheat_onlyresclimate_anom_ensmean_ts
    a2 = riverheat_anom_ensmean_ts
    a3 = lakeheat_climate_anom_ensmean_ts
    a4 = lakeheat_res_anom_ensmean_ts
    # a1 = lakeheat_onlyresclimate_anom_ensmean_ts
    # a2 = lakeheat_climate_anom_ensmean_ts
    # a3 = lakeheat_res_anom_ensmean_ts

    # reservoirs climate change
    line1, = ax1.plot(x_values,a1, color=colors_res[0])

    # rivers
    line2, = ax1.plot(x_values,a1+a2, color=colors_rivers[0])
    #area2  = ax1.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmean_ts,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts, color='sandybrown')

    # natural lakes
    line3, = ax1.plot(x_values,a1+a2+a3, color=colors_total)
    #area3  = ax1.fill_between(x_values,riverheat_anom_ensmean_ts,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts+riverheat_anom_ensmean_ts, color='lightsteelblue')

    area3 = ax1.fill_between(x_values,a1+a2,a1+a2+a3, color=colors_natlak[1])
    area2 = ax1.fill_between(x_values,a1,a1+a2, color=colors_rivers[1])
    area1 = ax1.fill_between(x_values,a1, color=colors_res[1])

    # Add joint uncertainty? 
    #if flag_uncertainty == 'envelope':
        # full envelope
        #area2 = ax1.fill_between(x_values,lakeheat_climate_anom_ensmin_ts,lakeheat_climate_anom_ensmax_ts, color='sandybrown',alpha=0.5)
    #elif flag_uncertainty =='2std':
    # 2x std error
        #under_2std = lakeheat_climate_anom_ensmean_ts - 2*lakeheat_climate_anom_std_ts
        #upper_2std = lakeheat_climate_anom_ensmean_ts + 2*lakeheat_climate_anom_std_ts
        #area2 = ax1.fill_between(x_values,under_2std,upper_2std, color='sandybrown',alpha=0.5)

    ax1.set_xlim(xlims)
    ax1.set_xticks(xticks)
    #ax1.set_ylim(-0.4e20,1e20)
    ax1.set_ylabel('Energy [J]')
    ax1.set_title('Heat accumulation from climate change', loc='right', fontsize=13)
    ax1.legend((area1,area3,area2, line3),['reservoir heat uptake','natural lake heat uptake','river heat uptake','total heat uptake'],frameon=False,loc='upper left', bbox_to_anchor = (-0.01,0.95),prop={"size":11})

    ax1.text(0.03, 0.93, '(a)', transform=ax1.transAxes, fontsize=12)

    # add arrows

    total = (lakeheat_climate_anom_ensmean_ts+lakeheat_onlyresclimate_anom_ensmean_ts+riverheat_anom_ensmean_ts)
    clim_frac = np.round(lakeheat_climate_anom_ensmean_ts[-1]/total[-1] *100,1)
    resclim_frac = np.round(lakeheat_onlyresclimate_anom_ensmean_ts[-1]/total[-1] *100,1)
    river_frac = np.round(riverheat_anom_ensmean_ts[-1]/total[-1] *100,1)

    # clim only
    y1 = a1[-1]+a2[-1]
    y2 = y1+a3[-1]
    add_arrow_value(ax1,y1,y2,colors_natlak[0],str(clim_frac))

    # rivers
    y1 = a1[-1]
    y2 = y1+a2[-1]
    print(y2-y1)
    add_arrow_value(ax1,y1-0.1e20,y2,colors_rivers[0],str(river_frac))
   
    # resclim
    y1 = 0
    y2 = a1[-1]
    add_arrow_value(ax1,y1,y2,colors_res[0],str(resclim_frac),flag_onlyres=1,y1_textpos_toadd=0.005e21)


    # -------------------------------------
    # subplot 2: reservoir heat uptake
    line_zero = ax2.plot(x_values, np.zeros(np.shape(x_values)), linewidth=0.5,color='darkgray')

    # reservoirs climate change
    line1, = ax2.plot(x_values,a1, color=colors_res[0])
    line2, = ax2.plot(x_values,a1+a2, color=colors_rivers[0])
    line3, = ax2.plot(x_values,a1+a2+a3, color=colors_natlak[0])
    line4, = ax2.plot(x_values,a1+a2+a3+a4, color=colors_total)


    area4 = ax2.fill_between(x_values,a1+a2+a3,a1+a2+a3+a4, color=colors_onlyres[1])   
    area3 = ax2.fill_between(x_values,a1+a2,a1+a2+a3, color=colors_natlak[1])
    area2 = ax2.fill_between(x_values,a1,a1+a2, color=colors_rivers[1])
    area1 = ax2.fill_between(x_values,a1, color=colors_res[1])

    # area3 = ax2.fill_between(x_values,lakeheat_climate_anom_ensmean_ts+lakeheat_onlyresclimate_anom_ensmean_ts,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts+lakeheat_res_anom_ensmean_ts, color=colors_onlyres[1])
    # area2  = ax2.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmean_ts,lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_climate_anom_ensmean_ts, color=colors_natlak[1])
    # area1 = ax1.fill_between(x_values,lakeheat_onlyresclimate_anom_ensmean_ts, color=colors_res[1])


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

    ax2.set_xlim(xlims)
    ax2.set_xticks(ticks= xticks)
    #ax1.set_ylim(-0.4e20,1e20)
    ax2.set_ylabel('Energy [J]')
    ax2.set_title('Heat accumulation with redistribution', loc='right', fontsize=13)
    ax2.text(0.03, 0.93, '(b)', transform=ax2.transAxes, fontsize=12)
    ax2.legend((area3,area2,area1,area4,line4),['natural lake heat uptake','reservoir heat uptake', 'river heat uptake', 'reservoir expansion', 'total uptake and expansion'],frameon=False,loc='upper left', bbox_to_anchor = (-0.01,0.95), prop={"size":11})

    total = (lakeheat_climate_anom_ensmean_ts+lakeheat_onlyresclimate_anom_ensmean_ts+lakeheat_res_anom_ensmean_ts+riverheat_anom_ensmean_ts)
    res_frac = np.round(lakeheat_res_anom_ensmean_ts[-1]/total[-1] *100,1)
    clim_frac = np.round(lakeheat_climate_anom_ensmean_ts[-1]/total[-1] *100,1)
    resclim_frac = np.round(lakeheat_onlyresclimate_anom_ensmean_ts[-1]/total[-1] *100,1)
    river_frac = np.round(riverheat_anom_ensmean_ts[-1]/total[-1] *100,1)


    # res only 
    y1 = a1[-1]+a2[-1]+a3[-1]
    y2 = y1+a4[-1]
    add_arrow_value(ax2,y1,y2,colors_onlyres[0],str(res_frac))

    # clim only
    y1 = a1[-1]+a2[-1]
    y2 = y1+a3[-1]
    add_arrow_value(ax2,y1+0.08e21,y2+0.08e21,colors_natlak[0],str(clim_frac))

    # river
    y1 = 0
    y2 = a1[-1]
    add_arrow_value(ax2,y1,y2,colors_rivers[0],str(river_frac),flag_onlyres=False,y1_textpos_toadd=0.025e21,flag_gobelow=True)

    # resclim
    y1 = a1[-1]
    y2 =y1+a2[-1]
    add_arrow_value(ax2,y1,y2,colors_res[0],str(resclim_frac),flag_onlyres=2,y1_textpos_toadd=0.025e21)


    #f.suptitle('Reference period 1900-1929, 5 year moving average')
    plt.tight_layout()
    if flag_save_plots:
        plt.savefig(plotdir+'fig2_heat_acc'+'.jpeg',dpi=1000, bbox_inches='tight')


#%%
