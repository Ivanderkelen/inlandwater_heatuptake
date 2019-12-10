"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Scripts with example of regression model for river stream temperatures from Puznet et al., 2012

"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rc('axes',edgecolor='grey')
mpl.rc('axes',labelcolor='dimgrey')
mpl.rc('xtick',color='dimgrey')
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',color='dimgrey')
mpl.rc('ytick',labelsize=12)
mpl.rc('axes',titlesize=15)
mpl.rc('axes',labelsize=12)
mpl.rc('legend',fontsize='large')
mpl.rc('text',color='dimgrey')


# Global standard regression equation values from Punzet et al. (2012)
c0 = 32 
c1 = -0.13
c2 = 1.94

# regression formula 
tas=np.arange(-30,50,1)
watertemp = c0/(1+np.exp(c1*tas+c2))
f,ax = plt.subplots()
line1 = ax.plot(tas,watertemp, color='coral')

ax.set_xlim(tas[0],tas[-1])
ax.set_ylim(watertemp[0],watertemp[-1])
#ax.grid(color='lightgrey')
ax.set_ylabel('Stream temperature [°C]')
ax.set_title('Global standard stream temperature regression', pad=15)
ax.set_xlabel('Air temperature [°C]')

# save the figure, adjusting the resolution 
plt.savefig('/home/inne/documents/phd/data/processed/isimip_lakeheat/plots/streamtemp_regression.png')
