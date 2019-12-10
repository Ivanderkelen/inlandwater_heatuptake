
"""
Author      : Inne Vanderkelen (inne.vanderkelen@vub.be)
Institution : Vrije Universiteit Brussel (VUB)
Date        : November 2019

Script containing all functions to aggregate values for lake heat calculations
    - calculate timeseries
    - calculate anomalies

"""

import numpy as np

from dict_functions import *


# ------------------------------------
# FUNCTIONS


# -------------------
# CALCULATE LAKE HEAT ANOMALIES

lakeheat_anom = calc_anomalies(lakeheat)


# -----------------------------------------------------------------------------
# Aggregate - calculate timeseries and averages

lakeheat_anom_ts = timeseries(lakeheat_anom)

lakeheat_anom_ens = ensmean(lakeheat_anom)

lakeheat_anom_ensmean_ts = moving_average(ensmean_ts(lakeheat_anom))
lakeheat_anom_ensmin_ts  = moving_average(ensmin_ts(lakeheat_anom))
lakeheat_anom_ensmax_ts  = moving_average(ensmax_ts(lakeheat_anom))

lakeheat_anom_ens_spmean = ens_spmean(lakeheat_anom)

lakeheat_anom_spcumsum = ensmean_spcumsum(lakeheat_anom)

lake_anom = [lakeheat_anom_ts['CLM45']['gfdl-esm2m'][-1], lakeheat_anom_ts['CLM45']['ipsl-cm5a-lr'][-1], lakeheat_anom_ts['CLM45']['hadgem2-es'][-1],lakeheat_anom_ts['CLM45']['miroc5'][-1] ]
np.std(lake_anom)
lakeheat_anom_ts['CLM45']['ipsl-cm5a-lr'][-1]