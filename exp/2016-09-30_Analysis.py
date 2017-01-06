# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:00:32 2017

@author: Agus
"""

import os
os.chdir(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\FRAP_Analysis\src')

from Frap_Analysis import *

#%% Define folders and constants

fp = r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Primeras Mediciones\2016-09-30\Smaug'

#%% Generate dictionary of files to be analyzed

df = generate_df(fp)

#%% add time series of  masks

df = add_foregroundSeries(df)

#%% add fluorescence time series

df =  add_fluorescence(df)

#%% Calculate and add corrected fluorescence normalized to initial FRAP intensity

df = add_f_corr(df)

#%% Fit with FRAP typical function

df =  add_fitParams(df, Plot=True)

#%% cells not working

non_cell = ['C3F2', 'C7F4', 'C7F2', 'C5F4', 'C6F1', 'C1F1', 'C4F1', 'C4F2', 'C3F3', 'C2F5', 'C6F5', 'C3F4', 'C3F1', 'C5F3', 'C1F5', 'C2F4', 'C2F1']

corr_df = pd.DataFrame()

for i in df.index:
    if df['cell'][i] not in non_cell:
        corr_df = corr_df.append({'Amp':df['Amp'][i], 'Imm':df['Imm'][i], 'k':1/df['tau'][i], 'area':df['pre_area'][i], 'pre_I':df['pre_I'][i]}, ignore_index=True)