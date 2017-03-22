# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:54:44 2017

@author: Agus
"""

import pathlib
import os

os.chdir(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\FRAP_Analysis\src') # TODO corregir esto
import Frap_Analysis as fa

#%% Set experiments data directory

data_dir = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\2017-03-17')
folders = {f.name: f for f in data_dir.iterdir() if f.is_dir()}

#%% 

this_folder = 'FL'
experiments = {f.name: f for f in folders[this_folder].iterdir() if f.is_dir()}

this_experiment = 'GR'

#%% Generate df for this experiment

df = fa.generate_df(experiments[this_experiment])

#%% add time series of  masks

df = fa.add_foregroundSeries(df)

#%% add fluorescence time series

df =  fa.add_fluorescence(df)

#%% Calculate and add corrected fluorescence normalized to initial FRAP intensity

df = fa.add_f_corr(df)

#%% Fit with FRAP typical function

df =  fa.add_fitParams(df, Plot=True)

#%% trials
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import correlate2d

for serie in df.series.values:
    for img in serie:
        this_mask = np.zeros(img.shape)
        for i in range(3, 4):
            mask = fa.generate_masks(img, i)
            if not mask.any():
                continue
            this_mask = mask + this_mask
        
        plt.imshow(img)
        try:
            plt.contour(this_mask)
        except:
            pass
        plt.show()
        print('a')