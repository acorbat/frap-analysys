# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:17:56 2016

@author: Agus
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif

from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_opening

os.chdir(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\FRAP_Analysis\src')
import oiffile as oif

#%% Define useful functions

# Functions to fit with

def Frap_Func(t, A, immobile_frac, tau):
    return (1-immobile_frac) - A * np.exp(-t / tau)

# Function to generate filepath dictionary

def generate_FileDict(filepath):
    os.chdir(filepath)
    files = [file for file in os.listdir() if file.endswith('.tif')]
    
    File_Dict = {}
    for file in files:
        file_parts = file.split('_')
        cell = file_parts[1]
        pinhole = file_parts[3]
        obj = file_parts[4][0:2]
        File_Dict[cell, pinhole, obj] = file
    
    return File_Dict

# Functions to get metadata from oif files

def get_timepoint(filepath):
    folder = r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Primeras Mediciones\2016-09-23\FRAP_ph'
    filepath = filepath.replace('tif', 'oif')
    file_parts = filepath.split('_')
    filepath = folder + '\FRAP_' + file_parts[1] + '_pre_01'
    if int(file_parts[3])==600:
        filepath = filepath + '_PH_600.oif.files\s_C001T001.pty'
    elif int(file_parts[3])==110:
        filepath = filepath + '.oif.files\s_C001T001.pty'
    try:
        with open(filepath, 'rb') as file:
            for row in file:
                try:
                    r = str(row.decode("utf-8"))
                    if 'T\00i\00m\00e\00 \00P\00e\00r\00 \00F\00r\00a\00m\00e' in r:
                        timepoint = ''
                        for character in r:
                            try:
                                float(character)
                                timepoint = timepoint + character
                            except:
                                continue 
                        timepoint = float(timepoint)
                        timepoint *= 1e-9 # transform to seconds
                        return timepoint
                except:
                    continue
    except:
        filepath = filepath.replace('01.oif','02.oif')
        with open(filepath, 'rb') as file:
            for row in file:
                try:
                    r = str(row.decode("utf-8"))
                    if 'T\00i\00m\00e\00 \00P\00e\00r\00 \00F\00r\00a\00m\00e' in r:
                        timepoint = ''
                        for character in r:
                            try:
                                float(character)
                                timepoint = timepoint + character
                            except:
                                continue 
                        timepoint = float(timepoint)
                        timepoint *= 1e-9 # transform to seconds
                        return timepoint
                except:
                    continue

# Functions to crop and mask images

def generate_masks(img, iterations):
    log_img = np.log(img+1)
    thresh = threshold_otsu(log_img)
    thresh = np.exp(thresh)-1
    mask = img>thresh
    if iterations>0:
        mask = binary_opening(mask, iterations=iterations)
    return mask

# Functions to calculate sum or means of intensities

def calculate_areas(img):
    Ints = []
    CPs = []
    areas = []
    for i in range(2, 4):
        mask = generate_masks(img, i)
        if not mask.any():
            continue
        ROI = img[mask]
        Ints.extend(ROI)
        CPs.extend(img[~mask])
        areas.append(len(ROI))
    mean_I = np.nanmean(Ints)
    std_I = np.nanstd(Ints)
    mean_CP = np.nanmean(CPs)
    std_CP = np.nanstd(CPs)
    area = np.mean(areas)
    if np.isnan(mean_I):
        mean_I = mean_CP
        std_I = std_CP
        area = 0
    return mean_I, std_I, mean_CP, std_CP, area

def calculate_series(series):
    means_I = []
    stds_I = []
    means_CP = []
    stds_CP = []
    areas = []
    for img in series:
        mean_I, std_I, mean_CP, std_CP, area = calculate_areas(img)
        means_I.append(mean_I)
        stds_I.append(std_I)
        means_CP.append(mean_CP)
        stds_CP.append(std_CP)
        areas.append(area)
    
    means_I =  np.asarray(means_I)
    stds_I =  np.asarray(stds_I)
    means_CP =  np.asarray(means_CP)
    stds_CP =  np.asarray(stds_CP)
    areas =  np.asarray(areas)
    return means_I, stds_I, means_CP, stds_CP, areas

def calculate_fluorescence(CP, GR, DK):
    CP = CP - DK
    GR = GR - DK
    return GR/CP

# Functions that create and add columns to pandas dataframe

def generate_df(fp):
    df = pd.DataFrame()
    for key, file in FileDict.items():
        if key[2] == 'FC':
            series = tif.imread(file)
            timepoint = get_timepoint(file)
            cell = key[0]
            pinhole = key[1]
            df = df.append({'cell':cell, 'pinhole':pinhole, 'series':series, 'timepoint':timepoint}, ignore_index=True)
    return df

def add_foregroundSeries(df):
    means_Is = []
    stds_Is = []
    means_CPs = []
    stds_CPs = []
    areass = []
    for i in df.index:
        series = df['series'][i]
        
        means_I, stds_I, means_CP, stds_CP, areas = calculate_series(series)
        
        means_Is.append(means_I)
        stds_Is.append(stds_I)
        means_CPs.append(means_CP)
        stds_CPs.append(stds_CP)
        areass.append(areas)
    
    df['mean_GR'] = means_Is
    df['std_GR'] = stds_Is
    df['mean_CP'] = means_CPs
    df['std_CP'] = stds_CPs
    df['area'] = areass

    return df

def add_far_BG(df):
    mean_farBGs = []
    for i in df.index:
        cell = df.cell[i]
        pinhole = df.pinhole[i]
        series = tif.imread(FileDict[cell, pinhole, 'BG'])
        mean_farBG = []
        for img in series:
            this_mean_farBG = np.nanmean(img.flatten())
            mean_farBG.append(this_mean_farBG)
        mean_farBGs.append(mean_farBG)
    df['far_BG'] = mean_farBGs
    return df

def add_DarkCounts(df):
    key = ('C7F2', '110', 'DK')
    DKimg = tif.imread(FileDict[key])
    dark = np.nanmean(DKimg)
    df['dark'] = dark
    return df

def add_fluorescence(df):
    fs = []
    for i in df.index:
        CP = df['mean_CP'][i]
        GR = df['mean_GR'][i]
        DK = df['dark'][i]
        f = calculate_fluorescence(CP, GR, DK)
        fs.append(f)
    df['f'] = fs
    return df

def add_f_corr(df):
    pre_bleachs = []
    post_bleachs = []
    pre_areas = []
    for i in df.index:
        this_f = df['f'][i]
        pre_bleach = np.mean(this_f[:20])
        pre_area = np.mean(df['area'][i][:20])
        post_bleach = this_f[20:]/pre_bleach
        
        pre_bleachs.append(pre_bleach)
        pre_areas.append(pre_area)
        post_bleachs.append(post_bleach)
    
    df['pre_I'] = pre_bleachs
    df['pre_area'] = pre_areas
    df['f_corr'] = post_bleachs
    return df

def add_fitParams(df, Plot=False):
    Amps = []
    Imms = []
    taus = []
    for i in df.index:
        print(df['cell'][i])
        this_f = df['f_corr'][i]
        
        timepoint = df['timepoint'][i]
        max_temp = len(this_f)*timepoint
        t = np.arange(0, max_temp, timepoint)
        t = t[:len(this_f)]
        popt, pcov = curve_fit(Frap_Func, t, this_f, p0=[0.2, 15, 5])
        
        Amp, Imm, tau = popt[0], popt[1], popt[2]
        
        Amps.append(Amp)
        Imms.append(Imm)
        taus.append(tau)
        
        if Plot:
            plt.plot(t, Frap_Func(t, Amp, Imm, tau), 'r')
            plt.scatter(t, this_f)
            plt.title(df['cell'][i])
            plt.xlabel('Time (s)')
            plt.ylabel('Fraction I (u.a.)')
            #plt.xlim((0,2))
            plt.show()
            print('Amplitude: '+str(Amp))
            print('Imm Frac: '+str(Imm))
            print('tau: '+str(tau))
            print('k: '+str(1/tau))
    
    df['Amp'] = Amps
    df['Imm'] = Imms
    df['tau'] = taus
    
    return df

def add_fitParams_CP(df, Plot=False):
    Amps = []
    Imms = []
    taus = []
    for i in df.index:
        print(df['cell'][i])
        this_CP = df['mean_CP'][i]-df['dark'][i]
        this_f = this_CP[20:]/np.nanmean(this_CP[:20])
        
        timepoint = df['timepoint'][i]
        max_temp = len(this_f)*timepoint
        t = np.arange(0, max_temp, timepoint)
        t = t[:len(this_f)]
        popt, pcov = curve_fit(Frap_Func, t, this_f, p0=[0.2, 15, 5])
        
        Amp, Imm, tau = popt[0], popt[1], popt[2]
        
        Amps.append(Amp)
        Imms.append(Imm)
        taus.append(tau)
        
        if Plot:
            plt.plot(t, Frap_Func(t, Amp, Imm, tau), 'r')
            plt.scatter(t, this_f)
            plt.title(df['cell'][i])
            plt.xlabel('Time (s)')
            plt.ylabel('Fraction I (u.a.)')
            #plt.xlim((0,2))
            plt.show()
            print('Amplitude: '+str(Amp))
            print('Imm Frac: '+str(Imm))
            print('tau: '+str(tau))
            print('k: '+str(1/tau))
    
    df['Amp_CP'] = Amps
    df['Imm_CP'] = Imms
    df['tau_CP'] = taus
    
    return df

#%% Define folders and constants

fp = r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Primeras Mediciones\2016-09-23\FRAP_ph_process'

#%% Generate dictionary of files to be analyzed

FileDict = generate_FileDict(fp)
df = generate_df(fp)

#%% add time series of  masks

df = add_foregroundSeries(df)

#%% add time series of  far backgrounds

df = add_far_BG(df)

#%% add Dark Counts

df = add_DarkCounts(df)

#%% add fluorescence time series

df =  add_fluorescence(df)

#%% Calculate and add corrected fluorescence normalized to initial FRAP intensity

df = add_f_corr(df)

#%% Fit with FRAP typical function

df =  add_fitParams(df, Plot=True)

#%% Fit with FRAP typical function

df =  add_fitParams_CP(df, Plot=True)

#%% cells not working

non_cell = ['C3F1', 'C4F1', 'C2F1']

corr_df = pd.DataFrame()

for i in df.index:
    if df['cell'][i] not in non_cell:
        corr_df = corr_df.append({'Amp':df['Amp'][i], 'Imm':df['Imm'][i], 'k':1/df['tau'][i], 'area':df['pre_area'][i], 'pre_I':df['pre_I'][i], 'pinhole':df['pinhole'][i], 'Amp_CP':df['Amp_CP'][i], 'Imm_CP':df['Imm_CP'][i], 'k_CP':1/df['tau_CP'][i]}, ignore_index=True)

#%% Import seaborn and calculate cross correlations

import seaborn as sns

sns.pairplot(corr_df, vars=['Amp', 'Imm', 'k', 'area', 'pre_I', 'Amp_CP', 'Imm_CP', 'k_CP'], hue='pinhole')