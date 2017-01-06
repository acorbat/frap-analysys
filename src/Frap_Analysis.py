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

os.chdir(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Analisis')
import oiffile as oif

#%% Define useful functions

# Functions to fit with

def Frap_Func(t, A, immobile_frac, tau):
    return (1-immobile_frac) - A * np.exp(-t / tau)

# Function to generate filepath dictionary

def generate_FileDict(filepath):
    os.chdir(filepath)
    files = [file for file in os.listdir() if file.endswith('.oif') and '_post_' in file or file.endswith('.oif') and '_pre_' in file]
    
    File_Dict = {}
    for file in files:
        file_parts = file.split('_')
        cell = file_parts[1]
        moment = file_parts[2]
        File_Dict[cell, moment] = file
    
    return File_Dict

# Functions to get metadata from oif files

def get_timepoint(filepath):
    filepath = filepath + '.files\s_C001T001.pty'
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

def get_clip(filepath):
    Axises = ['X', 'Y']
    clip = {}
    for Axis in Axises:
        if Axis == 'X':
            Axis_Name = '[\00A\00x\00i\00s\00 \0000\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        elif Axis == 'Y':
            Axis_Name = '[\00A\00x\00i\00s\00 \0001\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        
        with open(filepath, 'rb') as file:
            found_Axis_0 = False
            for row in file:
                try:
                    r = str(row.decode("utf-8"))
                    if Axis_Name in r:
                        found_Axis_0 = True
                    if found_Axis_0 and 'C\00l\00i\00p\00P\00o\00s\00i\00t\00i\00o\00n' in r:
                        Pos = ''
                        for character in r:
                            try:
                                float(character)
                                Pos = Pos + character
                            except:
                                continue
                        clip[Axis] = int(Pos)
                        break
                except:
                    continue
    return (clip['X'], clip['Y'])

def get_size(filepath):
    Axises = ['X', 'Y']
    Sizes = {}
    for Axis in Axises:
        if Axis == 'X':
            Axis_Name = '[\00A\00x\00i\00s\00 \0000\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        elif Axis == 'Y':
            Axis_Name = '[\00A\00x\00i\00s\00 \0001\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        
        with open(filepath, 'rb') as file:
            found_Axis_0 = False
            for row in file:
                try:
                    r = str(row.decode("utf-8"))
                    if Axis_Name in r:
                        found_Axis_0 = True
                    if found_Axis_0 and 'M\00a\00x\00S\00i\00z\00e' in r and 'G\00U\00I\00' not in r:
                        Size = ''
                        for character in r:
                            try:
                                float(character)
                                Size = Size + character
                            except:
                                continue
                        Sizes[Axis] = int(Size)
                        break
                except:
                    continue
    return (Sizes['X'], Sizes['Y'])

# Functions to crop and mask images

def crop_and_conc(cell):
    file_pre = FileDict[cell, 'pre']
    file_post = FileDict[cell, 'post']
    file_ble = file_post.replace('post', 'ble')
    
    Size = get_size(file_ble)
    start = get_clip(file_ble)
    end = np.asarray(start) + np.asarray(Size)
    
    pre_imgs = oif.imread(file_pre)
    post_imgs = oif.imread(file_post)
    
    pre_imgs_clipped = [img[start[1]:end[1],start[0]:end[0]] for img in pre_imgs[0][:]]
    post_imgs_clipped = [img[start[1]:end[1],start[0]:end[0]] for img in post_imgs[0][:]]
    
    pre_imgs_clipped = np.asarray(pre_imgs_clipped)
    post_imgs_clipped = np.asarray(post_imgs_clipped)
    
    imgs = np.concatenate((pre_imgs[0], post_imgs[0]))
    
    tot_Ints = []
    for img in imgs:
        img[start[1]-10:end[1]+10,start[0]-10:end[0]+10] = np.nan
        #img[0:start[1]-10,:] = np.nan
        #img[end[1]+5:,:] = np.nan
        #img[:,0:start[0]-10] = np.nan
        #img[:,end[0]+5] = np.nan
        #plt.imshow(img)
        #plt.show()
        img = img.flatten()
        tot_Int = np.nanmean(img)
        tot_Ints.append(tot_Int)
    
    imgs_clipped = np.concatenate((pre_imgs_clipped, post_imgs_clipped))
    
    return imgs_clipped, tot_Ints

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
    for i in range(1, 4):
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

def calculate_fluorescence(CP, GR):
    return GR/CP

# Functions that create and add columns to pandas dataframe

def generate_df(fp):
    cells = set()
    for key in FileDict.keys():
        cells.add(key[0])
    
    df = pd.DataFrame()
    for cell in cells:
        series, tot_Int = crop_and_conc(cell)
        timepoint = get_timepoint(FileDict[cell, 'post'])
        df = df.append({'cell':cell, 'series':series, 'timepoint':timepoint, 'total_Int':tot_Int}, ignore_index=True)
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

def add_fluorescence(df):
    fs = []
    for i in df.index:
        CP = df['mean_CP'][i]
        GR = df['mean_GR'][i]
        f = calculate_fluorescence(CP, GR)
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
        popt, pcov = curve_fit(Frap_Func, t, this_f, p0=[2000, 15, 5])
        
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


#%% Define folders and constants

fp = r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Primeras Mediciones\2016-09-30\Smaug'

#%% Generate dictionary of files to be analyzed

FileDict = generate_FileDict(fp)
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