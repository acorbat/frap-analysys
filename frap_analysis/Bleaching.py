# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:42:34 2016

@author: Agus
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif

from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu

os.chdir(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Primeras Mediciones\2016-09-30\Smaug')

#%% 

def get_timepoint(filepath):
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
                        
            except:
                continue
    timepoint = float(timepoint)
    timepoint *= 1e-9 # transform to seconds
    return timepoint

def Exp_decay(t, A, k, offset):
    return A * np.exp(-k * t) + offset

def double_exp_decay(t, A1, A2, k1, k2, offset):
    return Exp_decay(t, A1, k1, offset) + Exp_decay(t, A2, k2, 0)

def Frap_Func(t, A, immobile_frac, tau):
    return (1-immobile_frac) - A * np.exp(-t / tau)

def listar_archivos():
    files = [file for file in os.listdir() if file.endswith('.tif') and file.split('_')[2]=='ble']
    file_dict = {}
    for file in files:
        cell = file.split('_')[1]
        file_dict[cell] = file
    return file_dict

def generate_masks(img):
    log_img = np.log(img+1)
    first_thresh = threshold_otsu(log_img)
    first_mask = log_img>first_thresh
    second_thresh = threshold_otsu(log_img[first_mask])
    
    return np.exp(second_thresh)-1, np.exp(first_thresh)-1

def calculate_atr(img, mask):
    ROI = img[mask]
    mean = np.nanmean(ROI)
    std = np.nanstd(ROI)
    area = len(ROI)
    return mean, std, area

def calculate_ser(series, mask):
    means = []
    stds = []
    areas = []
    for serie in series:
        mean, std, area = calculate_atr(serie, mask)
        means.append(mean)
        stds.append(std)
        areas.append(area)
    
    means =  np.asarray(means)
    stds =  np.asarray(stds)
    areas =  np.asarray(areas)
    return means, stds, areas

def calculate_fluorescence(BG, CP, GR):
    return GR-CP

def add_foregroundSeries(df):
    foregrounds = ['BG', 'CP', 'GR']
    for mask_val, foreground in enumerate(foregrounds):
        means_fg = []
        stds_fg = []
        areas_fg = []
        for i in df.index:
            series = df['series'][i]
            mask = df['mask'][i]
        
            this_mask = mask == mask_val
            means, stds, areas = calculate_ser(series, this_mask)
            
            means_fg.append(means)
            stds_fg.append(stds)
            areas_fg.append(areas)
        
        df['mean_'+foreground] = means_fg
        df['std_'+foreground] = stds_fg
        df['area_'+foreground] = areas_fg
    
    return df

def add_Fluorescence(df):
    fs = []
    for i in df.index:
        BG = df['mean_BG'][i]
        CP = df['mean_CP'][i]
        GR = df['mean_GR'][i]
        f = calculate_fluorescence(BG, CP, GR)
        fs.append(f)
    df['f'] = fs
    return df

def add_fitParams(df, Plot=False):
    Amps = []
    ks = []
    offs = []
    for i in df.index:
        this_f = df['f'][i]
        
        timepoint = df['timepoint'][i]
        max_temp = len(this_f)*timepoint
        t = np.arange(0, max_temp, timepoint)
        t = t[:len(this_f)]
        popt, pcov = curve_fit(Exp_decay, t, this_f, p0=[2000, 15, 5])
        
        Amp, k, off = popt[0], popt[1], popt[2]
        
        Amps.append(Amp)
        ks.append(k)
        offs.append(off)
        
        if Plot:
            plt.semilogy(t, Exp_decay(t, Amp, k, off), 'r')
            plt.scatter(t, this_f)
            plt.title(df['cell'][i])
            #plt.xlim((0,2))
            plt.show()
            print('Amplitude: '+str(Amp))
            print('k: '+str(k))
            print('offset: '+str(off))
    
    df['Amp'] = Amps
    df['k'] = ks
    df['offset'] = offs
    
    return df


def add_double_fitParams(df, Plot=False):
    A1s = []
    A2s = []
    k1s = []
    k2s = []
    doffs = []
    for i in df.index:
        this_f = df['f'][i]
        
        timepoint = df['timepoint'][i]
        max_temp = len(this_f)*timepoint
        t = np.arange(0, max_temp, timepoint)
        t = t[:len(this_f)]
        popt, pcov = curve_fit(double_exp_decay, t, this_f, p0=[2000, 1000, 1, 15, 5])
        
        A1, A2, k1, k2, off = popt[0], popt[1], popt[2], popt[3], popt[4]
        
        A1s.append(A1)
        A2s.append(A2)
        k1s.append(k1)
        k2s.append(k2)
        doffs.append(off)
        
        if Plot:
            plt.semilogy(t, double_exp_decay(t, A1, A2, k1, k2, off), 'r')
            plt.scatter(t, this_f)
            plt.title(df['cell'][i])
            #plt.xlim((0,2))
            plt.show()
            print('Amplitudes: '+str(A1)+' '+str(A2))
            print('k: '+str(k1)+' '+str(k2))
            print('offset: '+str(off))
    
    df['A1'] = A1s
    df['A2'] = A2s
    df['k1'] = k1s
    df['k2'] = k2s
    df['d_offset'] = doffs
    
    return df

def Plot_all(df):
    for i in df.index:
        this_f = df['f'][i]
        timepoint = df['timepoint'][i]
        max_temp = len(this_f)*timepoint
        t = np.arange(0, max_temp, timepoint)
        t = t[:len(this_f)]
        
        plt.semilogy(t, this_f)
        #plt.xlim((0, 2))
        plt.xlabel('Time (s)')
        plt.ylabel('Fluorescence (u.a.)')
    plt.show()

#%% Generate masks of background, citoplasm and granule

file_dict = listar_archivos()
df = pd.DataFrame(columns=['cell', 'series', 'mask', 'timepoint'])

for cell, file in file_dict.items():
    img = tif.imread(file)
    this_img = img[0].copy()
    thresh_1, thresh_2 = generate_masks(this_img)
    BG_mask = this_img>thresh_1
    GR_mask = this_img>thresh_2
    Mask = BG_mask.astype(float) + GR_mask.astype(float)
    timepoint = get_timepoint(file)
    """
    plt.imshow(img[0])
    plt.contour(Mask)
    plt.show()
    """
    df = df.append({'cell':cell, 'series':img, 'mask':Mask, 'timepoint':timepoint}, ignore_index=True)

#%% Calculate mean and std of different regions

df = add_foregroundSeries(df)

#%% Calculate Fluorescence signal

df = add_Fluorescence(df)

#%% Fit with single decay exponential

df = add_fitParams(df, Plot=True)

#%% Fit with double decay exponential

df = add_double_fitParams(df, Plot=True)