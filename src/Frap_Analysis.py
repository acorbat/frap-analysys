# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:17:56 2016

@author: Agus
"""
import pathlib
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
import skimage.measure as meas

from scipy.signal import correlate2d
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.ndimage.morphology import binary_opening
from skimage.filters import threshold_otsu
from skimage.draw import circle

os.chdir(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\FRAP_Analysis\src')
import oiffile as oif

#%% Define useful functions

# Functions to fit with

def Frap_Func(t, A, immobile_frac, tau):
    """
    Returns (1-immobile_frac) - A * np.exp(-t / tau)
    """
    return (1-immobile_frac) - A * np.exp(-t / tau)

# Function to generate filepath dictionary

def generate_FileDict(filepath):
    """
    Generates a dictionary with paths for each cell and time period
    
    Inputs:
    filepath -- filepath to folder with all the .oif files
    Returns:
    File_Dict -- Dictionary where keys are [cell, period] and values are the corresponding full path
    """
    filepath = pathlib.Path(filepath)
    File_Dict = {(f.name.split('_')[0], f.name.split('_')[1][:-4]): f for f in filepath.glob('*.oif') if '_pos' in str(f.name) or '_pre' in str(f.name)}
    return File_Dict

# Functions to get metadata from oif files

def get_info(filepath):
    """
    Parses info as cell, cell number and foci number
    
    Inputs
    filepath -- filepath of the image file
    Returns
    cell   -- CnFn (key of the dictionary)
    cell_n -- Number of cell studied
    foci   -- Number of foci of the same cell studied
    """
    filepath = pathlib.Path(filepath)
    filename = filepath.name
    file_parts = filename.split('_')
    cell = file_parts[0]
    cell_n = re.search('C.*F', cell)
    cell_n = int(cell_n.group(0)[1:-1])
    foci = re.search('F.*', cell)
    foci = int(foci.group(0)[1:])
    
    return cell, cell_n, foci

def get_timepoint(filepath):
    """
    Retrieves timepoint from .files folder of the .oif file selected in filepath
    """
    filepath = pathlib.Path(filepath)
    this_filepath = filepath.parent
    this_filepath = this_filepath.joinpath(filepath.name + '.files\s_C001T001.pty')
    this_filepath = pathlib.Path(this_filepath)
    
    with open(str(this_filepath), 'rb') as file:
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
    """
    Retrieves (x, y) clip start from .files folder
    """
    Axises = ['X', 'Y']
    clip = {}
    for Axis in Axises:
        if Axis == 'X':
            Axis_Name = '[\00A\00x\00i\00s\00 \0000\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        elif Axis == 'Y':
            Axis_Name = '[\00A\00x\00i\00s\00 \0001\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        
        with open(str(filepath), 'rb') as file:
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
    """
    Retrieves clip size from .files folder
    """
    Axises = ['X', 'Y']
    Sizes = {}
    for Axis in Axises:
        if Axis == 'X':
            Axis_Name = '[\00A\00x\00i\00s\00 \0000\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        elif Axis == 'Y':
            Axis_Name = '[\00A\00x\00i\00s\00 \0001\00 \00P\00a\00r\00a\00m\00e\00t\00e\00r\00s\00 \00C\00o\00m\00m\00o\00n\00]'
        
        with open(str(filepath), 'rb') as file:
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

def crop_and_conc2(cell, FileDict):
    """
    Returns a concatenated series clipped from the bleaching clip.
    
    Inputs
    cell     -- cell name to concatenate and clip
    FileDict -- Filepath dictionary generated by generate_FileDict function with keys (cell, period)
    Returns
    imgs_clipped -- image series of pre and post bleaching clipped by the selected bleaching roi
    tot_Ints     -- list of mean intensity of every image without clip
    """
    file_pre = FileDict[cell, 'pre']
    file_post = FileDict[cell, 'pos']
    file_ble = file_post.parent
    file_ble = file_ble.joinpath(str(file_post.name).replace('_pos', '_ble'))
    
    Size = get_size(file_ble)
    start = get_clip(file_ble)
    end = np.asarray(start) + np.asarray(Size)
    
    pre_imgs = oif.imread(str(file_pre))
    post_imgs = oif.imread(str(file_post))
    
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


def imcrop(im, x1, x2, y1, y2):
    assert x1 < x2
    assert y1 < y2
    sh = im.shape
    x1, x2 = np.clip([x1, x2], 0, sh[0])
    y1, y2 = np.clip([y1, y2], 0, sh[1])
    return im[x1:x2, y1:y2]
    
    
def clip(x, mn, mx):
    return x if mn <= x <= mx else (mn if x < mn else mx)    
    
def imcrop_wh(im, x1, w, y1, h):
    assert w>0
    assert h>0
    sh = im.shape
    x1 = clip(x1, 0, sh[0])
    x2 = clip(x1 + w, 0, sh[1])
    y1 = clip(y1, 0, sh[1])
    y2 = clip(y1 + h, 0, sh[1])
    return im[x1:x2, y1:y2]
    

    
def crop_and_shift(imgs, xwyh, filter_width=5, D=5):
    """
    Returns a concatenated series clipped from the bleaching clip.
    
    Inputs
    cell     -- cell name to concatenate and clip
    FileDict -- Filepath dictionary generated by generate_FileDict function with keys (cell, period)
    Returns
    imgs_clipped -- image series of pre and post bleaching clipped by the selected bleaching roi
    tot_Ints     -- list of mean intensity of every image without clip
    """

    x, w, y, h = xwyh
    len_series, sh_y, sh_x = imgs.shape
    stack = np.full((len_series, w, h), np.nan)
    out_intensity = np.full((len_series, ), np.nan)
    offsets = np.empty((len_series, 2), dtype=np.uint)
    
    pre_img = np.zeros((w,h))
    rr, cc = circle(pre_img.shape[0]//2, pre_img.shape[1]//2, 5, pre_img.shape)
    pre_img[rr, cc] = 1
    
    for ndx in range(len_series):
        img = imgs[ndx, :, :]
        cropped = imcrop_wh(img, x, w, y, h)
        p1, p2 = np.percentile(cropped, [20, 80])
        if p2 - p1 > 75:
            correlation = correlate2d(cropped, pre_img)#smooth, smooth_pre)
            pos = np.unravel_index(np.argmax(correlation), correlation.shape)
            x += (pos[0] - correlation.shape[0]//2)
            y += (pos[1] - correlation.shape[1]//2)
            cropped = imcrop_wh(img, x, w, y, h)
            
        stack[ndx, :, :] = cropped.copy()
        out_intensity[ndx] = np.nansum(img) - np.nansum(imcrop_wh(img, x-D, w+D, y-D, h+D))
        offsets[ndx, :] = (x, y)
        
    return stack, out_intensity, offsets


def crop_and_conc(cell, FileDict):
    """
    Returns a concatenated series clipped from the bleaching clip.
    
    Inputs
    cell     -- cell name to concatenate and clip
    FileDict -- Filepath dictionary generated by generate_FileDict function with keys (cell, period)
    Returns
    imgs_clipped -- image series of pre and post bleaching clipped by the selected bleaching roi
    tot_Ints     -- list of mean intensity of every image without clip
    """
    file_pre = FileDict[cell, 'pre']
    file_post = FileDict[cell, 'pos']
    file_ble = file_post.parent
    file_ble = file_ble.joinpath(str(file_post.name).replace('_pos', '_ble'))

    Size = get_size(file_ble)
    start = get_clip(file_ble)
    
    stack1, out_intensity1, offsets = crop_and_shift(oif.imread(str(file_pre))[0], (start[1], Size[1], start[0], Size[0]))

    stack2, out_intensity2, offsets = crop_and_shift(oif.imread(str(file_post))[0], (offsets[-1, 0], Size[1], offsets[-1, 1], Size[0]))
    
    return np.concatenate((stack1, stack2)), np.concatenate((out_intensity1, out_intensity2))
    

def generate_masks(img, iterations):
    """
    Returns mask of image (img) after applying Otsu and binary_opening iterations times
    
    It applies otsu threshold over the log intensity image and 
    then iterates over erosion and dilation of the image to 
    delete lonely pixels or holes in mask.
    Inputs
    img        -- image to generate mask
    iterations -- times that binary_opening is applied over image
    Returns
    mask -- boolean mask generated
    """
    log_img = np.log(img+1)
    thresh = threshold_otsu(log_img)
    thresh = np.exp(thresh)-1
    mask = img>thresh
    if iterations>0:
        mask = binary_opening(mask, iterations=iterations)
    return mask

# Functions to calculate sum or means of intensities

def calculate_areas(img):
    """
    Calculates mean and std of foci and citoplasm intensity and area of foci after generating a mask.
    
    Takes an image and applies generate_mask function
    with 1 to 4 iterations of binary_opening and sums the masks.
    This gives a weighted mask that gives more weight to the center of the foci.
    Mean and std intensities of foci and citoplasms are returned
    Inputs
    img -- image to calculate intensities and area of foci
    Returns
    mean_I  -- mean weighted intensity of foci
    std_I   -- standard deviation of weighted intensity foci
    mean_CP -- mean weighted intensity of non-foci
    std_CP  -- standard deviation of weighted intensity non-foci
    area    -- mean of weighted mask area
    """
    Ints = []
    CPs = []
    areas = []
    for i in range(3, 4):
        mask = generate_masks(img, i)
        if not mask.any():
            continue
        ROI = img[mask]
        Ints.extend(ROI)
        CPs.extend(img[~mask])
        areas.append(len(ROI))
    mean_I = np.nansum(Ints)
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
    """
    Calculates intensities and area of foci in image series.
    
    Applies the calculate_areas function to every image in 
    the series returning the tuple of lists with the mean 
    and standard deviation of foci and non-foci intensities
    as well as foci mean area.
    Inputs
    series -- series of images to calculate foci intensities and area
    Returns
    means_I  -- list of mean weighted intensity of foci
    stds_I   -- list of standard deviation of weighted intensity of foci
    means_CP -- list of mean weighted intensity of non-foci
    stds_CP  -- list of standard deviation of weighted intensity of non-foci
    areas    -- list of mean weighted area of foci
    """
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
    """Returns GR/CP as normalization"""
    return GR/CP

# Functions that create and add columns to pandas dataframe

def generate_df(fp):
    """
    Generates dataframe with with images and attributes of each cell in the fp filepath.
    
    Sweeps the fp directory for oif files of cells and concatenates the pre and pos images
    cropping the image with the clip selection for bleaching found in the ble oif file.
    The DataFrame returned has cell name ('cell'), cell number ('cell_number'), foci number
    ('foci'), cropped image series ('series'), timepoint ('timepoint'), and total intensity 
    of non-cropped region of the image ('total_Int')
    Inputs
    fp -- filepath of the folder containing oif files
    Returns
    df -- pandas DataFrame containing the information of the oif files
    """
    FileDict = generate_FileDict(fp)
    
    cells = set()
    for key in FileDict.keys():
        cells.add(key[0])
    
    df = pd.DataFrame()
    for cell in cells:
        series, tot_Int = crop_and_conc(cell, FileDict)
        timepoint = get_timepoint(FileDict[cell, 'pre'])
        _, cell_n, foci = get_info(FileDict[cell, 'pre'])
        df = df.append({'cell':cell, 'cell_number':cell_n, 'foci':foci, 'series':series, 'timepoint':timepoint, 'total_Int':tot_Int}, ignore_index=True)
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
        try:
            popt, pcov = curve_fit(Frap_Func, t[np.isfinite(this_f)], this_f[np.isfinite(this_f)], p0=[2000, 15, 5])
        except TypeError:
            popt = [np.nan,np.nan,np.nan]
        
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
