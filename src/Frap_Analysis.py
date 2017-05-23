# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:17:56 2016

@author: Agus
"""
import pathlib
import os
import re
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
import skimage.measure as meas
import scipy.stats as st

from scipy.signal import correlate2d
from scipy import ndimage as ndi
from scipy.optimize import curve_fit, minimize
from scipy.ndimage.morphology import binary_opening
from skimage.filters import threshold_otsu
from skimage.draw import circle
from datetime import datetime, timedelta

os.chdir(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\FRAP_Analysis\src')
import oiffile as oif

#%% Define useful functions

# Functions to fit with

def Noise_Func(t, A, max_noise, tau):
    """
    Returns (max_noise) - A * np.exp(-t / tau)
    """
    return (max_noise) - A * np.exp(-t / tau)


def Frap_Func(t, A, immobile_frac, tau):
    """
    Returns (1-immobile_frac) - A * np.exp(-t / tau)
    """
    return (1-immobile_frac) - A * np.exp(-t / tau)


def Exp_decay(t, A, offset, tau):
    """Returns A * np.exp(-t/tau) + offset"""
    return A * np.exp(-t/tau) + offset


# Function to generate filepath dictionary


def generate_FileDict(filepath):
    """
    Generates a dictionary with paths for each cell and time period
    
    Inputs:
    filepath -- filepath to folder with all the .oif files
    Returns:
    File_Dict -- Dictionary where keys are filename split by '_' and values are the corresponding full path
    """
    filepath = pathlib.Path(filepath)
    File_Dict = {tuple(f.stem.split('_')[:-1]): f for f in filepath.glob('*.oif') if ('_pos' in str(f.name) or '_pre' in str(f.name))}
    return File_Dict


def generate_statVar(shape):
    """
    Initializes a structured array filled with nans to contain the statistics of data.
    """
    rec = np.recarray((shape, ), [('sum', float), ('mean', float), ('mode', float), ('median', float), ('p20', float), ('p80', float)])
    rec.fill(np.nan)
    return rec


def calculate_statvar(rec, ind, vect):
    """
    Adds statistics of vect to the structured array rec at the index ind. Ignores Nans.
    """
    if len(vect)>0:
        vect = np.asarray(vect)
        vect = vect[np.isfinite(vect)]
        rec['sum'][ind] = np.nansum(vect)
        rec['mean'][ind] = np.nanmean(vect)
        rec['mode'][ind] = st.mode(vect).mode[0]
        rec['median'][ind] = np.nanmedian(vect)
        rec['p20'][ind], rec['p80'][ind] = np.percentile(vect, [20, 80])
    else:
        for field in rec.dtype.names:
            rec[field] = np.nan


def rec_to_dict(this_dict, rec, suffix):
    """
    Transfers structured array rec to dictionary this_dict where each statistical field is preluded by the suffix.
    """
    for field in rec.dtype.names:
        this_dict[suffix+'_'+field] = rec[field]


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


def get_metadata(filepath):
    """
    Gets the whole metadata from the filepath series of .oif file.
    
    Input
    filepath -- The filepath to the image whos metadata is desired
    Return
    metadata -- A dictionary of dictionaries containing the metadata classified into subsets
    """
    filepath = pathlib.Path(filepath)
    
    with open(str(filepath), 'rb') as file:
        metadata = dict()
        present_dict = 'General'
        metadata[present_dict] = dict()
        for row in file:
            try:
                this_r = row.decode("utf-8")
                this_r = this_r.replace('\00', '')
                this_r = this_r.replace('\r', '')
                this_r = this_r.replace('\n', '')
                if this_r.startswith('['):
                    present_dict = this_r[1:-1]
                    metadata[present_dict] = dict()
                else:
                    this_r = this_r.split('=')
                    metadata[present_dict][this_r[0]] = this_r[1]
            except:
                pass
            
    return metadata


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
    Retrieves (y, x) clip start from .files folder
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
    return (clip['Y'], clip['X'])


def get_size(filepath):
    """
    Retrieves clip size (h, w) from .files folder
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
    return (Sizes['Y'], Sizes['X'])


def get_time(filepath):
    """
    Generates time vector for pos file with t=0 when bleaching has ended
    """
    filepath = pathlib.Path(filepath)
    file_ble = filepath.parent
    file_ble = file_ble.joinpath(str(filepath.name).replace('_pos', '_ble'))
    # Get timepoints and metadata from pos and ble files
    pos_timepoint = get_timepoint(filepath)
    ble_timepoint = get_timepoint(file_ble)
    pos_meta = get_metadata(filepath)
    ble_meta = get_metadata(file_ble)
    # Get Acquisition initial time
    time_format = "%Y-%m-%d %H:%M:%S %f"
    ble_ini = ble_meta['General']['ImageCaputreDate'][1:-1]+' '+ble_meta['General']['ImageCaputreDate+MilliSec']
    pos_ini = pos_meta['General']['ImageCaputreDate'][1:-1]+' '+pos_meta['General']['ImageCaputreDate+MilliSec']
    ble_ini = datetime.strptime(ble_ini, time_format)
    pos_ini = datetime.strptime(pos_ini, time_format)
    # Estimate end of bleaching
    ble_time = timedelta(seconds=float(ble_meta['Axis 4 Parameters Common']['MaxSize'])*ble_timepoint)
    pos_len = int(pos_meta['Axis 4 Parameters Common']['MaxSize'])
    pos_time = float(pos_len)*pos_timepoint
    ble_end = ble_ini + ble_time
    # time vector
    start = pos_ini-ble_end
    t = np.arange(start.total_seconds(), start.total_seconds()+pos_time, pos_timepoint)
    
    return t[:pos_len]


def open_ble(filepath):
    """Opens ble series because there is the time sequence and reference image"""
    filepath = pathlib.Path(filepath)
    this_filepath = filepath.parent
    this_filepath = this_filepath.joinpath(filepath.name + '.files')
    w = int(get_metadata(filepath)['Axis 0 Parameters Common']['MaxSize'])
    h = int(get_metadata(filepath)['Axis 1 Parameters Common']['MaxSize'])
    len_series = int(get_metadata(filepath)['Axis 4 Parameters Common']['MaxSize'])
    stack = np.full((len_series, h, w), np.nan)
    for i in range(len_series):
        j = i+1
        this_image_file = this_filepath.joinpath('s_C001T'+'%03d' %j+'.tif')
        img = tif.imread(str(this_image_file))
        stack[i] = img
    
    return stack


def get_ble_f(series):
    """Calculates stat_var of intentsities of ble images"""
    Ints = generate_statVar(series.shape[0])
    for ndx, img in enumerate(series):
        calculate_statvar(Ints, ndx, img.flatten())
    return Ints


# Background correction for images


def bkg_correct(series, t):
    """ Hard coded background correction function that depends on Nois_Func"""
    noise = Noise_Func(t, 3.8, 22, 33)
    new_series = np.asarray(series, dtype=float)
    for i, img in enumerate(new_series):
        new_series[i, :, :] = np.clip(img - noise[i], 0, 4096)
    
    return new_series


# Functions to crop and mask images

    
def clip(x, mn, mx):
    """Returns x clipped between mn and mx"""
    return x if mn <= x <= mx else (mn if x < mn else mx)    


def imcrop(im, y1, y2, x1, x2):
    """Crops the image im between x1 and x2 (vertical axis) and y1 and y2 (horizontal axis)"""
    assert x1 < x2
    assert y1 < y2
    sh = im.shape
    x1, x2 = np.clip([x1, x2], 0, sh[0])
    y1, y2 = np.clip([y1, y2], 0, sh[1])
    return im[y1:y2, x1:x2]

    
def imcrop_wh(im, y1, h, x1, w):
    """Crops the image im with a rectangle of height h and width w starting at point y1, x1"""
    assert w>0
    assert h>0
    sh = im.shape
    x1 = clip(x1, 0, sh[0])
    x2 = clip(x1 + w, 0, sh[1])
    y1 = clip(y1, 0, sh[1])
    y2 = clip(y1 + h, 0, sh[1])
    return im[y1:y2, x1:x2]


def get_farCP(img, yhxw, add=40):
    """
    Discard close bleaching zone and estimate citoplasm intensity.
    
    Dsicards bleaching area plus 'add' pixels and generates a vector of rest intensity pixels.
    commented: Discards the bleaching area plus 'add' pixels and then generates two Otsu masks
    to find granules and dark background.
    Inputs
    img  -- image to work on
    yhxw -- window cropped for analysis
    add  -- (optional) pixels added to bleaching zone
    Return
    CP_far     -- list of intensity of citoplasm pixels
    dark       -- deprecated(list of intensity of dark pixels)
    """
    # Discard granule in study
    y, h, x, w = yhxw
    y -= add
    h += 2*add
    x -= add
    w += 2*add
    sh = img.shape
    x1 = clip(x, 0, sh[0])
    x2 = clip(x + w, 0, sh[1])
    y1 = clip(y, 0, sh[1])
    y2 = clip(y + h, 0, sh[1])
    img[y1:y2, x1:x2] = np.nan
    """
    # Generate mask for citoplasm
    extra_thresh = 20
    upper_thresh = threshold_otsu(img)
    upper_thresh = upper_thresh -extra_thresh
    
    
    lower_thresh = threshold_otsu(np.log(img[img<upper_thresh]+1))
    lower_thresh = np.exp(lower_thresh) -1
    mask = np.where(np.logical_and(img>lower_thresh, img<upper_thresh), 1, 0)
    mask = binary_opening(mask, iterations=3)
    
    # Prepare values to return
    CP_far = img[mask]
    dark = img[img<lower_thresh]"""
    CP_far = img.flatten()
    dark = [1,2,3,4]
    return CP_far, dark

    
def crop_and_shift(imgs, yhxw, filter_width=5):
    """
    Returns the cropped series imgs starting from y,x with size h,w and centering the granule.
    
    Generates a crop at y,x with size h,w and centers the crop in the present granule using correlation with a centered disk.
    From then on, the image is centered by correlating to previous image. If difference in image intensity percentile 20 and 80
    is low, no correlation and tracking is done.
    Inputs
    imgs -- series to track and crop
    yhxw -- tuple with (y position, height of crop, x position, crop width)
    Returns
    stack         -- stack of cropped and centered images
    CP_far        -- stat_var of intensity of every image outside crop
    dark          -- deprecated: stat_var of intensity of dark background
    offsets       -- list of (y,x) positions of crop (can be used as trajectory of granule)
    """
    y, h, x, w = yhxw
    len_series, sh_y, sh_x = imgs.shape
    stack = np.full((len_series, h, w), np.nan)
    CP_far = generate_statVar(len_series)
    dark = generate_statVar(len_series)
    offsets = np.empty((len_series, 2), dtype=np.uint)
    
    pre_img = np.zeros((h,w))
    rr, cc = circle(pre_img.shape[0]//2, pre_img.shape[1]//2, 5, pre_img.shape)
    pre_img[rr, cc] = 1
    
    for ndx in range(len_series):
        img = imgs[ndx, :, :]
        cropped = imcrop_wh(img, y, h, x, w)
        p1, p2 = np.percentile(cropped, [20, 80])
        if p2 - p1 > 75:
            correlation = correlate2d(cropped, pre_img)#smooth, smooth_pre)
            pos = np.unravel_index(np.argmax(correlation), correlation.shape)
            y += (pos[0] - correlation.shape[0]//2)
            x += (pos[1] - correlation.shape[1]//2)
            cropped = imcrop_wh(img, y, h, x, w)
            
        # Check cropped image has the same size
        if cropped.shape != (h, w):
            dif_shape = np.asarray([h, w]) - np.asarray(cropped.shape)
            new_cropped = np.full((h,w), np.nan)
            y, x = dif_shape//2
            new_cropped[y:y+cropped.shape[0], x:x+cropped.shape[1]] = cropped
            cropped = new_cropped.copy()
        stack[ndx, :, :] = cropped.copy()
        this_CP_far, this_dark = get_farCP(img, yhxw)
        calculate_statvar(CP_far, ndx, this_CP_far)
        calculate_statvar(dark, ndx, this_dark)
        offsets[ndx, :] = (y, x)
        
    return stack, CP_far, dark, offsets


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
    thresh = np.exp(thresh)-1 +50
    mask = img>thresh
    if iterations>0:
        mask = binary_opening(mask, iterations=iterations)
    return mask


def circle_mask(img, radius):
    """
    Returns mask for image (img) consisting of centered circle of radius.
    
    Inputs
    img    -- image to generate mask
    radius -- radius of the centered circle mask
    Returns
    mask -- boolean mask generated
    """
    mask = np.zeros(img.shape)
    rr, cc = circle(img.shape[0]//2, img.shape[1]//2, radius, img.shape)
    mask[rr, cc] = 1
    return mask

# Functions to calculate sum or means of intensities

def calculate_areas(img, rad=None):
    """
    Returns a list of intensities of foci and citoplasm pixels and area and equivalent diameter of foci after generating a mask or a radius rad circle mask.
    
    Takes an image and applies generate_mask function
    with 3 iterations of binary_opening and sums the masks.
    deprecated: This gives a weighted mask that gives more weight to the center of the foci.
    Inputs
    img -- image to calculate intensities and area of foci
    rad -- (optional) radius of disk to be used as mask
    Returns
    Ints      -- list of intensities of granule pixels
    CPs       -- list of intensities of near citoplasm pixels
    areas     -- list of sinlge element weighted mask area
    diameters -- list of single element equivalent circle diameter
    """
    Ints = []
    CPs = []
    areas = []
    diameters = []
    for i in range(3, 4):
        if rad is None:
            mask = generate_masks(img, i)
        else:
            mask = circle_mask(img, rad)
        if not mask.any():
            areas.append(np.nan)
            diameters.append(np.nan)
            continue
        # Select only centered granule
        labeled = meas.label(mask)
        obj_num = labeled[labeled.shape[0]//2, labeled.shape[1]//2]
        mask = np.asarray(labeled==obj_num, dtype='int')
        props = meas.regionprops(mask)[0]
        area = props.area
        diameter = props.equivalent_diameter 
        ROI = img[mask==obj_num]
        Ints.extend(ROI)
        CPs.extend(img[labeled==0])
        areas.append(area)
        diameters.append(diameter)
    if np.isnan(Ints).all():
        Ints = CPs
    return Ints, CPs, areas, diameters

def calculate_series(series, rad=None):
    """
    Calculates intensities and area of foci in image series.
    
    Applies the calculate_areas function to every image in 
    the series returning the stat_var of foci and non-foci intensities
    as well as foci area and diameter time series.
    Inputs
    series -- series of images to calculate foci intensities and area
    rad    -- (optional) radius to be used if centered disk mask is to be used
    Returns
    GR         -- stat_var of intensities of granules
    CP_near    -- stat_var of intensities of near citoplasm
    areass     -- time series list of weighted area of foci
    diameterss -- time series of equivalent circle diameter
    """
    len_series, sh_y, sh_x = series.shape
    GR = generate_statVar(len_series)
    CP_near = generate_statVar(len_series)
    areass = []
    diameterss = []
    for ndx in range(len_series):
        img = series[ndx, :, :]
        this_GR, this_CP_near, areas, diameters = calculate_areas(img, rad)
        calculate_statvar(GR, ndx, this_GR)
        calculate_statvar(CP_near, ndx, this_CP_near)
        areass.append(areas)
        diameterss.append(diameters)
    
    areass =  np.asarray(areass)
    diameterss = np.asarray(diameterss)
    return GR, CP_near, areass, diameterss


def calculate_fluorescence(CP, GR):
    """Returns list of GR/CP as normalization"""
    return np.asarray(GR/CP)

# Functions that create and add columns to pandas dataframe


def process_frap(fp):
    """
    Generates dataframe with images and attributes of each cell in the fp filepath.
    
    Sweeps the fp directory for oif files of cells and analyzes the pre and pos images
    cropping the image with the clip selection for bleaching found in the ble oif file and
    then centering on the granule closest to the center. The returned DataFrame has all
    the attributes calculated for the granule of interest.
    Inputs
    fp -- filepath of the folder containing oif files
    Returns
    df -- pandas DataFrame containing the information of the oif files
    """
    FileDict = generate_FileDict(fp)
    
    cells = set()
    for key in FileDict.keys():
        cells.add(key[0])
    
    # Load all cropped images from folder into dataframe
    df = pd.DataFrame()
    for cell in cells:
        print(cell)
        # load complete image
        file_pre = FileDict[cell, 'pre']
        series = oif.imread(str(file_pre))[0]
        timepoint = get_timepoint(FileDict[cell, 'pre'])
        t_pre = np.arange(0, len(series)*timepoint, timepoint)[:len(series)]
        series = bkg_correct(series, t_pre)
        
        # track and crop images pre bleaching
        file_ble = file_pre.parent
        file_ble = file_ble.joinpath(str(file_pre.name).replace('_pre', '_ble'))
        ble_f = get_ble_f(open_ble(str(file_ble)))
        ble_timepoint = get_timepoint(file_ble)
        Size = get_size(file_ble)
        start = get_clip(file_ble)
        yhxw = (start[0], Size[0], start[1], Size[1])
        
        pre_series, pre_CP_far, pre_dark, pre_trajectory = crop_and_shift(series, yhxw)
        
        # track and crop images post bleaching
        file_pos = FileDict[cell, 'pos']
        series = oif.imread(str(file_pos))[0]
        t = get_time(FileDict[cell, 'pos'])
        series = bkg_correct(series, t)
        yhxw = (pre_trajectory[-1, 0], pre_series[0].shape[0], pre_trajectory[-1, 1], pre_series[0].shape[1])
        
        pos_series, pos_CP_far, pos_dark, pos_trajectory = crop_and_shift(series, yhxw)
        
        # get cell information
        _, cell_n, foci = get_info(FileDict[cell, 'pre'])
        metadata = get_metadata(FileDict[cell, 'pos'])
        h_umpxratio = float(metadata['Reference Image Parameter']['HeightConvertValue'])
        w_umpxratio = float(metadata['Reference Image Parameter']['WidthConvertValue'])
        laser = float(metadata['General']['LaserTransmissivity01'])
        PMT_Volt = float(metadata['Channel 1 Parameters']['AnalogPMTVoltage'])
        PMT_Gain = float(metadata['Channel 1 Parameters']['AnalogPMTGain'])/1000
        # prepare dataframe
        this_dict = {'cell':cell, 
                     'cell_number':cell_n, 
                     'foci':foci,  
                     'timepoint':timepoint,
                     'ble_timepoint':ble_timepoint,
                     'h_ratio':h_umpxratio,
                     'w_ratio':w_umpxratio,
                     'laser':laser,
                     'PMT_Volt':PMT_Volt,
                     'PMT_Gain':PMT_Gain,
                     'pre_series':pre_series,
                     'pre_trajectory':pre_trajectory, 
                     'pos_series':pos_series,
                     'pos_trajectory':pos_trajectory,
                     't':t}

        rec_to_dict(this_dict, pre_CP_far, 'pre_CP_far')
        rec_to_dict(this_dict, pre_dark, 'pre_dark')
        rec_to_dict(this_dict, pos_CP_far, 'pos_CP_far')
        rec_to_dict(this_dict, pos_dark, 'pos_dark')
        rec_to_dict(this_dict, ble_f, 'ble_f')
        df = df.append(this_dict, ignore_index=True)
    
    # Characterize granule from pre series
    pre_charac_df = pd.DataFrame()
    for i in df.index:
        print('pre characterization '+df.cell.values[i])
        this_GR, this_CP_near, areas, diameters = calculate_series(df.pre_series.values[i])
        pre_charac = {'cell':df.cell.values[i], 'area':areas, 'diameter':diameters}
        rec_to_dict(pre_charac, this_GR, 'pre_GR')
        rec_to_dict(pre_charac, this_CP_near, 'pre_CP_near')
        pre_charac_df = pre_charac_df.append(pre_charac, ignore_index=True)
    
    df = df.merge(pre_charac_df, on='cell')
    
    # Use radius to generate masks for post bleaching processing
    pos_charac_df = pd.DataFrame()
    for i in df.index:
        print('pos '+df.cell.values[i])
        rad = np.nanmean(df.diameter.values[i])//2 + 1
        this_GR, this_CP_near, _, _ = calculate_series(df.pos_series.values[i], rad) # if rad is not passed, previous segmentation is used
        pos_charac = {'cell':df.cell.values[i]}
        rec_to_dict(pos_charac, this_GR, 'pos_GR')
        rec_to_dict(pos_charac, this_CP_near, 'pos_CP_near')
        pos_charac_df = pos_charac_df.append(pos_charac, ignore_index=True)

    df = df.merge(pos_charac_df, on='cell')
    
    # add fluorescence calculation
    df['pre_f'] = list(map(calculate_fluorescence, df['pre_CP_far_mean'], df['pre_GR_sum']))
    df['pos_f'] = list(map(calculate_fluorescence, df['pos_CP_far_mean'], df['pos_GR_sum']))
    
    # Add pre bleach mean intensity and normalize post bleach fluorescence with it
    # non corrected version
    df['mean_area_px'] = list(map(np.nanmean, df['area']))
    df['mean_diameter_px'] = list(map(np.nanmean, df['diameter']))
    df['mean_pre_I_px'] = list(map(np.nanmean, df['pre_f']))
    # Corrected and translated to reality
    df['mean_area'] = list(map(lambda x, y, z: x*y*z, df['mean_area_px'], df['h_ratio'], df['w_ratio']))
    df['mean_diameter'] = list(map(lambda x, y: x*y, df['mean_diameter_px'].values, df['h_ratio'].values))
    df['mean_pre_I'] = list(map(lambda x, y, z, w: x*(700/y)*(1/z)*(0.1/w), df['mean_pre_I_px'], df['PMT_Volt'], df['PMT_Gain'], df['laser']))
    
    
    lambdafunc = lambda x, y: x/y
    df['f_corr'] = list(map(lambdafunc, df['pos_f'], df['mean_pre_I_px']))
    
    df = add_fitParams(df, Plot=True)
    df = fit_whole_frap_func(df, Plot=True)
    
    return df


# CP specific functions


def crop_and_shift_CP(imgs, yhxw, filter_width=5, D=5):
    """
    Returns the cropped series imgs starting from y,x with size h,w.
    
    Generates a crop at y,x with size h,w and centers the crop in the bleaching clip.
    Inputs
    imgs -- series to crop
    yhxw -- tuple with (y position, height of crop, x position, crop width)
    Returns
    stack   -- stack of cropped images
    CP_far  -- stat_var of intensity of whole image outside crop
    dark    -- (deprecated)
    offsets -- list of (y,x) positions of crop (can be used as trajectory of granule)
    """
    y, h, x, w = yhxw
    len_series, sh_y, sh_x = imgs.shape
    stack = np.full((len_series, h, w), np.nan)
    CP_far = generate_statVar(len_series)
    dark = generate_statVar(len_series)
    offsets = np.empty((len_series, 2), dtype=np.uint)
    
    for ndx in range(len_series):
        img = imgs[ndx, :, :]
        cropped = imcrop_wh(img, y, h, x, w)
        
        # Check cropped image has the same size
        if cropped.shape != (h, w):
            dif_shape = np.asarray([h, w]) - np.asarray(cropped.shape)
            new_cropped = np.full((h,w), np.nan)
            y, x = dif_shape//2
            new_cropped[y:y+cropped.shape[0], x:x+cropped.shape[1]] = cropped
            cropped = new_cropped.copy()
            
        stack[ndx, :, :] = cropped.copy()
        this_CP_far, this_dark = get_farCP(img, yhxw)
        calculate_statvar(CP_far, ndx, this_CP_far)
        calculate_statvar(dark, ndx, this_dark)
        offsets[ndx, :] = (y, x)
        
    return stack, CP_far, dark, offsets


def calculate_series_CP(series):
    """
    Calculates intensities and area of ROI bleached in image series.
    
    Calculates stat_var of every image in the series 
    as well as area time series.
    Inputs
    series -- series of images to calculate foci intensities and area
    Returns
    GR         -- stat_var of intensities of ROI
    CP_near    -- empty stat_var
    areass     -- time series list of area of ROI
    diameterss -- 0
    """
    len_series, sh_y, sh_x = series.shape
    GR = generate_statVar(len_series)
    CP_near = generate_statVar(len_series)
    areass = []
    diameterss = []
    for ndx in range(len_series):
        img = series[ndx, :, :]
        calculate_statvar(GR, ndx, img.flatten())
        calculate_statvar(CP_near, ndx, [])
        areass.append(img.shape[0]*img.shape[1])
        diameterss.append(0)
    
    areass =  np.asarray(areass)
    diameterss = np.asarray(diameterss)
    return GR, CP_near, areass, diameterss


def process_frap_CP(fp):
    """
    Generates dataframe with images and attributes of each cell in the fp filepath.
    
    Sweeps the fp directory for oif files of cells and analyzes the pre and pos images
    cropping the image with the clip selection for bleaching found in the ble oif file.
    The returned DataFrame has all the attributes calculated for the citoplasm of interest.
    Inputs
    fp -- filepath of the folder containing oif files
    Returns
    df -- pandas DataFrame containing the information of the oif files
    """
    FileDict = generate_FileDict(fp)
    
    cells = set()
    for key in FileDict.keys():
        if key[2]=='GR':
            cells.add(key[0])
    
    # Load all cropped images from folder into dataframe
    df = pd.DataFrame()
    for cell in cells:
        print(cell)
        # load complete image
        file_pre = FileDict[cell, 'pre', 'GR']
        series = oif.imread(str(file_pre))[0]
        timepoint = get_timepoint(FileDict[cell, 'pre', 'GR'])
        t_pre = np.arange(0, len(series)*timepoint, timepoint)[:len(series)]
        series = bkg_correct(series, t_pre)
        
        # track and crop images pre bleaching
        file_ble = file_pre.parent
        file_ble = file_ble.joinpath(str(file_pre.name).replace('_pre', '_ble'))
        Size = get_size(file_ble)
        start = get_clip(file_ble)
        yhxw = (start[0], Size[0], start[1], Size[1])
        
        pre_series, pre_CP_far, pre_dark, pre_trajectory = crop_and_shift_CP(series, yhxw)
        
        # track and crop images post bleaching
        file_pos = FileDict[cell, 'pos', 'GR']
        series = oif.imread(str(file_pos))[0]
        t = get_time(FileDict[cell, 'pos', 'GR'])
        series = bkg_correct(series, t)
        yhxw = (pre_trajectory[-1, 0], pre_series[0].shape[0], pre_trajectory[-1, 1], pre_series[0].shape[1])
        pos_series, pos_CP_far, pos_dark, pos_trajectory = crop_and_shift_CP(series, yhxw)
        
        # get cell information
        _, cell_n, foci = get_info(FileDict[cell, 'pre', 'GR'])
        metadata = get_metadata(FileDict[cell, 'pos', 'GR'])
        h_umpxratio = float(metadata['Reference Image Parameter']['HeightConvertValue'])
        w_umpxratio = float(metadata['Reference Image Parameter']['WidthConvertValue'])
        laser = float(metadata['General']['LaserTransmissivity01'])
        PMT_Volt = float(metadata['Channel 1 Parameters']['AnalogPMTVoltage'])
        PMT_Gain = float(metadata['Channel 1 Parameters']['AnalogPMTGain'])/1000
        # prepare dataframe
        this_dict = {'cell':cell, 
                     'cell_number':cell_n, 
                     'foci':foci,  
                     'timepoint':timepoint,
                     'h_ratio':h_umpxratio,
                     'w_ratio':w_umpxratio,
                     'laser':laser,
                     'PMT_Volt':PMT_Volt,
                     'PMT_Gain':PMT_Gain,
                     'pre_series':pre_series,
                     'pre_trajectory':pre_trajectory, 
                     'pos_series':pos_series,
                     'pos_trajectory':pos_trajectory,
                     't':t}

        rec_to_dict(this_dict, pre_CP_far, 'pre_CP_far')
        rec_to_dict(this_dict, pre_dark, 'pre_dark')
        rec_to_dict(this_dict, pos_CP_far, 'pos_CP_far')
        rec_to_dict(this_dict, pos_dark, 'pos_dark')
        df = df.append(this_dict, ignore_index=True)
    
    # Characterize granule from pre series
    pre_charac_df = pd.DataFrame()
    for i in df.index:
        print('pre characterization '+df.cell.values[i])
        this_GR, this_CP_near, areas, diameters = calculate_series_CP(df.pre_series.values[i])
        pre_charac = {'cell':df.cell.values[i], 'area':areas, 'diameter':diameters}
        rec_to_dict(pre_charac, this_GR, 'pre_GR')
        rec_to_dict(pre_charac, this_CP_near, 'pre_CP_near')
        pre_charac_df = pre_charac_df.append(pre_charac, ignore_index=True)
    
    df = df.merge(pre_charac_df, on='cell')
    
    # Use radius to generate masks for post bleaching processing
    pos_charac_df = pd.DataFrame()
    for i in df.index:
        print('pos '+df.cell.values[i])
        this_GR, this_CP_near, _, _ = calculate_series_CP(df.pos_series.values[i]) # if rad is not passed, previous segmentation is used
        pos_charac = {'cell':df.cell.values[i]}
        rec_to_dict(pos_charac, this_GR, 'pos_GR')
        rec_to_dict(pos_charac, this_CP_near, 'pos_CP_near')
        pos_charac_df = pos_charac_df.append(pos_charac, ignore_index=True)

    df = df.merge(pos_charac_df, on='cell')
    
    # add fluorescence calculation
    df['pre_f'] = list(map(calculate_fluorescence, df['pre_CP_far_mean'], df['pre_GR_sum']))
    df['pos_f'] = list(map(calculate_fluorescence, df['pos_CP_far_mean'], df['pos_GR_sum']))
    
    # Add pre bleach mean intensity and normalize post bleach fluorescence with it
    # non corrected version
    df['mean_area_px'] = list(map(np.nanmean, df['area']))
    df['mean_diameter_px'] = list(map(np.nanmean, df['diameter']))
    df['mean_pre_I_px'] = list(map(np.nanmean, df['pre_f']))
    # Corrected and translated to reality
    df['mean_area'] = list(map(lambda x, y, z: x*y*z, df['mean_area_px'], df['h_ratio'], df['w_ratio']))
    df['mean_diameter'] = list(map(lambda x, y: x*y, df['mean_diameter_px'].values, df['h_ratio'].values))
    df['mean_pre_I'] = list(map(lambda x, y, z, w: x*(700/y)*(1/z)*(0.1/w), df['mean_pre_I_px'], df['PMT_Volt'], df['PMT_Gain'], df['laser']))
    
    # Add pre bleach mean intensity and normalize post bleach fluorescence with it   
    #df['mean_pre_I'] = list(map(np.nanmean, df['pre_f']))
    lambdafunc = lambda x, y: x/y
    df['f_corr'] = list(map(lambdafunc, df['pos_f'], df['mean_pre_I_px']))
    
    df = add_fitParams(df, Plot=True)
    
    return df


def add_fitParams(df, Plot=False):
    Amps = []
    Imms = []
    taus = []
    for i in df.index:
        print(df['cell'][i])
        this_f = df['f_corr'][i]
        this_t = df['t'][i]
        
        try:
            popt, pcov = curve_fit(Frap_Func, this_t[np.isfinite(this_f)], this_f[np.isfinite(this_f)], p0=[2000, 15, 5], sigma=this_f[np.isfinite(this_f)])
        except (TypeError, RuntimeError):
            popt = [np.nan,np.nan,np.nan]
        
        Amp, Imm, tau = popt[0], popt[1], popt[2]
        
        Amps.append(Amp)
        Imms.append(Imm)
        taus.append(tau)
        
        if Plot:
            plt.plot(this_t, Frap_Func(this_t, Amp, Imm, tau), 'r')
            plt.scatter(this_t, this_f)
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


def fit_whole_frap_func(df, Plot=False):
    tau_imgs = []
    tau_bles = []
    tau_recs = []
    
    for i in df.index:
        print(df.cell[i])
        this_pre_f = df['pre_f'][i]/df['mean_pre_I_px'][i]
        this_pre_f = this_pre_f[np.isfinite(this_pre_f)]
        this_pre_t = np.arange(0, df.timepoint[i]*len(this_pre_f), df.timepoint[i])[0:len(this_pre_f)]
        this_pre_t = this_pre_t[np.isfinite(this_pre_f)]
        
        this_ble_f = df['ble_f_mean'][i]
        this_ble_f = this_ble_f[np.isfinite(this_ble_f)]        
        this_ble_t = np.arange(0, df.ble_timepoint[i]*len(this_ble_f), df.ble_timepoint[i])[0:len(this_ble_f)]
        this_ble_t = this_ble_t[np.isfinite(this_ble_f)]
        
        this_pos_f = df['f_corr'][i]
        this_pos_f = this_pos_f[np.isfinite(this_pos_f)]
        this_pos_t = df['t'][i]
        this_pos_t = this_pos_t[np.isfinite(this_pos_f)]
        
        def chi_2(taus):
            tau_img, tau_rec, tau_ble = taus
            def pre_func(t, A, offset):
                """Returns A * np.exp(-t/tau) + offset"""
                return A * np.exp(-t/(tau_img*1e4)) + offset
            
            def ble_func(t, A, B, offset):
                """Returns A * np.exp(-t/tau) + offset"""
                return A * np.exp(-t*(10/(tau_ble))) + B * np.exp(-t/tau_rec) + offset
            
            def pos_func(t, A, immobile_frac):
                """Returns (1-immobile_frac) - A * np.exp(-t / tau)"""
                return (1-immobile_frac) - A * np.exp(-t / tau_rec)
            
            pre_popt, _ = curve_fit(pre_func, this_pre_t, this_pre_f, sigma=this_pre_f)
            ble_popt, _ = curve_fit(ble_func, this_ble_t, this_ble_f, sigma=this_ble_f)
            pos_popt, _ = curve_fit(pos_func, this_pos_t, this_pos_f, sigma=this_pos_f)
            
            chi = np.sum(((pre_func(this_pre_t, *pre_popt)-this_pre_f)/this_pre_f)**2)
            chi += np.sum(((ble_func(this_ble_t, *ble_popt)-this_ble_f)/this_ble_f)**2)
            chi += np.sum(((pos_func(this_pos_t, *pos_popt)-this_pos_f)/this_pos_f)**2)
            
            return chi
        try:
            mini = minimize(chi_2, [1, 15, 1])
            tau_img, tau_rec, tau_ble = mini.x
        except (TypeError, RuntimeError):
            tau_img, tau_rec, tau_ble = (np.nan, )*3
            tau_img = tau_img*1e4
            tau_ble = tau_ble/10
        
        if Plot and np.isfinite(tau_img):
            def pre_func(t, A, offset):
                """Returns A * np.exp(-t/tau) + offset"""
                return A * np.exp(-t/(tau_img)) + offset
            
            def ble_func(t, A, B, offset):
                """Returns A * np.exp(-t/tau) + offset"""
                return A * np.exp(-t/tau_ble) + B * np.exp(-t/tau_rec) + offset
            
            def pos_func(t, A, immobile_frac):
                """Returns (1-immobile_frac) - A * np.exp(-t / tau)"""
                return (1-immobile_frac) - A * np.exp(-t / tau_rec)
            
            pre_popt, _ = curve_fit(pre_func, this_pre_t, this_pre_f, sigma=this_pre_f)
            ble_popt, _ = curve_fit(ble_func, this_ble_t, this_ble_f, sigma=this_ble_f)
            pos_popt, _ = curve_fit(pos_func, this_pos_t, this_pos_f, sigma=this_pos_f)
            
            sim_pre_f = pre_func(this_pre_t, *pre_popt)
            sim_ble_f = ble_func(this_ble_t, *ble_popt)/np.max(this_ble_f)
            sim_pos_f = pos_func(this_pos_t, *pos_popt)
            sim_f = list(sim_pre_f) + list(sim_ble_f) + list(sim_pos_f)
            
            this_ble_f = this_ble_f/np.max(this_ble_f)
            this_ble_t = this_ble_t + np.max(this_pre_t)
            this_pos_t = this_pos_t + np.max(this_ble_t)
            
            this_t = list(this_pre_t) + list(this_ble_t)+ list(this_pos_t)
            this_f = list(this_pre_f) + list(this_ble_f)+ list(this_pos_f)
            
            plt.scatter(this_t, this_f)
            plt.plot(this_t, sim_f, 'r')
            plt.title(df.cell[i])
            plt.show()
            print('tau imaging: '+str(tau_img*1e4))
            print('tau recovery: '+str(tau_rec))
            print('tau bleaching: '+str(tau_ble/10))
            
            plt.scatter(this_ble_t, this_ble_f)
            plt.plot(this_ble_t, sim_ble_f)
            plt.show()
            
            
        tau_imgs.append(tau_img*1e4)
        tau_recs.append(tau_rec)
        tau_bles.append(tau_ble/10)
        
    df['tau_img'] = tau_imgs
    df['tau_rec'] = tau_recs
    df['tau_ble'] = tau_bles
    
    return df


def extract_tracks(fp):
    """Extracts attributes of TrackStat.csv files produced by fiji and adds them to a DataFrame
    """
    
    fp = pathlib.Path(fp)
    
    df = pd.DataFrame()
    # TODO: add link extraction and maybe some trajectory parser
    for file in fp.iterdir():
        if str(file).endswith('TrackStat.csv'):
            this_csv = pd.read_csv(str(file))
            this_csv['cell'] = file.stem.split('_')[0]
            this_csv['file'] = str(file)
            
            df = df.append(this_csv, ignore_index=True)
    
    return df


#%% (G)Oldies

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


def crop_series(cell, FileDict):
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
    
    pre_imgs_clipped = [img[start[0]:end[0],start[1]:end[1]] for img in pre_imgs[0][:]]
    post_imgs_clipped = [img[start[0]:end[0],start[1]:end[1]] for img in post_imgs[0][:]]
    
    pre_imgs_clipped = np.asarray(pre_imgs_clipped)
    post_imgs_clipped = np.asarray(post_imgs_clipped)
    
    imgs = np.concatenate((pre_imgs[0], post_imgs[0]))
    
    tot_Ints = []
    for img in imgs:
        img[start[0]-10:end[0]+10,start[1]-10:end[1]+10] = np.nan
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
    
    stack1, out_intensity1, offsets = crop_and_shift(oif.imread(str(file_pre))[0], (start[0], Size[0], start[1], Size[1]))

    stack2, out_intensity2, offsets = crop_and_shift(oif.imread(str(file_post))[0], (offsets[-1, 0], Size[0], offsets[-1, 1], Size[1]))
    
    return np.concatenate((stack1, stack2)), np.concatenate((out_intensity1, out_intensity2))


def load_and_track(fp, yhxw=None):
    """
    Returns a concatenated series clipped from the bleaching clip size starting from the window yhxw or from bleaching clip.
    
    Inputs
    fp   -- filepath of oif file with images
    yhxw -- tiple containing (y position, height, x position, width) of clip
    Returns
    stack -- image series cropped after tracking and centering granule
    out_intensity -- list of mean intensity of every image without clip
    offsets       -- trajectory of (y, x) of clip
    """
    fp = pathlib.Path(fp)
    if yhxw is None:
        file_ble = fp.parent
        if '_pos' in fp.name:
            file_ble = file_ble.joinpath(str(fp.name).replace('_pos', '_ble'))
        elif '_pre' in fp.name:
            file_ble = file_ble.joinpath(str(fp.name).replace('_pre', '_ble'))
        
        Size = get_size(file_ble)
        start = get_clip(file_ble)
        
        yhxw = (start[0], Size[0], start[1], Size[1])
    
    stack, out_intensity, offsets = crop_and_shift(oif.imread(str(fp))[0], yhxw)
    
    return stack, out_intensity, offsets


def add_fitParams2(df, Plot=False):
    Amps = []
    Imms = []
    taus = []
    for i in df.index:
        print(df['cell'][i])
        this_f = df['f_corr'][i]
        this_t = df['t'][i]
        this_t = np.arange(0, df.timepoint[i]*len(this_f), df.timepoint[i])
        
        try:
            popt, pcov = curve_fit(Frap_Func, this_t[np.isfinite(this_f)], this_f[np.isfinite(this_f)], p0=[2000, 15, 5])
        except TypeError:
            popt = [np.nan,np.nan,np.nan]
        
        Amp, Imm, tau = popt[0], popt[1], popt[2]
        
        Amps.append(Amp)
        Imms.append(Imm)
        taus.append(tau)
        
        if Plot:
            plt.plot(this_t, Frap_Func(this_t, Amp, Imm, tau), 'r')
            plt.scatter(this_t, this_f)
            plt.title(df['cell'][i])
            plt.xlabel('Time (s)')
            plt.ylabel('Fraction I (u.a.)')
            #plt.xlim((0,2))
            plt.show()
            print('Amplitude: '+str(Amp))
            print('Imm Frac: '+str(Imm))
            print('tau: '+str(tau))
            print('k: '+str(1/tau))
    
    df['Amp_non'] = Amps
    df['Imm_non'] = Imms
    df['tau_non'] = taus
    
    return df


def add_foregroundSeries(df):
    means_Is = []
    stds_Is = []
    means_CPs = []
    stds_CPs = []
    areass = []
    for i in df.index:
        series = df['series'][i]
        
        means_I, stds_I, means_CP, stds_CP, areas, _ = calculate_series(series)
        
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


#%% Some testing functions

def _my_plot(variable, pp=None):
    for i in df.index:
        time = np.arange(0, len(df[variable+'_mean'].values[i])*df.timepoint.values[i], df.timepoint.values[i])
        fig, ax1 = plt.subplots()
        mean, = ax1.plot(time, df[variable+'_mean'].values[i]/np.nanmean(df[variable+'_mean'].values[i]), label='mean')
        mode, = ax1.plot(time, df[variable+'_mode'].values[i]/np.nanmean(df[variable+'_mode'].values[i]), label='mode')
        medi, = ax1.plot(time, df[variable+'_median'].values[i]/np.nanmean(df[variable+'_median'].values[i]), label='median')
        p20,  = ax1.plot(time, df[variable+'_p20'].values[i]/np.nanmean(df[variable+'_p20'].values[i]), label='p20')
        p80,  = ax1.plot(time, df[variable+'_p80'].values[i]/np.nanmean(df[variable+'_p80'].values[i]), label='p80')
        
        #ax2 = ax1.twinx()
        sums, = ax1.plot(time, df[variable+'_sum'].values[i]/np.nanmean(df[variable+'_sum'].values[i]), 'k', label='sum')
        fig.legend((mean, mode, medi, p20, p80, sums), ('mean', 'mode', 'medi', 'p20', 'p80', 'sums'), 'upper left')
        ax1.set_title(variable)
        if pp is not None:
            pp.savefig(fig)
        plt.show()
        print(i)

from matplotlib.backends.backend_pdf import PdfPages

intensities = ['pre_GR', 'pre_CP_near', 'pre_CP_far', 'pre_dark', 'pos_GR', 'pos_CP_near', 'pos_CP_far', 'pos_dark']

pp = PdfPages('stat_vars1.pdf')

for intensity in intensities:
    _my_plot(intensity, pp)

pp.close()


def _difMaskPlot(variable, pp=None):
    stats = ['_mean', '_mode', '_median', '_p20', '_p80', '_sum']
    for i in df.index:
        for stat in stats:
            time = np.arange(0, len(df[variable+stat].values[i])*df.timepoint.values[i], df.timepoint.values[i])
            plt.plot(time, df[variable+stat].values[i]/np.nanmean(df[variable+stat].values[i]), label='normal')
            plt.plot(time, df_menos10[variable+stat].values[i]/np.nanmean(df_menos10[variable+stat].values[i]), label='menos 10')
            plt.plot(time, df_menos50[variable+stat].values[i]/np.nanmean(df_menos50[variable+stat].values[i]), label='menos 50')
            plt.plot(time, df_mas10[variable+stat].values[i]/np.nanmean(df_mas10[variable+stat].values[i]), label='mas 10')
            plt.plot(time, df_mas50[variable+stat].values[i]/np.nanmean(df_mas50[variable+stat].values[i]), label='mas 50')
            
            plt.legend(loc=2)
            plt.title(str(i)+' '+variable+stat)
            if pp is not None:
                pp.savefig()
            plt.show()
            print(i)

from matplotlib.backends.backend_pdf import PdfPages

intensities = ['pre_GR', 'pre_CP_near', 'pre_CP_far', 'pre_dark', 'pos_GR', 'pos_CP_near', 'pos_CP_far', 'pos_dark']

pp = PdfPages('Vary_masks.pdf')

for intensity in intensities:
    _difMaskPlot(intensity, pp)

pp.close()

#%% Filtering functions


def ask_question(q_string):
    """
    Asks user for y (True) or n (False). If no valid answer is given in three trys a ValueError arises.
    """
    c=0
    while c<3:
        answer = input(q_string)
        if answer=='n':
            return False
        if answer=='y':
            return True
        else:
            c+=1
    raise ValueError('Answer was not in list of possible answers')

def cell_chooser(that_df):
    """
    Shows curve and fit for each cell in that_df and asks user if that curve should be dropped.
    """
    for i in that_df.index:
        print(that_df['cell'][i])
        this_f = that_df['f_corr'][i]
        t = that_df['t'][i]
        Amp = that_df['Amp'][i]
        Imm = that_df['Imm'][i]
        tau = that_df['tau'][i]
        
        plt.plot(t, Frap_Func(t, Amp, Imm, tau), 'r')
        plt.scatter(t[:len(this_f)], this_f)
        plt.title(that_df['cell'][i])
        plt.xlabel('Time (s)')
        plt.ylabel('Fraction I (u.a.)')
        #plt.xlim((0,2))
        plt.show()
        print('Amplitude: '+str(Amp))
        print('Imm Frac: '+str(Imm))
        print('tau: '+str(tau))
        print('k: '+str(1/tau))
        
        if not ask_question('Is it ok?'):
            that_df = that_df.drop(i)
        
    return that_df


def filter_df(df_all, is_CP=False):
    """
    Drops from df_all DataFrame every curve whos fit parameters aren't possible. (Imm<1, Amp<1, tau<100, pre_I_mean<3500)
    """
    if is_CP:
        for i in df_all.index:
            if abs(df_all.mean_area[i]-np.pi*((df_all.mean_diameter[i]/2)**2))/df_all.mean_area[i]>0.15:
                df_all = df_all.drop(i)
        """
        for i in df_all.index:
            if df_all.mean_pre_I_px[i]>3500:
                df_all = df_all.drop(i)
        """
    for i in df_all.index:
        if df_all.tau[i]>100:
            df_all = df_all.drop(i)
    for i in df_all.index:
        if df_all.Imm[i]>1:
            df_all = df_all.drop(i)
    for i in df_all.index:
        if df_all.Amp[i]>1:
            df_all = df_all.drop(i)
    return df_all


def complete_filter(old_df, is_CP=False):
    """
    Generates new Dataframe after quick and user filtering of old_df.
    """
    new_df = old_df.copy()
    
    # Quick filter obviously incorrect results
    new_df = filter_df(new_df)
    
    # Visual user filter
    new_df = cell_chooser(new_df)
    
    return new_df


#%% 

def boxplot(data_to_plot, title):
    bp = plt.boxplot(data_to_plot, patch_artist=True)
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='k', linewidth=2)
        # change fill color
        box.set( facecolor = 'b' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='k', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='k', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='r', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='r', alpha=0.5)
    
    ## Custom x-axis labels
    plt.xticks([1, 2, 3], ['FL', 'DSAM'])
    plt.title(title)


#%% Analyze full folder

def analyze_all(fp):
    """ Processes every date folder in the specified path for gr and cp analysis"""
    fp = pathlib.Path(fp)
    
    # Generate DataFrames
    df_cp = pd.DataFrame()
    df_gr = pd.DataFrame()
    df_track = pd.DataFrame()
    
    dates = [x for x in fp.iterdir() if x.is_dir()]
    for date_folder in dates:
        this_date = date_folder.name
        plasmids = [x for x in date_folder.iterdir() if x.is_dir()]
        for plasmid_folder in plasmids:
            this_plasmid = plasmid_folder.name
            exps = [x for x in plasmid_folder.iterdir() if x.is_dir()]
            for exp_folder in exps:
                this_exp = exp_folder.name
                print('analyzing '+this_date+' experiments transfected with '+this_plasmid+' and studying '+this_exp)
                
                if this_exp=='CP':
                    this_df = process_frap_CP(exp_folder)
                    this_df['date'] = this_date
                    this_df['exp']  = this_plasmid
                    
                    df_cp = df_cp.append(this_df, ignore_index=True)
                elif this_exp=='GR':
                    this_df = process_frap(exp_folder)
                    this_df['date'] = this_date
                    this_df['exp']  = this_plasmid
                    
                    df_gr = df_gr.append(this_df, ignore_index=True)
                elif this_exp=='Videos':
                    this_df = extract_tracks(exp_folder)
                    this_df['date'] = this_date
                    this_df['exp']  = this_plasmid
                    
                    df_track = df_track.append(this_df, ignore_index=True)
                else:
                    continue
    
    df_cp.to_pickle(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Resultados\cp.pandas')
    df_gr.to_pickle(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Resultados\gr.pandas')
    df_track.to_pickle(r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Resultados\tracks.pandas')


#%% Generate pdf with usual graphs

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def plot_all_curves(df, pp=None):
    that_df = df.copy()
    for i in that_df.index:
        print(that_df['cell'][i])
        this_f = that_df['f_corr'][i]
        t = that_df.t[i]
        
        if that_df['exp'][i]=='FL':
            this_color = 'b'
        elif that_df['exp'][i]=='DSAM':
            this_color='r'
        elif that_df['exp'][i]=='YFP':
            this_color='g'
        plt.scatter(t, this_f, color=this_color, alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Fraction I (u.a.)')
    plt.ylim((0,1))
    plt.xlim((0,100))
    if pp is not None:
        pp.savefig()
    plt.show()

    
def make_pdf(file, df):
    pp = PdfPages(file)
    
    # Plot tau histogram
    plt.hist(df.query('(exp=="FL")').tau.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').tau.values, color='b', alpha=0.5, label='DSAM')
    plt.title('$\\tau$')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot Imm histogram
    plt.hist(df.query('(exp=="FL")').Imm.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').Imm.values, color='b', alpha=0.5, label='DSAM')
    plt.title('Fracción Inmóvil')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot Amp histogram
    plt.hist(df.query('(exp=="FL")').Amp.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').Amp.values, color='b', alpha=0.5, label='DSAM')
    plt.title('Amplitud')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot diameter histogram
    plt.hist(df.query('(exp=="FL")').mean_diameter.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').mean_diameter.values, color='b', alpha=0.5, label='DSAM')
    plt.title('Diámetro')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot diameter histogram
    plt.hist(df.query('(exp=="FL")').mean_pre_I.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').mean_pre_I.values, color='b', alpha=0.5, label='DSAM')
    plt.title('Intensidad')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Pair Plot
    sns.pairplot(df, hue='exp', vars=['Amp', 'Imm', 'tau', 'mean_diameter', 'mean_area', 'mean_pre_I'], size=2)
    pp.savefig()
    plt.show()
    
    plot_all_curves(df, pp)
    plot_all_curves(df.query('exp=="FL"'), pp)
    plot_all_curves(df.query('exp=="DSAM"'), pp)
    
    pp.close()


def make_pdf_CP(file, df):
    pp = PdfPages(file)
    
    # Plot tau histogram
    plt.hist(df.query('(exp=="FL")').tau.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').tau.values, color='b', alpha=0.5, label='DSAM')
    plt.hist(df.query('(exp=="YFP")').tau.values, color='r', alpha=0.5, label='YFP')
    plt.title('$\\tau$')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot Imm histogram
    plt.hist(df.query('(exp=="FL")').Imm.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').Imm.values, color='b', alpha=0.5, label='DSAM')
    plt.hist(df.query('(exp=="YFP")').Imm.values, color='r', alpha=0.5, label='YFP')
    plt.title('Fracción Inmóvil')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot Amp histogram
    plt.hist(df.query('(exp=="FL")').Amp.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').Amp.values, color='b', alpha=0.5, label='DSAM')
    plt.hist(df.query('(exp=="YFP")').Amp.values, color='r', alpha=0.5, label='YFP')
    plt.title('Amplitud')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot diameter histogram
    plt.hist(df.query('(exp=="FL")').mean_area.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').mean_area.values, color='b', alpha=0.5, label='DSAM')
    plt.hist(df.query('(exp=="YFP")').mean_area.values, color='r', alpha=0.5, label='YFP')
    plt.title('Area')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot diameter histogram
    plt.hist(df.query('(exp=="FL")').mean_pre_I.values, color='g', alpha=0.5, label='FL')
    plt.hist(df.query('(exp=="DSAM")').mean_pre_I.values, color='b', alpha=0.5, label='DSAM')
    plt.hist(df.query('(exp=="YFP")').mean_pre_I.values, color='r', alpha=0.5, label='YFP')
    plt.title('Intensidad')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Pair Plot
    sns.pairplot(df, hue='exp', vars=['Amp', 'Imm', 'tau', 'mean_area', 'mean_pre_I'], size=2)
    pp.savefig()
    plt.show()
        
    plot_all_curves(df, pp)
    plot_all_curves(df.query('exp=="FL"'), pp)
    plot_all_curves(df.query('exp=="DSAM"'), pp)
    plot_all_curves(df.query('exp=="YFP"'), pp)
    
    pp.close()


def make_pdf_track(file, df):
    pp = PdfPages(file)
    
    # Plot histogram of track displacement
    plt.hist(df.query('exp=="FL"').TRACK_DISPLACEMENT.values, bins=20, label='FL', alpha=0.75, normed=True)
    plt.hist(df.query('exp=="DSAM"').TRACK_DISPLACEMENT.values, bins=20, label='DSAM', alpha=0.75, normed=True)
    plt.title('Displacement')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot cumulative histogram of track displacement
    plt.hist(df.query('exp=="FL"').TRACK_DISPLACEMENT.values, bins=len(df.query('exp=="FL"').TRACK_DISPLACEMENT.values), label='FL', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.hist(df.query('exp=="DSAM"').TRACK_DISPLACEMENT.values, bins=len(df.query('exp=="DSAM"').TRACK_DISPLACEMENT.values), label='DSAM', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.title('Displacement')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot histogram of track mean speed
    plt.hist(df.query('exp=="FL"').TRACK_MEAN_SPEED.values, bins=20, label='FL', alpha=0.75, normed=True)
    plt.hist(df.query('exp=="DSAM"').TRACK_MEAN_SPEED.values, bins=20, label='DSAM', alpha=0.75, normed=True)
    plt.title('Mean Speed')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot cumulative histogram of track mean speed
    plt.hist(df.query('exp=="FL"').TRACK_MEAN_SPEED.values, bins=len(df.query('exp=="FL"').TRACK_MEAN_SPEED.values), label='FL', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.hist(df.query('exp=="DSAM"').TRACK_MEAN_SPEED.values, bins=len(df.query('exp=="DSAM"').TRACK_MEAN_SPEED.values), label='DSAM', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.title('Mean Speed')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot histogram of track max speed
    plt.hist(df.query('exp=="FL"').TRACK_MAX_SPEED.values, bins=20, label='FL', alpha=0.75, normed=True)
    plt.hist(df.query('exp=="DSAM"').TRACK_MAX_SPEED.values, bins=20, label='DSAM', alpha=0.75, normed=True)
    plt.title('Max Speed')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot cumulative histogram of track max speed
    plt.hist(df.query('exp=="FL"').TRACK_MAX_SPEED.values, bins=len(df.query('exp=="FL"').TRACK_MAX_SPEED.values), label='FL', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.hist(df.query('exp=="DSAM"').TRACK_MAX_SPEED.values, bins=len(df.query('exp=="DSAM"').TRACK_MAX_SPEED.values), label='DSAM', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.title('Max Speed')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot histogram of track median speed
    plt.hist(df.query('exp=="FL"').TRACK_MEDIAN_SPEED.values, bins=20, label='FL', alpha=0.75, normed=True)
    plt.hist(df.query('exp=="DSAM"').TRACK_MEDIAN_SPEED.values, bins=20, label='DSAM', alpha=0.75, normed=True)
    plt.title('Median Speed')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot cumulative histogram of track median speed
    plt.hist(df.query('exp=="FL"').TRACK_MEDIAN_SPEED.values, bins=len(df.query('exp=="FL"').TRACK_MEDIAN_SPEED.values), label='FL', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.hist(df.query('exp=="DSAM"').TRACK_MEDIAN_SPEED.values, bins=len(df.query('exp=="DSAM"').TRACK_MEDIAN_SPEED.values), label='DSAM', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.title('Median Speed')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot histogram of track standard deviation speed
    plt.hist(df.query('exp=="FL"').TRACK_STD_SPEED.values, bins=20, label='FL', alpha=0.75, normed=True)
    plt.hist(df.query('exp=="DSAM"').TRACK_STD_SPEED.values, bins=20, label='DSAM', alpha=0.75, normed=True)
    plt.title('Standard Deviation')
    plt.legend()
    pp.savefig()
    plt.show()
    
    # Plot cumulative histogram of track standard deviation speed
    plt.hist(df.query('exp=="FL"').TRACK_STD_SPEED.values, bins=len(df.query('exp=="FL"').TRACK_STD_SPEED.values), label='FL', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.hist(df.query('exp=="DSAM"').TRACK_STD_SPEED.values, bins=len(df.query('exp=="DSAM"').TRACK_STD_SPEED.values), label='DSAM', alpha=0.75, cumulative=True, normed=True, histtype='step')
    plt.title('Standard Deviation')
    plt.legend()
    pp.savefig()
    plt.show()
    # Cross correlation pair plot
    sns.pairplot(df, hue='exp', vars=['TRACK_DURATION', 'TRACK_DISPLACEMENT', 'TRACK_MEAN_SPEED', 'TRACK_MAX_SPEED', 'TRACK_MEDIAN_SPEED', 'TRACK_STD_SPEED'], size=4)
    pp.savefig()
    plt.show()
    
    pp.close()