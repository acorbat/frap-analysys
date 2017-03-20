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