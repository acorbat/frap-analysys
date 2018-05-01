import pathlib
import os

from frap_analysis import oiffile as oif

def generate_FileDict(filepath):
    """
    Generates a dictionary with paths for each cell and time period

    Inputs:
    filepath -- filepath to folder with all the .oif files
    Returns:
    File_Dict -- Dictionary where keys are filename split by '_' and values are
    the corresponding full path
    """
    filepath = pathlib.Path(filepath)
    File_Dict = {tuple(f.stem.split('_')[:-1]): f for f in filepath.glob('*.oif') if ('_pos' in str(f.name) or '_pre' in str(f.name))}
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
