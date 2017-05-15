# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:47:57 2017

@author: Agus
"""
import os
import sys

from ij import IJ, ImagePlus, ImageStack
import fiji.plugin.trackmate.Settings as Settings
import fiji.plugin.trackmate.Model as Model
import fiji.plugin.trackmate.SelectionModel as SelectionModel
import fiji.plugin.trackmate.TrackMate as TrackMate
import fiji.plugin.trackmate.Logger as Logger
import fiji.plugin.trackmate.detection.DetectorKeys as DetectorKeys
import fiji.plugin.trackmate.detection.DogDetectorFactory as DogDetectorFactory
import fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory as SparseLAPTrackerFactory
import fiji.plugin.trackmate.tracking.LAPUtils as LAPUtils
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.FeatureAnalyzer as FeatureAnalyzer
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory as SpotContrastAndSNRAnalyzerFactory
import fiji.plugin.trackmate.action.ExportStatsToIJAction as ExportStatsToIJAction
import fiji.plugin.trackmate.io.TmXmlReader as TmXmlReader
import fiji.plugin.trackmate.action.ExportTracksToXML as ExportTracksToXML
import fiji.plugin.trackmate.io.TmXmlWriter as TmXmlWriter
import fiji.plugin.trackmate.features.ModelFeatureUpdater as ModelFeatureUpdater
import fiji.plugin.trackmate.features.SpotFeatureCalculator as SpotFeatureCalculator
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzer as SpotContrastAndSNRAnalyzer
import fiji.plugin.trackmate.features.spot.SpotIntensityAnalyzerFactory as SpotIntensityAnalyzerFactory
import fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer as TrackSpeedStatisticsAnalyzer
import fiji.plugin.trackmate.util.TMUtils as TMUtils

# Generate list of files to analyze
all_experiments = list()
all_exp_folder = r'C:\Users\Agus\Documents\Laboratorio\uVesiculas\Mediciones'
date_folders = os.listdir(all_exp_folder)
for date_folder in date_folders:
    this_date_folder = all_exp_folder + '\\' + date_folder
    exp_folders = [f for f in os.listdir(this_date_folder) if os.path.isdir(this_date_folder + '\\' + f)]
    for exp_folder in exp_folders:
        this_exp_folder = this_date_folder + '\\' + exp_folder
        type_folders = [f for f in os.listdir(this_exp_folder) if f=='Videos']
        for type_folder in type_folders:
            this_type_folder = this_exp_folder + '\\' + type_folder
            these_files = [this_type_folder+ '\\' + f for f in os.listdir(this_type_folder) if f.endswith('.oif')]
            all_experiments.extend(these_files)

print(all_experiments)

   
# Get currently selected image
#imp = WindowManager.getCurrentImage()
imp = IJ.openImage('http://fiji.sc/samples/FakeTracks.tif')
#imp.show()
   
   
#-------------------------
# Instantiate model object
#-------------------------
   
model = Model()
   
# Set logger
model.setLogger(Logger.IJ_LOGGER)
   
#------------------------
# Prepare settings object
#------------------------
      
settings = Settings()
settings.setFrom(imp)
      
# Configure detector
settings.detectorFactory = DogDetectorFactory()
settings.detectorSettings = {
    DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
    DetectorKeys.KEY_RADIUS : 2.5,
    DetectorKeys.KEY_TARGET_CHANNEL : 1,
    DetectorKeys.KEY_THRESHOLD : 5.,
    DetectorKeys.KEY_DO_MEDIAN_FILTERING : False,
} 
    
# Configure tracker
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
settings.trackerSettings['LINKING_MAX_DISTANCE'] = 10.0
settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE']=10.0
settings.trackerSettings['MAX_FRAME_GAP']= 3
   
# Add the analyzers for some spot features.
# You need to configure TrackMate with analyzers that will generate 
# the data you need. 
# Here we just add two analyzers for spot, one that computes generic
# pixel intensity statistics (mean, max, etc...) and one that computes
# an estimate of each spot's SNR. 
# The trick here is that the second one requires the first one to be in
# place. Be aware of this kind of gotchas, and read the docs. 
settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())
   
# Add an analyzer for some track features, such as the track mean speed.
settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
   
settings.initialSpotFilterValue = 1
   
print(str(settings))
      
#----------------------
# Instantiate trackmate
#----------------------
   
trackmate = TrackMate(model, settings)
      
#------------
# Execute all
#------------
   
     
ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
     
ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
     
      
      
#----------------
# Display results
#----------------
   
model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
    
selectionModel = SelectionModel(model)
displayer =  HyperStackDisplayer(model, selectionModel, imp)
displayer.render()
displayer.refresh()
   
# The feature model, that stores edge and track features.
fm = model.getFeatureModel()
   
for id in model.getTrackModel().trackIDs(True):
   
    # Fetch the track feature from the feature model.
    v = fm.getTrackFeature(id, 'TRACK_MEAN_SPEED')
    model.getLogger().log('')
    model.getLogger().log('Track ' + str(id) + ': mean velocity = ' + str(v) + ' ' + model.getSpaceUnits() + '/' + model.getTimeUnits())
       
    track = model.getTrackModel().trackSpots(id)
    for spot in track:
        sid = spot.ID()
        # Fetch spot features directly from spot. 
        x=spot.getFeature('POSITION_X')
        y=spot.getFeature('POSITION_Y')
        t=spot.getFeature('FRAME')
        q=spot.getFeature('QUALITY')
        snr=spot.getFeature('SNR') 
        mean=spot.getFeature('MEAN_INTENSITY')
        model.getLogger().log('\tspot ID = ' + str(sid) + ': x='+str(x)+', y='+str(y)+', t='+str(t)+', q='+str(q) + ', snr='+str(snr) + ', mean = ' + str(mean))