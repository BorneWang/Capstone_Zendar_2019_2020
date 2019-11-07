#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracklog timestamp range [1541539442.280844 1541539890.5113943]
translation_value = 32.465983399182754 meters
translation_POV around [-9 -31]

"""

import h5py
import time
import numpy as np
from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import interp1d
from statistics import stdev

class tracklog_unit:
    
    def __init__(self, item):
        self.timestamp = item['timestamp']
        self.position = item['position']
        self.distance_traveled = item['distance_traveled']
        self.attitude = (item['attitude'][1],
                         item['attitude'][2],
                         item['attitude'][3],
                         item['attitude'][0])
        self.attitude = rot.from_quat(self.attitude).inv()
        
class Tracklog:
    
    def __init__(self, src):
        self.translations_ECEF = dict()
        self.translations_POV = dict()
        self.translation_value = 0
        self.position_x = []
        self.position_y = []
        self.translations_POV_mean = 0
        self.translations_POV_stdev = 0
        
        tic = time.time()
        self.loaddata(src)
        self.get_translations(src)
        print("time consume:", time.time()-tic)
        
    def loaddata(self, src):
        # load h5 file
        hdf5 = h5py.File(src,'r')
        # radar image data
        aperture = hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys())
        N_img = len(times)
        print("radar images :", N_img)
        # tracklog data
        tracklog = hdf5['tracklog']
        timestamp = tracklog['timestamp']
        position_x = tracklog['position']['x']
        position_y = tracklog['position']['y']
        position_z = tracklog['position']['z']
        self.position_x = interp1d(timestamp, position_x)
        self.position_y = interp1d(timestamp, position_y)
        self.position_z = interp1d(timestamp, position_z)
        N_log = len(tracklog)
        print("tracklog units:", N_log)                        
        
    def get_translations(self, src):
        #load file
        hdf5 = h5py.File(src,'r')
        # radar image data
        aperture = hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys())
        N_img = len(times)
        # define list
        trans_list = []
        POV_x_list = []
        POV_y_list = []
        POV_z_list = []
        # loop
        for key in range(N_img):
            try:
                car_pos_x = self.position_x(float(times[key]))
                car_pos_y = self.position_y(float(times[key]))
                car_pos_z = self.position_z(float(times[key]))
            except:
                break
            radar_pos = aperture[times[key]].attrs['POSITION'][0]
            radar_att = (aperture[times[key]].attrs['ATTITUDE'][0][1],
                         aperture[times[key]].attrs['ATTITUDE'][0][2],
                         aperture[times[key]].attrs['ATTITUDE'][0][3],
                         aperture[times[key]].attrs['ATTITUDE'][0][0])
            radar_att = rot.from_quat(radar_att).inv()
            self.translations_ECEF[key] = (radar_pos[0] - car_pos_x,
                                           radar_pos[1] - car_pos_y,
                                           radar_pos[2] - car_pos_z)
            self.translations_POV[key] = radar_att.apply(self.translations_ECEF[key])
            POV_x_list.append(self.translations_POV[key][0])
            POV_y_list.append(self.translations_POV[key][1])
            POV_z_list.append(self.translations_POV[key][2])
            trans_list.append(np.linalg.norm(self.translations_ECEF[key]))
        
        # get results
        self.translation_value = sum(trans_list)/len(trans_list)
        self.translations_POV_mean = (sum(POV_x_list)/len(POV_x_list), 
                                      sum(POV_y_list)/len(POV_y_list),
                                      sum(POV_z_list)/len(POV_z_list))
        self.translations_POV_stdev = (stdev(POV_x_list),
                                       stdev(POV_y_list),
                                       stdev(POV_z_list))
        
            
        hdf5.close()