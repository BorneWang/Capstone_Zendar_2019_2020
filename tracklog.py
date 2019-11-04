#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracklog timestamp range [1541539442.280844 1541539890.5113943]
translation_value = 32.34626543940416 meters
translation_POV around [-9 -31]

"""

import h5py
import time
import numpy as np
from scipy.spatial.transform import Rotation as rot
from collections import defaultdict

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
        self.tracklog = defaultdict(list)
        self.translations_ECEF = dict()
        self.translations_POV = dict()
        self.translation_value = 0
        
        self.loaddata(src)
        self.get_translations(src)
        self.unitcount = self.test_correct()
        
    def loaddata(self, src):
        tic1 = time.time()
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
        N_log = len(tracklog)
        print("tracklog units:", N_log)                        
        
        start = 0
        for i in range(N_img):
            time_low = aperture[times[i]].attrs['TIMESTAMP_SPAN'][0][0]
            time_high = aperture[times[i]].attrs['TIMESTAMP_SPAN'][0][1]
            for j in range(start,N_log):
                time_log = timestamp[j]
                if time_log > time_high or time_log < time_low:
                    start = j
                    break
                self.tracklog[i].append(tracklog_unit(tracklog[j]))
        hdf5.close()
        tic2 = time.time()
        print("time consume:", tic2-tic1)
        
    def get_translations(self, src):
        hdf5 = h5py.File(src,'r')
        # radar image data
        aperture = hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys())
        
        trans_list = []
        
        for key in self.tracklog:
            car_pos = self.tracklog[key][0].position
            radar_pos = aperture[times[key]].attrs['POSITION'][0]
            self.translations_ECEF[key] = (radar_pos[0] - car_pos[0],
                                           radar_pos[1] - car_pos[1],
                                           radar_pos[2] - car_pos[2])
            self.translations_POV[key] = self.tracklog[key][0].attitude.apply(self.translations_ECEF[key])
            trans_list.append(np.linalg.norm(self.translations_ECEF[key]))
            
        self.translation_value = sum(trans_list)/len(trans_list)
            
        hdf5.close()
        
    def test_correct(self):
        count = 0
        for key in self.tracklog:
            count += len(self.tracklog[key])
        return count