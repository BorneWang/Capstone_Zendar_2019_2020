# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:30:01 2019

@author: Pierre-Louis
"""

import numpy as np
import h5py
from PIL import Image
from data import RadarData
import matplotlib.pyplot as plt

def calculate_norm_log(image):
    """ Calculate the log of the norm of an image with complex pixels """
    row, col = image.shape
    new_image = np.zeros((row,col))
    for i in range(row):
        new_image[i,:] = list(map(lambda x: np.linalg.norm((x['real'],x['imag'])), image[i,:]))
    log_image = np.log(new_image)
    return log_image

def heatmap2img(heatmap):
    """ Conversion of a heatmap into an image """
    fig, ax = plt.subplots()
    height, width = heatmap.shape 
    plt.imshow(heatmap)
    plt.axis('off')
    height, width = heatmap.shape 
    fig.set_size_inches(width/100.0/4.0, height/100.0/4.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(data)

class Reader:
    
    def __init__(self, src):
        self.src = src
        self.heatmaps = dict()
        self.load_heatmaps()
    
    def load_heatmaps(self):
        """ Function to load heatmaps from HDF5 file (and format data to heatmaps if needed) """
        hdf5 = h5py.File(self.src,'r')
        aperture = self.hdf5['radar']['broad01']['aperture2D']
        times = list(aperture.keys()).sort()
        for t in times:
            #check if data is already converted as a heatmap
            #TODO: decide if RadarData should store image or heatmap
            if isinstance(aperture[t][0,0], complex):
                aperture[t][...] = calculate_norm_log(aperture[t])
                self.heatmaps[t] = RadarData(heatmap2img(aperture[t]), aperture[t].attrs['position'], aperture[t].attrs['attitude'])
            else:
                self.heatmaps[t] = RadarData(heatmap2img(aperture[t]), aperture[t].attrs['position'], aperture[t].attrs['attitude'])
        hdf5.close()
        
    def play_video(self, t_ini, t_final):
        # TODO: play a video with image between t_ini and t_final
        return
    
    def find_timestamps(self, timestamp, timestamp_final):
        times = self.heatmaps.keys.sort()
        selection = []
        for t in times:
            if t>=timestamp and t<=timestamp_final:
                selection.push(t)
        return selection
    
    def get_radardata(self, timestamp, timestamp_final=None):
        """ Return radar data for time between timestamp and timestamp_final """
        if timestamp_final is None:
            return self.heatmaps[timestamp]
        else:
            times = self.find_timestamps(timestamp, timestamp_final)
            return np.array([self.heatmaps[t] for t in times])
            
    def get_heatmap_img(self, timestamp, timestamp_final=None):
        """ Return radar data image for time between timestamp and timestamp_final """
        if timestamp_final is None:
            return self.heatmaps[timestamp].img
        else:
            times = self.find_timestamps(timestamp, timestamp_final)
            return np.array([self.heatmaps[t].img for t in times])