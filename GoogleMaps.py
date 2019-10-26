#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:09:28 2019

@author: david
"""
import numpy as np
from reader import Reader
import gmplot
import pyproj

# bring in data from reader
reader = Reader('radar_images_new_quat_pos.h5')
gps_array = reader.get_gps_pos(0,np.inf)

# define x, y,z
x = gps_array[:, 0]
y = gps_array[:, 1]
z = gps_array[:, 2]

# transform data
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)


# First two arugments are the geogrphical coordinates .i.e. Latitude and Longitude
# & the zoom resolution.
gmap=gmplot.GoogleMapPlotter(37.4220, -122.1012, 15)

# Create Scatter Plot
gmap.scatter(lat, lon, 'FF0000', size = 10, marker = False )

# Draw line between cordinate points
# gmap.plot(latitude_list, longitude_list, 'cornflowerblue', edge_width = 2.5)
             
#gmap.apikey = "Your_API_KEY"
gmap.apikey = "AIzaSyB0UlIEiFl6IFtzz2_1WaDyYsXjscLVRWU"

# Location where you want to save your file.
gmap.draw( "C:\\Users\\David\\Desktop\\UC Berkeley\\Zendar Capstone Project\\Software Files\\Capstone_Zendar_2019_2020-master\\map.html" )
