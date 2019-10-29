import gmplot
import pyproj
import numpy as np
from copy import deepcopy
import scipy.stats as stat
import matplotlib.pyplot as plt

def pos2coord(pos):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, pos[:,0], pos[:,1], pos[:,2], radians=False)
    return np.array([lon, lat, alt]).T

class Recorder:
    
    def __init__(self, reader, kalman):
        self.kalman_record = dict()
        self.reader = reader
        self.kalman = kalman
         
    def record(self, ts):
        """ Record value in a kalmans dictionary for later use """
        self.kalman_record[ts] = deepcopy(self.kalman)
    
    def export_map(self):
        """ Plot reference GPS on a Google map as well as measured position and filtered position """
        coords = pos2coord(self.reader.get_gps_pos(0,np.inf))
        gmap=gmplot.GoogleMapPlotter(np.mean(coords[:,1]), np.mean(coords[:,0]), 15)
        gmap.plot(coords[:,1], coords[:,0], 'green', edge_width = 2.5)

        coords = pos2coord(self.get_measured_positions())
        gmap.plot(coords[:,1], coords[:,0], 'red', edge_width = 2.5)

        coords = pos2coord(self.get_positions())
        gmap.plot(coords[:,1], coords[:,0], 'cornflowerblue', edge_width = 2.5)
        gmap.apikey = "AIzaSyB0UlIEiFl6IFtzz2_1WaDyYsXjscLVRWU"
        gmap.draw("map.html")
        
    def plot_attitude(self):
        """ Plot the orientation in the map frame given by the GPS and after fusion """
        plt.figure()
        plt.plot(list(self.kalman_record.keys()), np.array([(kalman.mapdata.attitude.inv()*kalman.last_data.attitude).as_rotvec()[2] for kalman in self.kalman_record.values()]))
        plt.plot(list(self.kalman_record.keys()),[(self.kalman.mapdata.attitude.inv()*r).as_rotvec()[2] for r in self.reader.get_gps_att(0, np.inf)])
    
    def plot_innovation(self, individual=False, p=0.99):
        """ Return innovation made by cv2 measurement during fusion """
        plt.figure()
        if individual:
            innovation = [kalman.innovation for kalman in list(self.kalman_record.values())[1:]]
            plt.plot(list(self.kalman_record.keys())[1:], [np.array([Z[0]**2/S[0,0],Z[1]**2/S[1,1],Z[2]**2/S[2,2]])/stat.chi2.ppf(p, df=1) for Z,S in innovation])
        else:
            plt.plot(list(self.kalman_record.keys())[1:], [kalman.innovation[0]*np.linalg.inv(kalman.innovation[1])*kalman.innovation[0]/stat.chi2.ppf(p, df=len(kalman.innovation[0])) for kalman in list(self.kalman_record.values())[1:]])
        
    def get_positions(self):
        """ Return positions after fusion """
        return np.array([kalman.last_data.gps_pos for kalman in self.kalman_record.values()])  

    def get_measured_positions(self):
        """ Return positions obtained with cv2 translations """
        return self.kalman.mapdata.gps_pos + self.kalman.mapdata.earth2rbd(np.concatenate(([np.zeros(3)],np.cumsum(np.array([kalman.Y[0:3] for kalman in list(self.kalman_record.values())[1:]]), axis=0))), True)