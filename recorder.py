import gmplot
import pyproj
import numpy as np
from copy import deepcopy
import scipy.stats as stat
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot

def ecef2lla(pos):
    """ Convert position in ECEF frame to LLA """
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    if pos.ndim == 1:
        lon, lat, alt = pyproj.transform(ecef, lla, pos[0], pos[1], pos[2], radians=False)
    else:        
        lon, lat, alt = pyproj.transform(ecef, lla, pos[:,0], pos[:,1], pos[:,2], radians=False)
    return np.array([lon, lat, alt]).T

def ecef2enu(lat0, lon0):
    """ Compute quaternion of transformation between ECEF to ENU frame """
    MatNorthPole = np.array([[-1., 0., 0.],
                           [0., -1., 0.],
                           [ 0., 0. , 1.]])
    sColat = np.sin(np.pi/2-lat0)
    cColat = np.cos(np.pi/2-lat0)
    MatLat = np.array([[ cColat , 0. , sColat ],
                     [   0.   , 1. ,   0.   ],
                     [-sColat , 0. , cColat ]])
    sLon = np.sin(lon0)
    cLon = np.cos(lon0)
    Matlon = np.array([[ cLon , -sLon , 0. ],
                     [ sLon , cLon  , 0. ],
                     [  0.  ,  0.   , 1. ]])
    return rot.from_dcm(np.array([[0,-1,0],[1,0,0],[0,0,1]]).dot(Matlon.dot(MatLat.dot(MatNorthPole)).T))

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
        coords = ecef2lla(self.reader.tracklog_translate(self.reader.get_gps_pos(0,np.inf), self.reader.get_gps_att(0,np.inf)))
        gmap=gmplot.GoogleMapPlotter(np.mean(coords[:,1]), np.mean(coords[:,0]), 15)
        gmap.plot(coords[:,1], coords[:,0], 'green', edge_width = 2.5)

        coords = ecef2lla(self.reader.tracklog_translate(self.get_measured_positions(), self.get_measured_attitude()))
        gmap.plot(coords[:,1], coords[:,0], 'red', edge_width = 2.5)
        
        #coords = ecef2lla(self.reader.tracklog_translate(self.get_positions(), self.get_attitude()))
        #gmap.plot(coords[:,1], coords[:,0], 'cornflowerblue', edge_width = 2.5)
        
        #img_bounds = {}
        #img_bounds['west'] = (xmin - lon_midpt) * (grid_points / (grid_points - 1)) + lon_midpt
        #img_bounds['east'] = (xmax - lon_midpt) * (grid_points / (grid_points - 1)) + lon_midpt
        #img_bounds['north'] = (ymax - lat_midpt) * (grid_points / (grid_points - 1)) + lat_midpt
        #img_bounds['south'] = (ymin - lat_midpt) * (grid_points / (grid_points - 1)) + lat_midpt
        #gmap.ground_overlay('heatmap.png', img_bounds)
        
        gmap.apikey = "AIzaSyB0UlIEiFl6IFtzz2_1WaDyYsXjscLVRWU"
        gmap.draw("map.html")
    
    def plot_trajectory(self):
        """ Plot the trajectory in earth frame centered on initial position """
        pos = self.reader.tracklog_translate(self.reader.get_gps_pos(0,np.inf), self.reader.get_gps_att(0,np.inf))
        coord0 = ecef2lla(pos[0])
        q = ecef2enu(coord0[1]*np.pi/180, coord0[0]*np.pi/180)
        trajectory = q.apply(pos - pos[0])
        
        arrows = np.array([q.apply(data.earth2rbd([0,-1,0], True)) for data in self.reader.get_radardata(0,np.inf)])
        plt.figure()
        plt.plot(trajectory[:,0], trajectory[:,1])
        for i in range(0, len(arrows), 5):
            plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])        
        plt.xlabel('x_meters')
        plt.ylabel('y_meters')
        plt.axis('equal')
        plt.show()
    
    def plot_attitude(self):
        """ Plot the orientation in the map frame given by the GPS and after fusion """
        plt.figure()
        plt.title("Comparison between GPS attitude and attitude estimated in first data frame")
        plt.xlabel("Times (s)")
        plt.ylabel("Attitude (rad)")
        plt.plot(list(self.kalman_record.keys()),np.unwrap([r.as_euler('zxy')[0] for r in self.reader.get_gps_att(0, np.inf)]), label="GPS")
        plt.plot(list(self.kalman_record.keys()), np.unwrap([r.as_euler('zxy')[0] for r in self.get_measured_attitude()]), label="CV2")
        plt.plot(list(self.kalman_record.keys()), np.unwrap(np.array([kalman.attitude.as_euler('zxy')[0] for kalman in self.kalman_record.values()])), label="Kalman")
        plt.legend()    
        
    def plot_innovation(self, individual=False, p=0.99):
        """ Return innovation made by cv2 measurement during fusion """
        plt.figure()
        plt.title("Innovation in function of time")
        plt.xlabel("Time (s)")
        if individual:
            innovation = [kalman.innovation for kalman in list(self.kalman_record.values())[1:]]
            plt.plot(list(self.kalman_record.keys())[1:], [np.array([Z[0]**2/S[0,0],Z[1]**2/S[1,1],Z[2]**2/S[2,2]])/stat.chi2.ppf(p, df=1) for Z,S in innovation])
        else:
            plt.plot(list(self.kalman_record.keys())[1:], [kalman.innovation[0].dot(np.linalg.inv(kalman.innovation[1])).dot(kalman.innovation[0])/stat.chi2.ppf(p, df=len(kalman.innovation[0])) for kalman in list(self.kalman_record.values())[1:]])
        
    def get_positions(self):
        """ Return positions after fusion """
        return np.array([kalman.position for kalman in self.kalman_record.values()])  

    def get_attitude(self):
        """ Return attitude after fusion """
        return np.ravel([kalman.attitude for kalman in self.kalman_record.values()])  

    def get_measured_attitude(self):
        """ Return attitude in first image frame obtained with cv2 transformations """
        measured_att = [self.kalman.mapdata.attitude]
        times = self.reader.get_timestamps(0, np.inf)
        for i in range(1, len(times)):
            _, rotation = self.reader.get_radardata(times[i]).image_transformation_from(self.reader.get_radardata(times[i-1]))
            measured_att.append(measured_att[-1]*rotation)
        return np.ravel(measured_att)
    
    def get_measured_positions(self):
        """ Return positions obtained with cv2 transformations """
        measured_pos = [self.kalman.mapdata.gps_pos]
        measured_att = [self.kalman.mapdata.attitude]
        times = self.reader.get_timestamps(0, np.inf)
        for i in range(1, len(times)):
            translation, rotation = self.reader.get_radardata(times[i]).image_transformation_from(self.reader.get_radardata(times[i-1]))
            measured_pos.append(measured_pos[-1] + measured_att[-1].apply(translation, True))
            measured_att.append(measured_att[-1]*rotation)
        return np.array(measured_pos)