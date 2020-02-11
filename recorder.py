import gmplot
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

from utils import rotation_proj, ecef2enu, ecef2lla, rbd_translate

class Recorder:
    
    def __init__(self, reader, kalman):
        self.kalman_record = dict()
        self.reader = reader
        self.kalman = kalman
         
    def record(self, ts):
        """ Record value in a kalmans dictionary for later use """
        self.kalman_record[ts] = {'attitude': self.kalman.attitude, 'position': self.kalman.position, 'innovation': self.kalman.innovation}
    
    def export_map(self):
        """ Plot reference GPS on a Google map as well as measured position and filtered position """
        if hasattr(self.reader,"groundtruth"):
            coords = ecef2lla(rbd_translate(np.array(self.reader.groundtruth["POSITION"]), self.reader.groundtruth["ATTITUDE"], self.reader.groundtruth_translation))
            gmap=gmplot.GoogleMapPlotter(np.rad2deg(np.mean(coords[:,1])), np.rad2deg(np.mean(coords[:,0])), 15)
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'black', edge_width = 2.5)
        
        coords = ecef2lla(rbd_translate(self.reader.get_gps_pos(), self.reader.get_gps_att(), self.reader.tracklog_translation))
        if not hasattr(self.reader,"groundtruth"):
            gmap=gmplot.GoogleMapPlotter(np.mean(coords[:,1]), np.mean(coords[:,0]), 15)
        gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'green', edge_width = 2.5)

        coords = ecef2lla(rbd_translate(self.get_measured_positions(), self.get_measured_attitude(), self.reader.tracklog_translation))
        gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'red', edge_width = 2.5)
        
        if len(self.get_positions())!=0:
            coords = ecef2lla(rbd_translate(self.get_positions(), self.get_attitudes(), self.reader.tracklog_translation))
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'cornflowerblue', edge_width = 2.5)
            
        #img_bounds = {}
        #img_bounds['west'] = (xmin - lon_midpt) * (grid_points / (grid_points - 1)) + lon_midpt
        #img_bounds['east'] = (xmax - lon_midpt) * (grid_points / (grid_points - 1)) + lon_midpt
        #img_bounds['north'] = (ymax - lat_midpt) * (grid_points / (grid_points - 1)) + lat_midpt
        #img_bounds['south'] = (ymin - lat_midpt) * (grid_points / (grid_points - 1)) + lat_midpt
        #gmap.ground_overlay('heatmap.png', img_bounds)
        
        gmap.apikey = "AIzaSyB0UlIEiFl6IFtzz2_1WaDyYsXjscLVRWU"
        gmap.draw("map.html")
    
    def plot_trajectory(self, arrow=False):
        """ Plot the trajectory in earth frame centered on initial position """
        fig = plt.figure()
        if hasattr(self.reader,"groundtruth"):
            pos = rbd_translate(np.array(self.reader.groundtruth["POSITION"]), self.reader.groundtruth["ATTITUDE"], self.reader.groundtruth_translation)
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
            trajectory = att0.apply(pos - pos[0])   
            plt.plot(trajectory[:,0], trajectory[:,1], 'black', label="Groundtruth", picker=True)
        else:
            pos = rbd_translate(self.reader.get_gps_pos(), self.reader.get_gps_att(), self.reader.tracklog_translation)
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
          
        pos = rbd_translate(self.reader.get_gps_pos(), self.reader.get_gps_att(), self.reader.tracklog_translation)
        trajectory = att0.apply(pos - pos[0])       
        plt.plot(trajectory[:,0], trajectory[:,1], 'g', label="GPS", picker=True)
        if arrow:
            arrows = np.array([att0.apply(data.earth2rbd([0,-1,0], True)) for data in self.reader.get_radardata()])
            for i in range(0, len(arrows), 5):
                plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
        
        pos = rbd_translate(self.get_measured_positions(), self.get_measured_attitude(), self.reader.tracklog_translation)
        trajectory = att0.apply(pos - pos[0])        
        plt.plot(trajectory[:,0], trajectory[:,1], 'r', label="CV2", picker=True)
        if arrow:
            arrows = np.array([att0.apply(att.apply([0,-1,0], True)) for att in self.get_measured_attitude()])
            for i in range(0, len(arrows), 5):
                plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
        
        if len(self.get_positions())!=0:
            pos = rbd_translate(self.get_positions(), self.get_attitudes(), self.reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(trajectory[:,0], trajectory[:,1], 'cornflowerblue', label="Output", picker=True)
            if arrow:
                arrows = np.array([att0.apply(att.apply([0,-1,0], True)) for att in self.get_measured_attitude()])
                for i in range(0, len(arrows), 5):
                    plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
    
        def show_timestamp(event):
            print(str(round(self.reader.get_timestamps()[event.ind[0]],2))+"s")
    
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.axis('equal')
        plt.legend()
        fig.canvas.mpl_connect('pick_event', show_timestamp)  
    
    def plot_attitude(self):
        """ Plot the orientation in the map frame given by the GPS and after fusion """
        plt.figure()
        plt.title("Yaw")
        plt.xlabel("Times (s)")
        plt.ylabel("Yaw (rad)")
        if hasattr(self.reader,"groundtruth"):
            pos = rbd_translate(np.array(self.reader.groundtruth["POSITION"]), self.reader.groundtruth["ATTITUDE"], self.reader.groundtruth_translation)
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
            plt.plot(self.reader.groundtruth["TIME"],np.unwrap([r.as_euler('zxy')[0] for r in self.reader.groundtruth["ATTITUDE"]]), 'black', label="Groundtruth")

        else:
            pos = rbd_translate(self.reader.get_gps_pos(), self.reader.get_gps_att(), self.reader.tracklog_translation)
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
        plt.plot(self.reader.get_timestamps(), np.unwrap([rotation_proj(att0,r).as_euler('zxy')[0] for r in self.reader.get_gps_att()]), 'green', label="GPS")
        plt.plot(self.reader.get_timestamps(), np.unwrap([rotation_proj(att0,r).as_euler('zxy')[0] for r in self.get_measured_attitude()]), 'red', label="CV2")
        plt.plot(list(self.kalman_record.keys()), np.unwrap(np.array([rotation_proj(att0,kalman['attitude']).as_euler('zxy')[0] for kalman in self.kalman_record.values()])), 'cornflowerblue', label="Output")
        plt.legend()    
        
    def plot_innovation(self, individual=False, p=0.99):
        """ Return innovation made by cv2 measurement during fusion """
        plt.figure()
        plt.title("Innovation in function of time")
        plt.xlabel("Time (s)")
        if individual:
            innovation = [kalman['innovation'] for kalman in list(self.kalman_record.values())[1:]]
            plt.plot(list(self.kalman_record.keys())[1:], [np.array([Z[0]**2/S[0,0],Z[1]**2/S[1,1],Z[2]**2/S[2,2]])/stat.chi2.ppf(p, df=1) for Z,S in innovation])
        else:
            plt.plot(list(self.kalman_record.keys())[1:], [kalman['innovation'][0].dot(np.linalg.inv(kalman['innovation'][1])).dot(kalman['innovation'][0])/stat.chi2.ppf(p, df=len(kalman['innovation'][0])) for kalman in list(self.kalman_record.values())[1:]])
        
    def get_positions(self):
        """ Return positions after fusion """
        return np.array([kalman['position'] for kalman in self.kalman_record.values()])  

    def get_attitudes(self):
        """ Return attitude after fusion """
        return np.ravel([kalman['attitude'] for kalman in self.kalman_record.values()])  

    def get_measured_attitude(self):
        """ Return attitude in first image frame obtained with cv2 transformations """
        measured_att = [self.reader.get_gps_att(0)]
        times = self.reader.get_timestamps(0, np.inf)
        for i in range(1, len(times)):
            _, rotation = self.reader.get_radardata(times[i]).image_transformation_from(self.reader.get_radardata(times[i-1]))
            measured_att.append(measured_att[-1]*rotation)
        return np.ravel(measured_att)
    
    def get_measured_positions(self):
        """ Return positions obtained with cv2 transformations """
        measured_pos = [self.reader.get_gps_pos(0)]
        measured_att = [self.reader.get_gps_att(0)]
        times = self.reader.get_timestamps(0, np.inf)
        for i in range(1, len(times)):
            translation, rotation = self.reader.get_radardata(times[i]).image_transformation_from(self.reader.get_radardata(times[i-1]))
            measured_pos.append(measured_pos[-1] + measured_att[-1].apply(translation, True))
            measured_att.append(measured_att[-1]*rotation)
        return np.array(measured_pos)