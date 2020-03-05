import gmplot
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj, ecef2enu, ecef2lla, rbd_translate

class Recorder:
    
    def __init__(self, reader, kalman):
        self.kalman_record = dict()
        self.reader = reader
        self.kalman = kalman
        
        self.measured_pos = None
        self.measured_att = None
        self.measured_pos_corr = None
        self.measured_att_corr = None
         
    def record(self, ts):
        """ Record value in a kalmans dictionary for later use """
        self.kalman_record[ts] = {'ATTITUDE': self.kalman.attitude, 'POSITION': self.kalman.position, 'INNOVATION': self.kalman.innovation, 'TRANS': self.kalman.trans}
    
    def export_map(self):
        """ Plot reference GPS on a Google map as well as measured position and filtered position """
        print("Exporting Google map...")
        if hasattr(self.reader,"groundtruth"):
            coords = ecef2lla(rbd_translate(self.reader.get_groundtruth_pos(), self.reader.get_groundtruth_att(), self.reader.groundtruth_translation))
            gmap=gmplot.GoogleMapPlotter(np.rad2deg(np.mean(coords[:,1])), np.rad2deg(np.mean(coords[:,0])), 15)
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'black', edge_width = 2.5)
        
        coords = ecef2lla(rbd_translate(self.reader.get_gps_pos(), self.reader.get_gps_att(), self.reader.tracklog_translation))
        if not hasattr(self.reader,"groundtruth"):
            gmap=gmplot.GoogleMapPlotter(np.mean(coords[:,1]), np.mean(coords[:,0]), 15)
        gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'green', edge_width = 2.5)

        coords = ecef2lla(rbd_translate(self.get_measured_positions(), self.get_measured_attitudes(), self.reader.tracklog_translation))
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
    
    def plot_trajectory(self, arrow=False, corrected = False):
        """ Plot the trajectory in earth frame centered on initial position """
        fig = plt.figure()
        if hasattr(self.reader,"groundtruth"):
            pos = rbd_translate(self.reader.get_groundtruth_pos(), self.reader.get_groundtruth_att(), self.reader.groundtruth_translation)
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
            trajectory = att0.apply(pos - pos[0])   
            plt.plot(trajectory[:,0], trajectory[:,1], 'black', label="Groundtruth", picker=True)
        else:
            pos0 = rbd_translate(self.reader.get_gps_pos(0), self.reader.get_gps_att(0), self.reader.tracklog_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
          
        pos = rbd_translate(self.reader.get_gps_pos(), self.reader.get_gps_att(), self.reader.tracklog_translation)
        trajectory = att0.apply(pos - pos[0])       
        plt.plot(trajectory[:,0], trajectory[:,1], 'g', label="GPS", picker=True)
        if arrow:
            arrows = np.array([att0.apply(data.earth2rbd([0,-1,0], True)) for data in self.reader.get_radardata()])
            for i in range(0, len(arrows), 5):
                plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
        
        pos = rbd_translate(self.get_measured_positions(), self.get_measured_attitudes(), self.reader.tracklog_translation)
        trajectory = att0.apply(pos - pos[0])        
        plt.plot(trajectory[:,0], trajectory[:,1], 'r', label="CV2", picker=True)
        if arrow:
            arrows = np.array([att0.apply(att.apply([0,-1,0], True)) for att in self.get_measured_attitudes()])
            for i in range(0, len(arrows), 5):
                plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
                
        if corrected:
            pos = rbd_translate(self.get_measured_positions(corrected), self.get_measured_attitudes(corrected), self.reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(trajectory[:,0], trajectory[:,1], 'r--', label="CV2 corrected", picker=True)
            if arrow:
                arrows = np.array([att0.apply(att.apply([0,-1,0], True)) for att in self.get_measured_attitudes(corrected)])
                for i in range(0, len(arrows), 5):
                    plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
        
        if len(self.get_positions())!=0:
            pos = rbd_translate(self.get_positions(), self.get_attitudes(), self.reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(trajectory[:,0], trajectory[:,1], 'cornflowerblue', label="Output", picker=True)
            if arrow:
                arrows = np.array([att0.apply(att.apply([0,-1,0], True)) for att in self.get_attitudes()])
                for i in range(0, len(arrows), 5):
                    plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
    
        def show_timestamp(event):
            print(str(round(self.reader.get_timestamps()[event.ind[0]],2))+"s")
    
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.axis('equal')
        plt.legend()
        plt.title("Trajectory in ENU frame centered on initial position")
        fig.canvas.mpl_connect('pick_event', show_timestamp) 
    
    def plot_altitude(self, corrected = False):
        """ Plot the altitude in earth frame centered on initial position """
        plt.figure()
        if hasattr(self.reader,"groundtruth"):
            pos = rbd_translate(self.reader.get_groundtruth_pos(), self.reader.get_groundtruth_att(), self.reader.groundtruth_translation)
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
            trajectory = att0.apply(pos - pos[0])   
            plt.plot(self.reader.get_timestamps(), trajectory[:,2], 'black', label="Groundtruth", picker=True)
        else:
            pos0 = rbd_translate(self.reader.get_gps_pos(0), self.reader.get_gps_att(0), self.reader.tracklog_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
          
        pos = rbd_translate(self.reader.get_gps_pos(), self.reader.get_gps_att(), self.reader.tracklog_translation)
        trajectory = att0.apply(pos - pos[0])       
        plt.plot(self.reader.get_timestamps(), trajectory[:,2], 'g', label="GPS", picker=True)
     
        pos = rbd_translate(self.get_measured_positions(), self.get_measured_attitudes(), self.reader.tracklog_translation)
        trajectory = att0.apply(pos - pos[0])        
        plt.plot(self.reader.get_timestamps(), trajectory[:,2], 'r', label="CV2", picker=True)
             
        if corrected:
            pos = rbd_translate(self.get_measured_positions(corrected), self.get_measured_attitudes(corrected), self.reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(self.reader.get_timestamps(), trajectory[:,2], 'r--', label="CV2 corrected", picker=True)
      
        if len(self.get_positions())!=0:
            pos = rbd_translate(self.get_positions(), self.get_attitudes(), self.reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(self.reader.get_timestamps(), trajectory[:,2], 'cornflowerblue', label="Output", picker=True)

        plt.xlabel('Times (s)')
        plt.ylabel('z (meters)')
        plt.axis('equal')
        plt.legend()
        plt.title("Altitude in ENU frame centered on initial position")
        
    def plot_attitude(self, corrected=False):
        """ Plot the orientation in the map frame given by the GPS and after fusion """
        plt.figure()
        plt.title("Yaw")
        plt.xlabel("Times (s)")
        plt.ylabel("Yaw (rad)")
        q = rot.from_dcm([[0,-1,0],[-1,0,0],[0,0,-1]])
        if hasattr(self.reader,"groundtruth"):
            pos0 = rbd_translate(self.reader.get_groundtruth_pos(0), self.reader.get_groundtruth_att(0), self.reader.groundtruth_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
            plt.plot(self.reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.reader.get_groundtruth_att()], 'black', label="Groundtruth")
        else:
            pos0 = rbd_translate(self.reader.get_gps_pos(0), self.reader.get_gps_att(0), self.reader.tracklog_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
        
        plt.plot(self.reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.reader.get_gps_att()], 'green', label="GPS")
        plt.plot(self.reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.get_measured_attitudes()], 'red', label="CV2")
        if corrected:
            plt.plot(self.reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.get_measured_attitudes(corrected)], 'r--', label="CV2 corrected")
        plt.plot(list(self.kalman_record.keys()), np.array([rotation_proj(att0, q*kalman['ATTITUDE']).as_euler('zxy')[0] for kalman in self.kalman_record.values()]), 'cornflowerblue', label="Output")
        plt.legend()    
        
    def plot_innovation(self, individual=False, p=0.99):
        """ Return innovation made by cv2 measurement during fusion """
        if list(self.kalman_record.values())[1]['INNOVATION'] is None:
            print("Plot of innovation canceled due to lack of Kalman update")
        else:
            plt.figure()
            plt.title("Innovation in function of time")
            plt.xlabel("Time (s)")
            if individual:
                innovation = [kalman['INNOVATION'] for kalman in list(self.kalman_record.values())[1:]]
                plt.plot(list(self.kalman_record.keys())[1:], [np.array([Z[0]**2/S[0,0],Z[1]**2/S[1,1],Z[2]**2/S[2,2]])/stat.chi2.ppf(p, df=1) for Z,S in innovation])
            else:
                plt.plot(list(self.kalman_record.keys())[1:], [kalman['INNOVATION'][0].dot(np.linalg.inv(kalman['INNOVATION'][1])).dot(kalman['INNOVATION'][0])/stat.chi2.ppf(p, df=len(kalman['INNOVATION'][0])) for kalman in list(self.kalman_record.values())[1:]])
        
    def plot_kalman_evaluation(self):
        """ Return error of the Kalman in filter in the first image frame """
        # TODO: reuse groundtruth when available
        #if hasattr(self.reader,"groundtruth"):
        #    error_pos = np.array([self.reader.get_radardata(0).earth2rbd(self.kalman_record[ts]['POSITION']-self.reader.get_groundtruth_pos(ts)) for ts in list(self.kalman_record.keys())])
        #else:    
        error_pos = np.array([self.reader.get_radardata(0).earth2rbd(self.kalman_record[ts]['POSITION']-self.reader.get_gps_pos(ts)) for ts in list(self.kalman_record.keys())])
        plt.figure()
        plt.plot(self.reader.get_timestamps(), error_pos)
        plt.legend(["Right", "Backward", "Down"])
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.title("Error in position of the Kalman filter in first image frame")
        
        # TODO: reuse groundtruth when available
        #if hasattr(self.reader,"groundtruth"):
        #    error_att = np.array([rotation_proj(self.reader.get_groundtruth_att(ts), self.kalman_record[ts]['ATTITUDE']).as_euler('zxy')[0] for ts in list(self.kalman_record.keys())])
        #else:    
        error_att = np.array([rotation_proj(self.reader.get_gps_att(ts), self.kalman_record[ts]['ATTITUDE']).as_euler('zxy')[0] for ts in list(self.kalman_record.keys())])
        plt.figure()
        plt.plot(self.reader.get_timestamps(), error_att)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (rad)")
        plt.title("Error in attitude of the Kalman filter in first image frame")
        return error_pos, error_att
        
    def get_positions(self):
        """ Return positions after fusion """
        return np.array([kalman['POSITION'] for kalman in self.kalman_record.values()])  

    def get_attitudes(self):
        """ Return attitude after fusion """
        return np.ravel([kalman['ATTITUDE'] for kalman in self.kalman_record.values()])  

    def get_measurements(self):
        """ Return positions and attitude obtained with cv2 transformations """
        if hasattr(self.reader,"groundtruth"):
            self.measured_pos = [self.reader.get_groundtruth_pos(0)]
            self.measured_att = [self.reader.get_groundtruth_att(0)]
        else:
            self.measured_pos = [self.reader.get_gps_pos(0)]
            self.measured_att = [self.reader.get_gps_att(0)]
        times = self.reader.get_timestamps(0, np.inf)     
        for i in range(1, len(times)):
            translation, rotation = self.reader.get_radardata(times[i]).image_transformation_from(self.reader.get_radardata(times[i-1]))
            self.measured_pos.append(self.measured_pos[-1] + self.measured_att[-1].apply(translation, True))
            self.measured_att.append(rotation.inv()*self.measured_att[-1])
        self.measured_pos = np.array(self.measured_pos)
        self.measured_att = np.ravel(self.measured_att)
        return self.measured_pos, self.measured_att

    def get_corrected_measurements(self):
        """ Return positions and attitude obtained with cv2 transformations with 3D correction based on GPS"""
        if hasattr(self.reader,"groundtruth"):
            self.measured_pos_corr = [self.reader.get_groundtruth_pos(0)]
            self.measured_att_corr = [self.reader.get_groundtruth_att(0)]
        else:
            self.measured_pos_corr = [self.reader.get_gps_pos(0)]
            self.measured_att_corr = [self.reader.get_gps_att(0)]
        times = self.reader.get_timestamps(0, np.inf)     
        for i in range(1, len(times)):
            translation, rotation = self.reader.get_radardata(times[i]).image_transformation_from(self.reader.get_radardata(times[i-1]))
            translation[2] = self.measured_att_corr[-1].apply(self.reader.get_radardata(times[i]).gps_pos - self.reader.get_radardata(times[i-1]).gps_pos)[2]
            self.measured_pos_corr.append(self.measured_pos_corr[-1] + self.measured_att_corr[-1].apply(translation + self.reader.get_bias()[0], True))
            
            ort = self.reader.get_radardata(times[i]).attitude*self.reader.get_radardata(times[i-1]).attitude.inv()*rotation_proj(self.reader.get_radardata(times[i-1]).attitude, self.reader.get_radardata(times[i]).attitude)
            self.measured_att_corr.append(ort*(self.reader.get_bias()[1]*rotation).inv()*self.measured_att_corr[-1])
        self.measured_pos_corr = np.array(self.measured_pos_corr)
        self.measured_att_corr = np.ravel(self.measured_att_corr)
        return self.measured_pos_corr, self.measured_att_corr

    def get_measured_attitudes(self, corrected=False):
        """ Return attitudes obtained with cv2 transformations """
        if corrected:
            if self.measured_att_corr is None:
                return self.get_corrected_measurements()[1]
            else:
                return self.measured_att_corr
        else:           
            if self.measured_att is None:
                return self.get_measurements()[1]
            else:
                return self.measured_att
    
    def get_measured_positions(self, corrected=False):
        """ Return positions obtained with cv2 transformations """
        if corrected:
            if self.measured_pos_corr is None:
                return self.get_corrected_measurements()[0]
            else:
                return self.measured_pos_corr
        else:           
            if self.measured_pos is None:
                return self.get_measurements()[0]
            else:
                return self.measured_pos
    
    