import gmplot
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj, ecef2enu, ecef2lla, rbd_translate, stat_filter

class Plot_Handler:
    
    def define_reader(self):
        if type(self).__name__ =="Reader":
            return self
        else:
            return self.reader
    
    def add_trajectory_line(self, map_origin, map_orientation, gps_pos, attitudes, label, color, arrow):
        """ Add a trajectory in an oriented 2D map """
        reader = self.define_reader()
        pos = rbd_translate(gps_pos, attitudes, reader.tracklog_translation)
        trajectory = map_orientation.apply(pos - map_origin)       
        plt.plot(trajectory[:,0], trajectory[:,1], color, label=label, picker=True)
        if arrow:
            arrows = np.array([map_orientation.apply(data.earth2rbd([0,-1,0], True)) for data in reader.get_radardata()])
            for i in range(0, len(arrows), 5):
                plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
                
    def export_map(self, gps_only = False, cv2_corrected=False):
        """ Plot reference GPS on a Google map as well as measured position and filtered position
            corrected: if True apply bias correction to CV2 measurements
        """
        print("Exporting Google map...")
        reader = self.define_reader()
        if hasattr(reader,"groundtruth"):
            coords = ecef2lla(rbd_translate(reader.get_groundtruth_pos(), reader.get_groundtruth_att(), reader.groundtruth_translation))
            gmap=gmplot.GoogleMapPlotter(np.rad2deg(np.mean(coords[:,1])), np.rad2deg(np.mean(coords[:,0])), 15)
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'black', edge_width = 2.5)
        
        coords = ecef2lla(rbd_translate(reader.get_gps_pos(), reader.get_gps_att(), reader.tracklog_translation))
        if not hasattr(reader,"groundtruth"):
            gmap=gmplot.GoogleMapPlotter(np.mean(coords[:,1]), np.mean(coords[:,0]), 15)
        gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'green', edge_width = 2.5)
        
        if not gps_only:           
            coords = ecef2lla(rbd_translate(self.get_measured_positions(), self.get_measured_attitudes(), reader.tracklog_translation))
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'red', edge_width = 2.5)

        if not gps_only and cv2_corrected:
            coords = ecef2lla(rbd_translate(self.get_measured_positions(cv2_corrected), self.get_measured_attitudes(cv2_corrected), reader.tracklog_translation))
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'red', edge_width = 2.5)
            
        if not gps_only and len(self.get_positions())!=0:
            coords = ecef2lla(rbd_translate(self.get_positions(), self.get_attitudes(), self.reader.tracklog_translation))
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'cornflowerblue', edge_width = 2.5)

        gmap.apikey = "AIzaSyB0UlIEiFl6IFtzz2_1WaDyYsXjscLVRWU"
        gmap.draw("map.html")  
        
    def plot_trajectory(self, arrow=False, gps_only = False, cv2_corrected = False):
        """ Plot the trajectory in earth frame centered on initial position 
            arrow: if True plot arrows in order to visualize attitude
            gps_only: if True plot only GPS data from dataset
            cv2_corrected: if True add a trajectory with bias correction on CV2 measurements
        """
        reader = self.define_reader()
        def show_timestamp(event):
            print(str(round(reader.get_timestamps()[event.ind[0]],2))+"s")
       
        fig = plt.figure()
        if hasattr(reader,"groundtruth"):
            pos = rbd_translate(reader.get_groundtruth_pos(), reader.get_groundtruth_att(), reader.groundtruth_translation)
            pos0 = pos[0]
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
            trajectory = att0.apply(pos - pos[0])   
            plt.plot(trajectory[:,0], trajectory[:,1], 'black', label="Groundtruth", picker=True)
            if arrow:
                arrows = np.array([att0.apply(att.apply([0,-1,0], True)) for att in reader.get_groundtruth_att()])
                for i in range(0, len(arrows), 5):
                    plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])
        else:
            pos0 = rbd_translate(reader.get_gps_pos(0), reader.get_gps_att(0), reader.tracklog_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
          
        self.add_trajectory_line(pos0, att0, reader.get_gps_pos(), reader.get_gps_att(), "GPS", 'g', arrow)
        
        if not gps_only:
            self.add_trajectory_line(pos0, att0, self.get_measured_positions(), self.get_measured_attitudes(), "CV2", 'r', arrow)  
        if cv2_corrected and not gps_only:
            self.add_trajectory_line(pos0, att0, self.get_measured_positions(cv2_corrected), self.get_measured_attitudes(cv2_corrected), "CV2 corrected", 'r--', arrow)

        if not gps_only and len(self.get_positions())!=0:
            self.add_trajectory_line(pos0, att0, self.get_positions(), self.get_attitudes(), "Output", 'cornflowerblue', arrow)

        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.axis('equal')
        plt.legend()
        plt.title("Trajectory in ENU frame centered on initial position")
        fig.canvas.mpl_connect('pick_event', show_timestamp) 
        
    def plot_altitude(self, gps_only = False, cv2_corrected = False):
        """ Plot the altitude in earth frame centered on initial position 
            gps_only: if True plot only GPS data from dataset
            cv2_corrected: if True add a line with bias correction on CV2 measurements
        """
        reader = self.define_reader()
        plt.figure()
        if hasattr(reader,"groundtruth"):
            pos = rbd_translate(reader.get_groundtruth_pos(), reader.get_groundtruth_att(), reader.groundtruth_translation)
            coord0 = ecef2lla(pos[0])
            att0 = ecef2enu(coord0[1], coord0[0])
            trajectory = att0.apply(pos - pos[0])   
            plt.plot(reader.get_timestamps(), trajectory[:,2], 'black', label="Groundtruth", picker=True)
        else:
            pos0 = rbd_translate(reader.get_gps_pos(0), reader.get_gps_att(0), reader.tracklog_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
          
        pos = rbd_translate(reader.get_gps_pos(), reader.get_gps_att(), reader.tracklog_translation)
        trajectory = att0.apply(pos - pos[0])       
        plt.plot(reader.get_timestamps(), trajectory[:,2], 'g', label="GPS", picker=True)
     
        if not gps_only:            
            pos = rbd_translate(self.get_measured_positions(), self.get_measured_attitudes(), reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(reader.get_timestamps(), trajectory[:,2], 'r', label="CV2", picker=True)
             
        if not gps_only and cv2_corrected:
            pos = rbd_translate(self.get_measured_positions(cv2_corrected), self.get_measured_attitudes(cv2_corrected), reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(reader.get_timestamps(), trajectory[:,2], 'r--', label="CV2 corrected", picker=True)
      
        if not gps_only and len(self.get_positions())!=0:
            pos = rbd_translate(self.get_positions(), self.get_attitudes(), reader.tracklog_translation)
            trajectory = att0.apply(pos - pos[0])        
            plt.plot(reader.get_timestamps(), trajectory[:,2], 'cornflowerblue', label="Output", picker=True)

        plt.xlabel('Times (s)')
        plt.ylabel('z (meters)')
        plt.axis('equal')
        plt.legend()
        plt.title("Altitude in ENU frame centered on initial position")

    def plot_attitude(self, gps_only = False, cv2_corrected = False):
        """ Plot the orientation in the map frame given by the GPS and after fusion 
            gps_only: if True plot only GPS data from dataset
            cv2_corrected: if True add a line with bias correction on CV2 measurements
        """
        reader = self.define_reader()
        plt.figure()
        plt.title("Yaw")
        plt.xlabel("Times (s)")
        plt.ylabel("Yaw (rad)")
        q = rot.from_dcm([[0,-1,0],[-1,0,0],[0,0,-1]])
        if hasattr(reader,"groundtruth"):
            pos0 = rbd_translate(reader.get_groundtruth_pos(0), reader.get_groundtruth_att(0), reader.groundtruth_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
            plt.plot(reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in reader.get_groundtruth_att()], 'black', label="Groundtruth")
        else:
            pos0 = rbd_translate(reader.get_gps_pos(0), reader.get_gps_att(0), reader.tracklog_translation)
            coord0 = ecef2lla(pos0)
            att0 = ecef2enu(coord0[1], coord0[0])
        
        plt.plot(reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in reader.get_gps_att()], 'green', label="GPS")
        if not gps_only:            
            plt.plot(reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.get_measured_attitudes()], 'red', label="CV2")
        if cv2_corrected and not gps_only:
            plt.plot(reader.get_timestamps(), [rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.get_measured_attitudes(cv2_corrected)], 'r--', label="CV2 corrected")
        if not gps_only:
            plt.plot(list(self.kalman_record.keys()), np.array([rotation_proj(att0, q*kalman['ATTITUDE']).as_euler('zxy')[0] for kalman in self.kalman_record.values()]), 'cornflowerblue', label="Output")
        plt.legend()  
    

class Recorder(Plot_Handler):
    
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
        self.kalman_record[ts] = {'ATTITUDE': self.kalman.attitude, 'POSITION': self.kalman.position, 'INNOVATION': self.kalman.innovation}
            
    def plot_innovation(self, individual=False, p=0.99):
        """ Return innovation made by cv2 measurement during fusion """
        if list(self.kalman_record.values())[1]['INNOVATION'] is None:
            raise Exception("Plot of innovation canceled due to lack of Kalman update")
        else:
            plt.figure()
            plt.title("Innovation in function of time")
            plt.xlabel("Time (s)")
            if individual:
                innovation = [kalman['INNOVATION'] for kalman in list(self.kalman_record.values())[1:]]
                plt.plot(list(self.kalman_record.keys())[1:], [np.array([Z[0]**2/S[0,0],Z[1]**2/S[1,1],Z[2]**2/S[2,2]])/stat.chi2.ppf(p, df=1) for Z,S in innovation])
            else:
                plt.plot(list(self.kalman_record.keys())[1:], [kalman['INNOVATION'][0].dot(np.linalg.inv(kalman['INNOVATION'][1])).dot(kalman['INNOVATION'][0])/stat.chi2.ppf(p, df=len(kalman['INNOVATION'][0])) for kalman in list(self.kalman_record.values())[1:]])
        
    def plot_kalman_evaluation(self, use_groundtruth = True, grouped=True):
        """ Return a plot of the error of the Kalman in filter in the first image frame 
            use_groundtruth: if False compare Kalman filter performance with radar images GPS
            grouped: if True return norm of error instead of error of each component
        """
        error_pos, error_att = self.get_kalman_error(use_groundtruth = True)
        
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.title("Error in position of the Kalman filter in first image frame")
        if grouped:
            plt.plot(self.reader.get_timestamps(), np.linalg.norm(error_pos))
            print("Average position error (m): " + str(np.round(np.mean(np.linalg.norm(stat_filter(error_pos, 0.9), axis=1), axis=0), 5)) + " (" +str(np.round(np.std(np.linalg.norm(error_pos, axis=1), axis=0), 5))+ ")")
        else:
            plt.plot(self.reader.get_timestamps(), error_pos)
            plt.legend(["Right", "Backward", "Down"])
            print("Average position error (m): " + str(np.round(np.mean(stat_filter(error_pos, 0.9), axis=0), 5)) + " (" +str(np.round(np.std(stat_filter(error_pos, 0.9), axis=0), 5))+ ")")
       
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Error (rad)")
        plt.title("Error in attitude of the Kalman filter in first image frame")
        if grouped:
            plt.plot(self.reader.get_timestamps(), abs(error_att))
            print("Average rotation error (rad): " + str(np.round(np.rad2deg(np.mean(np.abs(stat_filter(error_att, 0.9)))), 5)) + " (" +str(np.round(np.rad2deg(np.std(np.abs(error_att))), 5))+ ")")
        else:
            plt.plot(self.reader.get_timestamps(), error_att)
            print("Average rotation error (rad): " + str(np.round(np.rad2deg(np.mean(stat_filter(error_att, 0.9))), 5)) + " (" +str(np.round(np.rad2deg(np.std(error_att)), 5))+ ")")
        return error_pos, error_att
     
    def get_kalman_error(self, use_groundtruth = True):
        """ Return error of the Kalman in filter in the first image frame 
            use_groundtruth: if False compare Kalman filter performance with radar images GPS
        """
        if hasattr(self.reader,"groundtruth") and use_groundtruth:
            error_pos = np.array([self.reader.get_radardata(0).earth2rbd(self.kalman_record[ts]['POSITION']-self.reader.get_groundtruth_pos(ts)) for ts in list(self.kalman_record.keys())])
            error_att = np.array([rotation_proj(self.reader.get_groundtruth_att(ts), self.kalman_record[ts]['ATTITUDE']).as_euler('zxy')[0] for ts in list(self.kalman_record.keys())])
        else:    
            error_pos = np.array([self.reader.get_radardata(0).earth2rbd(self.kalman_record[ts]['POSITION']-self.reader.get_gps_pos(ts)) for ts in list(self.kalman_record.keys())])  
            error_att = np.array([rotation_proj(self.reader.get_gps_att(ts), self.kalman_record[ts]['ATTITUDE']).as_euler('zxy')[0] for ts in list(self.kalman_record.keys())])
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