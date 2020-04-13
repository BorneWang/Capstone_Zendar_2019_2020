import os
import gmplot
import pickle
import datetime
import numpy as np
from copy import deepcopy
import scipy.stats as stat
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj, rotation_ort, ecef2enu, ecef2lla, rbd_translate, stat_filter, increase_saturation, projection

def define_reader(obj):
    if type(obj).__name__ =="Reader":
        return obj
    else:
        return obj.reader

def get_plot_origin(obj):
    reader = define_reader(obj)
    if hasattr(reader,"groundtruth"):
        pos0 = rbd_translate(reader.get_groundtruth_pos(0), reader.get_groundtruth_att(0), reader.tracklog_translation)
        coord0 = ecef2lla(pos0)
        att0 = ecef2enu(coord0[1], coord0[0])
    else:
        pos0 = rbd_translate(reader.get_gps_pos(0), reader.get_gps_att(0), reader.tracklog_translation)
        coord0 = ecef2lla(pos0)
        att0 = ecef2enu(coord0[1], coord0[0])
    return pos0, att0

def add_trajectory_line(obj, gps_pos, attitudes, label, color, arrow):
    """ Add a trajectory in an oriented 2D map """
    reader = define_reader(obj)
    map_origin, map_orientation = get_plot_origin(obj)
    pos = rbd_translate(gps_pos, attitudes, reader.tracklog_translation)
    trajectory = map_orientation.apply(pos - map_origin)       
    plt.plot(trajectory[:,0], trajectory[:,1], color, label=label, picker=True)
    if arrow:
        arrows = np.array([map_orientation.apply(data.earth2rbd([0,-1,0], True)) for data in reader.get_radardata()])
        for i in range(0, len(arrows), 5):
            plt.arrow(trajectory[i,0], trajectory[i,1],arrows[i,0],arrows[i,1])

def add_altitude_line(obj, gps_pos, attitudes, label, color):
    """ Add a line in figure of altitude from a 2D plane"""
    reader = define_reader(obj)
    map_origin, map_orientation = get_plot_origin(obj)
    pos = rbd_translate(gps_pos, attitudes, reader.tracklog_translation)
    trajectory = map_orientation.apply(pos - map_origin)       
    plt.plot(obj.get_timestamps(), trajectory[:,2], color, label=label)

class Plot_Handler:
               
    def export_map(self, gps_only = False, cv2_corrected=False):
        """ Plot reference GPS on a Google map as well as measured position and filtered position
            corrected: if True apply bias correction to CV2 measurements
        """
        print("Exporting Google map...")
        reader = define_reader(self)
        if hasattr(reader,"groundtruth"):
            coords = ecef2lla(rbd_translate(reader.get_groundtruth_pos(), reader.get_groundtruth_att(), reader.tracklog_translation))
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
            coords = ecef2lla(rbd_translate(self.get_measured_positions(corrected=cv2_corrected), self.get_measured_attitudes(corrected=cv2_corrected), reader.tracklog_translation))
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'red', edge_width = 2.5)
            
        if hasattr(self,"get_position") and len(self.get_position())!=0:
            coords = ecef2lla(rbd_translate(self.get_position(), self.get_attitude(), self.reader.tracklog_translation))
            gmap.plot(np.rad2deg(coords[:,1]), np.rad2deg(coords[:,0]), 'cornflowerblue', edge_width = 2.5)

        gmap.apikey = "AIzaSyB0UlIEiFl6IFtzz2_1WaDyYsXjscLVRWU"
        gmap.draw("map.html")  
         
    def plot_trajectory(self, arrow=False, gps_only = False, cv2_corrected = False):
        """ Plot the trajectory in ENU frame centered on initial position 
            arrow: if True plot arrows in order to visualize attitude
            gps_only: if True plot only GPS data from dataset
            cv2_corrected: if True add a trajectory with bias correction on CV2 measurements
        """
        fig = plt.figure()
        reader = define_reader(self)
          
        if hasattr(reader,"groundtruth"):
            add_trajectory_line(self, reader.get_groundtruth_pos(), reader.get_groundtruth_att(), "Groundtruth", 'black', arrow)   
        add_trajectory_line(self, reader.get_gps_pos(), reader.get_gps_att(), "GPS", 'g', arrow)
        
        if not gps_only:
            add_trajectory_line(self, self.get_measured_positions(), self.get_measured_attitudes(), "CV2", 'r', arrow)  
        if cv2_corrected and not gps_only:
            add_trajectory_line(self, self.get_measured_positions(corrected=cv2_corrected), self.get_measured_attitudes(corrected=cv2_corrected), "CV2 corrected", 'r--', arrow)

        if hasattr(self,"get_position") and len(self.get_position())!=0:
            add_trajectory_line(self, self.get_position(), self.get_attitude(), "Output", 'cornflowerblue', arrow)

        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.axis('equal')
        plt.legend()
        plt.title("Trajectory in ENU frame centered on initial position")
        
        def show_timestamp(event):
            print(str(round(reader.get_timestamps()[event.ind[0]],2))+"s")
        fig.canvas.mpl_connect('pick_event', show_timestamp) 
        
    def plot_altitude(self, gps_only = False, cv2_corrected = False):
        """ Plot the altitude in ENU frame centered on initial position 
            gps_only: if True plot only GPS data from dataset
            cv2_corrected: if True add a line with bias correction on CV2 measurements
        """
        plt.figure()
        reader = define_reader(self)
        
        if hasattr(reader,"groundtruth"):
            add_altitude_line(self, reader.get_groundtruth_pos(), reader.get_groundtruth_att(), "Groundtruth", 'black')   
        add_altitude_line(self, reader.get_gps_pos(), reader.get_gps_att(), "GPS", 'g')
     
        if not gps_only:
            add_altitude_line(self, self.get_measured_positions(), self.get_measured_attitudes(), "CV2", 'r')  
        if cv2_corrected and not gps_only:
            add_altitude_line(self, self.get_measured_positions(corrected=cv2_corrected), self.get_measured_attitudes(corrected=cv2_corrected), "CV2 corrected", 'r--')
   
        if hasattr(self,"get_position") and len(self.get_position())!=0:
            add_altitude_line(self, self.get_position(), self.get_attitude(), "Output", 'cornflowerblue')

        plt.xlabel('Times (s)')
        plt.ylabel('z (meters)')
        plt.legend()
        plt.title("Altitude in ENU frame centered on initial position")

    def plot_attitude(self, gps_only = False, cv2_corrected = False):
        """ Plot the orientation in the map frame given by the GPS and after fusion 
            gps_only: if True plot only GPS data from dataset
            cv2_corrected: if True add a line with bias correction on CV2 measurements
        """
        plt.figure()
        reader = define_reader(self)
        pos0, att0 = get_plot_origin(self)
        q = rot.from_dcm([[0,-1,0],[-1,0,0],[0,0,-1]])

        if hasattr(reader,"groundtruth"):
            plt.plot(reader.get_timestamps(), np.unwrap([rotation_proj(att0, q*r).as_euler('zxy')[0] for r in reader.get_groundtruth_att()]), 'black', label="Groundtruth")
        plt.plot(reader.get_timestamps(), np.unwrap([rotation_proj(att0, q*r).as_euler('zxy')[0] for r in reader.get_gps_att()]), 'green', label="GPS")
        if not gps_only:            
            plt.plot(reader.get_timestamps(), np.unwrap([rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.get_measured_attitudes()]), 'red', label="CV2")
        if cv2_corrected and not gps_only:
            plt.plot(reader.get_timestamps(), np.unwrap([rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.get_measured_attitudes(cv2_corrected)]), 'r--', label="CV2 corrected")
        if hasattr(self,"get_attitude") and len(self.get_attitude())!=0:
            plt.plot(self.get_timestamps(0, np.inf), np.unwrap([rotation_proj(att0, q*r).as_euler('zxy')[0] for r in self.get_attitude()]), 'cornflowerblue', label="Output")
        
        plt.legend()
        plt.title("Yaw")
        plt.xlabel("Times (s)")
        plt.ylabel("Yaw (rad)")
    

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
        """ Record value in a dictionary for later use """
        self.kalman_record[ts] = {'ATTITUDE': self.kalman.attitude, 'POSITION': self.kalman.position, 'INNOVATION': self.kalman.innovation}
        
    def save(self):
        """ Save values recorded by the recorder """
        name = "recorder_"+str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")+".pickle"
        print("Saving " + name)
        record = open(name,"wb")
        pickle.dump({"record": self.kalman_record, "src": self.reader.src, "kalman": self.kalman}, record)
        record.close()
        
    def import_record(self, name):
        """ Import recorded values from pickle """ 
        record = open(name+".pickle","rb")
        info = pickle.load(record)
        self.kalman_record = info["record"]
        self.reader.src = info["src"]
        self.kalman = info["kalman"]
        record.close()
        
        self.reader.heatmaps = dict()
        self.reader.load_heatmaps(self.get_timestamps(0), self.get_timestamps(np.inf))
        
        self.measured_pos, self.measured_att, self.measured_pos_corr, self.measured_att_corr = None, None, None, None
            
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
                plt.plot(self.get_timestamps(0, np.inf)[1:], [np.array([Z[0]**2/S[0,0],Z[1]**2/S[1,1],Z[2]**2/S[2,2]])/stat.chi2.ppf(p, df=1) for Z,S in innovation])
            else:
                plt.plot(self.get_timestamps(0, np.inf)[1:], [kalman['INNOVATION'][0].dot(np.linalg.inv(kalman['INNOVATION'][1])).dot(kalman['INNOVATION'][0])/stat.chi2.ppf(p, df=len(kalman['INNOVATION'][0])) for kalman in list(self.kalman_record.values())[1:]])
        
    def plot_kalman_evaluation(self, use_groundtruth = True, grouped=True):
        """ Return a plot of the error of the Kalman in filter in the map frame 
            use_groundtruth: if False compare Kalman filter performance with radar images GPS
            grouped: if True return norm of error instead of error of each component
        """
        error_pos, error_att = self.get_kalman_error(use_groundtruth = True)
        
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.title("Error in position of the Kalman filter in first image frame")
        if grouped:
            plt.plot(self.get_timestamps(0, np.inf), np.linalg.norm(error_pos[:,0:2],  axis=1))
            print("Average position error (m): " + str(np.round(np.mean(np.linalg.norm(stat_filter(error_pos[:,0:2], 0.9), axis=1), axis=0), 5)) + " (" +str(np.round(np.std(np.linalg.norm(error_pos[:,0:2], axis=1), axis=0), 5))+ ")")
        else:
            plt.plot(self.get_timestamps(0, np.inf), error_pos)
            plt.legend(["Right", "Backward", "Down"])
            print("Average position error (m): " + str(np.round(np.mean(stat_filter(error_pos, 0.9), axis=0), 5)) + " (" +str(np.round(np.std(stat_filter(error_pos, 0.9), axis=0), 5))+ ")")
       
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Error (deg)")
        plt.title("Error in attitude of the Kalman filter in first image frame")
        if grouped:
            plt.plot(self.get_timestamps(0, np.inf), abs(np.rad2deg(error_att)))
            print("Average rotation error (deg): " + str(np.round(np.rad2deg(np.mean(np.abs(stat_filter(error_att, 0.9)))), 5)) + " (" +str(np.round(np.rad2deg(np.std(np.abs(error_att))), 5))+ ")")
        else:
            plt.plot(self.get_timestamps(0, np.inf), np.rad2deg(error_att))
            print("Average rotation error (deg): " + str(np.round(np.rad2deg(np.mean(stat_filter(error_att, 0.9))), 5)) + " (" +str(np.round(np.rad2deg(np.std(error_att)), 5))+ ")")
    
    def play_video(self, t_ini=0, t_final=np.inf, save=False):
        """ Play a video of the car driving between t_ini and t_final
            save: if True, save the video as a .mp4
        """
        shape = (1000,2000)
        overlay_alpha = 0.5
        border = 2
        
        # Handling pause/resume event when clicking on the video
        anim_running = True
        def onClick(event):
            nonlocal anim_running
            if anim_running:
                ani.event_source.stop()
                anim_running = False
            else:
                ani.event_source.start()
                anim_running = True
                
        images = []
        fig = plt.figure()
        ax = plt.axes()
        ax.set_facecolor("black")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        def update(t):
            center = -self.kalman.mapdata.precision*np.array([0.5*shape[1], 0.5*shape[0],0])
            gps_pos = projection(self.kalman.mapdata.gps_pos, self.kalman.mapdata.attitude, rbd_translate(self.get_position(t), self.get_attitude(t), self.reader.tracklog_translation))
            img, _= self.kalman.mapdata.extract_from_map(gps_pos + self.kalman.mapdata.attitude.apply(center,True), self.kalman.mapdata.attitude, shape)

            data = deepcopy(self.reader.get_radardata(t))
            data.gps_pos, data.gps_att = self.get_position(t), self.get_attitude(t)
            img_border = 255*np.ones(np.shape(data.img))
            img_border[border:-border,border:-border] = data.img[border:-border,border:-border]
            data.img = img_border
            img_overlay = np.nan_to_num(data.predict_image(gps_pos + self.kalman.mapdata.attitude.apply(center,True), self.kalman.mapdata.attitude, shape))
            overlay_red = np.zeros((np.shape(img_overlay)[0], np.shape(img_overlay)[1], 4))
            overlay_red[:,:,0] = img_overlay
            overlay_red[:,:,3] = (img_overlay != 0)*overlay_alpha*255
            return [plt.imshow(increase_saturation(np.nan_to_num(img)), cmap='gray', vmin=0, vmax=255, zorder=1), 
                    plt.imshow(increase_saturation(overlay_red.astype(np.uint8)), alpha = 0.5, zorder=2, interpolation=None), 
                    plt.text(0.6,0.5,str(round(t,2)))]

        print("Creating video...")
        for t in self.get_timestamps(t_ini, t_final):
            images.append(update(t))

        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = ArtistAnimation(fig, images, interval=100, blit=False, repeat_delay=1000)
        plt.show()
        if save:
            name = str(self.kalman.mapdata.name) + str(datetime.datetime.now())[0:16].replace(" ","_").replace(":","").replace("-","")          
            print("Saving video: "+str(name) + '.mp4')
            os.makedirs(os.path.dirname('Videos/' + name + '.mp4'), exist_ok=True)
            ani.save('Videos/' + name + '.mp4')
        return ani
    
    def get_kalman_error(self, use_groundtruth = True):
        """ Return error of the Kalman in filter in the map frame 
            use_groundtruth: if False compare Kalman filter performance with radar images GPS
        """
        if hasattr(self.reader,"groundtruth") and use_groundtruth:
            error_pos = np.array([self.kalman.mapdata.attitude.apply(self.get_position(ts)-self.reader.get_groundtruth_pos(ts)) for ts in self.get_timestamps(0, np.inf)])
            error_att = np.array([rotation_proj(self.reader.get_groundtruth_att(ts), self.get_attitude(ts)).as_euler('zxy')[0] for ts in self.get_timestamps(0, np.inf)])
        else:    
            error_pos = np.array([self.kalman.mapdata.attitude.apply(self.get_position(ts)-self.reader.get_gps_pos(ts)) for ts in self.get_timestamps(0, np.inf)])  
            error_att = np.array([rotation_proj(self.reader.get_gps_att(ts), self.get_attitude(ts)).as_euler('zxy')[0] for ts in self.get_timestamps(0, np.inf)])
        return error_pos, error_att  

    def get_timestamps(self, t_ini=None, t_final=None):
        """ Return a list of data timestamps between t_ini and t_final """
        times = list(self.kalman_record.keys())
        if (t_ini is None) or (t_ini == 0 and t_final==np.inf):
            return times
        else:
            times.sort()
            if t_final is None:
                if t_ini==np.inf:
                    return times[-1]
                t_adj = times[min(range(len(times)), key = lambda i: abs(times[i]-t_ini))]
                return t_adj
            else:   
                if t_ini>t_final:
                    raise ValueError("Initial timestamp should be smaller than final timestamp")
                return [t for t in times if t>=t_ini and t<=t_final]

    def get_position(self, t_ini=None, t_final=None):
        """ Return positions after fusion """
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return self.kalman_record[times]['POSITION'] 
        else:
            return np.array([self.kalman_record[t]['POSITION'] for t in times])

    def get_attitude(self, t_ini=None, t_final=None):
        """ Return attitude after fusion """
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return self.kalman_record[times]['ATTITUDE'] 
        else:
            return [self.kalman_record[t]['ATTITUDE'] for t in times]

    def get_measurements(self, corrected=False, use_groundtruth = True):
        """ Return positions and attitude obtained with cv2 transformations """
        if hasattr(self.reader,"groundtruth") and use_groundtruth:
            measured_pos = [self.reader.get_groundtruth_pos(0)]
            measured_att = [self.reader.get_groundtruth_att(0)]
        else:
            measured_pos = [self.reader.get_gps_pos(0)]
            measured_att = [self.reader.get_gps_att(0)]
        times = self.reader.get_timestamps(0, np.inf)     
        for i in range(1, len(times)):
            translation, rotation = self.reader.get_radardata(times[i]).image_transformation_from(self.reader.get_radardata(times[i-1]))
            if not corrected:
                measured_pos.append(measured_pos[-1] + measured_att[-1].apply(translation, True))
                measured_att.append(rotation.inv()*measured_att[-1])
            else:
                if hasattr(self.reader,"groundtruth") and use_groundtruth:
                    translation[2] = measured_att[-1].apply(self.reader.get_groundtruth_pos(times[i]) - self.reader.get_groundtruth_pos(times[i-1]))[2]
                    ort = rotation_ort(self.reader.get_groundtruth_att(times[i-1]), self.reader.get_groundtruth_att(times[i]))
                else:
                    translation[2] = measured_att[-1].apply(self.reader.get_gps_pos(times[i]) - self.reader.get_gps_pos(times[i-1]))[2]
                    ort = rotation_ort(self.reader.get_gps_att(times[i-1]), self.reader.get_gps_att(times[i]))
                    
                measured_pos.append(measured_pos[-1] + measured_att[-1].apply(translation + self.reader.get_bias()[0], True))
                measured_att.append(ort*(self.reader.get_bias()[1]*rotation).inv()*measured_att[-1])
          
        if corrected:
            self.measured_pos_corr = np.array(measured_pos)
            self.measured_att_corr = np.ravel(measured_att)
        else:
            self.measured_pos = np.array(measured_pos)
            self.measured_att = np.ravel(measured_att)
        return np.array(measured_pos), np.ravel(measured_att)

    def get_measured_attitudes(self, corrected=False, use_groundtruth=True):
        """ Return attitudes obtained with cv2 transformations """
        if (not corrected and self.measured_att is None) or (corrected and self.measured_att_corr is None):
            return self.get_measurements(corrected, use_groundtruth)[1]
        elif corrected:
            return self.measured_att_corr
        else:           
            return self.measured_att
    
    def get_measured_positions(self, corrected=False, use_groundtruth=True):
        """ Return positions obtained with cv2 transformations """
        if (not corrected and self.measured_pos is None) or (corrected and self.measured_pos_corr is None):
            return self.get_measurements(corrected, use_groundtruth)[0]
        elif corrected:
            return self.measured_pos_corr
        else:           
            return self.measured_pos