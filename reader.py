import os
import h5py
import shelve
import numpy as np
from PIL import Image
from copy import deepcopy
from data import RadarData
import matplotlib.pyplot as plt
from recorder import Plot_Handler
from matplotlib.animation import ArtistAnimation
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj, rbd_translate, stat_filter

class Reader(Plot_Handler):
    
    def __init__(self, src, t_ini=0, t_final=np.inf):
        if t_ini>t_final:
            raise ValueError("Initial timestamp should be smaller than final timestamp")
            
        self.src = src
        self.heatmaps = dict()
        self.tracklog_translation = np.zeros(3)
        self.bias = None
        
        self.load_heatmaps(t_ini, t_final)
        
        cv2_transformations = shelve.open("cv2_transformations")
        cv2_transformations['use_dataset'] = self.src
        cv2_transformations.close()
    
    def __iter__(self):
        self.iter = 0
        self.prev_perc = -1
        return self
    
    def __next__(self):
        N = len(list(self.heatmaps.items()))
        if np.floor(self.iter/(N-1)*10) != self.prev_perc:
            print("Processing data: "+str(np.floor(self.iter/(N-1)*10)*10)+"%")
            self.prev_perc = np.floor(self.iter/(N-1)*10)      
        self.iter+=1 
        if self.iter<=N:
            return list(self.heatmaps.items())[self.iter-1]
        else:
            print("Data processed")
            raise StopIteration
    
    def __getitem__(self, key):
        reader = deepcopy(self)
        reader.heatmaps = {k:self.heatmaps[k] for k in list(self.heatmaps.keys())[key]}
        return reader
    
    def load_heatmaps(self, t_ini=0, t_final=np.inf):
        """ Function load radar data magnitude from HDF5 between t_ini and t_final"""
        hdf5 = h5py.File(self.src,'r+')
        
        try:
            aperture = hdf5['radar']['broad01']['aperture2D']
        except:
            raise Exception("The images should be in radar/broad01/aperture2D directory")
        if not(('preprocessed' in aperture.attrs) and aperture.attrs['preprocessed']):
            hdf5.close()
            raise Exception("The dataset should be preprocessed before with Preprocessor")
            
        try:            
            times = list(aperture.keys())
            
            # Importing radar images from dataset
            t0 = float(times[0])       
            times = [times[i] for i in range(len(times)) if float(times[i])-t0>=t_ini and float(times[i])-t0<=t_final]
            prev_perc = -1
            for i, t in enumerate(times):
                if np.floor(i/(len(times)-1)*10) != prev_perc:
                    print("Loading data: "+str(np.floor(i/(len(times)-1)*10)*10)+"%")
                    prev_perc = np.floor(i/(len(times)-1)*10)
                heatmap = aperture[t][...];       
                if not np.sum(heatmap) == 0:
                    gps_pos = np.array(list(aperture[t].attrs['POSITION'][0]))
                    att = np.array(list(aperture[t].attrs['ATTITUDE'][0]))
                    self.heatmaps[float(t)-t0] = RadarData(float(t), np.array(Image.fromarray(heatmap, 'L')), gps_pos, rot.from_quat(att))
            self.tracklog_translation = -aperture.attrs['tracklog_translation']
            
            # Importing groundtruth GPS information if available
            aperture_gt = hdf5['radar']['broad01']
            groundtruth = ("groundtruth" in aperture_gt)
            if groundtruth:
                print("Loading groundtruth")
                aperture_gt = hdf5['radar']['broad01']['groundtruth']
                self.groundtruth = dict()
                times_gt = list(aperture_gt.keys())
                times_gt = [times_gt[i] for i in range(len(times_gt)) if (t_ini <= float(times_gt[i])-t0 < t_final) or (i+1<len(times_gt) and t_ini <= float(times_gt[i+1])-t0 < t_final) or (i-1>=0 and t_ini <= float(times_gt[i-1])-t0 < t_final)]
                gt_att, gt_pos, gt_time = [], [], []
                for i, t in enumerate(times_gt):
                    gt_att.append(rot.from_quat(np.array(list(aperture_gt[t].attrs['ATTITUDE'][0]))))
                    gt_pos.append(np.array(list(aperture_gt[t].attrs['POSITION'][0])))             
                    gt_time.append(float(t)-t0)
                gt_pos = rbd_translate(gt_pos, gt_att, self.tracklog_translation - (-aperture_gt.attrs['tracklog_translation']))
                
                # Interpolating groundtruth positions to make them match with radar images positions
                if times_gt[0]> times[0] or times_gt[-1]<times[-1]:
                    for t in times:
                        if times_gt[0] > t or times_gt[-1] < t:
                            self.heatmaps.pop(float(t)-t0)
                times = list(self.heatmaps.keys())
                slerp = Slerp(gt_time, rot.from_quat([r.as_quat() for r in gt_att]))
                gt_att = [rot.from_quat(q) for q in slerp(times).as_quat()]
                gt_pos = np.array([np.interp(times, gt_time, np.array(gt_pos)[:,i]) for i in range(0,3)]).T
                for i in range(len(times)):
                    self.groundtruth[times[i]] = {'POSITION': gt_pos[i], 'ATTITUDE': gt_att[i]}
            else:
                print("No groundtruth data found")
            
            hdf5.close()
            print("Data loaded")
        except:  
            hdf5.close()
            raise Exception("A problem occured when importing data")           
        
    def play_video(self, t_ini=0, t_final=np.inf, grayscale = True, save=False):
        """ Play a video of radar images between t_ini and t_final
            grayscale: if False automatic coloration of images is used
            save: if True, save the video as a .mp4
        """
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
                    
        times = self.get_timestamps(t_ini, t_final)
        images = []       
        fig = plt.figure()
        print("Creating video...")
        for t in times:
            plt.axis('off')
            if grayscale:              
                images.append([plt.imshow(Image.fromarray(self.heatmaps[t].img), cmap='gray', vmin=0, vmax=255), plt.text(0.5,0.5,str(round(t,2)))])
            else:
                images.append([plt.imshow(Image.fromarray(self.heatmaps[t].img)), plt.text(0.6,0.5,str(round(t,2)))])
        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = ArtistAnimation(fig, images, interval=100, blit=False, repeat_delay=1000)
        if save:
            print("Saving video: "+str(self.src) + '.mp4')
            os.makedirs(os.path.dirname('Videos/' + str(self.src) + '.mp4'), exist_ok=True)
            ani.save('Videos/' + str(self.src) + '.mp4')
        return ani
    
    def plot_evaluation(self, corrected = False, grouped = True):
        """ Evaluate the transformation given in GPS data information compared to CV2 image analysis
            corrected: if True calculate biases and remove them from displayed error
            grouped: if True return norm of error instead of error of each component
        """ 
        times = self.get_timestamps(0, np.inf)
        if hasattr(self,"groundtruth"):
            pos_error_gps = np.zeros((len(times)-1,3))
            pos_error_cv2 = np.zeros((len(times)-1,3))
            att_error_gps = np.zeros(len(times)-1)
            att_error_cv2 = np.zeros(len(times)-1)
        else:         
            pos_error = np.zeros((len(times)-1,3))
            att_error = np.zeros(len(times)-1)
            
        for i in range(1,len(times)):           
            trans_cv, rotation_cv = self.heatmaps[times[i]].image_transformation_from(self.heatmaps[times[i-1]])
            theta_cv = rotation_cv.as_euler('zxy')[0]
            
            theta_gps = rotation_proj(self.get_gps_att(times[i-1]), self.get_gps_att(times[i])).as_euler('zxy')[0]
            trans_gps = self.heatmaps[times[i]].earth2rbd(self.get_gps_pos(times[i]) - self.get_gps_pos(times[i-1]))
            
            if hasattr(self,"groundtruth"):
                theta_gt = rotation_proj(self.get_groundtruth_att(times[i-1]), self.get_groundtruth_att(times[i])).as_euler('zxy')[0]
                trans_gt = self.get_groundtruth_att(times[i]).apply(self.get_groundtruth_pos(times[i]) - self.get_groundtruth_pos(times[i-1]))                
                pos_error_gps[i-1] = trans_gt - trans_gps
                att_error_gps[i-1] = theta_gt - theta_gps
                if corrected:   
                    pos_error_cv2[i-1] = trans_gt - (trans_cv + self.get_bias()[0])
                    att_error_cv2[i-1] = theta_gt - (theta_cv + self.get_bias()[1].as_euler('zxy')[0])
                else:
                    pos_error_cv2[i-1] = trans_gt - trans_cv
                    att_error_cv2[i-1] = theta_gt - theta_cv
            else:     
                if corrected:   
                    pos_error[i-1] = trans_gps - (trans_cv + self.get_bias()[0])
                    att_error[i-1] = theta_gps - (theta_cv + self.get_bias()[1].as_euler('zxy')[0])
                else:
                    pos_error[i-1] = trans_gt - trans_cv
                    att_error[i-1] = theta_gps - theta_cv
                    
        if grouped:  
            plt.figure()
            plt.xlabel("Time (s)")
            plt.ylabel("Error (m)")
            if hasattr(self,"groundtruth"):
                plt.title("Square root error of GPS and CV2 translations with groundtruth")
                plt.plot(times[1:], np.linalg.norm(pos_error_gps[:,0:2], axis=1), label="GPS")
                plt.plot(times[1:], np.linalg.norm(pos_error_cv2[:,0:2], axis=1), label="CV2")
                plt.legend()
            else:
                plt.title("Square root error between GPS and CV2 translations")
                plt.plot(times[1:], np.linalg.norm(pos_error[:,0:2], axis=1))
        else:
            axis = ["x-axis", "y-axis", "z-axis"]
            for i in range(len(axis)):  
                plt.figure()  
                plt.xlabel("Time (s)")
                plt.ylabel("Error (m)")
                if hasattr(self,"groundtruth"):
                    plt.title("Error of GPS and CV2 translations with groundtruth along " + axis[i])
                    plt.plot(times[1:], np.array(pos_error_gps)[:,i], label="GPS")
                    plt.plot(times[1:], np.array(pos_error_cv2)[:,i], label="CV2")                    
                    plt.legend()
                else:
                    plt.title("Error between GPS and CV2 translations along " + axis[i])
                    plt.plot(times[1:], np.array(pos_error)[:,i])
                        
        plt.figure()
        if grouped:            
            if hasattr(self,"groundtruth"):
                plt.title("Squared error of GPS and CV2 rotations with groundtruth")
                plt.plot(times[1:], abs(np.rad2deg(att_error_gps)), label="GPS")
                plt.plot(times[1:], abs(np.rad2deg(att_error_cv2)), label="CV2")
                plt.legend()
            else:
                plt.title("Squared error between GPS and CV2 rotations")
                plt.plot(times[1:], abs(np.rad2deg(att_error)))
        else:           
            if hasattr(self,"groundtruth"):
                plt.title("Error of GPS and CV2 rotations with groundtruth")
                plt.plot(times[1:], np.rad2deg(att_error_gps), label="GPS")
                plt.plot(times[1:], np.rad2deg(att_error_cv2), label="CV2")
                plt.legend()
            else:
                plt.title("Error between GPS and CV2 rotations") 
                plt.plot(times[1:], np.rad2deg(att_error))
        plt.xlabel("Time (s)")
        plt.ylabel("Error (deg)")
        
        if grouped:
            if hasattr(self,"groundtruth"):
                print("Average GPS translation error (m): " + str(np.round(np.mean(np.linalg.norm(stat_filter(pos_error_gps[:,0:2], 0.9), axis=1), axis=0), 5)) + " (" +str(np.round(np.std(np.linalg.norm(pos_error_gps[:,0:2], axis=1), axis=0), 5))+ ")")
                print("Average cv2 translation error (m): " + str(np.round(np.mean(np.linalg.norm(stat_filter(pos_error_cv2[:,0:2], 0.9), axis=1), axis=0), 5)) + " (" +str(np.round(np.std(np.linalg.norm(pos_error_cv2[:,0:2], axis=1), axis=0), 5))+ ")")
                print("Average GPS rotation error (deg): " + str(np.round(np.rad2deg(np.mean(np.abs(stat_filter(att_error_gps, 0.9)))),5)) + " (" +str(np.round(np.rad2deg(np.std(np.abs(att_error_gps))), 5))+ ")")
                print("Average cv2 rotation error (deg): " + str(np.round(np.rad2deg(np.mean(np.abs(stat_filter(att_error_cv2, 0.9)))), 5)) + " (" +str(np.round(np.rad2deg(np.std(np.abs(att_error_cv2))), 5))+ ")")
            else:
                print("Average cv2 translation error (m): " + str(np.round(np.mean(np.linalg.norm(stat_filter(pos_error[:,0:2], 0.9), axis=1), axis=0),5)) + " (" +str(np.round(np.std(np.linalg.norm(pos_error[:,0:2], axis=1), axis=0), 5))+ ")")
                print("Average cv2 rotation error (deg): " + str(np.round(np.rad2deg(np.mean(np.abs(stat_filter(att_error, 0.9)))), 5)) + " (" +str(np.round(np.rad2deg(np.std(np.abs(att_error))), 5))+ ")")

        else:
            if hasattr(self,"groundtruth"):
                print("Average GPS translation error (m): " + str(np.round(np.mean(stat_filter(pos_error_gps, 0.9), axis=0), 5)) + " (" +str(np.round(np.std(pos_error_gps, axis=0), 5))+ ")")
                print("Average cv2 translation error (m): " + str(np.round(np.mean(stat_filter(pos_error_cv2, 0.9), axis=0), 5)) + " (" +str(np.round(np.std(pos_error_cv2, axis=0), 5))+ ")")
                print("Average GPS rotation error (deg): " + str(np.round(np.rad2deg(np.mean(stat_filter(att_error_gps, 0.9))),5)) + " (" +str(np.round(np.rad2deg(np.std(att_error_gps)), 5))+ ")")
                print("Average cv2 rotation error (deg): " + str(np.round(np.rad2deg(np.mean(stat_filter(att_error_cv2, 0.9))), 5)) + " (" +str(np.round(np.rad2deg(np.std(att_error_cv2)), 5))+ ")")
            else:
                print("Average cv2 translation error (m): " + str(np.round(np.mean(stat_filter(pos_error, 0.9), axis=0),5)) + " (" +str(np.round(np.std(pos_error), 5), axis=0)+ ")")
                print("Average cv2 rotation error (deg): " + str(np.round(np.rad2deg(np.mean(stat_filter(att_error, 0.9))), 5)) + " (" +str(np.round(np.rad2deg(np.std(att_error)), 5))+ ")")
    
    def get_bias(self):
        """ Calculate the bias in CV2 measurement from comparaison with GPS measurements """
        if self.bias is None:                
            times = self.get_timestamps(0, np.inf)       
            t_gps = np.zeros((len(times)-1,2))
            t_cv2 = np.zeros((len(times)-1,2))
            r_gps = np.zeros(len(times)-1)
            r_cv2 = np.zeros(len(times)-1)           
            for i in range(1,len(times)):           
                trans_cv, rotation_cv = self.heatmaps[times[i]].image_transformation_from(self.heatmaps[times[i-1]])
                r_cv2[i-1] = rotation_cv.as_euler('zxy')[0]
                t_cv2[i-1] = trans_cv[0:2]
                
                if hasattr(self,"groundtruth"):
                    r_gps[i-1] = rotation_proj(self.get_groundtruth_att(times[i-1]), self.get_groundtruth_att(times[i])).as_euler('zxy')[0]
                    t_gps[i-1] = self.get_groundtruth_att(times[i]).apply(self.get_groundtruth_pos(times[i]) - self.get_groundtruth_pos(times[i-1]))[0:2]             
                else:                    
                    r_gps[i-1] = rotation_proj(self.get_gps_att(times[i-1]), self.get_gps_att(times[i])).as_euler('zxy')[0]
                    t_gps[i-1] = self.heatmaps[times[i]].earth2rbd(self.get_gps_pos(times[i]) - self.get_gps_pos(times[i-1]))[0:2]
            
            # bias = np.array(t_cv2 - np.mean(t_cv2, axis=0)).T.dot(np.array(t_gps - np.mean(t_gps, axis=0)))
            # U, _, V = np.linalg.svd(bias)
            # R = V.dot(np.diag([1, np.linalg.det(V.dot(U.T))])).dot(U.T)
            # self.bias = (np.mean(t_gps, axis=0) - R.dot(np.mean(t_cv2, axis=0)), rot.from_dcm(np.array([[R[0,0], R[0,1], 0], [R[1,0], R[1,1], 0], [0,0,1]])))
            
            # Brutal mean
            bias_trans = np.append(np.mean(stat_filter(t_gps - t_cv2, 0.9), axis=0), 0) 
            bias_rot = np.mean(stat_filter(r_gps-r_cv2, 0.9), axis=0)
            self.bias = (bias_trans, rot.from_dcm(np.array([[np.cos(bias_rot), -np.sin(bias_rot), 0], [np.sin(bias_rot), np.cos(bias_rot), 0], [0,0,1]])))
        return self.bias
           
    def get_timestamps(self, t_ini=None, t_final=None):
        """ Return a list of data timestamps between t_ini and t_final """
        times = list(self.heatmaps.keys())
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
    
    def get_radardata(self, t_ini=None, t_final=None):
        """ Return radar data for time between t_ini and t_final """
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return self.heatmaps[times]
        else:
            return np.array([self.heatmaps[t] for t in times])
            
    def get_img(self, t):
        """ Return radar data image at time t """
        times = self.get_timestamps(t)
        return Image.fromarray(self.heatmaps[times].img)

    def get_groundtruth_pos(self,t_ini=None, t_final=None):
        """ Return groundtruth position for time between t_ini and t_final """        
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return self.groundtruth[times]['POSITION']
        else:
            return np.array([self.groundtruth[t]['POSITION'] for t in times])

    def get_groundtruth_att(self,t_ini=None, t_final=None):
        """ Return groundtruth attitude for time between t_ini and t_final """        
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return self.groundtruth[times]['ATTITUDE']
        else:
            return [self.groundtruth[t]['ATTITUDE'] for t in times]
            
    def get_gps_pos(self,t_ini=None, t_final=None):
        """ Return GPS position for time between t_ini and t_final """    
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return self.heatmaps[times].gps_pos
        else:
            return np.array([self.heatmaps[t].gps_pos for t in times])
        
    def get_gps_att(self,t_ini=None, t_final=None):
        """ Return GPS attitude for time between t_ini and t_final """
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return self.heatmaps[times].attitude
        else:
            return [self.heatmaps[t].attitude for t in times]
            
    def get_gps_speed(self, t_ini=None, t_final=None):
        """ Return GPS speed for time between t_ini and t_final """
        times = self.get_timestamps(t_ini, t_final)
        if not t_ini is None and t_final is None:
            return np.linalg.norm(self.heatmaps[self.get_timestamps(times+0.1)].gps_pos - self.heatmaps[self.get_timestamps(times)].gps_pos)/0.1
        else:
            out = []
            for i in range(len(times)-1):
                out.append(np.linalg.norm(self.heatmaps[times[i+1]].gps_pos - self.heatmaps[times[i]].gps_pos)/(times[i+1]- times[i]))
            return out
            
    def plot_trajectory(self, arrow=False):
        """ Redefine Plot_Handler plot_trajectory function to plot only GPS trajectories """
        super().plot_trajectory(arrow, True, False)
        
    def export_map(self):
        """ Redefine Plot_Handler export_map function to plot only GPS trajectories """
        super().export_map(True, False)
        
    def plot_altitude(self):
        """ Redefine Plot_Handler plot_altitude function to plot only GPS trajectories """
        super().plot_altitude(True, False)
    
    def plot_attitude(self):
        """ Redefine Plot_Handler plot_attitude function to plot only GPS trajectories """
        super().plot_attitude(True, False)