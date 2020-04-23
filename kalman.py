import numpy as np
from map import Map
from copy import deepcopy
from data import RadarData
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj, rotation_ort, stat_test, projection

class Kalman:
    """
    Class used as a basis for all Kalman filter classes
    """   
    def __init__(self, mapping = True, name = None, bias_estimation = False):
        self.mapping = mapping
        if mapping:
            self.mapdata = Map(name)
        else:
            self.mapdata = None
        self.last_data = None
        
        self.position = None            # position in ECEF
        self.attitude = None            # from ECEF to rbd
        self.pos2D = np.zeros(2)        # position in 2D map
        self.att2D = 0                  # attitude in 2D map

        self.innovation = None
        
        self.bias_estimation = bias_estimation
        self.bias = np.zeros(3)
        
        self.init_default_covariance()
      
    def init_default_covariance(self):
        """ Initialize the covariances with default values """
        self.Q = np.block([[np.array([[0.00051736, -0.00091164],[-0.00091164,  0.00519678]]), np.zeros((2,1))],[np.zeros(2), 3.54317141e-06]])  # determined from reader.plot_gps_evaluation
        self.R = np.diag([0.01**2, 0.01**2, np.deg2rad(0.1)**2])
        if self.bias_estimation:
            self.P = np.block([[self.R, np.zeros((3,3))],[np.zeros((3,3)),np.diag([0.01**2, 0.01**2, np.deg2rad(0.001)**2])]])
        else:
            self.P = deepcopy(self.R)
                
    def set_initial_position(self, gps_pos, attitude):
        """ Set the initial position of the map in ECEF """
        self.position = deepcopy(gps_pos)
        self.attitude = deepcopy(attitude)
        
    def add(self, new_data, fusion=True):
        """ Add a new radar data on the map 
            fusion: if false, add raw data to the map
        """
        if self.last_data is None:
            # use position/attitude custom initialisation
            if not (self.position is None):
                new_data = RadarData(new_data.id, new_data.img, self.position, new_data.attitude)
            if not (self.attitude is None): 
                new_data = RadarData(new_data.id, new_data.img, new_data.gps_pos, self.attitude)
            
            if self.mapping:
                self.mapdata.add_data(new_data)
            else:
                self.mapdata = deepcopy(new_data)
                 
            self.last_data = deepcopy(new_data)
            self.position = deepcopy(self.mapdata.gps_pos)
            self.attitude = deepcopy(self.mapdata.attitude)
        else:
            if fusion:                
                self.predict(new_data)
                self.update(new_data)

                self.position = self.process_position(new_data)
                self.attitude = self.process_attitude(new_data)
            else:
                self.position = new_data.gps_pos
                self.attitude = new_data.attitude

            self.last_data = RadarData(new_data.id, new_data.img, self.position, self.attitude)
            if self.mapping:
                self.mapdata.add_data(self.last_data)
        return deepcopy(self.position), deepcopy(self.attitude)

class Kalman_Mapper_CV2GPS(Kalman):
    """
    Prediction : GPS
    Measurement : CV2
    """
    def set_covariances(self, gps_pos_std, gps_att_std, cv2_trans_std, cv2_rot_std, bias_trans_std=None, bias_rot_std=None):
        """ Set covariances Q and R of the Kalman Filter """
        self.Q = np.diag([cv2_trans_std**2, cv2_trans_std**2, cv2_rot_std**2])
        self.R = np.diag([gps_pos_std**2, gps_pos_std**2, gps_att_std**2])
        if self.bias_estimation:
            if not bias_rot_std is None:
                self.P = np.diag([gps_pos_std**2, gps_pos_std**2, gps_att_std**2, bias_trans_std**2, bias_trans_std**2, bias_rot_std**2])
        else:
            self.P = deepcopy(self.R)
        
    def predict(self, new_data):
        """ Based on image transformation measurement, predict new state X """
        trans, rotation = new_data.image_transformation_from(self.last_data) 
        
        R = np.array([[np.cos(self.att2D), -np.sin(self.att2D)],[np.sin(self.att2D), np.cos(self.att2D)]])
        if not np.any(np.isnan(trans)):
            self.pos2D = self.pos2D + R.dot(trans[0:2] + self.bias[0:2])
            self.att2D = self.att2D + rotation.as_euler('zxy')[0] + self.bias[2]
        else:
            self.pos2D = self.mapdata.attitude.apply(new_data.gps_pos - self.mapdata.gps_pos)[0:2]
            self.att2D = rotation_proj(self.mapdata.attitude, new_data.attitude).as_euler("zxy")[0]
        
        F = np.array([[1, 0, -np.sin(self.att2D)*trans[0]-np.cos(self.att2D)*trans[1]],
                      [0, 1, np.cos(self.att2D)*trans[0]-np.sin(self.att2D)*trans[1]],
                      [0, 0, 1]])
        M = np.array([[np.cos(self.att2D), -np.sin(self.att2D), 0],
                      [np.sin(self.att2D), np.cos(self.att2D), 0],
                      [0, 0, 1]])
        if self.bias_estimation:
            F = np.block([[F, np.block([[R, np.zeros((2,1))],[np.zeros(2), 1]])],[np.zeros((3,3)), np.eye(3)]])
            M = np.block([[M], [np.zeros((3,3))]])
            self.P = F.dot(self.P).dot(F.T) + M.dot(self.Q).dot(M.T)
        else:
            self.P = F.dot(self.P).dot(F.T) + M.dot(self.Q).dot(M.T)
        
    def update(self, new_data):
        """ Update the state X thanks to GPS information """
        pos2D = self.mapdata.attitude.apply(new_data.gps_pos - self.mapdata.gps_pos)[0:2]
        att2D = rotation_proj(self.mapdata.attitude, new_data.attitude).as_euler("zxy")[0]
        Y = np.append(pos2D, att2D)
        Yhat = np.append(self.pos2D, self.att2D)

        if self.bias_estimation:
            H = np.block([np.eye(3), np.zeros((3,3))])
        else:
            H = np.eye(3)
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        #Z = stat_test(Y, Yhat, S, 0.99)*(Y - Yhat)
        Z = (Y - Yhat)
        e = K.dot(Z)

        self.pos2D = self.pos2D  + e[0:2]
        self.att2D = self.att2D + e[2]
        if self.bias_estimation:
            self.bias = self.bias + e[3:6]

        self.P = (np.eye(len(self.P)) - K.dot(H)).dot(self.P)
        self.innovation = (Z, S)

class Kalman_Mapper_CV2GPS_3D(Kalman_Mapper_CV2GPS):
    """
    Prediction : CV2
    Measurement : GPS with 3D correction from GPS
    """   
    def process_position(self, new_data):
        return self.mapdata.gps_pos + self.mapdata.attitude.apply(np.append(self.pos2D, self.mapdata.attitude.apply(new_data.gps_pos - self.mapdata.gps_pos)[2]), True)
    
    def process_attitude(self, new_data):
        ort = rotation_ort(self.mapdata.attitude,new_data.attitude)
        return ort*rot.from_euler('zxy', [self.att2D, 0, 0]).inv()*self.mapdata.attitude
    
class Kalman_Mapper_CV2GPS_2D(Kalman_Mapper_CV2GPS):
    """
    Prediction : CV2
    Measurement : GPS without 3D correction
    """
    def process_position(self, new_data):
        return self.mapdata.gps_pos + self.mapdata.attitude.apply(np.append(self.pos2D, 0), True)
          
    def process_attitude(self, new_data):
        return rot.from_euler('zxy', [self.att2D, 0, 0]).inv()*self.mapdata.attitude

# =============================================================================
# Kalman Localizer
# =============================================================================

class Kalman_Localizer(Kalman):
    
    def __init__(self, mapping = False, name = None):          
        super().__init__(mapping, name)
        self.mapdata = Map(name)
    
    def set_covariances(self, gps_pos_std, gps_att_std, cv2_trans_std, cv2_rot_std):
        """ Set covariances Q and R of the Kalman Filter """
        self.Q = np.block([[np.zeros((3,3)), np.zeros((3,3))],[ np.zeros((3,3)), np.diag([gps_pos_std**2, gps_pos_std**2, gps_att_std**2])]])
        self.R = np.diag([cv2_trans_std**2, cv2_trans_std**2, cv2_rot_std**2])
           
    def set_initial_position(self, gps_pos, attitude):
        """ Initialize the position of the car as a first guess """
        self.position, self.attitude = projection(self.mapdata.gps_pos, self.mapdata.attitude, gps_pos, attitude)
                
    def localize(self, new_data, gps_guess=False):
        """ Find the position of a image thanks to the map """
        if gps_guess:            
            mapping_img, _ = self.mapdata.extract_from_map(new_data.gps_pos, new_data.attitude, np.shape(new_data.img))
            gps_pos, attitude = projection(self.mapdata.gps_pos, self.mapdata.attitude, new_data.gps_pos, new_data.attitude)
            mapping_data = RadarData(None, mapping_img, gps_pos, attitude) 
        else:
            mapping_img, _ = self.mapdata.extract_from_map(self.position, self.attitude, np.shape(new_data.img))
            mapping_data = RadarData(None, mapping_img, self.position, self.attitude) 

        self.position, self.attitude = new_data.image_position_from(mapping_data)
        self.last_data = RadarData(new_data.id, new_data.img, self.position, self.attitude)

        if self.mapping:  
            self.mapdata.add_data(self.last_data)
            
        return deepcopy(self.position), deepcopy(self.attitude)
    