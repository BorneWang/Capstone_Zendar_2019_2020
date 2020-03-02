import numpy as np
from map import Map
from copy import deepcopy
from data import RadarData
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj, R, stat_test

class Kalman_Mapper:
    """
    Class used as a basis for all Kalman filter classes
    """   
    def __init__(self, mapping = True, name = None):
        self.mapping = mapping
        if mapping:
            self.mapdata = Map(name)
        else:
            self.mapdata = None
        self.last_data = None
        
        self.position = None # in ECEF
        self.attitude = None # from ECEF to rbd
        
        self.prev_pos2D = np.zeros(2) # (x,y) position on the map
        self.prev_att2D = 0 # orientation in the map frame
        self.trans = np.zeros(2) # translation from the previous image (not in the map frame !)
        self.rot = 0 # orientation from the previous image (not in the map frame !)
        
        self.innovation = None
        
        self.init_default_covariance()
      
    def init_default_covariance(self):
        """ Initialize the covariances with default values """
        gps_pos_std = 0.05          # standard deviation of GPS position measurement
        gps_att_std = np.deg2rad(1) # standard deviation of GPS attitude measurement
        cv2_trans_std = 0.04        # standard deviation of CV2 translation measurement
        cv2_rot_std = np.deg2rad(1) # standard deviation of CV2 rotation measurement
        self.set_covariances(gps_pos_std, gps_att_std, cv2_trans_std, cv2_rot_std)
        self.P = np.zeros((6,6)) # the initial 2D position and orientation is known to be (0,0)
        
    def set_initial_position(self, gps_pos, attitude):
        """ Set the initial position of the map in ECEF """
        self.position = deepcopy(gps_pos)
        self.attitude = deepcopy(attitude)
        
    def add(self, new_data):
        """ Add a new radar data on the map """
        if self.last_data is None:
            # use position/attitude custom initialisation
            if not (self.position is None):
                new_data = RadarData(new_data.id, new_data.img, self.position, new_data.attitude)
            if not (self.attitude is None): 
                new_data = RadarData(new_data.id, new_data.img, new_data.position, self.attitude)
            
            if self.mapping:
                self.mapdata.add_data(new_data)
            else:
                self.mapdata = deepcopy(new_data)
                 
            self.last_data = deepcopy(new_data)
            self.position = deepcopy(self.mapdata.gps_pos)
            self.attitude = deepcopy(self.mapdata.attitude)
        else:
            self.predict(new_data)
            self.update(new_data)

            self.position = self.process_position(new_data)
            self.attitude = self.process_attitude(new_data)

            self.last_data = RadarData(new_data.id, new_data.img, self.position, self.attitude)
            if self.mapping:
                self.mapdata.add_data(self.last_data)
        return deepcopy(self.position), deepcopy(self.attitude)


class Kalman_Mapper_GPSCV2(Kalman_Mapper):
    """
    Prediction : GPS
    Measurement : CV2
    """
    def set_covariances(self, gps_pos_std, gps_att_std, cv2_trans_std, cv2_rot_std):
        """ Set covariances Q and R of the Kalman Filter """
        self.Q = np.block([[np.zeros((3,3)), np.zeros((3,3))],[ np.zeros((3,3)), np.diag([gps_pos_std**2, gps_pos_std**2, gps_att_std**2])]])
        self.R = np.diag([cv2_trans_std**2, cv2_trans_std**2, cv2_rot_std**2])
        
    def predict(self, new_data):
        """ Based on GPS pos of the new data, predict new state X """
        pos2D = self.mapdata.attitude.apply(new_data.gps_pos - self.mapdata.gps_pos)[0:2]
        att2D = rotation_proj(self.mapdata.attitude, new_data.attitude).as_euler('zxy')[0]

        F1 =np.array([[1, 0, -self.trans[0]*np.sin(self.prev_att2D)-self.trans[1]*np.cos(self.prev_att2D), np.cos(self.prev_att2D), -np.sin(self.prev_att2D), 0],
                      [0, 1, self.trans[0]*np.cos(self.prev_att2D)-self.trans[1]*np.sin(self.prev_att2D), np.sin(self.prev_att2D), np.cos(self.prev_att2D), 0],
                      [0, 0, 1, 0, 0, 1]])
        #F2 = np.array([[0, 0, -(pos2D-self.prev_pos2D)[0]*np.sin(self.prev_att2D+self.rot)+(pos2D-self.prev_pos2D)[0]*np.cos(self.prev_att2D+self.rot), 0, 0, -(pos2D-self.prev_pos2D)[0]*np.sin(self.prev_att2D+self.rot)+(pos2D-self.prev_pos2D)[0]*np.cos(self.prev_att2D+self.rot)],
        #               [0, 0, -(pos2D-self.prev_pos2D)[1]*np.cos(self.prev_att2D+self.rot)-(pos2D-self.prev_pos2D)[1]*np.sin(self.prev_att2D+self.rot), 0, 0, -(pos2D-self.prev_pos2D)[1]*np.cos(self.prev_att2D+self.rot)-(pos2D-self.prev_pos2D)[1]*np.sin(self.prev_att2D+self.rot)],
        #               [0, 0, -1, 0, 0, 0]])
        F2 = np.array([[-np.cos(self.prev_att2D+self.rot), -np.sin(self.prev_att2D+self.rot), -(pos2D-self.prev_pos2D)[0]*np.sin(self.prev_att2D+self.rot)+(pos2D-self.prev_pos2D)[0]*np.cos(self.prev_att2D+self.rot), -np.cos(self.rot), -np.sin(self.rot), -(pos2D-self.prev_pos2D)[0]*np.sin(self.prev_att2D+self.rot)+(pos2D-self.prev_pos2D)[0]*np.cos(self.prev_att2D+self.rot) + self.trans[0]*np.sin(self.rot) - self.trans[1]*np.cos(self.rot)],
                       [ np.sin(self.prev_att2D+self.rot), -np.cos(self.prev_att2D+self.rot), -(pos2D-self.prev_pos2D)[1]*np.cos(self.prev_att2D+self.rot)-(pos2D-self.prev_pos2D)[1]*np.sin(self.prev_att2D+self.rot), np.sin(self.rot), -np.cos(self.rot), -(pos2D-self.prev_pos2D)[1]*np.cos(self.prev_att2D+self.rot)-(pos2D-self.prev_pos2D)[1]*np.sin(self.prev_att2D+self.rot) + self.trans[0]*np.cos(self.rot) + self.trans[1]*np.sin(self.rot)],
                       [0, 0, -1, 0, 0, -1]])
        F = np.block([[F1],[F2-F1]])
    
        self.prev_pos2D = self.prev_pos2D + R(-self.prev_att2D).dot(self.trans)
        self.prev_att2D = self.prev_att2D + self.rot
        
        self.trans = R(self.prev_att2D).dot(pos2D - self.prev_pos2D)
        self.rot = att2D - self.prev_att2D
                
        self.P = F.dot(self.P).dot(F.T) + self.Q
    
    def update(self, new_data):
        """ Update the state X thanks to image transformation measurement """
        trans, rotation = new_data.image_transformation_from(self.last_data)       
        Y = np.append(trans[0:2], rotation.as_euler('zxy')[0])
        Yhat = np.append(self.trans, self.rot)
        
        H = np.block([np.zeros((3,3)), np.eye(3)])
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        Z = stat_test(Y, Yhat, S, 0.99)*(Y - Yhat)
        e = K.dot(Z)

        self.prev_pos2D = self.prev_pos2D  + e[0:2]
        self.prev_att2D = self.prev_att2D + e[2]
        self.trans = self.trans + e[3:5]
        self.rot = self.rot + e[5]

        self.P = (np.eye(len(self.P)) - K.dot(H)).dot(self.P)
        self.innovation = (Z, S)
        
class Kalman_Mapper_GPSCV2_3D(Kalman_Mapper_GPSCV2):
    """
    Prediction : GPS
    Measurement : CV2 with 3D correction
    """   
    def process_position(self, new_data):
        return self.mapdata.gps_pos + self.mapdata.attitude.apply(np.append(self.prev_pos2D + R(-self.prev_att2D).dot(self.trans), self.mapdata.attitude.apply(new_data.gps_pos - self.mapdata.gps_pos)[2]), True)
    
    def process_attitude(self, new_data):
        ort = new_data.attitude*self.mapdata.attitude.inv()*rotation_proj(self.mapdata.attitude, new_data.attitude)
        return ort*rot.from_euler('zxy', [self.prev_att2D + self.rot, 0, 0]).inv()*self.mapdata.attitude
    
class Kalman_Mapper_GPSCV2_2D(Kalman_Mapper_GPSCV2):
    """
    Prediction : GPS
    Measurement : CV2 without 3D correction
    """
    def process_position(self, new_data):
        return self.mapdata.gps_pos + self.mapdata.attitude.apply(np.append(self.prev_pos2D + R(-self.prev_att2D).dot(self.trans), 0), True)
          
    def process_attitude(self, new_data):
        return rot.from_euler('zxy', [self.prev_att2D + self.rot, 0, 0]).inv()*self.mapdata.attitude
 
    
class Kalman_Mapper_CV2GPS(Kalman_Mapper):
    """
    Prediction : CV2
    Measurement : GPS at a given frequency
    """
    def __init__(self, mapping = False, name = None):          
        super().__init__(mapping, name)
        self.frequency = 0 #by default all GPS position will be taken
        
    def set_covariances(self, gps_pos_std, gps_att_std, cv2_trans_std, cv2_rot_std):
        """ Set covariances Q and R of the Kalman Filter """
        pass
        
    def predict(self, new_data):
        """ Based on GPS pos of the new data, predict new state X """
        pass
    
    def update(self, new_data):
        """ Update the state X thanks to image transformation measurement """
        pass
    
    
class Kalman_Localizer(Kalman_Mapper):
    
    def __init__(self, mapping = False, name = None):          
        super().__init__(mapping, name)
        self.mapdata = Map(name)
        self.pos2D = None
        self.att2D = None
    
    def set_covariances(self, gps_pos_std, gps_att_std, cv2_trans_std, cv2_rot_std):
        """ Set covariances Q and R of the Kalman Filter """
        self.Q = np.block([[np.zeros((3,3)), np.zeros((3,3))],[ np.zeros((3,3)), np.diag([gps_pos_std**2, gps_pos_std**2, gps_att_std**2])]])
        self.R = np.diag([cv2_trans_std**2, cv2_trans_std**2, cv2_rot_std**2])
        
    
    def set_initial_position(self, gps_pos, attitude):
        """ Initialize the position of the car as a first guess """
        self.position = deepcopy(gps_pos)
        self.attitude = deepcopy(attitude)
        self.pos2D = self.mapdata.attitude.apply(gps_pos - self.mapdata.gps_pos)[0:2]
        self.att2D = rotation_proj(self.mapdata.attitude, attitude).as_euler('zxy')[0]
        
    def localize(self, new_data):
        """ Find the position of a image thanks to the map """
        mapping_img, _ = self.mapdata.extract_from_map(self.position, self.attitude, np.shape(new_data.img))
        mapping_data = RadarData(-1, mapping_img, self.position, self.attitude) 

        self.position, self.attitude = new_data.image_position_from(mapping_data)
        self.last_data = RadarData(new_data.id, new_data.img, self.position, self.attitude)

        if self.mapping:  
            self.mapdata.add_data(self.last_data)
            
        return deepcopy(self.position), deepcopy(self.attitude)
    