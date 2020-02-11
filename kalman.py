import numpy as np
from map import Map
from copy import deepcopy
from data import RadarData
from scipy.spatial.transform import Rotation as rot

from utils import rotation_proj

class Kalman_Mapper:
    
    def __init__(self, mapping = True, name = None):
        self.mapping = mapping
        if mapping:
            self.mapdata = Map(name)
        else:
            self.mapdata = None
        self.last_data = None
        self.position = None
        self.attitude = None
        
        self.prev_pos2D = np.zeros(2)
        self.prev_att2D = 0
        self.pos2D = np.zeros(2)
        self.att2D = 0
        
        self.innovation = None
        
        self.init_default_covariance()
       
    def init_default_covariance(self):
        """ Initialize the covariances with default values """
        gps_std = 0.05
        att_std = np.deg2rad(1)
        cv2_trans_std = 0.04
        cv2_att_std = np.deg2rad(1)
        self.set_covariance(gps_std, att_std, cv2_trans_std, cv2_att_std)    
                
    def set_covariance(self, gps_std, att_std, cv2_trans_std, cv2_att_std):
        """ Set covariance of the Kalman Filter """
        self.P = np.zeros((6,6)) #the position and orientation of the map is fixed
        self.Q = np.block([[np.zeros((3,3)), np.zeros((3,3))],[ np.zeros((3,3)), np.diag([gps_std**2, gps_std**2, att_std**2])]])
        self.R = np.diag([cv2_trans_std**2, cv2_trans_std**2, cv2_att_std**2])
    
    def predict(self, new_data):
        """ Based on GPS pos of the new data, predict new state X """
        F = np.block([[np.zeros((3,3)), np.eye(3)], [np.zeros((3,6))]])
    
        self.prev_pos2D = deepcopy(self.pos2D)
        self.prev_att2D = self.att2D
        self.pos2D = self.mapdata.attitude.apply(new_data.gps_pos - self.mapdata.gps_pos)[0:2]
        self.att2D = rotation_proj(self.mapdata.attitude, new_data.attitude).as_euler('zxy')[0]
                
        self.P = F.dot(self.P).dot(F.T) + self.Q
    
    def update(self, new_data):
        """ Update the state X thanks to image transformation measurement """
        # TODO: Implement robust update
        trans, rotation = new_data.image_transformation_from(self.last_data)       
        self.Y = np.append(trans[0:2], rotation.as_euler('zxy')[0])
        Yhat = np.append(self.pos2D - self.prev_pos2D, self.att2D - self.prev_att2D)
     
        H = np.block([np.eye(3), -np.eye(3)])
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        Z = self.Y - Yhat     
        e = K.dot(Z)

        self.prev_pos2D = self.prev_pos2D  + e[0:2]
        self.prev_att2D = self.prev_att2D + e[2]
        self.pos2D = self.pos2D + e[3:5]
        self.att2D = self.att2D + e[5]

        self.P = (np.eye(len(self.P)) - K.dot(H)).dot(self.P)
        self.innovation = (Z, S)
        
    def add(self, new_data):
        """ Add a new radar data on the map """
        if self.last_data is None:
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

            self.position = self.mapdata.gps_pos + self.mapdata.attitude.apply(np.append(self.pos2D, 0), True)
            self.attitude = self.mapdata.attitude*rot.from_euler('zxy', [self.att2D, 0, 0])

            self.last_data = RadarData(new_data.id, new_data.img, self.position, self.attitude)
            if self.mapping:
                self.mapdata.add_data(self.last_data)
        return deepcopy(self.position), deepcopy(self.attitude)
    
class Kalman_Localizer(Kalman_Mapper):
    
    def __init__(self, mapping = False, name = None):          
        super().__init__(mapping, name)
        self.mapdata = Map(name)
     
    def init_position(self, gps_pos, attitude):
        """ Initialize the position of the car as a first guess """
        self.position = deepcopy(gps_pos)
        self.attitude = deepcopy(attitude)
        
    def localize(self, new_data):
        """ Find the position of a image thanks to the map """
        mapping_img, _ = self.mapdata.extract_from_map(self.position, self.attitude, np.shape(new_data.img))
        mapping_data = RadarData(-1, mapping_img, self.position, self.attitude) 

        self.position, self.attitude = new_data.image_position_from(mapping_data)
        self.last_data = RadarData(new_data.id, new_data.img, self.position, self.attitude)
        
        if self.mapping:  
            self.mapdata.add_data(self.last_data)
            
        return deepcopy(self.position), deepcopy(self.attitude)
    