import numpy as np
from map import Map
from copy import deepcopy
from data import RadarData
from scipy.spatial.transform import Rotation as rot

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

class Kalman_Mapper:
    
    def __init__(self, name = None):          
        self.mapdata = Map(name)
        self.last_data = None
        self.innovation = None
        
        self.prev_position = np.zeros(3)
        self.prev_attitude = rot.from_quat([0,0,1,0])
        self.position = np.zeros(3)
        self.attitude = rot.from_quat([0,0,1,0])
        
    def set_covariance(self, gps_std, orientation_std, cv2_trans_std, cv2_att_std):
        """ Set covariance of the Kalman Filter """
        self.P = np.block([[np.diag([gps_std**2, gps_std**2, gps_std**2, orientation_std**2, orientation_std**2, orientation_std**2]), np.zeros((6,6))], [np.zeros((6,6)), np.zeros((6,6))]])
        self.Q = np.block([[np.zeros((6,6)), np.zeros((6,6))],[ np.zeros((6,6)), np.diag([gps_std**2, gps_std**2, gps_std**2, orientation_std**2, orientation_std**2, orientation_std**2])]])
        #self.R = np.diag([cv2_trans_std**2, cv2_trans_std**2, cv2_att_std**2])
        self.R = np.diag([cv2_trans_std**2, cv2_trans_std**2])
    
    def predict(self, new_data):
        """ Based on GPS pos of the new data, predict new state X """
        F = np.block([[np.zeros((6,6)), np.eye(6)], [np.zeros((6,12))]])
    
        self.prev_position = self.last_data.gps_pos
        self.prev_attitude = self.last_data.attitude
        self.position = new_data.gps_pos
        self.attitude = new_data.attitude
                
        self.P = F.dot(self.P).dot(F.T) + self.Q
    
    def update(self, new_data):
        """ Update the state X thanks to image transformation measurement """
        trans, rotation = new_data.image_transformation_from(self.last_data)       
        #self.Y = np.append(trans[0:2], rotation.as_rotvec()[2])
        self.Y = trans[0:2]
        #Yhat = np.append(self.last_data.earth2rbd(self.position-self.prev_position)[0:2], self.prev_attitude.inv()*self.attitude.as_rotvec()[2])
        Yhat = self.last_data.earth2rbd(self.position-self.prev_position)[0:2]

        #H = np.block([[-self.prev_attitude.as_dcm(), skew(self.position-self.prev_position), self.prev_attitude.as_dcm(), np.zeros((3,3))][0:2],[np.zeros((1,3)), self.prev_attitude.inv()*self.attitude*self.prev_attitude.inv() , np.zeros((1,3)), self.prev_attitude.inv()]])
        H = np.block([-self.prev_attitude.as_dcm(), -skew(self.position-self.prev_position), self.prev_attitude.as_dcm(), np.zeros((3,3))])[0:2,:]
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        Z = self.Y - Yhat     
        e = K.dot(Z)
        
        self.prev_position = self.prev_position + e[0:3]
        #theta = np.linalg.norm(e[3:6])
        #dR_cross = skew(e[3:6])
        #self.prev_attitude = self.prev_attitude*rot.from_dcm(np.eye(3)+np.sin(theta)*dR_cross+(1-np.cos(theta))*dR_cross.dot(dR_cross))
        self.prev_attitude = self.prev_attitude*rot.from_rotvec(e[3:6])
        
        self.position = self.position + e[6:9]
        #theta = np.linalg.norm(e[9:12])
        #dR_cross = skew(e[9:12])
        #self.attitude = self.attitude*rot.from_dcm(np.eye(3)+np.sin(theta)*dR_cross+(1-np.cos(theta))*dR_cross.dot(dR_cross))
        self.attitude = self.attitude*rot.from_rotvec(e[9:12])
        
        self.P = (np.eye(len(self.P)) - K.dot(H)).dot(self.P)
        self.innovation = (Z, S)
        
    def add(self, new_data):
        """ Add a new radar data on the map """
        if self.last_data is None:
            self.mapdata.add_data(new_data)
            self.last_data = deepcopy(new_data)
            self.position = deepcopy(self.mapdata.gps_pos)
            self.attitude = deepcopy(self.mapdata.attitude)
        else: 
            self.predict(new_data)
            #self.update(new_data)

            self.last_data = RadarData(new_data.id, new_data.img, self.position, self.attitude)
            self.mapdata.add_data(self.last_data)
        return deepcopy(self.position), deepcopy(self.attitude)