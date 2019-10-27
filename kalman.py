import numpy as np
from copy import deepcopy
from data import RadarData
from scipy.spatial.transform import Rotation as rot

class Kalman_Mapper:
    
    def __init__(self):
        self.mapdata = None
        #TODO: add covariance map
        
        self.last_data = None
        self.X = np.zeros(12) #(x,y,z,theta_x, theta_y, theta_z, dx, dy, dz, dtheta_x, dtheta_y, dtheta_z) in mapdata frame
        #TODO: set covariance default
        
    def set_covariance(self, gps_std, orientation_std, cv2_trans_std, cv2_att_std):
        """ Set covariance of the Kalman Filter """
        self.P = np.block([[np.block([[(gps_std**2)*np.eye(3), np.zeros((3,3))], [np.zeros((3,3)), (orientation_std**2)*np.eye(3)]]), np.zeros((6,6))], [np.zeros((6,6)), np.zeros((6,6))]])
        self.Q = np.block([[np.zeros((6,6)), np.zeros((6,6))],[ np.zeros((6,6)), np.block([[(gps_std**2)*np.eye(3), np.zeros((3,3))], [np.zeros((3,3)), (orientation_std**2)*np.eye(3)]])]])
        self.R = np.block([[(cv2_trans_std**2)*np.eye(3), np.zeros((3,3))], [np.zeros((3,3)), (cv2_att_std**2)*np.eye(3)]])
    
    def predict(self, new_data):
        """ Based on GPS pos of the new data, predict new state X """
        F = np.block([[np.eye(6), np.eye(6)],[-np.eye(6), -np.eye(6)]])
        B = np.block([[np.zeros((6,6))], [np.eye(6)]])
        
        #TODO: complete 3D transformation
        gps_pos = np.append(self.mapdata.earth2rbd(new_data.gps_pos - self.mapdata.gps_pos)[0:2], 0) #(x,y,0)
        gps_att = np.array([0,0,rot.as_rotvec(self.mapdata.attitude.inv()*new_data.attitude)[2]]) #(0,0, theta_z)       
        U = np.append(gps_pos, gps_att)
        
        self.X_pred = F.dot(self.X) + B.dot(U)
        self.P_pred = F.dot(self.P).dot(F.T) + self.Q
        return deepcopy(self.X_pred), deepcopy(self.P_pred)
    
    def update(self, new_data):
        """ Update the state X thanks to image transformation measurement """
        trans, rotation = new_data.image_transformation_from(self.last_data)
        
        self.Y = np.append(np.append(self.mapdata.earth2rbd(self.last_data.earth2rbd(trans,True))[0:2],0), rotation.as_rotvec())
        H =  np.block([np.zeros((6,6)), np.eye(6)])
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        Z = self.Y-H.dot(self.X)
        X = self.X + K.dot(Z)
        P = (np.eye(len(self.P)) - K.dot(H)).dot(self.P)
        
        self.innovation = (Z.T.dot(np.linalg.inv(S)).dot(Z))
        return deepcopy(X), deepcopy(P)
        
    def add(self, new_data):
        """ Add a new radar data on the map """
        if self.mapdata is None:
            self.mapdata = deepcopy(new_data)
            self.last_data = deepcopy(new_data)
        else: 
            self.X, self.P = self.predict(new_data)
            self.X, self.P = self.update(new_data)
            
            #TODO: add last_data to mapdata by Johan
            self.last_data = RadarData(new_data.img, self.get_position(), self.get_attitude())
    
    def get_position(self):
        """ Return the updated position """
        gps_pos = self.mapdata.gps_pos + self.mapdata.earth2rbd(self.X[0:3]+self.X[6:9],True)
        return gps_pos
    
    def get_attitude(self):
        """ Return the updated position """
        attitude = self.mapdata.attitude*rot.from_rotvec(self.X[3:6])*rot.from_rotvec(self.X[9:12])
        return attitude