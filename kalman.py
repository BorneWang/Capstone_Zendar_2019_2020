import pickle
import numpy as np
from os import path
from copy import deepcopy
from data import RadarData
from scipy.spatial.transform import Rotation as rot

class Kalman_Mapper:
    
    def __init__(self):
        if path.exists("cv2_transformations.pickle"):        
            cv2_transformations = open("cv2_transformations.pickle","rb")
            self.cv2_transformations = pickle.load(cv2_transformations)
            cv2_transformations.close()
        else:
            self.cv2_transformations = dict()
            
        self.mapdata = None
        #TODO: add covariance map
        
        self.last_data = None
        self.X = np.zeros(6) #(x,y, theta_z, dx, dy, dtheta_z) in mapdata frame
        #TODO: set covariance default
        
    def set_covariance(self, gps_std, orientation_std, cv2_trans_std, cv2_att_std):
        """ Set covariance of the Kalman Filter """
        self.P = np.block([[np.diag([gps_std**2, gps_std**2, orientation_std**2]), np.zeros((3,3))], [np.zeros((3,3)), np.zeros((3,3))]])
        self.Q = np.block([[np.zeros((3,3)), np.zeros((3,3))],[ np.zeros((3,3)), np.diag([gps_std**2, gps_std**2, orientation_std**2])]])
        self.R = np.diag([cv2_trans_std**2, cv2_trans_std**2, cv2_att_std**2])
    
    def predict(self, new_data):
        """ Based on GPS pos of the new data, predict new state X """
        F = np.block([[np.eye(3), np.eye(3)],[-np.eye(3), -np.eye(3)]])
        B = np.block([[np.zeros((3,3))], [np.eye(3)]])
        
        #TODO: complete 3D transformation
        pos = self.mapdata.earth2rbd(new_data.gps_pos - self.mapdata.gps_pos)[0:2] #(x,y)
        att = rot.from_dcm(np.block([[rot.as_dcm(self.mapdata.attitude.inv()*new_data.attitude)[:2,:2], np.zeros((2,1))],[np.zeros((1,2)), 1]])).as_rotvec()[2] #(theta_z)       
        U = np.append(pos, att)
        
        self.X_pred = F.dot(self.X) + B.dot(U)
        self.P_pred = F.dot(self.P).dot(F.T) + self.Q
        return deepcopy(self.X_pred), deepcopy(self.P_pred)
    
    def update(self, new_data):
        """ Update the state X thanks to image transformation measurement """
        if new_data.id in self.cv2_transformations:
            trans, rotation = self.cv2_transformations[str(self.last_data.id)+"-"+str(new_data.id)]
        else:
            trans, rotation = new_data.image_transformation_from(self.last_data)
            cv2_transformations = open("cv2_transformations.pickle","wb")
            self.cv2_transformations[str(self.last_data.id)+"-"+str(new_data.id)] = (trans, rotation)
            pickle.dump(self.cv2_transformations, cv2_transformations)
            cv2_transformations.close()      
        self.Y = np.append(trans[0:2], rotation.as_rotvec()[2])
        
        theta = self.X[2]
        R = np.array([[np.cos(theta), np.sin(theta), 0],[-np.sin(theta), np.cos(theta), 0],[0,0,1]])

        H = np.block([np.zeros((3,3)), np.eye(3)])
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        Z = R.dot(self.Y) - H.dot(self.X)
       
        e = K.dot(Z)
        d_theta = e[2]
        d_pos = e[0:2]
        d_trans = e[3:5]
        d_rot = e[5]
        
        delta_x = np.array([np.sin(d_theta)*d_pos[0] + (np.cos(d_theta)-1)*d_pos[1],np.sin(d_theta)*d_pos[1] - (np.cos(d_theta)-1)*d_pos[0]])/d_theta
        delta_trans = np.array([np.sin(d_theta)*d_trans[0] + (np.cos(d_theta)-1)*d_trans[1],np.sin(d_theta)*d_trans[1] - (np.cos(d_theta)-1)*d_trans[0]])/d_theta
        
        R = np.array([[np.cos(d_theta), np.sin(d_theta)],[-np.sin(d_theta), np.cos(d_theta)]])
        X = np.concatenate((R.dot(self.X[0:2])+delta_x, [self.X[2]+ d_theta], R.dot(self.X[3:5])+delta_trans, [self.X[5]+ d_rot]))
        X = self.X + K.dot(Z)
        P = (np.eye(len(self.P)) - K.dot(H)).dot(self.P)
        
#        self.innovation = (Z.T.dot(np.linalg.inv(S)).dot(Z))
        self.innovation = (Z, S)
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
            self.last_data = RadarData(new_data.id, new_data.img, self.get_position(), self.get_attitude())
    
    def get_position(self):
        """ Return the updated position """
        gps_pos = self.mapdata.gps_pos + self.mapdata.earth2rbd(np.append(self.X[0:2]+self.X[3:5],0),True)
        return gps_pos
    
    def get_attitude(self):
        """ Return the updated position """
        attitude = self.mapdata.attitude*rot.from_rotvec(np.array([0,0,self.X[2]]))*rot.from_rotvec(np.array([0,0,self.X[5]]))
        return attitude