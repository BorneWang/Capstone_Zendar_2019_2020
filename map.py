import numpy as np
from data import RadarData
from scipy.spatial.transform import Rotation as rot

class Map:
    
    def __init__(self, radardata):
        self.origin = radardata.gps_pos
        self.orientation = radardata.attitude
        
        self.map = radardata.img      
        self.cov = np.zeros(self.map.size)
        
    def set_initial_covariance(self, R):
        self.cov = R*np.ones(self.cov.shape)
        self.last_cov = R*np.ones(self.cov.shape)
        
    def earth2map(self,pos, inverse=False):
        """ Change of frame from earth frame to map frame """
        return self.orientation.apply(pos, inverse)
        
    def to_mapframe(self, radardata):
        theta = rot.as_rotvec(self.orientation.inv()*radardata.attitude)[2]
        gps_pos = self.earth2map(radardata.gps_pos)
        if theta > 0:   
            gps_pos[0] = gps_pos[0] - np.sin(theta)*radardata.height()
        elif theta < 0:
            gps_pos[1] = gps_pos[1] + np.sin(theta)*radardata.width()
        return RadarData(radardata.img.rotate(theta, expand=True), self.earth2map(gps_pos, True), self.orientation)
    
    def add(self, radardata):
        new_radar = 

