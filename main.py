from PIL import Image
import numpy as np
from scipy.optimize import fmin_powell, fminbound, differential_evolution
from copy import deepcopy
from data import RadarData
import scipy.ndimage as snd
from scipy.spatial.transform import Rotation as rot
import warnings
warnings.filterwarnings("ignore") 

""" 
CASE WHERE THE FIRST IMAGE POSITION IS SURE AND THE SECUND IMAGE POSITION IS IMPERFECT
MEASURED DATA IS CONSIDERED AS PERFECTLY KNOWN
"""

def cost_function_rot(theta):
    new_data = deepcopy(data2)
    new_data.attitude = rot.from_rotvec(rot.from_quat(data2.attitude).as_rotvec()+[0,0,np.deg2rad(theta)]).as_quat()
    intersect_data_2, intersect_data_1 = new_data.intersection(data1)
    intersect_data_1 = intersect_data_1.circle()
    intersect_data_2 = intersect_data_2.circle()
    return -np.corrcoef(np.array(intersect_data_2.img.rotate(-np.rad2deg(rot.from_quat(data2.attitude).as_rotvec()[2])-theta)).ravel(), np.array(intersect_data_1.img).ravel())[0, 1]

def cost_function_trans(trans):
    new_data = deepcopy(data2)
    new_data.img = Image.fromarray(snd.affine_transform(data2.img, [1, 1], [trans[1],-trans[0]], order=1))
    intersect_data_2, intersect_data_1 = new_data.intersection(data1)
    intersect_data_1 = intersect_data_1.circle()
    intersect_data_2 = intersect_data_2.circle()
    return -np.corrcoef(np.array(intersect_data_2.img.rotate(-np.rad2deg(rot.from_quat(data2.attitude).as_rotvec()[2]))).ravel(), np.array(intersect_data_1.img).ravel())[0, 1]

def cost_function_trans_rot(trans_rot):
    theta = trans_rot[2]
    new_data = deepcopy(data2)
    new_data.attitude = rot.from_rotvec(rot.from_quat(data2.attitude).as_rotvec()+[0,0,np.deg2rad(theta)]).as_quat()
    new_data.img = Image.fromarray(snd.affine_transform(data2.img, [1, 1], [trans_rot[1], -trans_rot[0]], order=1))    
    intersect_data_2, intersect_data_1 = new_data.intersection(data1)
    intersect_data_1 = intersect_data_1.circle()
    intersect_data_2 = intersect_data_2.circle()
    return -np.corrcoef(np.array(intersect_data_2.img.rotate(-np.rad2deg(rot.from_quat(data2.attitude).as_rotvec()[2]))).ravel(), np.array(intersect_data_1.img).ravel())[0, 1]

# =============================================================================
# Creation of datasets
# =============================================================================

img = Image.open("Smiley.png").convert('1')
data1 = RadarData(img.crop((0, 200, 600, 600)), [0,0,0], [0,0,0,1])

rotation = np.pi/6
gps_pos = np.array([*(np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]).dot(np.array([-data1.width()/2, -data1.height()/2])) + np.array([data1.precision*(img.width-1)/2, data1.precision*(img.height-1)/2])),0])
data2 = RadarData(img.rotate(-np.rad2deg(rotation)).crop((0, 100, 600, 500)), gps_pos, rot.from_rotvec([0,0,-rotation]).as_quat())

# =============================================================================
# Estimation of rotation
# =============================================================================

# Errors
#error_rot = np.deg2rad(3)
#data2.attitude = rot.from_rotvec([0,0,-rotation+error_rot]).as_quat()
#
## Find rotation and GPS error
#theta = fminbound(cost_function_rot,-10,10)
#print("Rotation = "+str(theta))

# =============================================================================
# Estimation of translation
# =============================================================================

# Errors
#error_pos = np.array([0.5,0.7,0])
#data2.gps_pos = gps_pos + error_pos
#
#trans = fmin_powell(cost_function_trans, [0,0])
#print("Trans = "+str(rot.from_quat(data2.attitude).apply(np.array([trans[0],trans[1],0])*data2.precision, True)))

# =============================================================================
# Estimation of both translation and rotation
# =============================================================================
 
#TODO: problem highly non-convex, difficult to find the global optimum !

# Errors
error_rot = np.deg2rad(4)
data2.attitude = rot.from_rotvec([0,0,-rotation+error_rot]).as_quat()
error_pos = np.array([0.5,0.7,0])
data2.gps_pos = gps_pos + error_pos

trans_rot = differential_evolution(cost_function_trans_rot, ((-2,2),(-2,2),(-5,5))).x
print("Rotation = "+str(trans_rot[2]))
print("Trans = "+str(rot.from_quat(data2.attitude).apply(np.array([trans_rot[0],trans_rot[1],0])*data2.precision, True)))