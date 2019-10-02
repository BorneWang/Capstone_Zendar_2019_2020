from PIL import Image
import numpy as np
from scipy.optimize import fmin_powell
from copy import deepcopy
from data import Data
import data
import scipy.ndimage as snd

""" 
CASE WHERE THE FIRST IMAGE POSITION IS SURE AND THE SECUND IMAGE POSITION IS IMPERFECT
MEASURED DATA IS CONSIDERED AS PERFECTLY KNOWN
"""

def cost_function(trans_rot):
    theta = trans_rot[2]
    trans_x = trans_rot[0]
    trans_y = trans_rot[1]
    new_img = snd.affine_transform(intersect_data_2.img.rotate(theta), [1, 1], [-trans_x, -trans_y], order=1)
    return -np.corrcoef(np.array(new_img).ravel(), np.array(intersect_data_1.img).ravel())[0, 1]

# =============================================================================
# Creation of datasets
# =============================================================================

data1 = Data(Image.open("Smiley.png").convert('1'), [0,0,0], [1,0,0,0])
data1 = data1.crop(0,500,500,0)

data2 = Data(Image.open("Smiley.png").convert('1').rotate(20), [0,0,0], [1,0,0,0])
data2 = data2.crop(100,600,600,100)
# Error in GPS of 10cm
data2.gps_pos += np.array([0.1,0,0])

# =============================================================================
# Estimation of rotation and translation
# =============================================================================

# Find intersection
intersect_data_2, intersect_data_1 = data2.intersection(data1)
intersect_data_1 = intersect_data_1.circle()
intersect_data_2 = intersect_data_2.circle()

# Find rotation and GPS error
trans_rot = fmin_powell(cost_function, [0,0,0]) 

trans = data.fur2earth(data2.precision*np.array([trans_rot[1],0,trans_rot[0]]), data2.attitude)
print("Translation = "+str(trans))
print("Rotation = "+str(trans_rot[2]))