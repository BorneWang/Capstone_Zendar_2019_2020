from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as rot
from data import RadarData
import warnings
warnings.filterwarnings("ignore") 

""" 
CASE WHERE THE FIRST IMAGE POSITION IS SURE AND THE SECUND IMAGE POSITION IS IMPERFECT
MEASURED DATA IS CONSIDERED AS PERFECTLY KNOWN
"""

# =============================================================================
# Creation of datasets
# =============================================================================

gps_pos1 = np.array([-2695404.12206001, -4296068.84848991, 3854695.20485694])
attitude1 = rot.from_quat(np.array([0.38383466,  0.20363679, -0.7925903, 0.42778924 ]))
data1 = RadarData(Image.open('radar1.png'), gps_pos1, attitude1)

gps_pos2 = np.array([-2695404.33922905, -4296069.12386884,  3854694.80476515])
attitude2 = rot.from_quat(np.array([0.38428563,  0.2029266 , -0.79157001,  0.42960712]))
data2 = RadarData(Image.open('radar2.png'), gps_pos2, attitude2)

gps_pos3 = np.array([-2695404.56525643, -4296069.4010868 ,  3854694.3506411 ])
attitude3 = rot.from_quat(np.array([0.38522398,  0.20174871, -0.79037394,  0.43151939]))
data3 = RadarData(Image.open('radar3.png'), gps_pos3, attitude3)

# Expected transformation
exp_trans = data1.earth2flu(data2.gps_pos - data1.gps_pos)[0:2]
exp_rot = rot.as_rotvec(data1.attitude.inv()*data2.attitude)[2]
print("Expected translation: "+str(exp_trans))
print("Expected rotation (rad): "+str(exp_rot))

# Prediction (array) of what should be seen according to next position and orientation
prediction = data1.predict_image(data2.gps_pos, data2.attitude)
# Result can be seen in image radar2_2.png (to compare with real measurment radar2.png)

# Measured transformation
vc_trans, mx_rot = data2.transformation_from(data1)
meas_trans = vc_trans[0:2]
meas_rot = rot.from_dcm(mx_rot).as_rotvec()[2]
print("Measured translation: "+str(meas_trans))
print("Measured rotation (rad): "+str(meas_rot))

# quality of transformation estimation can be evaluated by comparing radar1.png 
# with radar1_1.png (reverse transformation of radar2.png)
