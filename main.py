import warnings
from reader import Reader
from scipy.spatial.transform import Rotation as rot
warnings.filterwarnings("ignore") 

""" 
CASE WHERE THE FIRST IMAGE POSITION IS SURE AND THE SECUND IMAGE POSITION IS IMPERFECT
MEASURED DATA IS CONSIDERED AS PERFECTLY KNOWN
"""

# =============================================================================
# Creation of datasets
# =============================================================================

# Loading data   
#reader = Reader('radardata.h5')
    
data1 = reader.get_radardata(20)
data2 = reader.get_radardata(20.1)
data1.img.save("radar1.png")
data2.img.save("radar2.png")

# Expected transformation
exp_trans = data1.earth2rbd(data2.gps_pos - data1.gps_pos)[0:2]
exp_rot = -rot.as_rotvec(data1.attitude.inv()*data2.attitude)[2]
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
