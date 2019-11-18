import warnings
import numpy as np
from reader import Reader
from recorder import Recorder
from kalman import Kalman_Mapper

warnings.filterwarnings("ignore") 


# Loading data   
reader = Reader('radardata.h5', 6, 15, option='None')

# Creating Kalman filter for mapping
kalman = Kalman_Mapper()
#TODO: decide more precisely covariances
kalman.set_covariance(0.05, np.deg2rad(1), 0.04, np.deg2rad(1))
#kalman.set_covariance(0, 0, 900, 900)

# Creating recorder 
recorder = Recorder(reader, kalman)

for ts, radardata in reader:
    kalman.add(radardata)
    recorder.record(ts)
    
# Plots
recorder.export_map()
recorder.plot_innovation()
recorder.plot_attitude()