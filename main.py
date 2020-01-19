import warnings
import numpy as np
from reader import Reader
from recorder import Recorder
from kalman import Kalman_Mapper

warnings.filterwarnings("ignore") 


# Loading data   
reader = Reader('radardata2.h5', 6, 15)

# Creating Kalman filter for mapping
kalman = Kalman_Mapper()
#TODO: decide more precisely covariances
kalman.set_covariance(0.05, np.deg2rad(1), 0.04, np.deg2rad(1))
#kalman.set_covariance(0, 0, 900, 900)

# show the map during mapping
kalman.mapdata.show()

# Creating recorder 
recorder = Recorder(reader, kalman)

for ts, radardata in reader:
    # add a new image
    pos, att = kalman.add(radardata)
    # save Kalman output
    recorder.record(ts)
    # update the map during mapping
    kalman.mapdata.show(pos)

# Extracting map after fusion
m = kalman.mapdata

# Plots
recorder.export_map()
#recorder.plot_innovation()
#recorder.plot_attitude()