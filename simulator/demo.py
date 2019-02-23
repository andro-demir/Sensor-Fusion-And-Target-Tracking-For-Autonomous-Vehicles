import scipy.io as sio
from objectClasses.objectClasses import SimSensor

cam_rear = SimSensor('./cam_rear.mat')
cam_front = SimSensor('./cam_front.mat')
radar_rear = SimSensor('./radar_rear.mat')
radar_front = SimSensor('./radar_front.mat')

asd=1
