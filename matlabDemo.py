# matlabDemo.py
import numpy as np
from objectClasses.objectClasses import Obstacle
from objectClasses.objectClasses import fusionList as fusionListCls
import objectAssociation as assc
from helper_functions import kf_measurement_update, temporal_alignment


def matExec(time, Measurements, States):
    '''
        param: time (float)
        param: Measurements (2d array) -- Sensor Obstacle List at t
        param: States (2d array) -- Fusion List at t-1
        return: stateEstimates (2d array)-- Fusion List at t
        Measurements in Matlab corresponds to sensorObjList in Python and
        States in Matlab corresponds to fusionList in Python and
    '''
    # We created the fusionList at time,
    # Get the sensorObjectList at time+1
    sensorObjList = []
    tmp_noise = np.array([[22.1,0,0,0,0,0], [0,2209,0,0,0,0], [0,0,1,0,0,0],
                          [0,0,0,22.1,0,0], [0,0,0,0,2209,0], [0,0,0,0,0,1]])
    for measurement in Measurements.T:
        sensorObjList.append(Obstacle(pos_x=measurement[0], 
                                      pos_y=measurement[1],
                                      pos_z=None, v_x=measurement[2], 
                                      v_y=measurement[3], v_z=None, 
                                      a_x=None, a_y=None, a_z=None,
                                      yaw=None, r_yaw=None, P=tmp_noise))
    fusionList = fusionListCls(time)
    for state in States.T:
        fusionList.append(Obstacle(pos_x=state[0], pos_y=state[1],
                                   pos_z=None, v_x=state[2], 
                                   v_y=state[3], v_z=None, 
                                   a_x=None, a_y=None, a_z=None,
                                   yaw=None, r_yaw=None, P=tmp_noise))

    mahalanobisMatrix = assc.getMahalanobisMatrix(fusionList, 
                                                  sensorObjList)
    rowInd, colInd = assc.matchObjs(mahalanobisMatrix)  
    kf_measurement_update(fusionList, sensorObjList, (rowInd, colInd))
    
    # Probability of existence of obstacles is updated:
    fusionList = assc.updateExistenceProbability(fusionList,
                                                 sensorObjList,
                                                 rowInd, colInd)
    N_obstacles = len(fusionList)
    stateEstimates = np.zeros((4, N_obstacles)) # (pos_x, pos_y, vel_x, vel_y)
    for i in range(N_obstacles):
        stateEstimates[0,i] = fusionList[i].s_vector[0] 
        stateEstimates[1,i] = fusionList[i].s_vector[1] 
        stateEstimates[2,i] = fusionList[i].s_vector[3] 
        stateEstimates[3,i] = fusionList[i].s_vector[4] 

    print(50 * "**")
    print("Time: %f" %time)
    print("Measurements:\n", Measurements)
    print("Mahalanobis Matrix", mahalanobisMatrix)
    print("State Estimates:\n", stateEstimates)
    return stateEstimates

