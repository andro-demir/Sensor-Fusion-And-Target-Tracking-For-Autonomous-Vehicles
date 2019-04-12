# matlabDemo.py
import numpy as np
from Classes.objectClasses import Obstacle, ObjectListCls
import objectAssociation as assc
from helper_functions import kf_measurement_update, temporal_alignment

def matExec(time, Measurements, States, last_update_times):
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
    # Note: In Eatron's code Measurements = [pos_x, v_x, pos_y, v_y]'
    sensorObjList = ObjectListCls(time)
    measurementNoise = np.array([[22.1, 0, 0, 0, 0, 0], [0, 22.1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0], [0, 0, 0, 2209, 0, 0],
                                 [0, 0, 0, 0, 2209, 0], [0, 0, 0, 0, 0, 1]])
    for measurement in Measurements.T:
        sensorObjList.append(Obstacle(pos_x=measurement[0],
                                      pos_y=measurement[2],
                                      pos_z=None, v_x=measurement[1],
                                      v_y=measurement[3], v_z=None,
                                      a_x=None, a_y=None, a_z=None,
                                      yaw=None, r_yaw=None, P=measurementNoise))
    fusionList = ObjectListCls(time)
    for idx, state in enumerate(States.T):
        fusionList.append(Obstacle(pos_x=state[0], pos_y=state[2],
                                   pos_z=None, v_x=state[1],
                                   v_y=state[3], v_z=None,
                                   a_x=None, a_y=None, a_z=None,
                                   yaw=None, r_yaw=None, P=measurementNoise,
                                   last_update_time=last_update_times[idx]))

    mahalanobisMatrix = assc.getMahalanobisMatrix(fusionList, sensorObjList)
    rowInd, colInd = assc.matchObjs(mahalanobisMatrix)
    kf_measurement_update(fusionList, sensorObjList, (rowInd, colInd))

    # Probability of existence of obstacles is updated:
    fusionList = assc.updateExistenceProbability(fusionList,
                                                 sensorObjList,
                                                 rowInd, colInd)
    N_obstacles = len(fusionList)
    stateEstimates = np.zeros((4, N_obstacles))  # (pos_x, vel_x, pos_y, vel_y)
    last_update_times = np.zeros((1, N_obstacles))
    for i in range(N_obstacles):
        stateEstimates[0, i] = fusionList[i].s_vector[0]  # pos_x
        stateEstimates[1, i] = fusionList[i].s_vector[3]  # v_x
        stateEstimates[2, i] = fusionList[i].s_vector[1]  # pos_y
        stateEstimates[3, i] = fusionList[i].s_vector[4]  # v_y
        last_update_times[0, i] = fusionList[i].last_update_time
    
    print(50 * "**")
    print("Time: %f" % time)
    print("Measurements:\n", Measurements)
    print("Mahalanobis Matrix", mahalanobisMatrix)
    print("State Estimates:\n", stateEstimates)
    print("Last Update Times:\n", last_update_times)
    return stateEstimates
