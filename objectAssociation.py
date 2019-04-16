# objectAssociation.py

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.linalg import inv, pinv
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import normalize
from trackManagement import initialize_fusion_objects, drop_objects
from objectClasses import ObjectListCls

def getMahalanobisMatrix(fusionList, sensorObjList):
    '''
    :param fusionList (list): objects in the global list. 
    :param sensorObjList (List): objects in the radar / vision sensor 
                                 measurements.
    the statistical distance (Mahalanobis distance) between state vectors 
    from a fusion object and a sensor object is evaluated. 
    mahalanobisMatrix (ndarray): The cost matrix of the bipartite graph.
    '''
    measurementNoise = np.array([[22.1,0,0,0], [0,22.1,0,0], 
                                 [0,0,2209,0], [0,0,0,2209]])
    numFusionObjs, numSensorObjs = len(fusionList), len(sensorObjList)
    mahalanobisMatrix = np.zeros((numSensorObjs, numFusionObjs))
    for i in range(numFusionObjs):
        for j in range(numSensorObjs):
            x = np.concatenate((fusionList[i].s_vector[:2].reshape((1,2)),
                                fusionList[i].s_vector[3:5].reshape((1,2))),
                                axis=1) 
            y = np.concatenate((sensorObjList[j].s_vector[:2].reshape((1,2)),
                                sensorObjList[j].s_vector[3:5].reshape((1,2))),
                                axis=1)
            # Get the covariance V:
            V = np.concatenate((x,y))
            V = V - (np.ones((2,2)) @ V) / 2.0
            V = (V.T @ V) / 2.0 + measurementNoise
            IV = inv(V)
            mahDist = np.sqrt((x-y) @ IV @ (x-y).T)
            mahalanobisMatrix[j,i] = mahDist
    return mahalanobisMatrix

def matchObjs(mahalanobisMatrix, clutter_threshold):
    '''
    :param: mahalanobisMatrix(np.array): The cost matrix of the 
                                          bipartite graph.
    :param: clutter_threshold(double): if the mahalanobis distance is greater
                                       than this threshold value, classify this
                                       rowInd-colInd match as a false positive
                                       or clutter
    :return rowInd, colInd (np.array): An array of row indices and one of  
                                        corresponding column indices giving 
                                        the optimal assignment. 
    This function applies the linear sum assignment problem, also known as 
    minimum weight matching in bipartite graphs using Hungarian Method. 
    Given a problem instance is described by a matrix cost matrix, 
    where each C[i,j] is the cost of matching vertex i of the first partite
    set (a “worker”) and vertex j of the second set (a “job”). 
    The goal is to find a complete assignment of workers to jobs of 
    minimal cost.
    '''
    rowInd, colInd = linear_sum_assignment(mahalanobisMatrix)
    matched_tuples = list(zip(rowInd, colInd))
    print("Matched tuples (rowInd: Meas at t+1, colInd: State Est at t)")
    print(rowInd)
    print(colInd)
    
    # get the rows and columns where mahalanobis distance is greater than 
    # clutter threshold
    cluttered_matches = [] # list of false positive matches as tuples
    cleaned_matches = [] # list of true postive matches as tuples
    for (x,y) in list(zip(rowInd, colInd)):
        if mahalanobisMatrix[x,y] >= clutter_threshold:
            cluttered_matches.append((x,y))
        else:
            cleaned_matches.append((x,y))

    print("False positive / Clutter tuples (rowInd: Meas at t+1, "
                                            "colInd: State Est at t):")
    print(cluttered_matches)
    print("Cleaned matches (rowInd: Meas at t+1, colInd: State Est at t)")
    print(cleaned_matches)
    num_true_postive = len(cleaned_matches)

    rowInd = [i[0] for i in cleaned_matches]
    colInd = [i[1] for i in cleaned_matches]
    print("Number of cluttered sensor readings:", len(cluttered_matches))
    return rowInd, colInd, cluttered_matches, num_true_postive


def updateExistenceProbability(fusionList, sensorObjList, rowInd, colInd, 
                               cluttered_matches, last, D):
    '''
    :param fusionList (list): objects in the global list. 
    :param sensorObjList (List): objects in the radar / vision sensor 
                                 measurements.
    :return rowInd, colInd (np.array): An array of row indices and one of  
                                        corresponding column indices giving 
                                        the optimal assignment.
    :param last(double): time for being last seen
    :param D(double): distance to ego (L1 norm of the state vector)
    :return fusionList (list): updated global list of obstacles
    '''
    # new initialization function
    cluttered_sensors = [i[0] for i in cluttered_matches]
    new_object_idx = list(set(range(len(sensorObjList))) - set(rowInd) - set(cluttered_sensors))
    print("New sensor object indices: (Meas at t+1)")
    print(new_object_idx)
    notAssignedSensor_objects = ObjectListCls(sensorObjList.timeStamp, 
                                              sensorObjList.sensor_specs)
    notAssignedSensor_objects.extend([i for idx, i in enumerate(sensorObjList) if
                                      idx in new_object_idx])
    fusionList.extend(initialize_fusion_objects(notAssignedSensor_objects))
    # drop the obj from the fusion list
    fusionList = drop_objects(fusionList, cluttered_matches, 
                              last_seen=last, distance_to_ego=D)
    return fusionList
    



