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
    # get the rows and columns where mahalanobis distance is greater than 
    # clutter threshold
    fp_rowInd, fp_colInd = np.where(mahalanobisMatrix >= clutter_threshold)
    fp_tuples = list(zip(fp_rowInd, fp_colInd))
    # remove the clutter detections from the matches
    cleaned_matches = list(set(matched_tuples)-set(fp_tuples))
    # return the row index and column index of true positive 
    # object associations as numpy arrays.
    cleaned_matches = np.array(cleaned_matches, dtype=np.dtype('int,int'))
    rowInd, colInd = cleaned_matches['f0'], cleaned_matches['f1']
    return rowInd, colInd

def updateExistenceProbability(fusionList, sensorObjList, rowInd, colInd,
                               last, D):
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
    notAssignedSensor_objects = ObjectListCls(sensorObjList.timeStamp, 
                                              sensorObjList.sensor_specs)
    notAssignedSensor_objects.extend([i for idx, i in enumerate(sensorObjList) if
                                      idx not in rowInd])
    fusionList.extend(initialize_fusion_objects(notAssignedSensor_objects))

    # drop the obj from the fusion list
    fusionList = drop_objects(fusionList, last_seen=last, distance_to_ego=D)
    return fusionList



