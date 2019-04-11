# objectAssociation.py

import numpy as np
import math
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.linalg import inv, pinv
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import normalize
from random import uniform

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

def matchObjs(mahalanobisMatrix):
    '''
    :param: mahalanobisMatrix(np.array): The cost matrix of the 
                                          bipartite graph.
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
    return rowInd, colInd

def updateExistenceProbability(fusionList, sensorObjList, rowInd, colInd):
    '''
    :param fusionList (list): objects in the global list. 
    :param sensorObjList (List): objects in the radar / vision sensor 
                                 measurements.
    :param: mahalanobisMatrix(np.array): The cost matrix of the 
                                          bipartite graph.
    Variables called from the helper functions:
    :param rowInd, colInd (np.array): An array of row indices and one of  
                                        corresponding column indices giving 
                                        the optimal assignment. 
    :param thresh (double): threshold level. If the cost is bigger than
                            this, it might be a clutter - reduce the 
                            probability of existence by alpha.
    :param alpha, beta, gamma (double): coeffs to update the probabilities  
                                        of existence
    If a row (a sensor object) is not assigned to a column (an object in 
    the global(fusion) list), it may be a new object in the environment. 
    Initialize a new object with probability of existence: beta.
    :return fusionList (list): updated global list of obstacles
    '''
    numFusionObjs, numSensorObjs = len(fusionList), len(sensorObjList)
    # TODO:
    # reduce the probability of existence if it might be a clutter, reduce
    # its probability of existence by alpha
      
    # # Update the state vectors of the obstacles in the fusionList:
    # for x, y in zip(rowInd, colInd):
    #     fusionList[y].s_vector = sensorObjList[x].s_vector

    # initilialize a new object in the global list by assigning a 
    # probability of existence (gamma), if the sensor object doesn't match
    # any objects in the globalList
    if numSensorObjs >= rowInd.shape[0]:
        notAssignedSensors = sorted(set(list(range(numSensorObjs))) -
                                                 set(list(rowInd))) 
        for i in notAssignedSensors:
            sensorObjList[i].p_existence = 1.0  
            fusionList.append(sensorObjList[i])  

    return fusionList 
   
    

