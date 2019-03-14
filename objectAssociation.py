'''
****************************************************************
Contributors to this code
** Andac Demir (andacdemir@gmail.com) (main developer)
****************************************************************
'''

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from scipy.linalg import inv, pinv
from sklearn.preprocessing import normalize

def getMahalanobisMatrix(fusionList, sensorObjList):
    '''
    :param fusionList (list): objects in the global list. 
    :param sensorObjList (List): objects in the radar / vision sensor 
                                 measurements.
    the statistical distance (Mahalanobis distance) between state vectors 
    from a fusion object and a sensor object is evaluated. 
    mahalanobisMatrix: The cost matrix of the bipartite graph.
    '''
    numFusionObjs, numSensorObjs = len(fusionList), len(sensorObjList)
    mahalanobisMatrix = np.zeros((numSensorObjs, numFusionObjs))
    #print(numSensorObjs, numFusionObjs)
    #print("Mahalanobis matrix:\n", mahalanobisMatrix)
    for i in range(numFusionObjs):
        # first remove None elements from the state vector:
        fusionList[i].s_vector = remove_none(fusionList[i].s_vector)
        #print("Fusion List state vector:\n", fusionList[i].s_vector)
        for j in range(numSensorObjs):
            # first remove None elements from the state vector:
            sensorObjList[j].s_vector = remove_none(sensorObjList[j].s_vector)
            #print("Sensor state vector:\n", sensorObjList[j].s_vector)
            # innovation covariance between 2 state estimates (3.14):
            V = np.stack((np.asarray(fusionList[i].s_vector), 
                          np.asarray(sensorObjList[j].s_vector)), axis=0)
            V = np.cov(V.T)
            #print("covariance matrix:\n", V)
            IV = pinv(V) 
            #print("inverse of the covariance matrix:\n", IV)
            mahalanobisMatrix[j,i] = mahalanobis(fusionList[i].s_vector, 
                                                 sensorObjList[j].s_vector, IV)
            #print("Mahalanobis matrix:\n", mahalanobisMatrix)
            #print(40*"--")
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

def updateExistenceProbability(fusionList, sensorObjList, mahalanobisMatrix, 
                               rowInd, colInd):
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
    thresh = getThreshold()
    alpha = getAlpha()
    beta = getBeta()
    gamma = getGamma()
    # reduce the probability of existence if it might be a clutter, reduce
    # its probability of existence by alpha
    for i, j in zip(rowInd, colInd):
        if mahalanobisMatrix[i, j] > thresh:
            fusionList[j].p_existence -= alpha
    
    # reduce the probability of existence of an object in the globalList 
    # by beta if it doesn't match with any sensor objs
    notAssignedGlobals = np.setdiff1d(colInd, 
                                      np.arange(numFusionObjs))
    for i in notAssignedGlobals:
        fusionList[i].p_existence -= beta
    
    # initilialize a new object in the global list by assigning a 
    # probability of existence (gamma), if the sensor object doesn't match
    # any objects in the globalList
    notAssignedSensors = np.setdiff1d(rowInd, 
                                      np.arange(numSensorObjs))
    for i in notAssignedSensors:
        sensorObjList[i].p_existence = gamma  
        fusionList.append(sensorObjList[i])  

    return fusionList 

# Assigned 0 and 1 for simplicity in the first scenario.
def getThreshold():
    return 0.0

def getAlpha():
    return 0.0

def getBeta():
    return 0.0

def getGamma():
    return 1.0
    
def remove_none(l):
    return [x for x in l if x is not None]
    

