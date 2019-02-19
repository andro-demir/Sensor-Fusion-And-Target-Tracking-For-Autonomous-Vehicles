'''
****************************************************************
Contributors to this code
** Andac Demir (andacdemir@gmail.com) (main developer)
****************************************************************
Radar sees so many objects and we need to delete some.
Most of these things seen by the radar are clutter.
For computationally efficient sensor fusion we remove the clutters.
Of the rest of the candidate detections, we detect the best detections 
coming based on the predicted track and the Mahalanobis distance.

Step 1: Predict next position along the track of the vehicle
Step 2: Matches should be close to the next state and some matches
        are unlikely (clutter). We prune those detection using
        the Global Nearest Neighbor method based on Mahalanobis
        distance. This evaluates each observation in track gating
        region and chooses the ones to incorporate into the track.
Step 3: Apply Hungarian algorithm to the resulting cost matrix and 
        the data association with the lowest cost associated to it
        is generated. 
'''
import numpy as np
from scipy.spatial.distance import Mahalanobis
from scipy.optimize import linear_sum_assignment

def getMahalanobisMatrix(globalList, radarObjList, visionObjList):
    '''
    :param globalList (list): objects in the global list 
    :param radarObjList (List): objects in the radar sensor measurements
    :param visionObjList (List): objects in the vision sensor measurements
    :return (np.array): the statistical distance (Mahalanobis distance) between
                        state vectors from a global object and a sensor object 
    '''
    sensorObjList = radarObjList + visionObjList
    numGlobalObjs = len(globalList)
    numSensorObjs = len(sensorObjList)
    mahalanobisMatrix = np.zeros((numSensorObjs, numGlobalObjs))
    for i in globalList:
        for j in sensorObjList:
            # innovation covariance between 2 state estimates (3.14):
            V = np.cov(np.array(globalList[i].stateVector, 
                                sensorObjList[j].stateVector).T)
            IV = np.linalg.inv(V)
            mahalanobisMatrix[j, i] = mahalanobis(globalList[i].stateVector, 
                                                  sensorObjList[j].stateVector,
                                                  IV)
    return mahalanobisMatrix

def updateExistenceProbability(mahalanobisMatrix):
