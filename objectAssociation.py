'''
****************************************************************
Contributors to this code
** Andac Demir (andacdemir@gmail.com) (main developer)
****************************************************************
'''

import numpy as np
from scipy.spatial.distance import Mahalanobis
from scipy.optimize import linear_sum_assignment

class Association():
    def __init__(self, globalList, radarObjList, visionObjList):
        '''
        :param globalList (list): objects in the global list 
        :param radarObjList (List): objects in the radar sensor measurements
        :param visionObjList (List): objects in the vision sensor measurements
        '''
        self.globalList = globalList
        self.radarObjList = radarObjList
        self.visionObjList = visionObjList
        self.sensorObjList = radarObjList + visionObjList
        self.numGlobalObjs = len(globalList)
        self.numSensorObjs = len(self.sensorObjList)
        self.mahalanobisMatrix = np.zeros((self.numSensorObjs, 
                                           self.numGlobalObjs))

    def getMahalanobisMatrix(self):
        '''
        the statistical distance (Mahalanobis distance) between state vectors 
        from a global object and a sensor object is evaluated. 
        '''
        for i in self.numGlobalObjs:
            for j in self.numSensorObjs:
                # innovation covariance between 2 state estimates (3.14):
                V = np.cov(np.array(self.globalList[i].stateVector, 
                                    self.sensorObjList[j].stateVector).T)
                IV = np.linalg.inv(V)
                self.mahalanobisMatrix[j, i] = Mahalanobis(self.globalList[i].
                                                                  stateVector, 
                                                          self.sensorObjList[j]
                                                              .stateVector, IV)

    def matchObjs(self):
        '''
        :return rowInd, colInd (np.array): An array of row indices and one of  
                                           corresponding column indices giving 
                                           the optimal assignment. 
        This function applies the linear sum assignment problem, also known as 
        minimum weight matching in bipartite graphs using Hungarian Method. 
        Given a problem instance is described by a matrix cost matrix, 
        where each C[i,j] is the cost of matching vertex i of the first partite set 
        (a “worker”) and vertex j of the second set (a “job”). 
        The goal is to find a complete assignment of workers to jobs of 
        minimal cost.
        '''
        self.rowInd, self.colInd = linear_sum_assignment(self.mahalanobisMatrix)

    def updateExistenceProbability(self, mahalanobisMatrix, globalList, thresh, 
                                   alpha, beta, gamma, rowInd, colInd):
        '''
        :param mahalonobisMatrix (np.array): The cost matrix of the bipartite graph
        :param globalList (list): objects in the global list 
        :param thresh (double): threshold level. If the cost is bigger than this,
                                it might be a clutter - reduce the probability of
                                existence by alpha.
        :param alpha, beta, gamma (double): coeffs to update the probabilities of 
                                            existence
        If a row (a sensor object) is not assigned to a column (an object in 
        the global list), it may be a new object in the environment. Initialize a 
        new object with probability of existence: beta.
        :return globalList (list): updated globalList of obstacles
        '''
        alpha = self.getAlpha()
        beta = self.getBeta()
        thresh = self.getThreshold()
        # reduce the probability of existence if it might be a clutter, reduce
        # its probability of existence by alpha
        for i, j in zip(self.rowInd, self.colInd):
            if self.mahalanobisMatrix[i, j] > thresh:
                self.globalList[j].pExistence -= alpha
        
        # reduce the probability of existence of an object in the globalList 
        # by beta if it doesn't match with any sensor objs
        notAssignedGlobals = np.setdiff1d(self.colInd, 
                                          np.arange(len(self.globalList)))
        for i in notAssignedGlobals:
            self.globalList[i].pExistence -= beta
        
        # initilialize a new object in the global lists by assigning a 
        # probability of existence (gamma), if the sensor object doesn't match
        # any objects in the globalList
        notAssignedSensors = np.setdiff1d(self.rowInd, np.arange(
                                self.mahalanobisMatrix.shape[0]))
        numRadarObjs = len(self.radarObjList)
        for i in notAssignedSensors:
            if i < numRadarObjs:
                self.radarObjList[i].pExistence = gamma  
                self.globalList.append(self.radarObjList[i])
            else:
                self.visionObjList[i-numRadarObjs].pExistence = gamma
                self.globalList.append(self.visionObjList[i-numRadarObjs])     

    def getThreshold(self):
        pass

    def getAlpha(self):
        pass

    def getBeta(self):
        pass

    def getGamma(self):
        pass
