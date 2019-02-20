'''
****************************************************************
Contributors to this code
** Andac Demir (andacdemir@gmail.com) (main developer)
****************************************************************
'''

import numpy as np
from scipy.spatial.distance import Mahalanobis
from scipy.optimize import linear_sum_assignment

class Association:
    __slots__ = ['fusionList', 'sensorObjList', 'numfusionObjs', 
                 'numSensorObjs', 'mahalanobisMatrix']
    def __init__(self, fusionList, sensorObjList):
        '''
        :param fusionList (list): objects in the global list. 
        :param sensorObjList (List): objects in the radar / vision sensor 
                                     measurements.
        :mahalanobisMatrix: The cost matrix of the bipartite graph.
        '''
        self.fusionList = fusionList
        self.sensorObjList = sensorObjList
        self.numfusionObjs = len(fusionList)
        self.numSensorObjs = len(self.sensorObjList)
        self.mahalanobisMatrix = np.zeros((self.numSensorObjs, 
                                         self.numfusionObjs))

    def getMahalanobisMatrix(self):
        '''
        the statistical distance (Mahalanobis distance) between state vectors 
        from a fusion object and a sensor object is evaluated. 
        '''
        for i in self.numfusionObjs:
            for j in self.numSensorObjs:
                # innovation covariance between 2 state estimates (3.14):
                V = np.cov(np.array(self.fusionList[i].stateVector, 
                                    self.sensorObjList[j].stateVector).T)
                IV = np.linalg.inv(V)
                self.mahalanobisMatrix[j, i] = Mahalanobis(self.fusionList[i].
                                                                  stateVector, 
                                                        self.sensorObjList[j].
                                                             stateVector, IV)

    def matchObjs(self):
        '''
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
        self.getMahalanobisMatrix()
        rowInd, colInd = linear_sum_assignment(self.mahalanobisMatrix)
        return rowInd, colInd

    def updateExistenceProbability(self, thresh, alpha, beta, gamma):
        '''
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
        rowInd, colInd = self.matchObjs()
        alpha = self.getAlpha()
        beta = self.getBeta()
        thresh = self.getThreshold()
        # reduce the probability of existence if it might be a clutter, reduce
        # its probability of existence by alpha
        for i, j in zip(rowInd, colInd):
            if self.mahalanobisMatrix[i, j] > thresh:
                self.fusionList[j].pExistence -= alpha
        
        # reduce the probability of existence of an object in the globalList 
        # by beta if it doesn't match with any sensor objs
        notAssignedGlobals = np.setdiff1d(colInd, 
                                          np.arange(len(self.fusionList)))
        for i in notAssignedGlobals:
            self.fusionList[i].pExistence -= beta
        
        # initilialize a new object in the global list by assigning a 
        # probability of existence (gamma), if the sensor object doesn't match
        # any objects in the globalList
        notAssignedSensors = np.setdiff1d(rowInd, np.arange(
                                            self.mahalanobisMatrix.shape[0]))
        for i in notAssignedSensors:
            self.sensorObjList[i].pExistence = gamma  
            self.fusionList.append(self.sensorObjList[i])   

    # Assigned 0 and 1 for simplicity in the first scenario.
    def getThreshold(self):
        return 0.0

    def getAlpha(self):
        return 0.0

    def getBeta(self):
        return 0.0

    def getGamma(self):
        return 1.0


'''
For data association --> do:

# fusionList is the obstacles(objs) in the global list
# and sensorObjList is the obstacles(objs) seen by a single sensor only
Iterate for all sensorObjList(sensors): 
    assc = Association(fusionList, sensorObjList)
    assc.updateExistenceProbability()
    # updated fusion list then becomes:
    fusionList = assc.fusionList
'''