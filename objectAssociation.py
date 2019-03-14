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

class Association:
    def __init__(self, fusionList, sensorObjList):
        '''
        :param fusionList (list): objects in the global list. 
        :param sensorObjList (List): objects in the radar / vision sensor 
                                     measurements.
        '''
        self.fusionList = fusionList
        self.sensorObjList = sensorObjList
        self.numFusionObjs = len(fusionList)
        self.numSensorObjs = len(sensorObjList)

    def getMahalanobisMatrix(self):
        '''
        the statistical distance (Mahalanobis distance) between state vectors 
        from a fusion object and a sensor object is evaluated. 
        mahalanobisMatrix: The cost matrix of the bipartite graph.
        '''
        mahalanobisMatrix = np.zeros((self.numSensorObjs, 
                                      self.numFusionObjs))
        #print(self.numSensorObjs, self.numFusionObjs)
        #print("Mahalanobis matrix:\n", mahalanobisMatrix)
        for i in range(self.numFusionObjs):
            # first remove None elements from the state vector:
            self.fusionList[i].s_vector = remove_none(
                                          self.fusionList[i].s_vector)
            #print("Fusion List state vector:\n", self.fusionList[i].s_vector)
            for j in range(self.numSensorObjs):
                # first remove None elements from the state vector:
                self.sensorObjList[j].s_vector = remove_none(
                                             self.sensorObjList[j].s_vector)
                #print("Sensor state vector:\n", self.sensorObjList[j].s_vector)
                # innovation covariance between 2 state estimates (3.14):
                V = np.stack((np.asarray(self.fusionList[i].s_vector), 
                              np.asarray(self.sensorObjList[j].s_vector)), 
                              axis=0)
                V = np.cov(V.T)
                #print("covariance matrix:\n", V)
                IV = pinv(V) 
                #print("inverse of the covariance matrix:\n", IV)
                mahalanobisMatrix[j,i] = mahalanobis(self.fusionList[i].
                                                               s_vector, 
                                                  self.sensorObjList[j].
                                                          s_vector, IV)
                #print("Mahalanobis matrix:\n", mahalanobisMatrix)
                #print(40*"--")
        return mahalanobisMatrix

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
        mahalanobisMatrix = self.getMahalanobisMatrix()
        rowInd, colInd = linear_sum_assignment(mahalanobisMatrix)
        return mahalanobisMatrix, rowInd, colInd

    def updateExistenceProbability(self):
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
        mahalanobisMatrix, rowInd, colInd = self.matchObjs()
        thresh = self.getThreshold()
        alpha = self.getAlpha()
        beta = self.getBeta()
        gamma = self.getGamma()
        # reduce the probability of existence if it might be a clutter, reduce
        # its probability of existence by alpha
        for i, j in zip(rowInd, colInd):
            if mahalanobisMatrix[i, j] > thresh:
                self.fusionList[j].p_existence -= alpha
        
        # reduce the probability of existence of an object in the globalList 
        # by beta if it doesn't match with any sensor objs
        notAssignedGlobals = np.setdiff1d(colInd, 
                                          np.arange(self.numFusionObjs))
        for i in notAssignedGlobals:
            self.fusionList[i].p_existence -= beta
        
        # initilialize a new object in the global list by assigning a 
        # probability of existence (gamma), if the sensor object doesn't match
        # any objects in the globalList
        notAssignedSensors = np.setdiff1d(rowInd, 
                                          np.arange(self.numSensorObjs))
        for i in notAssignedSensors:
            self.sensorObjList[i].p_existence = gamma  
            self.fusionList.append(self.sensorObjList[i])  

        return self.fusionList 

    # Assigned 0 and 1 for simplicity in the first scenario.
    def getThreshold(self):
        return 0.0

    def getAlpha(self):
        return 0.0

    def getBeta(self):
        return 0.0

    def getGamma(self):
        return 1.0
    

def remove_none(l):
    return [x for x in l if x is not None]
    

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