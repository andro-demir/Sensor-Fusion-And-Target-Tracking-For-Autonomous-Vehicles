'''
****************************************************************
Contributors to this code
** Andac Demir (andacdemir@gmail.com) (main developer)
****************************************************************
Radar sees so many objects and we need to delete the some.
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import Mahalanobis

def 
