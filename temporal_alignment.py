import numpy as np


def temporal_alignment(obj, current_time):
    """

    :param obj:
    :param current_time:
    :return:
    """
    for _ in range(current_time, obj.timeStamp): # :TODO not complete, add u and w 
        obj.last_state = obj.x
        obj.x = np.dot(obj.F, obj.x)
        obj.P = np.dot(np.dot(obj.F, obj.P), obj.F.T) + obj.Q

    return
