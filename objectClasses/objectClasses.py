from numpy import asarray
'''
    posX, posY, velX, velY, accX, accY: 
        Obstacle's location and dynamic information relative to the ego 
        vehicle's coordinate system.
    yaw: Orientation (yaw angle)
    yawRt: Yaw angle rate (angular velocity)
    P: State covariance matrix
    dim: Estimated object dimension vector
    dimUncertainty: Dimension uncertainty vector
    pExistence: Probability of existence
    c: Classification vector
    f: Feature vector
'''
class obstacle:
    __slots__ = ['posX', 'posY', 'velX', 'velY', 'accX', 'accY', 'yaw', 
                 'yawRt', 'P', 'dim', 'dimUncertainty', 'pExistence', 'c', 'f',
                 'stateVector']
    def __init__(self, posX, posY, velX, velY, accX, accY, yaw, yawRt, P, dim,
                 dimUncertainty, pExistence, c, f, stateVector):
        self.posX = posX
        self.posY = posY
        self.velX = velX
        self.velY = velY
        self.accX = accX
        self.accY = accY
        self.yaw = yaw
        self.yawRt = yawRt
        self.stateVector = asarray([posX, posY, velX, velY, accX, accY, 
                                    yaw, yawRt])
        self.P = P
        self.dim = dim
        self.dimUncertainty = dimUncertainty
        self.pExistence = pExistence
        self.c = c 
        self.f = f


class radar:  
    __slots__ = ['timeStamp', 'obj', 'numObjects']
    def __init__(self, timeStamp, obj, numObjects):
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects


class vision:  
    __slots__ = ['timeStamp', 'obj', 'numObjects']
    def __init__(self, timeStamp, obj, numObjects):
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects


class lane:  
    __slots__ = ['left', 'right']
    def __init__(self, left, right):
        self.left = left
        self.right = right


class IMU:  
    __slots__ = ['timeStamp', 'velocity', 'yawRate']
    def __init__(self, timeStamp, velocity, yawRate):
        self.timeStamp = timeStamp
        self.velocity = velocity
        self.yawRate = yawRate