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
class Obstacle:
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


'''
Sensor Class and Subclasses Require Edits...
'''
class Sensor:
    def __init__(self):
        pass

    def spatialAlignment(self):
        pass

class Radar(Sensor):  
    def __init__(self, timeStamp, obj, numObjects):
        super().__init__()
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects
    
    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(Radar, self).spatialAlignment()


class Vision(Sensor):  
    def __init__(self, timeStamp, obj, numObjects):
        super().__init__()
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects
    
    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(Vision, self).spatialAlignment()


class Lane(Sensor):  
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(Lane, self).spatialAlignment()


class IMU(Sensor):  
    def __init__(self, timeStamp, velocity, yawRate):
        super().__init__()
        self.timeStamp = timeStamp
        self.velocity = velocity
        self.yawRate = yawRate
    
    # calls Sensor.spatialAlignment()
    def spatialAlignment(self):
        super(IMU, self).spatialAlignment()