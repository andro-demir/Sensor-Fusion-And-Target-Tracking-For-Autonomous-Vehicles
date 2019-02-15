class vehicle:
    __slots__ = ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y']
    def __init__(self, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.acc_x = acc_x
        self.acc_y = acc_y


class radar():  
    __slots__ = ['timeStamp', 'obj', 'numObjects']
    def __init__(self, timeStamp, obj, numObjects):
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects


class vision():  
    __slots__ = ['timeStamp', 'obj', 'numObjects']
    def __init__(self, timeStamp, obj, numObjects):
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects


class lane():  
    __slots__ = ['left', 'right']
    def __init__(self, left, right):
        self.left = left
        self.right = right


class IMU():  
    __slots__ = ['timeStamp', 'velocity', 'yawRate']
    def __init__(self, timeStamp, velocity, yawRate):
        self.timeStamp = timeStamp
        self.velocity = velocity
        self.yawRate = yawRate