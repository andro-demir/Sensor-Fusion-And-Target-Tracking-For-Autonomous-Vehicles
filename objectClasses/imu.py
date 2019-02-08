class IMU:  
    __slots__ = ['timeStamp', 'velocity', 'yawRate']
    def __init__(self, timeStamp, velocity, yawRate):
        self.timeStamp = timeStamp
        self.velocity = velocity
        self.yawRate = yawRate