class radar:  
    __slots__ = ['timeStamp', 'obj', 'numObjects']
    def __init__(self, timeStamp, obj, numObjects):
        self.timeStamp = timeStamp
        self.obj = obj
        self.numObjects = numObjects