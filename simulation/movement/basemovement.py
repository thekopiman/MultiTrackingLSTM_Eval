import numpy as np


class BaseMovement:
    def __init__(self):
        pass

    def additive_vector(self, time):
        return np.array([0, 0, 0])
