import numpy as np
from .baseobject import BaseObject


class Sensor(BaseObject):
    def __init__(
        self,
        initial_location: np.array = np.zeros((3)),
        interval=0.01,
        error=0,
        id=None,
        checkpoints=[],
    ):
        super().__init__(initial_location, interval, id, checkpoints=checkpoints)
        self.error = error
