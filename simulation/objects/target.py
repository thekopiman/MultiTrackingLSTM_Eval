import numpy as np
from .baseobject import BaseObject


class Target(BaseObject):
    def __init__(
        self,
        initial_location: np.array = np.zeros((3)),
        interval=0.01,
        id=None,
        checkpoints=[],
    ):
        super().__init__(initial_location, interval, id, checkpoints=checkpoints)
