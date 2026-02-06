import numpy as np
from .basemovement import BaseMovement


class LinearConstantVelocity(BaseMovement):
    def __init__(
        self, velocity: np.float32 = 10, direction=np.array([1, 1, 1], dtype=np.float32)
    ):
        self.velocity = velocity
        self.direction = self.direction_normalisation(direction)

    def update_velocity(self, velocity: np.float32):
        self.velocity = velocity

    def direction_normalisation(self, direction):
        norm = np.linalg.norm(direction)

        if norm != 0:
            v = direction / norm
        else:
            v = np.zeros_like(direction)

        return v

    def additive_vector(self, time):
        return self.direction * time * self.velocity
