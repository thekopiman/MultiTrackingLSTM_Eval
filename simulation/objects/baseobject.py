import numpy as np

from ..movement.basemovement import BaseMovement


class BaseObject:
    def __init__(
        self,
        initial_location: np.array = np.zeros((3)),
        interval=0.01,
        id=None,
        checkpoints=[],
    ):
        self.location = initial_location

        # [(duration, movement)...]
        self.sequential = []
        self.interval = interval
        self.timestamp_coordinates = None
        self.timestamp_velocities = None
        self.id = id
        self.current_velocity = np.zeros_like(initial_location)
        self.checkpoints = checkpoints

    def _is_alive(self):
        # This is effectively a transition matrix
        if self.alive and np.random.uniform(0, 1) <= self.death_rate:
            self.alive = False
        elif not self.alive and np.random.uniform(0, 1) <= self.birth_rate:
            self.alive = True

    def current_location(self):
        return self.location

    def update_interval(self, interval):
        self.interval = interval

    def __repr__(self) -> np.array:
        return f"{self.current_location()}"

    def update_location(self, new_location: np.array):
        self.current_velocity = (new_location - self.location) / self.interval
        self.location = new_location

    def update_sequential_movement(self, lst, auto_generate: bool = False):
        assert all(
            isinstance(i[0], (np.integer, np.float32, int, float))
            and isinstance(i[1], BaseMovement)
            for i in lst
        ), "Each element in the lst should be in the format of (duration, BaseMovement)"

        self.sequential = lst

        if auto_generate:
            self.generate_timestamps()
            self.generate_velocities()
            return self.timestamp_coordinates, self.timestamp_velocities

        return None

    def generate_timestamps(self, truncation=None):
        if not self.sequential:
            self.timestamp_coordinates = np.expand_dims(self.location, axis=0)
            return

        total_duration = sum(map(lambda x: x[0], self.sequential))

        max_total_duration = int(total_duration / self.interval)
        if truncation is not None:
            max_total_duration = min(truncation, int(total_duration / self.interval))

        self.timestamp_coordinates = np.zeros(
            (max_total_duration, self.location.shape[0])
        )

        curr_idx = 0
        for duration, movement in self.sequential:
            timestamps = np.arange(0, int(duration / self.interval)) * self.interval
            result = np.array(
                [self.location + movement.additive_vector(t) for t in timestamps]
            )
            if result.size > 0:
                final_idx = curr_idx + min(
                    int(duration / self.interval),
                    self.timestamp_coordinates[curr_idx:, :].shape[0],
                )
                try:
                    self.timestamp_coordinates[curr_idx:final_idx, :] = result

                    self.location = np.add(
                        self.location, movement.additive_vector(duration)
                    )
                except ValueError:
                    self.timestamp_coordinates[curr_idx:final_idx, :] = result[
                        : final_idx - curr_idx, :
                    ]
                    self.location = np.add(
                        self.location,
                        movement.additive_vector(
                            (final_idx - curr_idx) * self.interval
                        ),
                    )
                    break

                curr_idx = final_idx
        try:
            self.timestamp_coordinates[-1, :] = self.location
        except IndexError:
            self.timestamp_coordinates = np.expand_dims(self.location, axis=0)

    def return_timestamp_coordinates(self):
        # assert not isinstance(self.timestamp_coordinates, (None))
        return self.timestamp_coordinates

    def generate_velocities(self):
        self.timestamp_velocities = np.zeros_like(self.timestamp_coordinates)
        self.timestamp_velocities[1:, :] = (
            self.timestamp_coordinates[1:, :] - self.timestamp_coordinates[:-1, :]
        ) / self.interval

    def return_timestamp_velocities(self):
        # assert not isinstance(self.timestamp_coordinates, (None))
        return self.timestamp_velocities


if __name__ == "__main__":
    obj = BaseObject()
