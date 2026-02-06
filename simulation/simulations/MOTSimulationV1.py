import numpy as np
from typing import Union
from ..objects import *
import itertools
import random
from ..movement.linearconstantvelocity import LinearConstantVelocity


def find_azimuth(s: np.array, t: np.array):
    dx, dy = s - t
    return np.arctan2(dy, dx)


def find_azimuth_elevation(s: np.array, t: np.array):
    dx, dy, dz = s - t
    return np.arctan2(dy, dx), np.arctan(dz / np.sqrt(dx**2 + dy**2))


def gaussian_noise(noise: float, angle: np.ndarray):
    noisy_angle = angle + np.random.normal(0, noise, size=angle.shape)
    return (noisy_angle + np.pi) % (2 * np.pi) - np.pi  # Keep in range [-π, π]


class MOTSimulationV1:
    def __init__(
        self,
        interval=0.1,
        dimension=np.array([[0, 100], [0, 100], [0, 100]]),
        sensor_radius=np.array([[-5, 5], [-5, 5], [-5, 5]]),
        target_radius=np.array([[-5, 5], [-5, 5], [-5, 5]]),
        ThreeD=True,
    ):
        assert dimension.shape[0] == (3 if ThreeD else 2)
        assert sensor_radius.shape[0] == (3 if ThreeD else 2)
        assert target_radius.shape[0] == (3 if ThreeD else 2)

        self.ThreeD = ThreeD

        self.dimension = dimension
        self.sensor_radius = sensor_radius
        self.target_radius = target_radius
        self.unique_id_count = itertools.count()
        self.sensors_checkpoints = []
        self.targets_checkpoints = []
        self.interval = interval

        self.sensors = []
        self.targets = []

    def reset(self):
        self.sensors_checkpoints = []
        self.targets_checkpoints = []
        self.unique_id_count = itertools.count()

        self.sensors = []
        self.targets = []

    def generate_checkpoints(
        self,
        no_sensors_checkpoints=np.random.poisson(10),
        no_targets_checkpoints=np.random.poisson(10),
    ):
        # Minimum of 2 checkpoints
        no_sensors_checkpoints = max(no_sensors_checkpoints, 2)
        no_targets_checkpoints = max(no_targets_checkpoints, 2)

        # Sensors
        for i in range(no_sensors_checkpoints):
            x = np.random.uniform(low=self.dimension[0, 0], high=self.dimension[0, 1])
            y = np.random.uniform(low=self.dimension[1, 0], high=self.dimension[1, 1])

            if self.ThreeD:
                z = np.random.uniform(
                    low=self.dimension[2, 0], high=self.dimension[2, 1]
                )
                self.sensors_checkpoints.append(np.array((x, y, z)))
            else:
                self.sensors_checkpoints.append(np.array((x, y)))

        # Targets
        for i in range(no_targets_checkpoints):
            x = np.random.uniform(low=self.dimension[0, 0], high=self.dimension[0, 1])
            y = np.random.uniform(low=self.dimension[1, 0], high=self.dimension[1, 1])

            if self.ThreeD:
                z = np.random.uniform(
                    low=self.dimension[2, 0], high=self.dimension[2, 1]
                )
                self.targets_checkpoints.append(np.array((x, y, z)))
            else:
                self.targets_checkpoints.append(np.array((x, y)))

    def spawn_sensors(
        self,
        distribution=lambda: max(np.random.poisson(3), 3),
        error=lambda: np.random.poisson(5),
    ):
        """
        Spawn Sensors.

        Each sensor will visit U(2, N) checkpoints - inclusive of the starting location.

        Args:
            distribution (lambda random distribution, optional): Number of sensors being spawned. Defaults to lambda:max(np.random.poisson(3), 3).
            error (lambda random distribution, optional): Errors in terms of Degree, not radians. Defaults to lambda:np.random.poisson(5).
        """
        for _ in range(distribution()):
            no_of_checkpoints = random.randint(2, len(self.sensors_checkpoints))
            checkpoints = random.sample(self.sensors_checkpoints, no_of_checkpoints)

            # Deviaition from the initial preset checkpoints
            for checkpoint in checkpoints:
                if self.ThreeD:
                    checkpoint += np.array(
                        [
                            np.random.uniform(
                                low=self.sensor_radius[0, 0],
                                high=self.sensor_radius[0, 1],
                            ),
                            np.random.uniform(
                                low=self.sensor_radius[1, 0],
                                high=self.sensor_radius[1, 1],
                            ),
                            np.random.uniform(
                                low=self.sensor_radius[2, 0],
                                high=self.sensor_radius[2, 1],
                            ),
                        ]
                    )
                else:
                    checkpoint += np.array(
                        [
                            np.random.uniform(
                                low=self.sensor_radius[0, 0],
                                high=self.sensor_radius[0, 1],
                            ),
                            np.random.uniform(
                                low=self.sensor_radius[1, 0],
                                high=self.sensor_radius[1, 1],
                            ),
                        ]
                    )

            self.sensors.append(
                Sensor(
                    error=error(),
                    id=next(self.unique_id_count),
                    initial_location=checkpoints[0],
                    interval=self.interval,
                    checkpoints=checkpoints[1:],
                )
            )

    def spawn_targets(
        self,
        distribution=lambda: max(np.random.poisson(3), 3),
    ):
        """
        Spawn targets

        Each target will visit U(2, N) checkpoints - inclusive of the starting location.

        Args:
            distribution (lambda random distribution, optional): Number of sensors being spawned. Defaults to lambda:max(np.random.poisson(3), 3).
        """
        for _ in range(distribution()):
            no_of_checkpoints = random.randint(2, len(self.targets_checkpoints))
            checkpoints = random.sample(self.targets_checkpoints, no_of_checkpoints)

            for checkpoint in checkpoints:
                if self.ThreeD:
                    checkpoint += np.array(
                        [
                            np.random.uniform(
                                low=self.target_radius[0, 0],
                                high=self.target_radius[0, 1],
                            ),
                            np.random.uniform(
                                low=self.target_radius[1, 0],
                                high=self.target_radius[1, 1],
                            ),
                            np.random.uniform(
                                low=self.target_radius[2, 0],
                                high=self.target_radius[2, 1],
                            ),
                        ]
                    )
                else:
                    checkpoint += np.array(
                        [
                            np.random.uniform(
                                low=self.target_radius[0, 0],
                                high=self.target_radius[0, 1],
                            ),
                            np.random.uniform(
                                low=self.target_radius[1, 0],
                                high=self.target_radius[1, 1],
                            ),
                        ]
                    )

            self.targets.append(
                Target(
                    id=next(self.unique_id_count),
                    initial_location=checkpoints[0],
                    interval=self.interval,
                    checkpoints=checkpoints[1:],
                )
            )

    def generate_paths(
        self,
        sensor_speed_distribution=lambda: np.random.normal(20, 5),
        target_speed_distribution=lambda: np.random.normal(20, 5),
        truncation=None,
    ):
        """
        Autogenerate the paths of each object based on the checkpoints it has to traverse to.
        """
        for obj in self.sensors + self.targets:
            speed_dist = None
            if isinstance(obj, Sensor):
                speed_dist = sensor_speed_distribution
            elif isinstance(obj, Target):
                speed_dist = target_speed_distribution
            else:
                assert False, "The object has to be either Target or Sensor"

            if not obj.checkpoints:
                obj.update_sequential_movement([], True)
                continue

            points = [obj.location] + obj.checkpoints
            sequentials = []

            for i in range(len(points) - 1):
                start = np.array(points[i])
                end = np.array(points[i + 1])
                displacement = end - start

                distance = np.linalg.norm(displacement)  # Euclidean distance
                velocity = max(0.00001, speed_dist())  # Minimally 1m/s

                if distance == 0:
                    direction = np.zeros_like(start)  # No movement
                    duration = 0
                else:
                    direction = displacement / distance  # Unit vector
                    duration = distance / velocity  # Time = Distance / Speed

                sequentials.append(
                    (
                        duration,
                        LinearConstantVelocity(
                            velocity=velocity,
                            direction=direction,
                        ),
                    )
                )
            obj.update_sequential_movement(sequentials, False)
            obj.generate_timestamps(truncation)
            obj.generate_velocities()

    def run(self):
        self.max_length = 0
        for i in self.sensors:
            self.max_length = max(
                i.return_timestamp_coordinates().shape[0], self.max_length
            )
        for i in self.targets:
            self.max_length = max(
                i.return_timestamp_coordinates().shape[0], self.max_length
            )

        self.sensors_timestamps = np.zeros(
            (len(self.sensors), self.max_length, 3 if self.ThreeD else 2)
        )
        self.targets_timestamps = np.zeros(
            (len(self.targets), self.max_length, 3 if self.ThreeD else 2)
        )
        self.sensors_velocities = np.zeros(
            (len(self.sensors), self.max_length, 3 if self.ThreeD else 2)
        )
        self.targets_velocities = np.zeros(
            (len(self.targets), self.max_length, 3 if self.ThreeD else 2)
        )

        for idx, i in enumerate(self.sensors):
            # Copy the first few [,:,]
            self.sensors_timestamps[
                idx, : i.return_timestamp_coordinates().shape[0], :
            ] = i.return_timestamp_coordinates()
            self.sensors_velocities[
                idx, : i.return_timestamp_velocities().shape[0], :
            ] = i.return_timestamp_velocities()

            # Duplicate the last [,-1,] row
            self.sensors_timestamps[
                idx, i.return_timestamp_coordinates().shape[0] :, :
            ] = i.return_timestamp_coordinates()[-1, :]

        for idx, i in enumerate(self.targets):
            # Copy the first few [,:,]
            self.targets_timestamps[
                idx, : i.return_timestamp_coordinates().shape[0], :
            ] = i.return_timestamp_coordinates()
            self.targets_velocities[
                idx, : i.return_timestamp_velocities().shape[0], :
            ] = i.return_timestamp_velocities()

            # Duplicate the last [,-1,] row
            self.targets_timestamps[
                idx, i.return_timestamp_coordinates().shape[0] :, :
            ] = i.return_timestamp_coordinates()[-1, :]

    def __repr__(self):
        return f"Sensors : {len(self.sensors)} | Targets : {len(self.targets)}."

    def find_bearings(self):
        assert len(self.sensors) > 0
        assert self.max_length > 0

        self.angles = np.zeros(
            (
                len(self.sensors),
                len(self.targets),
                self.max_length,
                2 if self.ThreeD else 1,
            )
        )

        # Calculate the angles
        for i, sensor_positions in enumerate(self.sensors_timestamps):
            for j, target_positions in enumerate(self.targets_timestamps):
                for k in range(self.max_length):  # Iterate over timestamps

                    if self.ThreeD:
                        angle = find_azimuth_elevation(
                            sensor_positions[k], target_positions[k]
                        )

                    else:
                        angle = find_azimuth(sensor_positions[k], target_positions[k])

                    try:
                        self.angles[i, j, k] = gaussian_noise(
                            np.radians(self.sensors[i].error), angle
                        )
                    except AttributeError as e:
                        print(find_azimuth(sensor_positions[k], target_positions[k]))
                        print(e)
                        raise Exception(
                            "Debug: ", angle, sensor_positions[k], target_positions[k]
                        )

        return self.angles


if __name__ == "__main__":
    pass
