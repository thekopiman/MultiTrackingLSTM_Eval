import numpy as np
from typing import Union
from ..objects import *


# def find_theta_phi(s: Sensor, t: Target):
#     dx, dy, dz = s.current_location() - t.current_location()
#     return np.arctan2(dy, dx), np.arctan(dz / np.sqrt(dx**2 + dy**2))


def find_theta_phi(s: np.array, t: np.array):
    dx, dy, dz = s - t
    return np.arctan2(dy, dx), np.arctan(dz / np.sqrt(dx**2 + dy**2))


def gaussian_noise(noise: float, theta_phi: tuple[float, float]):
    theta = theta_phi[0]
    phi = theta_phi[1]

    # Add Gaussian noise
    theta_noisy = theta + np.random.normal(0, noise)
    phi_noisy = phi + np.random.normal(0, noise)

    return theta_noisy, phi_noisy


class SimpleSimulation:
    def __init__(
        self,
        boundary: np.array = np.array([[-10, 10], [-10, 10], [-10, 10]]),
        interval=0.01,
    ):
        """
        This simulation assumes that the objects will spawn at time = 0. Deaths will not occur.
        After each object completed its sequential movements, it will remain stationary until the simulation is completed.

        Args:
            boundary (np.array, optional): _description_. Defaults to np.array([[-10, 10], [-10, 10], [-10, 10]]).
            interval (float, optional): _description_. Defaults to 0.01.
        """
        self.boundary = boundary
        self.sensors = []
        self.targets = []
        self.calculate_angles = None
        self.interval = interval

        # Run Simulation
        self.sensors_timestamps = None  # (Number of sensors, N, 3)
        self.targets_timestamps = None  # (Number of targets, N, 3)
        self.max_length = -1

        # Obtain Bearing

        self.angles = None  # (Number of sensors, Number of targets, N, 2)

    def add_sensors(self, sensors: Union[Sensor, list[Sensor]]):
        if isinstance(sensors, Sensor):
            sensors.update_interval(self.interval)
            self.sensors.append(sensors)
        elif isinstance(sensors, list):
            for sensor in sensors:
                self.add_sensors(sensor)

    def remove_sensor(self, index):
        assert len(self.sensors) > 0
        self.sensors.remove(self.sensors[index])

    def add_targets(self, targets: Union[Target, list[Target]]):
        if isinstance(targets, Target):
            targets.update_interval(self.interval)
            self.targets.append(targets)
        elif isinstance(targets, list):
            for target in targets:
                self.add_targets(target)

    def remove_target(self, index):
        assert len(self.targets) > 0
        self.targets.remove(self.targets[index])

    def __repr__(self):
        return f"Sensors: {len(self.sensors)} | Targets: {len(self.targets)}"

    def find_bearings(self, noise: float = 0.0):
        assert len(self.sensors) > 0
        assert self.max_length > 0

        self.angles = np.zeros(
            (len(self.sensors), len(self.targets), self.max_length, 2)
        )

        # Calculate the angles
        for i, sensor_positions in enumerate(self.sensors_timestamps):
            for j, target_positions in enumerate(self.targets_timestamps):
                for k in range(self.max_length):  # Iterate over timestamps
                    self.angles[i, j, k] = gaussian_noise(
                        noise, find_theta_phi(sensor_positions[k], target_positions[k])
                    )

        return self.angles

    def run(self):
        for i in self.sensors:
            i.generate_timestamps()
            self.max_length = max(
                i.return_timestamp_coordinates().shape[0], self.max_length
            )
        for i in self.targets:
            i.generate_timestamps()
            self.max_length = max(
                i.return_timestamp_coordinates().shape[0], self.max_length
            )

        self.sensors_timestamps = np.zeros((len(self.sensors), self.max_length, 3))
        self.targets_timestamps = np.zeros((len(self.targets), self.max_length, 3))

        for idx, i in enumerate(self.sensors):
            # Copy the first few [,:,]
            self.sensors_timestamps[
                idx, : i.return_timestamp_coordinates().shape[0], :
            ] = i.return_timestamp_coordinates()

            # Duplicate the last [,-1,] row
            self.sensors_timestamps[
                idx, i.return_timestamp_coordinates().shape[0] :, :
            ] = i.return_timestamp_coordinates()[-1, :]

        for idx, i in enumerate(self.targets):
            # Copy the first few [,:,]
            self.targets_timestamps[
                idx, : i.return_timestamp_coordinates().shape[0], :
            ] = i.return_timestamp_coordinates()

            # Duplicate the last [,-1,] row
            self.targets_timestamps[
                idx, i.return_timestamp_coordinates().shape[0] :, :
            ] = i.return_timestamp_coordinates()[-1, :]

    def export(self, folder_name=None):
        assert isinstance(self.sensors_timestamps, np.ndarray)
        assert isinstance(self.targets_timestamps, np.ndarray)
        assert isinstance(self.angles, np.ndarray)

        import os
        from datetime import datetime

        directory = (
            folder_name if folder_name != None else datetime.now().strftime("%Y%m%d")
        )
        os.makedirs(directory, exist_ok=True)

        np.save(
            os.path.join(directory, "sensors_coordinates.npy"), self.sensors_timestamps
        )
        np.save(
            os.path.join(directory, "targets_coordinates.npy"), self.targets_timestamps
        )
        np.save(os.path.join(directory, "angles.npy"), self.angles)

        print("Export Completed!")
