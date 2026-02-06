# import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor

from .MOTSimulationV1 import MOTSimulationV1


def generate_labels(training_tensor, truncated_targets_timestamps, interval=0.1):
    labels = []
    for i in training_tensor:
        B = i.shape[0]
        idx = i[0, 0, -1] / interval
        idx = idx.cpu().int()
        expanded_tensor = (
            truncated_targets_timestamps[:, idx, :].unsqueeze(0).expand(B, -1, -1)
        )
        labels.append(expanded_tensor)
    return labels


def assignment_from_unique_id(unique_ids: list, M, N):
    res = []
    for id in unique_ids:
        _, cur_M = id.shape
        after_one_hot = torch.nn.functional.one_hot(id.long(), num_classes=N).permute(
            0, 2, 1
        )

        pad_size = M - cur_M + 1
        if pad_size > 0:
            padded_one_hot = torch.nn.functional.pad(after_one_hot, (0, pad_size))
        else:
            padded_one_hot = after_one_hot

        total_sum = padded_one_hot.sum(dim=-1, keepdim=True)
        nonzero_mask = total_sum != 0

        # Normalize where sum is nonzero
        normalized = torch.where(
            nonzero_mask, padded_one_hot / total_sum, padded_one_hot
        )

        # Set the last element to 1 where sum is zero
        normalized[..., -1] = torch.where(
            nonzero_mask.squeeze(-1), normalized[..., -1], 1
        )
        res.append(normalized)

    return res


def pad_input_tensor(training_tensor: list, M):
    res = []
    for i in training_tensor:
        cur_M = i.shape[1]
        pad_size = M - cur_M
        if pad_size > 0:
            t1 = i.permute(0, 2, 1)
            t1_pad = torch.nn.functional.pad(t1, (0, pad_size)).permute(0, 2, 1)
        else:
            t1_pad = i
        res.append(t1_pad)
    return res


def attach_time(data: np.ndarray, interval):
    B, M, N, t, k = data.shape  # Extract current shape
    new_row = np.arange(t) * interval  # Shape (t,)

    # Reshape to (M, N, t, 1) for broadcasting
    new_row = new_row.reshape(1, 1, 1, t, 1)

    # Repeat across M and N to match the shape
    new_row = np.tile(new_row, (B, M, N, 1, 1))

    # Concatenate along the last axis (k)
    return np.concatenate([data, new_row], axis=-1)


def split_tensor(A, ids):
    """
    Splits tensor A of shape (B, T, 4) into a list of tensors (B, t_varies, 4),
    where each segment corresponds to a unique timestamp block.
    """
    B, T, C = A.shape  # C should be 4

    # Extract timestamps from the last column of any batch element (e.g., A[0, :, -1])
    timestamps = A[0, :, -1]

    # Find indices where timestamps increase
    break_indices = torch.where(timestamps[1:] > timestamps[:-1])[0] + 1

    # Include start and end indices
    indices = [0] + break_indices.tolist() + [T]

    # Split tensor based on indices
    split_tensors = [
        A[:, indices[i] : indices[i + 1], :] for i in range(len(indices) - 1)
    ]
    ids_split_tensors = [
        ids[:, indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
    ]

    return split_tensors, ids_split_tensors


class DataGenerator:
    def __init__(
        self,
        params,
    ):
        self.params = params
        self.device = params.training.device
        # self.pool = multiprocessing.Pool()
        self.truncation = params.data_generation.truncation
        self.batch = int(params.training.batch_size)
        self.interval = params.data_generation.interval
        self.p = params.data_generation.p

        # Put this in Params in the future
        np.random.seed(params.training.seed)

        if params.data_generation.simulation_generator == "MOTSimulationV1":
            self.datagen = MOTSimulationV1(
                dimension=np.array(params.data_generation.dimension),
                sensor_radius=np.array(params.data_generation.sensor_radius),
                target_radius=np.array(params.data_generation.target_radius),
                ThreeD=params.data_generation.ThreeD,
                interval=self.interval,
            )

    def get_measurements(self, raw_data: tuple):
        """_summary_

        Args:
            raw_data (tuple): Containing =
            (
            truncated_sensors_timestamps,  # (M, t, 2/3)
            truncated_targets_timestamps,  # (N, t, 2/3)
            truncated_sensors_velocities,  # (M, t, 2/3)
            truncated_targets_velocities,  # (N, t, 2/3)
            truncated_angles_array,  # (B, M, N, t, 1/2)
            )

        Returns:
            training_nested_tensor (list): Each element consist of bearing-only
             measurements from the sensor together with the timestamp
            labels (list): Each element contains the expected coordinates and velocities
            of the targets
            unique_measuurement_ids (list): Target id corresponding to each bearing
            reading.
        """
        (
            truncated_sensors_timestamps,  # (M, t, 2/3)
            truncated_targets_timestamps,  # (N, t, 2/3)
            truncated_sensors_velocities,  # (M, t, 2/3) Redundant
            truncated_targets_velocities,  # (N, t, 2/3)
            truncated_angles_array,  # (B, M, N, t, 1/2)
        ) = raw_data
        truncated_angles_array = attach_time(truncated_angles_array, self.interval)

        training_nested_tensor = []
        labels = []
        unique_measuurement_ids = []
        targets_coordinates = []

        t1, l1, u_id, target = self._step(
            truncated_sensors_timestamps,
            truncated_targets_timestamps,
            truncated_targets_velocities,
            truncated_angles_array,
        )

        training_nested_tensor = t1
        labels = [l1] * len(t1)
        unique_measuurement_ids = [u_id] * len(t1)
        targets_coordinates = [target] * len(t1)

        return (
            training_nested_tensor,
            labels,
            unique_measuurement_ids,
            targets_coordinates,
        )

    def _step(
        self,
        split_sensors_timestamps,  # (M, t, 2/3)
        split_targets_timestamps,  # (N, t, 2/3)
        split_targets_velocities,  # (N, t, 2/3)
        split_angles_array,  # (B, M, N, t, 2/3) Time have been attached at the end
        #  of [:,:,:,-1]
    ):
        """Shuffling the mini dataset at each t"""
        total_duration = split_sensors_timestamps.shape[1]
        total_sensors = split_sensors_timestamps.shape[0]
        total_targets = split_targets_timestamps.shape[0]
        batch_size = split_angles_array.shape[0]

        final_measurement = [np.array([]) for _ in range(batch_size)]
        final_target_coordinates = np.array([])
        final_unique_ids = np.array([], dtype="int64")

        for time in range(total_duration):
            measurement_array = [[] for _ in range(batch_size)]
            target_array = []
            unique_target_ids = []
            for s in range(total_sensors):
                for t in range(total_targets):
                    # Target is not selected
                    if not self._bool_select_bearing():
                        break

                    sensor_coordinate = split_sensors_timestamps[s, time, :]
                    target_coordinate = split_targets_timestamps[t, time, :]
                    target_array.append(target_coordinate)
                    unique_target_ids.append(t)

                    # Batch based bearingg split
                    for b in range(batch_size):
                        bearing_and_time = split_angles_array[b, s, t, time, :]
                        sensor_measurements = np.concatenate(
                            [sensor_coordinate, bearing_and_time]
                        )
                        measurement_array[b].append(sensor_measurements)  # wrong here

            random_idx = np.random.permutation(len(measurement_array[0]))
            measurement_array = [np.array(i)[random_idx] for i in measurement_array]
            unique_target_ids = np.array(unique_target_ids)[random_idx]
            target_array = np.array(target_array)[random_idx]

            # There might be situations where there are no bearing measurements in certain t
            for idx, batch_measurement_array in enumerate(measurement_array):
                if batch_measurement_array.size > 0:
                    final_measurement[idx] = (
                        np.vstack([final_measurement[idx], batch_measurement_array])
                        if final_measurement[idx].size > 0
                        else batch_measurement_array
                    )

            if measurement_array[0].size > 0:
                final_target_coordinates = (
                    np.vstack([final_target_coordinates, target_array])
                    if final_target_coordinates.size > 0
                    else target_array
                )

                final_unique_ids = np.hstack([final_unique_ids, unique_target_ids])

        label = np.concatenate(
            [
                split_targets_timestamps[:, -1, :],
                split_targets_velocities[:, -1, :],
            ],
            axis=-1,
        )

        return final_measurement, label, final_unique_ids, final_target_coordinates

    def get_batch(self):
        self.raw_data = get_single_training_example(
            self.params, self.datagen, self.truncation
        )

        training_data, labels, unique_measurement_ids, target_coordinates = (
            self.get_measurements(self.raw_data)
        )
        labels = [Tensor(l).to(torch.device(self.device)) for l in labels]
        unique_measurement_ids = [list(u) for u in unique_measurement_ids]

        # Pad training data
        max_len = max(list(map(len, training_data)))
        training_data, _ = pad_to_batch_max(training_data, max_len)
        target_coordinates, _ = pad_to_batch_max(target_coordinates, max_len)

        # Pad unique ids
        for i in range(len(unique_measurement_ids)):
            unique_id = unique_measurement_ids[i]
            n_items_to_add = max_len - len(unique_id)
            unique_measurement_ids[i] = np.concatenate(
                [unique_id, [-2] * n_items_to_add]
            )[None, :]
        unique_measurement_ids = np.concatenate(unique_measurement_ids)

        training_nested_tensor = (
            Tensor(training_data).to(torch.float32).to(torch.device(self.device))
        )

        unique_measurement_ids = Tensor(unique_measurement_ids).to(self.device)
        target_coordinates = Tensor(target_coordinates).to(self.device)

        new_training_tensor, new_unique_measurement_ids = split_tensor(
            training_nested_tensor, unique_measurement_ids
        )

        return (
            new_training_tensor,
            new_unique_measurement_ids,
            target_coordinates,
        )

    def _bool_select_bearing(self) -> bool:
        x = np.random.uniform(0, 1)
        return x <= self.p


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i, : len(ex), :] = ex
        mask[i, : len(ex)] = 0

    return training_data_padded, mask


def get_single_training_example(params, data_generator, truncation):
    """Generates a single training example

    Returns:
        training_data   : A single training example
        true_data       : Ground truth for example
    """
    data_generator.reset()
    data_generator.generate_checkpoints(
        no_targets_checkpoints=np.random.poisson(
            params.data_generation.checkpoints.targets
        ),
        no_sensors_checkpoints=np.random.poisson(
            params.data_generation.checkpoints.sensors
        ),
    )
    data_generator.spawn_sensors(
        distribution=lambda: np.clip(
            np.random.poisson(params.data_generation.no_of_objects.sensors_lambda),
            params.data_generation.no_of_objects.min_sensors,
            params.data_generation.no_of_objects.max_sensors,
        ),
        error=lambda: np.random.poisson(
            params.data_generation.no_of_objects.sensor_error
        ),
    )
    data_generator.spawn_targets(
        distribution=lambda: np.clip(
            np.random.poisson(params.data_generation.no_of_objects.targets_lambda),
            params.data_generation.no_of_objects.min_targets,
            params.data_generation.no_of_objects.max_targets,
        ),
    )
    data_generator.generate_paths(
        sensor_speed_distribution=lambda: np.random.normal(
            params.data_generation.speed.sensors[0],
            params.data_generation.speed.sensors[1],
        ),
        target_speed_distribution=lambda: np.random.normal(
            params.data_generation.speed.targets[0],
            params.data_generation.speed.targets[1],
        ),
        truncation = params.data_generation.truncation_generated
    )
    data_generator.run()

    truncated_sensors_timestamps = truncate_array(
        data_generator.sensors_timestamps, truncation
    )
    truncated_targets_timestamps = truncate_array(
        data_generator.targets_timestamps, truncation
    )
    truncated_sensors_velocities = truncate_array(
        data_generator.sensors_velocities, truncation
    )
    truncated_targets_velocities = truncate_array(
        data_generator.targets_velocities, truncation
    )

    truncated_angles_stack = []
    for i in range(params.training.batch_size):
        truncated_angles = truncate_angles_array(
            data_generator.find_bearings(), truncation
        )
        truncated_angles_stack.append(truncated_angles)

    truncated_angles_stack = np.stack(truncated_angles_stack, axis=0)

    return (
        truncated_sensors_timestamps,
        truncated_targets_timestamps,
        truncated_sensors_velocities,
        truncated_targets_velocities,
        truncated_angles_stack,
    )


def truncate_array(arr, m):
    if arr.shape[1] > m:
        return arr[:, :m, :]
    return arr


def truncate_angles_array(arr, m):
    if arr.shape[2] > m:
        return arr[:, :, :m, :]
    return arr
