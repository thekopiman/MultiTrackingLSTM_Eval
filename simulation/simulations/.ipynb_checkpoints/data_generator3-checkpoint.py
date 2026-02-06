# import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor
import torch.nn.functional as F


from .MOTSimulationV1 import MOTSimulationV1


def pseudo_assignment(state, target_tensor):

    c = torch.cdist(state[..., :2], target_tensor.unsqueeze(1).to(torch.float)).squeeze(
        -1
    )

    sm_layer = torch.nn.Softmax(dim=-1)

    output = sm_layer(-c).unsqueeze(-1)
    assignment_nomeasurement = torch.ones_like(output) - output
    assignment = torch.concat([output, assignment_nomeasurement], -1)
    return assignment

def pseudo_assignment_hard_argmin(state, target_tensor):

    c = torch.cdist(state[..., :2], target_tensor.unsqueeze(1).to(torch.float)).squeeze(
        -1
    )

    min_indices = torch.argmin(c, dim=-1)
    
    # Create a one-hot encoding for assignments
    assignment = F.one_hot(min_indices, num_classes=c.shape[-1]).to(c.dtype).unsqueeze(-1)

    assignment_nomeasurement = torch.ones_like(assignment) - assignment
    assignment = torch.concat([assignment, assignment_nomeasurement], -1)
    return assignment


def generate_labels(
    training_tensor: np.ndarray, truncated_targets_timestamps, interval=0.1
):
    idx = training_tensor[:, -1] / interval
    idx = idx.astype(int)

    truncated_targets_timestamps = np.transpose(
        truncated_targets_timestamps, (1, 0, 2)
    )  # (t, N, 4)
    expanded_tensor = truncated_targets_timestamps[idx, :, :]
    return expanded_tensor


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
    __version__ = 3.0

    def __init__(
        self,
        params,
    ):
        """
        This version only allows for fixed number of targets.
        Mixed variable targets will be available in Version 3.1

        Args:
            params (DocDict): Read from yaml file
        """
        self.params = params
        self.device = params.training.device
        self.truncation = params.data_generation.truncation
        self.batch = int(params.training.batch_size)
        self.interval = params.data_generation.interval
        self.p = params.data_generation.p
        self.training_dim = (
            params.rnn.cartesian_dim
            + params.rnn.cartesian_dim // 2
            + params.rnn.cartesian_dim % 2
            + 1
        )
        self.max_targets = params.data_generation.no_of_objects.max_targets
        self.max_sensors = params.data_generation.no_of_objects.max_sensors
        self.tracks = params.rnn.tracks

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

        print("DataGenerator Version 2: Async")

    def get_measurements(self, raw_data: dict):
        truncated_angles_array = attach_time(
            raw_data["truncated_angles"][np.newaxis, :], self.interval
        )

        t1, u_id, t2 = self._step(
            raw_data["truncated_sensors_timestamps"],
            raw_data["truncated_targets_timestamps"],
            raw_data["truncated_targets_velocities"],
            truncated_angles_array,
        )

        training_nested_tensor = t1
        unique_measuurement_ids = u_id
        final_target_tensor = t2

        return (training_nested_tensor, unique_measuurement_ids, final_target_tensor)

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

        final_measurement = np.array([])
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
                    final_measurement = (
                        np.vstack([final_measurement, batch_measurement_array])
                        if final_measurement.size > 0
                        else batch_measurement_array
                    )

            if measurement_array[0].size > 0:
                final_target_coordinates = (
                    np.vstack([final_target_coordinates, target_array])
                    if final_target_coordinates.size > 0
                    else target_array
                )

                final_unique_ids = np.hstack([final_unique_ids, unique_target_ids])

        return final_measurement, final_unique_ids, final_target_coordinates

    def get_batch(self):
        """
        The `get_batch` function processes training data, generates labels, pads the data to create
        batches, and computes time deltas before returning the final tensors and masks.
        :return: The `get_batch` method returns the following tensors:
        1. `final_training_tensor`: Tensor containing the training data after padding to the maximum
        batch size.
        2. `final_mask`: Tensor containing the mask for the padded training data.
        3. `final_target_tensor`: Tensor containing the target data after padding to the maximum batch
        size.
        4. `final_labels`: Tensor containing the labels after padding to
        """

        self.raw_data = get_single_training_example(self.params, self.datagen)  # dict

        batch_no = len(self.raw_data)

        training_tensor_array = []
        target_array = []
        labels_array = []
        sensors_labels_array = []

        for idx in self.raw_data:
            training_data, _, measured_target_data = self.get_measurements(
                self.raw_data[idx]
            )
            dirty_label = np.concatenate(
                [
                    self.raw_data[idx]["truncated_targets_timestamps"],
                    self.raw_data[idx]["truncated_targets_velocities"],
                ],
                axis=-1,
            )
            labels = generate_labels(
                training_data, dirty_label, self.interval
            )  # (T, N, 4)
            
            sensors_labels = generate_labels(
                training_data, self.raw_data[idx]["truncated_sensors_timestamps"], self.interval
            )  # (T, N, 4)

            assert training_data.shape[0] == labels.shape[0], "Mismatch T shape"

            training_tensor_array.append(training_data)
            labels_array.append(labels)
            target_array.append(measured_target_data)
            sensors_labels_array.append(sensors_labels)

        final_training_tensor, _ = pad_to_batch_max(
            training_tensor_array, self.truncation
        )
        final_sensor_tensor, _ = pad_to_batch_max_labels(sensors_labels_array, self.truncation, self.max_sensors)

        # We need a label mask as N might differ for the various batches
        # Mask of shape (B, N, t)
        final_labels, final_mask = pad_to_batch_max_labels(
            labels_array, self.truncation, self.tracks
        )
        
        final_target_tensor, _ = pad_to_batch_max(target_array, self.truncation)
        
        

        final_training_tensor = torch.from_numpy(final_training_tensor).to(self.device)
        final_mask = torch.from_numpy(final_mask).to(self.device)
        final_target_tensor = torch.from_numpy(final_target_tensor).to(self.device)
        final_labels = torch.from_numpy(final_labels).to(self.device)
        final_sensor_tensor = torch.from_numpy(final_sensor_tensor).to(self.device)

        # Compute time deltas
        time_deltas = torch.diff(final_training_tensor[..., -1], dim=1)

        # Pad with zero at the beginning to maintain shape
        time_deltas = torch.cat(
            [torch.zeros(batch_no, 1).to(self.device), time_deltas], dim=1
        )

        # Replace the timestamp column with time deltas
        final_training_tensor[..., -1] = time_deltas
        return final_training_tensor, final_mask, final_target_tensor, final_labels, final_sensor_tensor

    def _bool_select_bearing(self) -> bool:
        x = np.random.uniform(0, 1)
        return x <= self.p


def pad_to_batch_max(training_data, max_len=None):
    batch_size = len(training_data)

    # Determine feature shape dynamically
    feature_shape = training_data[0].shape[1:]  # Everything except T

    if max_len is None:
        max_len = max(len(ex) for ex in training_data)  # Use the longest sequence

    # Initialize padded array and mask
    training_data_padded = np.zeros((batch_size, max_len, *feature_shape))
    mask = np.zeros((batch_size, max_len))

    for i, ex in enumerate(training_data):
        ex_len = min(len(ex), max_len)  # Prevent overflow
        training_data_padded[i, :ex_len, ...] = ex[
            :ex_len, ...
        ]  # Preserve extra dimensions
        mask[i, :ex_len] = 1  # Adjust mask

    return training_data_padded, mask


def pad_to_batch_max_labels(label_data, max_len=None, max_targets=None):
    batch_size = len(label_data)

    if max_targets is None:
        max_targets = max(i.shape[1] for i in label_data)

    if max_len is None:
        max_len = max(len(ex) for ex in label_data)  # Use the longest sequence

    # Sanity Check
    for i in label_data:
        assert i.ndim == 3, "Fit in data with shape (T, N, ?) only"
        assert i.shape[1] <= max_targets, "There are more targets than max targets"

    # initialize padded array and mask
    data_padded = np.zeros((batch_size, max_len, max_targets, label_data[0].shape[-1]))

    mask = np.zeros((batch_size, max_len, max_targets))

    for i, ex in enumerate(label_data):
        ex_len = min(len(ex), max_len)
        ex_max = min(ex.shape[1], max_targets)

        data_padded[i, :ex_len, :ex_max, ...] = ex[:ex_len, :ex_max, ...]
        mask[i, :ex_len, :ex_max] = 1

    return data_padded, mask


def get_single_training_example(params, data_generator):
    """Generates a single training example

    Returns:
        training_data   : A single training example
        true_data       : Ground truth for example
    """
    results = dict()

    for batch in range(params.training.batch_size):
        results[batch] = dict()
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
            truncation=params.data_generation.truncation_generated,
        )
        data_generator.run()

        results[batch][
            "truncated_sensors_timestamps"
        ] = data_generator.sensors_timestamps
        results[batch][
            "truncated_targets_timestamps"
        ] = data_generator.targets_timestamps
        results[batch][
            "truncated_sensors_velocities"
        ] = data_generator.sensors_velocities
        results[batch][
            "truncated_targets_velocities"
        ] = data_generator.targets_velocities
        results[batch]["truncated_angles"] = data_generator.find_bearings()

    return results
