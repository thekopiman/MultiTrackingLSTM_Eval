# import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor
import torch.nn.functional as F


from .MOTSimulationV1 import MOTSimulationV1


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

def pad_tensor(tensor, N):
    n = tensor.shape[-1]
    if N < n:
        raise ValueError("N must be greater than or equal to n.")
    
    # Pad the last dimension (n -> N)
    pad_size = N - n
    return F.pad(tensor, (0, pad_size), mode='constant', value=0)

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


def safe_one_hot(final_unique_id, num_classes):
    """
    One-hot encodes `final_unique_id`. If a value is -1, the entire row becomes all zeros.

    Args:
        final_unique_id (torch.Tensor): Tensor of shape (B, T) containing indices.
        N (int): Number of classes for one-hot encoding.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (B, T, N) with zero rows where values were -1.
    """
    # Create mask: True where valid (not -1), False where -1
    valid_mask = final_unique_id != -1  

    # Replace -1 with 0 (safe for one-hot encoding)
    sanitized_ids = torch.where(valid_mask, final_unique_id, torch.tensor(0, device=final_unique_id.device))

    # One-hot encode
    one_hot_encoded = F.one_hot(sanitized_ids.to(torch.int64), num_classes=num_classes).to(dtype=torch.float32)

    # Zero out entire row where original value was -1
    one_hot_encoded *= valid_mask.unsqueeze(-1)  

    return one_hot_encoded

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
        self.birth_rate = params.data_generation.birth
        self.death_rate = params.data_generation.death
        self.noise = params.data_generation.noise
        self.revival_cooldown = params.data_generation.revival_cooldown

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

        (
            training_nested_tensor,
            unique_measuurement_ids,
            final_target_tensor,
            final_life,
        ) = self._step(
            raw_data["truncated_sensors_timestamps"],
            raw_data["truncated_targets_timestamps"],
            raw_data["truncated_targets_velocities"],
            truncated_angles_array,
        )

        return (
            training_nested_tensor,
            unique_measuurement_ids,
            final_target_tensor,
            final_life,
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

        final_measurement = np.array([])
        final_target_coordinates = np.array([])
        final_unique_ids = np.array([], dtype="int64")
        final_life = self.simulate_life(
            total_targets, total_duration, self.birth_rate, self.death_rate, self.revival_cooldown
        )

        for time in range(total_duration):
            measurement_array = []
            target_array = []
            unique_target_ids = []
            for s in range(total_sensors):
                for t in range(total_targets):
                    # Target is not selected

                    sensor_coordinate = split_sensors_timestamps[s, time, :]
                    target_coordinate = split_targets_timestamps[t, time, :]

                    # Noise
                    if self._bool_select_p(self.noise):
                        target_array.append(np.zeros_like(target_coordinate))
                        unique_target_ids.append(-1)

                        only_time = split_angles_array[0, s, t, time, -1:None]

                        sensor_measurements = np.concatenate(
                            [
                                sensor_coordinate,
                                np.random.uniform(-np.pi, np.pi, size=(1,)),
                                only_time,
                            ]
                        )
                        measurement_array.append(sensor_measurements)
                        # print("noise", unique_target_ids)

                    # Did not detect target
                    if not self._bool_select_p(self.p) or not final_life[time, t]:
                        continue

                    target_array.append(target_coordinate)
                    unique_target_ids.append(t)

                    bearing_and_time = split_angles_array[0, s, t, time, :]
                    sensor_measurements = np.concatenate(
                        [sensor_coordinate, bearing_and_time]
                    )
                    measurement_array.append(sensor_measurements)

            random_idx = np.random.permutation(len(measurement_array))
            measurement_array = np.array(measurement_array)[random_idx]
            unique_target_ids = np.array(unique_target_ids)[random_idx]
            target_array = np.array(target_array)[random_idx]
            

            # There might be situations where there are no bearing measurements in certain t
            if measurement_array.size > 0:
                final_measurement = (
                    np.vstack([final_measurement, measurement_array])
                    if final_measurement.size > 0
                    else measurement_array
                )

            if measurement_array.size > 0:
                final_target_coordinates = (
                    np.vstack([final_target_coordinates, target_array])
                    if final_target_coordinates.size > 0
                    else target_array
                )

                final_unique_ids = np.hstack([final_unique_ids, unique_target_ids])

        return final_measurement, final_unique_ids, final_target_coordinates, final_life

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
        life_array = []
        unique_id_array = []

        for idx in self.raw_data:
            training_data, unique_id, measured_target_data, life = (
                self.get_measurements(self.raw_data[idx])
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
                training_data,
                self.raw_data[idx]["truncated_sensors_timestamps"],
                self.interval,
            )  # (T, N, 4)

            life_label = generate_labels(
                training_data,
                np.expand_dims(life, axis=-1).transpose(1, 0, 2),
                self.interval,
            )  # (T, N, 1)
            

            assert training_data.shape[0] == labels.shape[0], "Mismatch T shape"

            training_tensor_array.append(training_data)
            unique_id_array.append(unique_id)
            labels_array.append(labels)
            target_array.append(measured_target_data)
            sensors_labels_array.append(sensors_labels)
            life_array.append(life_label)

        final_training_tensor, _ = pad_to_batch_max(
            training_tensor_array, self.truncation
        )

        final_unique_id, _ = pad_to_batch_max(unique_id_array, self.truncation)

        # We need a label mask as N might differ for the various batches
        # Mask of shape (B, N, t)
        final_sensor_tensor, _ = pad_to_batch_max_labels(
            sensors_labels_array, self.truncation, self.max_sensors
        )

        final_labels, final_mask = pad_to_batch_max_labels(
            labels_array, self.truncation, self.tracks
        )
        final_life, _ = pad_to_batch_max_labels(life_array, self.truncation)

        final_target_tensor, _ = pad_to_batch_max(target_array, self.truncation)

        final_training_tensor = torch.from_numpy(final_training_tensor).to(self.device)
        final_mask = torch.from_numpy(final_mask).to(self.device)
        final_target_tensor = torch.from_numpy(final_target_tensor).to(self.device)
        final_labels = torch.from_numpy(final_labels).to(self.device)
        final_sensor_tensor = torch.from_numpy(final_sensor_tensor).to(self.device)
        final_unique_id = torch.from_numpy(final_unique_id).to(self.device)
        final_life = (
            torch.from_numpy(final_life).to(self.device).squeeze(-1).to(torch.bool)
        )

        # Compute time deltas
        time_deltas = torch.diff(final_training_tensor[..., -1], dim=1)

        # Pad with zero at the beginning to maintain shape
        time_deltas = torch.cat(
            [torch.zeros(batch_no, 1).to(self.device), time_deltas], dim=1
        )

        # Replace the timestamp column with time deltas
        final_training_tensor[..., -1] = time_deltas
        return (
            final_training_tensor,
            final_mask,
            final_target_tensor,
            final_labels,
            final_sensor_tensor,
            final_unique_id,
            final_life,
        )

    def _bool_select_p(self, p) -> bool:
        x = np.random.uniform(0, 1)
        return x <= p

    def simulate_life(self, n, steps, birth_rate, death_rate, revival_cooldown = 100, initial_alive=None):
        """
        Simulates life states over time using a transition matrix approach.
        Entities can come back to life after being dead for at least 100 time steps.

        Parameters:
        - n: Number of entities (size of state array)
        - steps: Number of time steps
        - birth_rate: Probability of becoming alive if currently dead
        - death_rate: Probability of dying if currently alive
        - initial_alive: Optional initial state array (1 for alive, 0 for dead)

        Returns:
        - A 2D NumPy array of shape (steps, n) showing state evolution
        """

        # Initialize the state randomly if not provided
        if initial_alive is None:
            state = np.random.choice([1, 0], size=n, p =[birth_rate, 1 - birth_rate])  # 1=alive, 0=dead
        else:
            state = np.array(initial_alive, dtype=int)

        # Track how long each entity has been dead
        dead_duration = np.zeros(n, dtype=int)

        # Store state history
        history = np.zeros((steps, n), dtype=int)

        for t in range(steps):
            history[t] = state  # Save current state

            # Transition rules
            birth_prob = np.random.uniform(0, 1, n)
            death_prob = np.random.uniform(0, 1, n)

            # Apply death probability
            state = np.where((state == 1) & (death_prob <= death_rate), 0, state)

            # Update dead duration counter
            dead_duration = np.where(state == 0, dead_duration + 1, 0)

            # Apply birth probability only if dead for at least 100 steps
            can_revive = dead_duration >= revival_cooldown
            state = np.where((state == 0) & (birth_prob <= birth_rate) & can_revive, 1, state)
            
            # Reset dead duration if revived
            dead_duration = np.where(state == 1, 0, dead_duration)

        return history

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
