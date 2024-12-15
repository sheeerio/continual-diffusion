import zarr
import torch
import torch.utils.data
import numpy as np

def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i, end in enumerate(episode_ends):
        start = 0 if i == 0 else episode_ends[i-1]
        episode_length = end - start
        for idx in range(-pad_before, episode_length - sequence_length + pad_after + 1):
            buffer_start = max(start + idx, start)
            buffer_end = min(buffer_start + sequence_length, end)
            sample_start = max(0, -idx)
            sample_end = min(sequence_length, buffer_end - (start + idx))
            indices.append([buffer_start, buffer_end, sample_start, sample_end])
    return np.array(indices)

def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = {}
    for key, input_arr in train_data.items():
        buffer = input_arr[buffer_start_idx:buffer_end_idx]
        sample = buffer[sample_start_idx:sample_end_idx]

        actual_length = sample.shape[0]
        expected_length = sample_end_idx - sample_start_idx

        if actual_length != expected_length:
            print(f"Key '{key}': sample_length={actual_length}, expected_length={expected_length}")
            if actual_length > expected_length:
                sample = sample[:expected_length]
            else:
                padding = expected_length - actual_length
                if input_arr.ndim == 4:
                    pad_before = sample[0].reshape(1, -1, 1, 1).repeat(padding, axis=0)
                elif input_arr.ndim == 2:
                    pad_before = sample[0].reshape(1, -1).repeat(padding, axis=0)
                sample = np.concatenate([pad_before, sample], axis=0)

        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            if input_arr.ndim == 4:
                padded = np.zeros((sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    padded[:sample_start_idx] = sample[0]
                if sample_end_idx < sequence_length:
                    padded[sample_end_idx:] = sample[-1]
            elif input_arr.ndim == 2:
                padded = np.zeros((sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    padded[:sample_start_idx] = sample[0]
                if sample_end_idx < sequence_length:
                    padded[sample_end_idx:] = sample[-1]
            padded[sample_start_idx:sample_end_idx] = sample
            result[key] = padded
        else:
            result[key] = sample
    return result

def get_data_stats(data):
    if data.ndim == 4:
        # For image data: (N, C, H, W)
        stats = {
            'min': np.min(data, axis=(0, 2, 3)),  # Shape: (C,)
            'max': np.max(data, axis=(0, 2, 3))   # Shape: (C,)
        }
    elif data.ndim == 2:
        # For action data: (N, 15)
        stats = {
            'min': np.min(data, axis=0),  # Shape: (15,)
            'max': np.max(data, axis=0)   # Shape: (15,)
        }
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    return stats

def normalize_data(data, stats):
    scale = stats['max'] - stats['min']
    scale[scale == 0] = 1 
    if data.ndim == 4:
        scale = scale.reshape(-1, 1, 1)
        min_vals = stats['min'].reshape(-1, 1, 1)
    elif data.ndim == 2:
        scale = scale.reshape(1, -1)
        min_vals = stats['min'].reshape(1, -1)
    ndata = (data - min_vals) / scale
    ndata = ndata * 2 - 1 
    return ndata

def unnormalize_data(ndata, stats):
    scale = stats['max'] - stats['min']
    scale[scale == 0] = 1 
    if ndata.ndim == 4:
        scale = scale.reshape(-1, 1, 1)
        min_vals = stats['min'].reshape(-1, 1, 1)
    elif ndata.ndim == 2:
        scale = scale.reshape(1, -1)
        min_vals = stats['min'].reshape(1, -1)
    data = (ndata + 1) / 2 * scale + min_vals
    return data

class MazeImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        dataset_root = zarr.open(dataset_path, 'r')
        train_image_data = dataset_root['images'][:]
        train_image_data = np.moveaxis(train_image_data, -1, 1) 

        actions = dataset_root['actions'][:]
        episode_ends = dataset_root['meta']['episode_ends'][:]

        train_data = {
            'image': train_image_data,
            'action': actions
        }

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )

        stats = {}
        normalized_train_data = {}
        for key, data in train_data.items():
            if data.size == 0:
                raise ValueError(f"Dataset '{key}' is empty.")
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        nsample['image'] = nsample['image'][:self.obs_horizon, :]
        nsample['action'] = nsample['action'][:self.pred_horizon, :]

        return nsample

def test_dataset(dataset):
    try:
        sample = dataset[0]
        print("Sample 'image' shape:", sample['image'].shape)
        print("Sample 'action' shape:", sample['action'].shape)

        image = sample['image']
        image_stats = dataset.stats['image']
        original_image = unnormalize_data(image, image_stats)

        assert np.all(original_image >= image_stats['min'].reshape(-1, 1, 1) - 1e-5), "Image data below min after unnormalization."
        assert np.all(original_image <= image_stats['max'].reshape(-1, 1, 1) + 1e-5), "Image data above max after unnormalization."
        print("Image normalization and unnormalization successful.")

        action = sample['action']
        action_stats = dataset.stats['action']
        original_action = unnormalize_data(action, action_stats)

        assert np.all(original_action >= action_stats['min'].reshape(1, -1) - 1e-5), "Action data below min after unnormalization."
        assert np.all(original_action <= action_stats['max'].reshape(1, -1) + 1e-5), "Action data above max after unnormalization."
        print("Action normalization and unnormalization successful.")

    except AssertionError as e:
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"Error: {e}")

dataset = MazeImageDataset(
    dataset_path='maze.zarr',
    pred_horizon=16,
    obs_horizon=16,
    action_horizon=1
)

test_dataset(dataset)