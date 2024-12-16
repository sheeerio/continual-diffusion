import argparse
import zarr
import torch
import torch.utils.data
import numpy as np
import os

def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i, end in enumerate(episode_ends):
        start = 0 if i == 0 else episode_ends[i - 1]
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
                pad_shape = [padding] + list(sample.shape[1:])
                pad_before = np.tile(sample[0:1], (padding,) + (1,) * (sample.ndim - 1))
                sample = np.concatenate([pad_before, sample], axis=0)

        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            padded = np.zeros((sequence_length,) + sample.shape[1:], dtype=sample.dtype)
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
        # For action data: (N, action_dim)
        stats = {
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    return stats

def normalize_data(data, stats):
    scale = stats['max'] - stats['min']
    scale[scale == 0] = 1
    normalized_data = (data - stats['min']) / scale * 2 - 1
    return normalized_data

def unnormalize_data(normalized_data, stats):
    scale = stats['max'] - stats['min']
    scale[scale == 0] = 1
    data = (normalized_data + 1) / 2 * scale + stats['min']
    return data

class MazeImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        dataset_root = zarr.open(dataset_path, 'r')
        train_image_data = dataset_root['images'][:]
        train_image_data = np.moveaxis(train_image_data, -1, 1)  # (N, H, W, C) -> (N, C, H, W)

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
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        sample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        sample['image'] = sample['image'][:self.obs_horizon, :]
        sample['action'] = sample['action'][:self.pred_horizon, :]
        return sample

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

        assert np.all(original_action >= action_stats['min'] - 1e-5), "Action data below min after unnormalization."
        assert np.all(original_action <= action_stats['max'] + 1e-5), "Action data above max after unnormalization."
        print("Action normalization and unnormalization successful.")

    except AssertionError as e:
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"Error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Data Collection for MazeImageDataset")
    parser.add_argument('--dataset_path', type=str, default='maze.zarr', help='Path to the dataset')
    parser.add_argument('--pred_horizon', type=int, default=16, help='Prediction horizon')
    parser.add_argument('--obs_horizon', type=int, default=16, help='Observation horizon')
    parser.add_argument('--action_horizon', type=int, default=1, help='Action horizon')
    parser.add_argument('--test', action='store_true', help='Run dataset testing')
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = MazeImageDataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon
    )

    if args.test:
        test_dataset(dataset)
    else:
        print(f"Dataset created with {len(dataset)} samples.")

if __name__ == "__main__":
    main()
