import random

import numpy as np
import torch
import torch.nn.functional as F


class FMriDatasetAug(torch.utils.data.Dataset):
    def __init__(self, data, labels, augment=False,
                 noise_std=0.015,
                 time_mask_ratio=0.05,
                 temporal_dropout_ratio=0.05,
                 feature_mask_ratio=0.05,
                 jitter_max_shift=1,
                 amp_scale_range=(0.95, 1.05),
                 smooth_kernel_size=3,
                 # apply_augmentations probabilities (notebook variants)
                 p_gaussian_noise=0.6,
                 p_time_mask=0.3,
                 p_temporal_dropout=0.3,
                 p_feature_mask=0.2,
                 p_time_jitter=0.4,
                 p_amplitude_scale=0.4,
                 p_smooth_signal=0.3):
        """
        Safe fMRI augmentation settings for resting-state data.

        data: np.array shape (N, time_steps, features)
        labels: np.array shape (N,) or (N,1)
        augment: whether to augment on __getitem__
        """

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()

        self.augment = augment

        self.noise_std = noise_std
        self.time_mask_ratio = time_mask_ratio
        self.temporal_dropout_ratio = temporal_dropout_ratio
        self.feature_mask_ratio = feature_mask_ratio
        self.jitter_max_shift = jitter_max_shift
        self.amp_scale_range = tuple(amp_scale_range)
        self.smooth_kernel_size = smooth_kernel_size

        self.p_gaussian_noise = p_gaussian_noise
        self.p_time_mask = p_time_mask
        self.p_temporal_dropout = p_temporal_dropout
        self.p_feature_mask = p_feature_mask
        self.p_time_jitter = p_time_jitter
        self.p_amplitude_scale = p_amplitude_scale
        self.p_smooth_signal = p_smooth_signal

        self.original_len = data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].clone()
        label = self.labels[idx]

        if self.augment:
            sample = self.apply_augmentations(sample)

        return sample, label

    # === AUGMENTATION FUNCTIONS ===

    def gaussian_noise(self, x):
        "add a random noise with noies_std standard deviation to one subject fmri data"
        return x + torch.randn_like(x) * self.noise_std

    def time_mask(self, x):
        "make mask(zero) some of the time points"
        num_mask_steps = int(self.time_mask_ratio * x.size(0))
        mask_idx = np.random.choice(x.size(0), num_mask_steps, replace=False)
        x[mask_idx] = 0.0
        return x

    def temporal_dropout(self, x):

        chunk_len = int(random.uniform(0.02, self.temporal_dropout_ratio) * x.size(0))
        if chunk_len > 0:
            start = random.randint(0, x.size(0) - chunk_len)
            x[start:start+chunk_len] = 0.0
        return x

    def feature_mask(self, x):
        "make mask(zero) some of the region points"
        num_mask_features = int(self.feature_mask_ratio * x.size(1))
        mask_idx = np.random.choice(x.size(1), num_mask_features, replace=False)
        x[:, mask_idx] = 0.0
        return x

    def time_jitter(self, x):
        shift = random.randint(-self.jitter_max_shift, self.jitter_max_shift)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=0)
            if shift > 0:
                x[:shift] = 0.0
            elif shift < 0:
                x[shift:] = 0.0
        return x

    def amplitude_scale(self, x, per_roi=False):
        if per_roi:
            scales = torch.empty(x.size(1)).uniform_(*self.amp_scale_range)
            x = x * scales
        else:
            x = x * random.uniform(*self.amp_scale_range)
        return x

    def smooth_signal(self, x):
        channels = x.size(1)
        kernel = torch.ones(channels, 1, self.smooth_kernel_size, dtype=x.dtype) / self.smooth_kernel_size
        x_smooth = F.conv1d(
            x.unsqueeze(0).transpose(1, 2),
            kernel,
            padding=self.smooth_kernel_size // 2,
            groups=channels
        ).transpose(1, 2).squeeze(0)
        return x_smooth

    # === SAFE AUGMENTATION PIPELINE ===
    def apply_augmentations(self, x):
        if random.random() < self.p_gaussian_noise:
            x = self.gaussian_noise(x)   # mild noise
        if random.random() < self.p_time_mask:
            x = self.time_mask(x)        # small fraction
        if random.random() < self.p_temporal_dropout:
            x = self.temporal_dropout(x) # very short gaps
        if random.random() < self.p_feature_mask:
            x = self.feature_mask(x)     # rare ROI masking
        if random.random() < self.p_time_jitter:
            x = self.time_jitter(x)      # +/- 1 TR shift
        if random.random() < self.p_amplitude_scale:
            x = self.amplitude_scale(x, per_roi=False)
        if random.random() < self.p_smooth_signal:
            x = self.smooth_signal(x)    # mild smoothing
        return x
