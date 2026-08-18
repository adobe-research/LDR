"""
Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.

This file constitutes Licensed Materials under the Adobe Research License.
Use is limited to noncommercial research purposes.
See the LICENSE file at the project root for the complete license terms and disclaimer.

PhyWorld HDF5 video dataset (returns frame tensors in [-1, 1]).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v3 as iio


class PhyWorldDataset(Dataset):
    def __init__(self, hdf5_path, num_cond=3, img_size=128, augment=False):
        self.hdf5_path = hdf5_path
        self.num_cond = num_cond
        self.img_size = img_size
        self.augment = augment
        with h5py.File(hdf5_path, 'r') as f:
            self.index = []
            for gk in sorted(f['video_streams'].keys()):
                n = len(f['video_streams'][gk])
                for i in range(n):
                    self.index.append((gk, i))
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r', swmr=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        self._open()
        gk, li = self.index[idx]

        raw = self._file['video_streams'][gk][li]
        frames_np = iio.imread(raw.tobytes(), index=None, format_hint='.mp4')

        frames = torch.from_numpy(frames_np.astype(np.float32))
        frames = frames.permute(0, 3, 1, 2)
        frames = frames / 127.5 - 1.0

        if frames.shape[-1] != self.img_size:
            import torch.nn.functional as F
            frames = F.interpolate(frames, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)

        if self.augment and torch.rand(1).item() < 0.5:
            frames = frames.flip(-1)

        cond_frames = frames[:self.num_cond]
        target_frames = frames[self.num_cond:]

        init = torch.from_numpy(
            self._file['init_streams'][gk][li].astype(np.float32)
        )
        if init.numel() < 4:  # pad per-task init to a fixed len-4 so mixed-task batches collate
            init = torch.cat([init, init.new_zeros(4 - init.numel())])

        return {
            'cond_frames': cond_frames,
            'target_frames': target_frames,
            'init': init,
        }
