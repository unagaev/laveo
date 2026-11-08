import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LavenderDataset(Dataset):

    def __init__(self, img_paths, mask_paths, mode="train", lavender_ratio=0.7):

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.mode = mode

        self.lavender_ratio = lavender_ratio

        # -------------------------
        # SPLIT INDEXES
        # -------------------------
        self.lavender_indices = []
        self.empty_indices = []

        for i, m_path in enumerate(self.mask_paths):

            mask = np.load(m_path).astype(np.float32)
            mask = np.nan_to_num(mask)

            # 🔥 enforce binary for indexing
            mask_bin = (mask > 0).astype(np.uint8)

            if mask_bin.sum() > 0:
                self.lavender_indices.append(i)
            else:
                self.empty_indices.append(i)

        print(f"[Dataset] lavender patches: {len(self.lavender_indices)}")
        print(f"[Dataset] empty patches: {len(self.empty_indices)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # -------------------------
        # BALANCED SAMPLING
        # -------------------------
        if self.mode == "train":

            if np.random.rand() < self.lavender_ratio and len(self.lavender_indices) > 0:
                idx = np.random.choice(self.lavender_indices)
            else:
                idx = np.random.choice(self.empty_indices)

        # -------------------------
        # LOAD DATA
        # -------------------------
        img = np.load(self.img_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.float32)

        img = np.nan_to_num(img)
        mask = np.nan_to_num(mask)

        # 🔥 FORCE STRICT BINARY (CRITICAL FIX)
        mask = (mask > 0).astype(np.float32)
        

        # ensure shape = (1, H, W)
        if mask.ndim == 2:
            mask = mask[None, :, :]

        # -------------------------
        # TO TENSOR
        # -------------------------
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return img, mask