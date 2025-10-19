#!/usr/bin/env python
"""Dataset and data-related utilities (group-safe)."""

import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Any, Tuple, Optional


def default_group_from_filename(fn: str) -> str:
    """
    Derive a stable group id from filename so patches from the same source
    (slide/patient) are kept together. Adjust if your naming differs.
    Example: 'PID12_slideA_patch003.png' -> 'PID12'
    """
    stem = os.path.splitext(os.path.basename(fn))[0]
    token = re.split(r"[_\-]", stem)[0]
    return token or stem


class PatchClassificationDataset(Dataset):
    """
    Custom dataset for patch classification.

    Args:
        data (DataFrame or CSV path): columns include image_col and label_col,
            and optionally group_col (slide/patient id).
        image_dir (str): Folder containing images.
        transforms (albumentations.Compose): augment/transform pipeline.
        image_col (str): Column with filenames.
        label_col (str): Column with int labels.
        group_col (str|None): Optional grouping column name.
    """

    def __init__(
        self,
        data: Any,
        image_dir: str,
        transforms: Optional[Any] = None,
        image_col: str = "filename",
        label_col: str = "label",
        group_col: Optional[str] = None,
        indices: Optional[Any] = None,
    ) -> None:
        df = pd.read_csv(data) if isinstance(data, str) else data.copy()

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        # Keep only rows that physically exist
        rows = []
        for _, row in df.iterrows():
            p = os.path.join(image_dir, row[image_col])
            if os.path.exists(p):
                rows.append(row)
            else:
                print(f"[data] missing → {p} (skip)")
        df = pd.DataFrame(rows).reset_index(drop=True)

        # Guaranteed group id
        if group_col and group_col in df.columns:
            df["_group_id"] = df[group_col].astype(str)
        else:
            df["_group_id"] = df[image_col].astype(str).map(default_group_from_filename)

        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row[self.image_col])
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unreadable image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            image = self.transforms(image=image)["image"]
        else:
            image = torch.from_numpy(image / 255.0).permute(2, 0, 1).contiguous()

        label = int(row[self.label_col])
        return image, label
