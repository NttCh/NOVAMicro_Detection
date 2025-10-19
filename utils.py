#!/usr/bin/env python
"""Utility functions."""

import datetime
import random
import numpy as np
import torch
import importlib
import os
import pandas as pd
import gc
from typing import Any, Iterable, Optional

def thai_time():
    """Return the current time in Thailand (UTC+7)."""
    return datetime.datetime.utcnow() + datetime.timedelta(hours=7)


def set_seed(seed: int = 666) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_obj(obj_path: str):
    """
    Dynamically load an object by its string path.

    Args:
        obj_path (str): The module path in dot notation.

    Returns:
        Any: The loaded object (class, function, etc.).
    """
    parts = obj_path.split(".")
    module_path = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def generate_detection_csv(negative_dir: str,
                           positive_dir: str,
                           output_csv: str) -> None:
    """
    Walks the two class folders, collects (filepath, label) rows, 
    writes them to output_csv.
    """
    data_rows = []

    for label, root_dir in [(0, negative_dir), (1, positive_dir)]:
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', ".tif", ".tiff")):
                    full_path = os.path.join(root, fname)
                    data_rows.append({"filename": full_path, "label": label})

    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)
    print(f"[CSV] Generated detection CSV: {output_csv}")

def make_run_dirs(root: str):
    """
    Create and return a dict of subdirectories:
      - best_model/
      - eval/
      - multi_predictions/
    """
    subdirs = {
        "best_model":        os.path.join(root, "best_model"),
        "eval":              os.path.join(root, "eval"),
        "multi_predictions": os.path.join(root, "multi_predictions"),
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)
    return subdirs

def record_trial_history(eval_folder: str):
    def _callback(study, trial):
        df = study.trials_dataframe(attrs=(
            'number','value','params','state',
            'datetime_start','datetime_complete','intermediate_values'
        )).iloc[[trial.number]]

        history_path = os.path.join(eval_folder, "optuna_trials.xlsx")
        if os.path.exists(history_path):
            old = pd.read_excel(history_path)
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["number"], keep="last")
        else:
            combined = df
        combined.to_excel(history_path, index=False)
    return _callback

def cleanup_cuda(
    *,
    logger: Optional[Any] = None,
    trainer: Optional[Any] = None,
    models: Optional[Iterable[Any]] = None,
    dataloaders: Optional[Iterable[Any]] = None,
) -> None:
    """
    Best-effort cleanup of GPU/loader resources between CV folds.
    - Closes TensorBoard writer if present
    - Deletes strong refs to models/loaders/trainer
    - Forces GC
    - Empties CUDA caching allocator (and IPC handles)

    This function does not touch random seeds or determinism flags,
    so metric values remain identical given the same seed/config.
    """
    # 1) Close TB writer to release file handles
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        if logger and isinstance(logger, TensorBoardLogger):
            exp = getattr(logger, "experiment", None)
            if hasattr(exp, "flush"): exp.flush()
            if hasattr(exp, "close"): exp.close()
    except Exception:
        pass

    # 2) Break references so workers/handles can die
    try:
        if dataloaders:
            for dl in dataloaders:
                # Help Python drop any worker iterator
                if hasattr(dl, "_iterator"):
                    setattr(dl, "_iterator", None)
    except Exception:
        pass

    # 3) Drop strong references
    try:
        if models:
            for m in models:
                del m
        if trainer is not None:
            del trainer
        if dataloaders:
            for dl in dataloaders:
                del dl
    except Exception:
        pass

    # 4) Collect Python garbage
    gc.collect()

    # 5) Flush CUDA caches
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Return cached memory to OS (esp. after many folds)
            torch.cuda.ipc_collect()
    except Exception:
        pass
