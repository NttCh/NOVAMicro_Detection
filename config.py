#!/usr/bin/env python
"""Configuration settings and global variables."""

from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]

def env_path(var, default_rel):
    # use env if set, otherwise a repo-relative default
    return os.getenv(var, str((REPO_ROOT / default_rel).as_posix()))

CFG_DICT = {
    "run_mode": "test",     # "train" or "tune" or "test"
    "use_cv": True,          # If True, use cross-validation
    "reuse": {
            "enable": False,
            "retrain_trial_id": -1, #if x < 0, choose the best trial in the study
            "study_name": "bacteria_optuna_objF2_26080900",
    },

    "general": {
        "save_dir": env_path("MICROCLF_SAVE_DIR", "novamicro_project"),
        "project_name": "bacteria"
    },

    "trainer": {
        "devices": 1,
        "accelerator": "auto",
        "precision": "16-mixed",
        "gradient_clip_val": 0.5
    },

    "training": {
        "seed": 666,
        "mode": "max",
        "tuning_epochs_detection": 3,
        "num_folds": 2,
        "repeated_cv": 1,
        "early_stopping": {
            "enabled": True,
            "early_stop_metric": "val_f2",
            "patience_hpo": 15,
            "patience_final": 15
        },
        "freeze_strategy": "warmup_unfreeze",   # "none" | "feature_extractor" | "warmup_unfreeze"
        "freeze_backbone_epochs": 10,
        "final_unfreeze": True,
        "finalize_after_hpo": False,
        "clear_cuda_before_stage": True
    },

    "xai": {
        "enabled": True,
        "target_layer": ["layer2","layer3","layer4"],
        "k_patches": 3,
        "patch_size": 96,
        "save_heatmap": True,
        "save_overlay": True,
        "save_original_overlay": True,
        "panel": False,
        "class_names": ["negative", "positive"],
        "tta": True,
        "smooth": True,
        "smooth_n": 6,
        "smooth_sigma": 0.05,
        "mask_scale_bar": False,
        "mask_rect_wh": [0.18, 0.08]
    },


    "optimizer": {
        "class_name": "torch.optim.AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.001
        }
    },

    "scheduler": {
        "class_name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "step": "epoch",
        "monitor": "val_f2",
        "params": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10
        }
    },

    "model": {
        "backbone": {
            "class_name": "torchvision.models.resnet50",
            "params": {
                "weights": "IMAGENET1K_V2"
            }
        }
    },

    "data": {
        "negative_dir": env_path("MICROCLF_NEG_DIR", "dataset/train_nonbac"),
        "positive_dir": env_path("MICROCLF_POS_DIR", "dataset/train_bac"),
        "detection_csv":  env_path("MICROCLF_CSV", "dataset/train_model.csv"),
        "folder_path": env_path("MICROCLF_DATA_ROOT", "dataset"),
        "num_workers": 0,
        "generate_csv": False,
        "batch_size": 8,
        "label_col": "label",
        "valid_split": 0.2
    },

    "augmentation": {
        "train": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "albumentations.Rotate", "params": {"limit": 10, "p": 0.5}},
                {"class_name": "albumentations.HorizontalFlip", "params": {"p": 0.5}},
                {"class_name": "albumentations.VerticalFlip", "params": {"p": 0.5}},
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        },

        "valid": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        },
        "hpo_resize": {"height": 400, "width": 400},
        "final_resize": {"height": 400, "width": 400}
    },

    "test": {
        "folder_path": [env_path("MICROCLF_TEST_DIR", "test/test_data")],

    },

    "pretrained_ckpt": [env_path("MICROCLF_CKPT", "test/pretrained_model/best_from_best_trial.ckpt")],      


    "optuna": {
        "n_trials": 6,
        "sampler": {
            "class_name": "optuna.samplers.TPESampler",
            "params": {"multivariate": True}
        },
        "pruner": {
            "class_name": "optuna.pruners.MedianPruner",
            "params": {"n_startup_trials": 5, "n_warmup_steps": 15}
        },
        
        "params": {
            "lr": {"type": "loguniform", "low": 1e-6, "high": 3e-4},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
            "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
            "gradient_clip_val": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.05},

        },
    }

}

cfg: DictConfig = OmegaConf.create(CFG_DICT)

BASE_SAVE_DIR = CFG_DICT["general"]["save_dir"]
