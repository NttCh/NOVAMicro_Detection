#!/usr/bin/env python
"""Configuration settings and global variables."""

from omegaconf import OmegaConf, DictConfig

CFG_DICT = {
    "run_mode": "train",     # "train" or "tune" or "test"
    "use_cv": True,          # If True, use cross-validation

    "general": {
        "save_dir": "final_6",
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
        "tuning_epochs_detection": 50,
        "num_folds": 5,
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
        "save_overlay": False,
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
            "betas": [0.9, 0.999],
            "eps": 1.0e-8
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
        "negative_dir": r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection.ver6-1\data\train_nonbac",
        "positive_dir": r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection.ver6-1\data\train_bac",
        "detection_csv":  r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection.ver6-1\data\train_model.csv",
        "folder_path": r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection.ver6-1\data",
        "num_workers": 0,
        "generate_csv": True,
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
        "folder_path": [
            r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection-3\bacai_ver2\bac_1_4_Copy_3\test_bac_Copy_3_recat\multi-microbe",
            r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection-3\bacai_ver2\bac_1_4_Copy_3\test_bac_Copy_3_recat\Less-microbe",
            r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection-3\Data\test_non_bac"

]


    },

    "pretrained_ckpt": [
        r"C:\Users\Natthacha\Downloads\NOVAMicroOps_Detection.ver6-1\final_6\20251018\133807_train\best_model\best_from_best_trial.ckpt"   
    ],        


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
        "reuse": {
            "enable": True,
            "retrain_trial_id": -1, #if x < 0, choose the best trial in the study
            "study_name": "bacteria_optuna_objF2_26080900",
            "use_cv": True,
            "repeats": 1
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
