#!/usr/bin/env python
"""Optuna search and utilities."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import optuna
import pandas as pd
from omegaconf import OmegaConf

from src.config import cfg as global_cfg
from .train import repeated_cross_validation, train_stage
from .utils import load_obj, set_seed


# ---------------- helpers ----------------
def record_trial_history(eval_folder: str):
    """Return a callback that appends the latest trial to an Excel log."""
    def _callback(study, trial):
        df = study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "params",
                "state",
                "datetime_start",
                "datetime_complete",
                "intermediate_values",
            )
        ).iloc[[trial.number]]

        history_path = os.path.join(eval_folder, "optuna_trials.xlsx")
        if os.path.exists(history_path):
            old = pd.read_excel(history_path)
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["number"], keep="last")
        else:
            combined = df

        combined.to_excel(history_path, index=False)

    return _callback


def _suggest_from_space(
    trial: optuna.Trial,
    space: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Read cfg.optuna.params and produce a dict of sampled params.

    Supports keys:
      type: [float|loguniform|int|categorical], low/high or min/max, step/choices.
    """
    out: Dict[str, Any] = {}
    for name, spec in (space or {}).items():
        t = str(spec.get("type", "float")).lower()
        low = spec.get("low", spec.get("min", None))
        high = spec.get("high", spec.get("max", None))

        if t in ("float", "uniform"):
            step = spec.get("step", None)
            out[name] = trial.suggest_float(name, float(low), float(high), step=step)
        elif t in ("loguniform", "logfloat"):
            out[name] = trial.suggest_float(name, float(low), float(high), log=True)
        elif t in ("int", "integer"):
            step = int(spec.get("step", 1))
            out[name] = trial.suggest_int(name, int(low), int(high), step=step)
        elif t in ("categorical", "choice"):
            choices = spec["choices"]
            out[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unknown optuna param type for '{name}': {t}")
    return out


def space_hash(space: Dict[str, Any]) -> str:
    """Short stable hash of the search space dict."""
    blob = json.dumps(space, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    import hashlib as _h  # local import to avoid top-level dependency at import time

    return _h.md5(blob.encode("utf-8")).hexdigest()[:8]


def _make_study_name(base_name: str, space: Dict[str, Any]) -> str:
    """Encode objective + space hash in the study name."""
    return f"{base_name}_objF2_{space_hash(space)}"


# ---------------- public api ----------------
def run_optuna_search(
    cfg=None,
    study_name: Optional[str] = None,
    direction: str = "maximize",           # keep maximize for F2
    storage: Optional[str] = None,         # e.g., "sqlite:///optuna.db"
    load_if_exists: bool = True,
    show_progress_bar: bool = True,
    callbacks: Optional[list] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run Optuna and return (best_params, study).

    - Honors cfg.optuna.{n_trials, params, sampler, pruner}
    - Stores `metrics` and `best_ckpt` in trial.user_attrs
    - Study name auto-encodes search space + objective (F2)
    """
    if cfg is None:
        cfg = global_cfg

    # Optional seed
    try:
        if hasattr(cfg.training, "seed"):
            set_seed(int(cfg.training.seed))
    except Exception:
        pass

    n_trials = int(getattr(cfg.optuna, "n_trials", 20))
    space = (
        OmegaConf.to_container(cfg.optuna.params, resolve=True)
        if hasattr(cfg.optuna, "params")
        else {}
    )

    # Sampler / pruner from cfg
    sampler = None
    pruner = None
    try:
        if (
            hasattr(cfg, "optuna")
            and hasattr(cfg.optuna, "sampler")
            and "class_name" in cfg.optuna.sampler
        ):
            SamplerCls = load_obj(cfg.optuna.sampler["class_name"])
            sampler = SamplerCls(**cfg.optuna.sampler.get("params", {}))
    except Exception as e:
        print(f"[Optuna] Sampler setup failed; using default. Reason: {e}")

    try:
        if (
            hasattr(cfg, "optuna")
            and hasattr(cfg.optuna, "pruner")
            and "class_name" in cfg.optuna.pruner
        ):
            PrunerCls = load_obj(cfg.optuna.pruner["class_name"])
            pruner = PrunerCls(**cfg.optuna.pruner.get("params", {}))
    except Exception as e:
        print(f"[Optuna] Pruner setup failed; disabling pruning. Reason: {e}")

    # Study setup
    if study_name is None:
        project = getattr(cfg.general, "project_name", "project")
        base = f"{project}_optuna"
    else:
        base = study_name

    derived_name = _make_study_name(base, space)

    def _create_study(name: str) -> optuna.Study:
        return optuna.create_study(
            direction=direction,
            study_name=name,
            storage=storage,
            load_if_exists=load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

    study = _create_study(derived_name)

    try:
        study.set_user_attr(
            "_search_space_json", json.dumps(space, sort_keys=True, ensure_ascii=False)
        )
        study.set_user_attr("_objective", "val_f2_max")
    except Exception:
        pass

    try:
        print(
            "[Optuna] Study:", study.study_name,
            "| Sampler:", type(study.sampler).__name__,
            "| Pruner:", type(study.pruner).__name__,
            "| Objective: val_f2 (maximize)",
        )
    except Exception:
        pass

    # ----- inner helpers -----
    def _run_one_training(
        local_cfg,
        trial_params: Dict[str, Any],
        trial: optuna.Trial,
    ) -> Dict[str, Any]:
        """
        Run one training/CV round with the sampled params.

        Return {"score": <float or None>, "best_ckpt": <str>}.
        The score should be the monitored metric (default val_f2).
        """
        # Keep HPO light by default
        local_cfg.training.suppress_artifacts = True
        local_cfg.run_mode = "tune"

        # Map sampled params → cfg
        if "lr" in trial_params:
            local_cfg.optimizer.params.lr = float(trial_params["lr"])
        if "weight_decay" in trial_params:
            local_cfg.optimizer.params.weight_decay = float(
                trial_params["weight_decay"]
            )
        if "batch_size" in trial_params:
            local_cfg.data.batch_size = int(trial_params["batch_size"])
        if "gradient_clip_val" in trial_params:
            if not hasattr(local_cfg, "trainer") or local_cfg.trainer is None:
                local_cfg.trainer = OmegaConf.create({})
            local_cfg.trainer.gradient_clip_val = float(
                trial_params["gradient_clip_val"]
            )

        opt_name = trial_params.get("optimizer_name", None)
        if opt_name is not None:
            if opt_name == "AdamW":
                local_cfg.optimizer.class_name = "torch.optim.AdamW"
                local_cfg.optimizer.params.pop("momentum", None)
                local_cfg.optimizer.params.pop("nesterov", None)
            elif opt_name == "SGD":
                local_cfg.optimizer.class_name = "torch.optim.SGD"
                mom = float(trial_params.get("sgd_momentum", 0.9))
                nes = bool(trial_params.get("sgd_nesterov", False))
                local_cfg.optimizer.params.momentum = mom
                local_cfg.optimizer.params.nesterov = nes

        if hasattr(local_cfg.training, "tuning_epochs_detection"):
            if not hasattr(local_cfg, "trainer") or local_cfg.trainer is None:
                local_cfg.trainer = OmegaConf.create({})
            local_cfg.trainer.max_epochs = int(
                local_cfg.training.tuning_epochs_detection
            )

        # Train
        csv_path = local_cfg.data.detection_csv
        stage = "detection"
        n_class = 2
        repeats = int(getattr(local_cfg.training, "repeated_cv", 1))
        use_cv = bool(
            getattr(local_cfg, "use_cv", False)
            or getattr(local_cfg.training, "cross_validation", False)
        )

        if use_cv:
            model, score = repeated_cross_validation(
                local_cfg,
                csv_path,
                num_classes=n_class,
                stage_name=stage,
                repeats=repeats,
                trial=trial,
            )
        else:
            model, score = train_stage(
                local_cfg,
                csv_path,
                num_classes=n_class,
                stage_name=stage,
                trial=trial,
            )

        return {
            "score": float(score) if score is not None else None,
            "best_ckpt": getattr(model, "best_ckpt_path", ""),
        }

    # ----- objective -----
    def objective(trial: optuna.Trial) -> float:
        trial_params = _suggest_from_space(trial, space)

        local_cfg = deepcopy(cfg)
        metrics = _run_one_training(local_cfg, trial_params, trial)
        score = metrics.get("score", None)

        # Prune if metric missing/NaN
        if score is None or not math.isfinite(score):
            trial.set_user_attr("error", "score_missing_or_nan")
            raise optuna.exceptions.TrialPruned()

        # Annotate trial
        trial.set_user_attr("metrics", metrics)
        for k, v in trial_params.items():
            trial.set_user_attr(f"param__{k}", v)
        if "best_ckpt" in metrics:
            trial.set_user_attr("best_ckpt", metrics["best_ckpt"])

        return float(score)

    # Optimize
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks or [],
        )
    except ValueError as e:
        msg = str(e)
        if "does not support dynamic value space" in msg:
            alt_name = derived_name + "_v2"
            print(
                f"[Optuna] Detected dynamic space clash. "
                f"Creating a new study: {alt_name}"
            )
            study = _create_study(alt_name)
            try:
                study.set_user_attr(
                    "_search_space_json",
                    json.dumps(space, sort_keys=True, ensure_ascii=False),
                )
                study.set_user_attr("_objective", "val_f2_max")
            except Exception:
                pass
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=show_progress_bar,
                callbacks=callbacks or [],
            )
        else:
            raise

    best_params = study.best_trial.params
    return best_params, study


def apply_best_params_to_cfg(
    cfg,
    best_params: Dict[str, Any],
    target_nodes: Optional[Dict[str, str]] = None,
):
    """Merge best_params into specific nodes in cfg."""
    if target_nodes:
        for k, path in target_nodes.items():
            if k in best_params:
                OmegaConf.update(cfg, path, best_params[k], merge=True)
    else:
        OmegaConf.update(cfg, "hparams", best_params, merge=True)


def _choose_reuse_trial(study, trial_id: int | None):
    """
    Pick a trial from the study:
      - If trial_id is not None and >= 0, return that trial (must be COMPLETE).
      - Otherwise, return the best COMPLETE trial.

    Robust to pruned/incomplete trials.
    """
    import optuna as _opt

    if trial_id is not None and trial_id >= 0:
        matches = [t for t in study.trials if t.number == trial_id]
        if not matches:
            raise RuntimeError(
                f"[Reuse] Trial {trial_id} not found in '{study.study_name}'."
            )
        t = matches[0]
        if t.state != _opt.trial.TrialState.COMPLETE:
            raise RuntimeError(
                f"[Reuse] Trial {trial_id} is not COMPLETE (state={t.state})."
            )
        return t

    complete = [
        t
        for t in study.trials
        if t.state == _opt.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError(
            f"[Reuse] No COMPLETE trials with value found in '{study.study_name}'."
        )

    try:
        return study.best_trial
    except Exception:
        reverse = (
            getattr(study, "direction", None)
            == _opt.study.StudyDirection.MAXIMIZE
        )
        return sorted(complete, key=lambda z: z.value, reverse=reverse)[0]


def _apply_trial_params_to_cfg(cfg, params: dict):
    """Apply a trial's params into cfg just like finalize_after_hpo does."""
    mapping = {
        "lr": "optimizer.params.lr",
        "weight_decay": "optimizer.params.weight_decay",
        "batch_size": "data.batch_size",
        "gradient_clip_val": "optimizer.params.gradient_clip_val",
    }

    try:
        from optuna_tuner import apply_best_params_to_cfg

        apply_best_params_to_cfg(cfg, params, target_nodes=mapping)
    except Exception:
        if "lr" in params:
            cfg.optimizer.params.lr = float(params["lr"])
        if "weight_decay" in params:
            cfg.optimizer.params.weight_decay = float(params["weight_decay"])
        if "batch_size" in params:
            cfg.data.batch_size = int(params["batch_size"])
        if "gradient_clip_val" in params:
            if not hasattr(cfg, "trainer") or cfg.trainer is None:
                cfg.trainer = OmegaConf.create({})
            cfg.trainer.gradient_clip_val = float(params["gradient_clip_val"])
