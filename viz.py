#!/usr/bin/env python
"""Visualization helpers for HPO, CV learning curves, and PR/Calibration."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---- Optuna plotting (with HTML fallback) ----
def export_optuna_plots(study, out_dir: str) -> None:
    """Save Optuna optimization history + param importance (and extras) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )
        # Prefer HTML to avoid external static exporters
        plot_optimization_history(study).write_html(os.path.join(out_dir, "optuna_optimization_history.html"))
        plot_param_importances(study).write_html(os.path.join(out_dir, "optuna_param_importance.html"))
        # Strong add-ons (appendix)
        plot_parallel_coordinate(study).write_html(os.path.join(out_dir, "optuna_parallel_coordinate.html"))
        plot_slice(study).write_html(os.path.join(out_dir, "optuna_slice.html"))
        print("[Viz] Optuna HTML plots saved.")
    except Exception as e:
        print(f"[Viz] Optuna plots failed ({e}). Ensure optuna.visualization is available.")

# ======================================
# viz.py — only the two plotting funcs
# ======================================
def _plot_mean_std_curve(
    df: pd.DataFrame,
    metric_cols: list,          # e.g. ["val_loss","train_loss"] or ["val_f2","train_f2"]
    out_png: str,
    ylabel: str,
    title: str = None
) -> None:
    """
    Plot CV mean ± std over epochs for one or more metric columns.
    Expects columns: ['epoch','fold', <metric_cols...>]
    If any metric in metric_cols is missing, it's skipped gracefully.
    """
    if df.empty:
        print(f"[Viz] Empty DF for {metric_cols}. Skipping.")
        return

    present = [m for m in metric_cols if m in df.columns]
    if not present:
        print(f"[Viz] No requested metrics present: {metric_cols}")
        return

    plt.figure(figsize=(7.2, 4.8))
    g = df.groupby("epoch")

    for m in present:
        mean = g[m].mean()
        std  = g[m].std(ddof=0)

        epochs = mean.index.values
        mean_v = mean.values
        std_v  = std.values

        plt.plot(epochs, mean_v, label=f"{m} (mean)")
        plt.fill_between(epochs, mean_v - std_v, mean_v + std_v, alpha=0.25, label=f"{m} ±1 SD")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title or f"{', '.join(present)} (CV mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[Viz] Saved curve → {out_png}")

def plot_cv_mean_std_curves(all_eval_excel: str, out_dir: str) -> None:
    """
    Read per-epoch, per-fold metrics (Excel) and produce:
      - cv_f2_mean_std.png          (val_f2 + overlays train_f2 if present)
      - cv_loss_mean_std.png        (val_loss + overlays train_loss if present)
    Required columns: 'epoch','fold','val_loss','val_f2'
    Optional columns: 'train_loss','train_f2'
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    cols = set(df.columns)

    needed_core = {"epoch", "fold"}
    if not needed_core.issubset(cols):
        print(f"[Viz] Missing core columns in {all_eval_excel}. Need {needed_core}, have {list(cols)}")
        return

    # --- F2 (val + optional train) ---
    if "val_f2" in cols:
        f2_cols = ["val_f2"] + (["train_f2"] if "train_f2" in cols else [])
        _plot_mean_std_curve(
            df[["epoch", "fold"] + f2_cols],
            metric_cols=f2_cols,
            out_png=os.path.join(out_dir, "cv_f2_mean_std.png"),
            ylabel="F2 (β=2)",
            title="Train vs Val F2 (CV mean ± std)" if "train_f2" in cols else "Validation F2 (CV mean ± std)"
        )
    else:
        print("[Viz] Skipped F2 plot: 'val_f2' not found.")

    # --- Loss (val + optional train) ---
    if "val_loss" in cols:
        loss_cols = ["val_loss"] + (["train_loss"] if "train_loss" in cols else [])
        _plot_mean_std_curve(
            df[["epoch", "fold"] + loss_cols],
            metric_cols=loss_cols,
            out_png=os.path.join(out_dir, "cv_loss_mean_std.png"),
            ylabel="Loss",
            title="Train vs Val Loss (CV mean ± std)" if "train_loss" in cols else "Validation Loss (CV mean ± std)"
        )
    else:
        print("[Viz] Skipped loss plot: 'val_loss' not found.")

def plot_per_fold_curves(all_eval_excel: str, out_dir: str, metric_col: str = "val_f2") -> None:
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    df = pd.read_excel(all_eval_excel)
    if not {"epoch", "fold", metric_col}.issubset(df.columns):
        print(f"[Viz] Missing columns for per-fold curves.")
        return
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7.2, 4.8))
    for f, g in df.groupby("fold"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g[metric_col], alpha=0.5, linewidth=1, label=f"fold {f}")
    # overlay mean
    mean = df.groupby("epoch")[metric_col].mean()
    plt.plot(mean.index, mean.values, linewidth=2.5, label="mean", zorder=10)
    plt.xlabel("Epoch"); plt.ylabel(metric_col); plt.title(f"Per-fold {metric_col}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"per_fold_{metric_col}.png"))
    plt.close()
    print(f"[Viz] Saved per-fold {metric_col} curves.")

def save_confusion_matrix(
    y_true,
    y_pred,
    out_path: str,
    title: str = "Confusion Matrix",
    labels=("0", "1"),
    cmap: str = "Blues",
    annotate_fmt: str = "d",
    figsize=(8, 6),
    probs=None,
    threshold=None,
):
    # make sure dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # to numpy (handles torch tensors/lists)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # compute CM with fixed label order [0, 1] for consistent color mapping
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=annotate_fmt,
        cmap=cmap,              # unified palette
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_cv_f2_and_loss_combined(
    all_eval_excel: str,
    out_dir: str,
    f2_col_val: str = "val_f2",
    f2_col_train: str = "train_f2",
    loss_col_val: str = "val_loss",
    loss_col_train: str = "train_loss",
    out_name: str = "cv_f2_and_loss_combined.png",
) -> None:
    """
    Create a single plot with CV mean±std curves:
      - Left y-axis: F2 (validation + optional train)
      - Right y-axis: Loss (validation + optional train)

    Expects columns (any subset is OK): ['epoch','fold', f2_col_val, f2_col_train, loss_col_val, loss_col_train]
    Saves: <out_dir>/<out_name>
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)

    if not {"epoch", "fold"}.issubset(df.columns):
        print(f"[Viz] Missing core columns. Need ['epoch','fold'], have {list(df.columns)}")
        return

    # Figure out which metrics exist
    has_val_f2   = f2_col_val   in df.columns
    has_train_f2 = f2_col_train in df.columns
    has_val_loss = loss_col_val in df.columns
    has_train_loss = loss_col_train in df.columns

    if not (has_val_f2 or has_train_f2 or has_val_loss or has_train_loss):
        print("[Viz] No F2 or Loss columns found — nothing to plot.")
        return

    # Group by epoch to get CV mean ± std (ddof=0 so folds=1 doesn't NaN)
    g = df.groupby("epoch")
    epochs = g.size().index.values

    def _mean_std(colname):
        if colname not in df.columns:
            return None, None
        m = g[colname].mean().values
        s = g[colname].std(ddof=0).values
        # Replace NaN std (e.g., single fold at some epoch) with zeros for fill_between
        s = np.where(np.isnan(s), 0.0, s)
        return m, s

    f2_val_mean,   f2_val_std   = _mean_std(f2_col_val)
    f2_train_mean, f2_train_std = _mean_std(f2_col_train)
    loss_val_mean, loss_val_std = _mean_std(loss_col_val)
    loss_train_mean, loss_train_std = _mean_std(loss_col_train)

    # ---- Plot ----
    fig, ax_f2 = plt.subplots(figsize=(7.6, 4.8))

    lines = []
    labels = []

    # Left axis (F2)
    ax_f2.set_xlabel("Epoch")
    ax_f2.set_ylabel("F2 (β=2)")
    ax_f2.set_ylim(0.0, 1.0)

    if f2_val_mean is not None:
        l, = ax_f2.plot(epochs, f2_val_mean, label="val_f2 (mean)")
        ax_f2.fill_between(epochs, f2_val_mean - f2_val_std, f2_val_mean + f2_val_std, alpha=0.25)
        lines.append(l); labels.append("val_f2 (mean ± sd)")

    if f2_train_mean is not None:
        l, = ax_f2.plot(epochs, f2_train_mean, linestyle="--", label="train_f2 (mean)")
        ax_f2.fill_between(epochs, f2_train_mean - f2_train_std, f2_train_mean + f2_train_std, alpha=0.20)
        lines.append(l); labels.append("train_f2 (mean ± sd)")

    # Right axis (Loss)
    ax_loss = ax_f2.twinx()
    ax_loss.set_ylabel("Loss")

    if loss_val_mean is not None:
        l, = ax_loss.plot(epochs, loss_val_mean, label=f"{loss_col_val} (mean)")
        ax_loss.fill_between(epochs, loss_val_mean - loss_val_std, loss_val_mean + loss_val_std, alpha=0.25)
        lines.append(l); labels.append(f"{loss_col_val} (mean ± sd)")

    if loss_train_mean is not None:
        l, = ax_loss.plot(epochs, loss_train_mean, linestyle="--", label=f"{loss_col_train} (mean)")
        ax_loss.fill_between(epochs, loss_train_mean - loss_train_std, loss_train_mean + loss_train_std, alpha=0.20)
        lines.append(l); labels.append(f"{loss_col_train} (mean ± sd)")

    # Title & legend
    title_bits = []
    if (f2_val_mean is not None) or (f2_train_mean is not None):
        title_bits.append("F2")
    if (loss_val_mean is not None) or (loss_train_mean is not None):
        title_bits.append("Loss")
    plt.title(f"CV Mean ± Std: {' & '.join(title_bits)}")

    ax_f2.legend(lines, labels, loc="best")

    fig.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[Viz] Saved combined curve → {out_png}")

def _group_by_epoch_mean(df: pd.DataFrame, col: str):
    g = df.groupby("epoch")
    return g[col].mean().index.values, g[col].mean().values

def plot_cv_mean_f2(
    all_eval_excel: str,
    out_dir: str,
    f2_col_val: str = "val_f2",
    f2_col_train: str = "train_f2",
    out_name: str = "cv_f2_mean_only.png",
):
    """Mean curves only: val_f2 (+ optional train_f2) on left axis."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    if not {"epoch","fold"}.issubset(df.columns):
        print("[Viz] Need columns ['epoch','fold'].")
        return

    plt.figure(figsize=(7.6, 4.8))
    lines, labels = [], []

    if f2_col_val in df.columns:
        e, y = _group_by_epoch_mean(df, f2_col_val)
        l, = plt.plot(e, y, label="val_f2 (mean)")
        lines.append(l); labels.append("val_f2 (mean)")
    if f2_col_train in df.columns:
        e, y = _group_by_epoch_mean(df, f2_col_train)
        l, = plt.plot(e, y, label="train_f2 (mean)")
        lines.append(l); labels.append("train_f2 (mean)")

    if not lines:
        print("[Viz] No F2 columns found to plot.")
        return

    plt.xlabel("Epoch")
    plt.ylabel("F2 (β=2)")
    plt.ylim(0.0, 1.0)
    plt.title("Train vs Val F2 (CV mean)")
    plt.legend(lines, labels, loc="best")
    plt.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Viz] Saved → {out_png}")

def plot_cv_mean_loss(
    all_eval_excel: str,
    out_dir: str,
    loss_col_val: str = "val_loss",
    loss_col_train: str = "train_loss",
    out_name: str = "cv_loss_mean_only.png",
):
    """Mean curves only: val_loss (+ optional train_loss)."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    if not {"epoch","fold"}.issubset(df.columns):
        print("[Viz] Need columns ['epoch','fold'].")
        return

    plt.figure(figsize=(7.6, 4.8))
    lines, labels = [], []

    if loss_col_val in df.columns:
        e, y = _group_by_epoch_mean(df, loss_col_val)
        l, = plt.plot(e, y, label=f"{loss_col_val} (mean)")
        lines.append(l); labels.append(f"{loss_col_val} (mean)")
    if loss_col_train in df.columns:
        e, y = _group_by_epoch_mean(df, loss_col_train)
        l, = plt.plot(e, y, label=f"{loss_col_train} (mean)")
        lines.append(l); labels.append(f"{loss_col_train} (mean)")

    if not lines:
        print("[Viz] No Loss columns found to plot.")
        return

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss (CV mean)")
    plt.legend(lines, labels, loc="best")
    plt.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Viz] Saved → {out_png}")

def plot_cv_f2_and_loss_mean_only(
    all_eval_excel: str,
    out_dir: str,
    f2_col_val: str = "val_f2",
    f2_col_train: str = "train_f2",
    loss_col_val: str = "val_loss",
    loss_col_train: str = "train_loss",
    out_name: str = "cv_f2_and_loss_mean_only.png",
) -> None:
    """
    Plot CV mean curves (NO std) on one figure:
      - Left y-axis: F2 (val + optional train)
      - Right y-axis: Loss (val + optional train)
    Requires columns: 'epoch','fold' and any subset of the metric columns.
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)

    if not {"epoch", "fold"}.issubset(df.columns):
        print(f"[Viz] Missing core columns. Need ['epoch','fold'], have {list(df.columns)}")
        return

    cols = set(df.columns)
    has = {
        "val_f2":   f2_col_val in cols,
        "train_f2": f2_col_train in cols,
        "val_loss": loss_col_val in cols,
        "train_loss": loss_col_train in cols
    }
    if not any(has.values()):
        print("[Viz] No plottable columns found.")
        return

    g = df.groupby("epoch")
    epochs = g.size().index.values

    def _mean(col):
        return g[col].mean().values if col in df.columns else None

    f2_val_mean   = _mean(f2_col_val)
    f2_train_mean = _mean(f2_col_train)
    loss_val_mean = _mean(loss_col_val)
    loss_train_mean = _mean(loss_col_train)

    fig, ax_f2 = plt.subplots(figsize=(7.6, 4.8))
    lines, labels = [], []

    # Left axis: F2
    ax_f2.set_xlabel("Epoch")
    ax_f2.set_ylabel("F2 (β=2)")
    ax_f2.set_ylim(0.0, 1.0)

    if f2_val_mean is not None:
        l, = ax_f2.plot(epochs, f2_val_mean, label="val_f2 (mean)")
        lines.append(l); labels.append("val_f2 (mean)")
    if f2_train_mean is not None:
        l, = ax_f2.plot(epochs, f2_train_mean, label="train_f2 (mean)")
        lines.append(l); labels.append("train_f2 (mean)")

    # Right axis: Loss
    ax_loss = ax_f2.twinx()
    ax_loss.set_ylabel("Loss")
    if loss_val_mean is not None:
        l, = ax_loss.plot(epochs, loss_val_mean, label=loss_col_val + " (mean)")
        lines.append(l); labels.append(loss_col_val + " (mean)")
    if loss_train_mean is not None:
        l, = ax_loss.plot(epochs, loss_train_mean, label=loss_col_train + " (mean)")
        lines.append(l); labels.append(loss_col_train + " (mean)")

    # Title & legend
    title_bits = []
    if (f2_val_mean is not None) or (f2_train_mean is not None): title_bits.append("F2")
    if (loss_val_mean is not None) or (loss_train_mean is not None): title_bits.append("Loss")
    plt.title("CV Mean: " + " & ".join(title_bits) if title_bits else "CV Mean")

    ax_f2.legend(lines, labels, loc="best")
    fig.tight_layout()

    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[Viz] Saved mean-only curve → {out_png}")
