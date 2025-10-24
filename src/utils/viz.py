#!/usr/bin/env python
"""Visualization helpers for HPO, CV learning curves, and Confusion matrix"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------- optuna plots ----------------
def export_optuna_plots(study, out_dir: str) -> None:
    """Save Optuna optimization history + param importance (and extras) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )

        plot_optimization_history(study).write_html(
            os.path.join(out_dir, "optuna_optimization_history.html")
        )
        plot_param_importances(study).write_html(
            os.path.join(out_dir, "optuna_param_importance.html")
        )
        plot_parallel_coordinate(study).write_html(
            os.path.join(out_dir, "optuna_parallel_coordinate.html")
        )
        plot_slice(study).write_html(os.path.join(out_dir, "optuna_slice.html"))
        print("[Viz] Optuna HTML plots saved.")
    except Exception as e:
        print(f"[Viz] Optuna plots failed ({e}). Ensure optuna.visualization is available.")


# ---------------- internal helpers ----------------
def _plot_mean_std_curve(
    df: pd.DataFrame,
    metric_cols: List[str],
    out_png: str,
    ylabel: str,
    title: str | None = None,
) -> None:
    """
    Plot CV mean ± std over epochs for one or more metric columns.

    Expects: columns ['epoch', 'fold', *metric_cols]
    Missing metrics are skipped.
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
        std = g[m].std(ddof=0)

        epochs = mean.index.values
        mean_v = mean.values
        std_v = np.where(np.isnan(std.values), 0.0, std.values)

        plt.plot(epochs, mean_v, label=f"{m} (mean)")
        plt.fill_between(epochs, mean_v - std_v, mean_v + std_v, alpha=0.25, label=f"{m} ±1 SD")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title or f"{', '.join(present)} (CV mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Viz] Saved curve → {out_png}")


# ---------------- CV overview plots ----------------
def plot_cv_mean_std_curves(all_eval_excel: str, out_dir: str) -> None:
    """
    From per-epoch, per-fold metrics (Excel) produce:
      - cv_f2_mean_std.png
      - cv_loss_mean_std.png

    Required: 'epoch','fold','val_loss','val_f2'
    Optional: 'train_loss','train_f2'
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    cols = set(df.columns)

    if not {"epoch", "fold"}.issubset(cols):
        print(f"[Viz] Missing core columns in {all_eval_excel}. Need ['epoch','fold'].")
        return

    if "val_f2" in cols:
        f2_cols = ["val_f2"] + (["train_f2"] if "train_f2" in cols else [])
        _plot_mean_std_curve(
            df[["epoch", "fold"] + f2_cols],
            metric_cols=f2_cols,
            out_png=os.path.join(out_dir, "cv_f2_mean_std.png"),
            ylabel="F2 (β=2)",
            title="Train vs Val F2 (CV mean ± std)" if "train_f2" in cols else "Validation F2 (CV mean ± std)",
        )
    else:
        print("[Viz] Skipped F2 plot: 'val_f2' not found.")

    if "val_loss" in cols:
        loss_cols = ["val_loss"] + (["train_loss"] if "train_loss" in cols else [])
        _plot_mean_std_curve(
            df[["epoch", "fold"] + loss_cols],
            metric_cols=loss_cols,
            out_png=os.path.join(out_dir, "cv_loss_mean_std.png"),
            ylabel="Loss",
            title="Train vs Val Loss (CV mean ± std)" if "train_loss" in cols else "Validation Loss (CV mean ± std)",
        )
    else:
        print("[Viz] Skipped loss plot: 'val_loss' not found.")


def plot_per_fold_curves(all_eval_excel: str, out_dir: str, metric_col: str = "val_f2") -> None:
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    df = pd.read_excel(all_eval_excel)
    if not {"epoch", "fold", metric_col}.issubset(df.columns):
        print("[Viz] Missing columns for per-fold curves.")
        return
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7.2, 4.8))
    for f, g in df.groupby("fold"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g[metric_col], alpha=0.5, linewidth=1, label=f"fold {f}")

    mean = df.groupby("epoch")[metric_col].mean()
    plt.plot(mean.index, mean.values, linewidth=2.5, label="mean", zorder=10)
    plt.xlabel("Epoch")
    plt.ylabel(metric_col)
    plt.title(f"Per-fold {metric_col}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"per_fold_{metric_col}.png"), dpi=150)
    plt.close()
    print(f"[Viz] Saved per-fold {metric_col} curves.")


# ---------------- confusion matrix ----------------
def save_confusion_matrix(
    y_true,
    y_pred,
    out_path: str,
    title: str = "Confusion Matrix",
    labels: Iterable[str] = ("0", "1"),
    cmap: str = "Blues",
    annotate_fmt: str = "d",
    figsize: Tuple[int, int] = (8, 6),
    probs=None,
    threshold=None,
) -> None:
    """Save a confusion matrix heatmap."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=annotate_fmt,
        cmap=cmap,
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


# ---------------- combined F2+loss plots ----------------
def plot_cv_f2_and_loss_combined(
    all_eval_excel: str,
    out_dir: str,
    f2_col_val: str = "val_f2",
    f2_col_train: str = "train_f2",
    loss_col_val: str = "val_loss",
    loss_col_train: str = "train_loss",
    out_name: str = "cv_f2_and_loss_combined.png",
) -> None:
    """Single figure with CV mean±std curves: left axis F2, right axis Loss."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    if not {"epoch", "fold"}.issubset(df.columns):
        print("[Viz] Missing core columns. Need ['epoch','fold'].")
        return

    g = df.groupby("epoch")
    epochs = g.size().index.values

    def _mean_std(col: str):
        if col not in df.columns:
            return None, None
        m = g[col].mean().values
        s = g[col].std(ddof=0).values
        return m, np.where(np.isnan(s), 0.0, s)

    f2_val_m, f2_val_s = _mean_std(f2_col_val)
    f2_tr_m, f2_tr_s = _mean_std(f2_col_train)
    loss_val_m, loss_val_s = _mean_std(loss_col_val)
    loss_tr_m, loss_tr_s = _mean_std(loss_col_train)

    if not any(x is not None for x in (f2_val_m, f2_tr_m, loss_val_m, loss_tr_m)):
        print("[Viz] No F2 or Loss columns found — nothing to plot.")
        return

    fig, ax_f2 = plt.subplots(figsize=(7.6, 4.8))
    lines, labels = [], []

    ax_f2.set_xlabel("Epoch")
    ax_f2.set_ylabel("F2 (β=2)")
    ax_f2.set_ylim(0.0, 1.0)

    if f2_val_m is not None:
        ln, = ax_f2.plot(epochs, f2_val_m, label="val_f2 (mean)")
        ax_f2.fill_between(epochs, f2_val_m - f2_val_s, f2_val_m + f2_val_s, alpha=0.25)
        lines.append(ln)
        labels.append("val_f2 (mean ± sd)")
    if f2_tr_m is not None:
        ln, = ax_f2.plot(epochs, f2_tr_m, linestyle="--", label="train_f2 (mean)")
        ax_f2.fill_between(epochs, f2_tr_m - f2_tr_s, f2_tr_m + f2_tr_s, alpha=0.20)
        lines.append(ln)
        labels.append("train_f2 (mean ± sd)")

    ax_loss = ax_f2.twinx()
    ax_loss.set_ylabel("Loss")

    if loss_val_m is not None:
        ln, = ax_loss.plot(epochs, loss_val_m, label=f"{loss_col_val} (mean)")
        ax_loss.fill_between(epochs, loss_val_m - loss_val_s, loss_val_m + loss_val_s, alpha=0.25)
        lines.append(ln)
        labels.append(f"{loss_col_val} (mean ± sd)")
    if loss_tr_m is not None:
        ln, = ax_loss.plot(epochs, loss_tr_m, linestyle="--", label=f"{loss_col_train} (mean)")
        ax_loss.fill_between(epochs, loss_tr_m - loss_tr_s, loss_tr_m + loss_tr_s, alpha=0.20)
        lines.append(ln)
        labels.append(f"{loss_col_train} (mean ± sd)")

    title_bits = []
    if (f2_val_m is not None) or (f2_tr_m is not None):
        title_bits.append("F2")
    if (loss_val_m is not None) or (loss_tr_m is not None):
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
) -> None:
    """Mean curves only: val_f2 (+ optional train_f2)."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    if not {"epoch", "fold"}.issubset(df.columns):
        print("[Viz] Need columns ['epoch','fold'].")
        return

    plt.figure(figsize=(7.6, 4.8))
    lines, labels = [], []

    if f2_col_val in df.columns:
        e, y = _group_by_epoch_mean(df, f2_col_val)
        ln, = plt.plot(e, y, label="val_f2 (mean)")
        lines.append(ln)
        labels.append("val_f2 (mean)")
    if f2_col_train in df.columns:
        e, y = _group_by_epoch_mean(df, f2_col_train)
        ln, = plt.plot(e, y, label="train_f2 (mean)")
        lines.append(ln)
        labels.append("train_f2 (mean)")

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
) -> None:
    """Mean curves only: val_loss (+ optional train_loss)."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    if not {"epoch", "fold"}.issubset(df.columns):
        print("[Viz] Need columns ['epoch','fold'].")
        return

    plt.figure(figsize=(7.6, 4.8))
    lines, labels = [], []

    if loss_col_val in df.columns:
        e, y = _group_by_epoch_mean(df, loss_col_val)
        ln, = plt.plot(e, y, label=f"{loss_col_val} (mean)")
        lines.append(ln)
        labels.append(f"{loss_col_val} (mean)")
    if loss_col_train in df.columns:
        e, y = _group_by_epoch_mean(df, loss_col_train)
        ln, = plt.plot(e, y, label=f"{loss_col_train} (mean)")
        lines.append(ln)
        labels.append(f"{loss_col_train} (mean)")

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
    Plot CV mean curves (no std) on one figure:
      - Left axis: F2 (val + optional train)
      - Right axis: Loss (val + optional train)
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    if not {"epoch", "fold"}.issubset(df.columns):
        print("[Viz] Missing core columns. Need ['epoch','fold'].")
        return

    cols = set(df.columns)
    if not any(
        [
            f2_col_val in cols,
            f2_col_train in cols,
            loss_col_val in cols,
            loss_col_train in cols,
        ]
    ):
        print("[Viz] No plottable columns found.")
        return

    g = df.groupby("epoch")
    epochs = g.size().index.values

    def _mean(col: str):
        return g[col].mean().values if col in df.columns else None

    f2_val_m = _mean(f2_col_val)
    f2_tr_m = _mean(f2_col_train)
    loss_val_m = _mean(loss_col_val)
    loss_tr_m = _mean(loss_col_train)

    fig, ax_f2 = plt.subplots(figsize=(7.6, 4.8))
    lines, labels = [], []

    ax_f2.set_xlabel("Epoch")
    ax_f2.set_ylabel("F2 (β=2)")
    ax_f2.set_ylim(0.0, 1.0)

    if f2_val_m is not None:
        ln, = ax_f2.plot(epochs, f2_val_m, label="val_f2 (mean)")
        lines.append(ln)
        labels.append("val_f2 (mean)")
    if f2_tr_m is not None:
        ln, = ax_f2.plot(epochs, f2_tr_m, label="train_f2 (mean)")
        lines.append(ln)
        labels.append("train_f2 (mean)")

    ax_loss = ax_f2.twinx()
    ax_loss.set_ylabel("Loss")

    if loss_val_m is not None:
        ln, = ax_loss.plot(epochs, loss_val_m, label=f"{loss_col_val} (mean)")
        lines.append(ln)
        labels.append(f"{loss_col_val} (mean)")
    if loss_tr_m is not None:
        ln, = ax_loss.plot(epochs, loss_tr_m, label=f"{loss_col_train} (mean)")
        lines.append(ln)
        labels.append(f"{loss_col_train} (mean)")

    title_bits = []
    if (f2_val_m is not None) or (f2_tr_m is not None):
        title_bits.append("F2")
    if (loss_val_m is not None) or (loss_tr_m is not None):
        title_bits.append("Loss")
    plt.title("CV Mean: " + " & ".join(title_bits) if title_bits else "CV Mean")

    ax_f2.legend(lines, labels, loc="best")
    fig.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[Viz] Saved mean-only curve → {out_png}")

# --- Boxplot: best epoch per fold pulled from Excel ---
def plot_cv_boxplot(metric_dict: dict[str, list[float]], out_png: str) -> None:
    """
    Render a boxplot for {metric_name: [values per fold]} and save to out_png.
    """
    if not metric_dict:
        print("[Viz] Boxplot: empty metric_dict.")
        return

    metrics = list(metric_dict.keys())
    vals = [metric_dict[m] for m in metrics]
    means = [np.mean(v) for v in vals]

    pos = np.arange(1, len(metrics) + 1)
    plt.figure(figsize=(7.6, 4.8))
    bp = plt.boxplot(vals, positions=pos, widths=0.45, patch_artist=True, showfliers=False)

    # style
    for b in bp["boxes"]:
        b.set_facecolor("#5fa8d3"); b.set_alpha(0.45); b.set_edgecolor("black")
    for w in bp["whiskers"]: w.set_color("black")
    for c in bp["caps"]:     c.set_color("black")
    for m in bp["medians"]:  m.set_color("black"); m.set_linewidth(1.5)

    # mean dots + labels
    for i, mu in enumerate(means, start=1):
        plt.scatter(i, mu, color="black", s=20, zorder=5)

    plt.xlim(0.5, len(metrics) + 0.5)
    plt.ylim(0.80, 1.00)
    plt.xticks(pos, metrics, fontsize=12)
    plt.ylabel("Score", fontsize=13)
    plt.title("Cross-Validation Results Across Metrics", fontsize=14, pad=16)
    for s in ["top", "right", "left", "bottom"]:
        plt.gca().spines[s].set_visible(False)
    plt.gca().yaxis.grid(True, linestyle="-", alpha=0.2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved CV boxplot → {out_png}")


def plot_cv_boxplot_from_excel(
    all_eval_excel: str,
    out_dir: str,
    monitor_col: str = "val_f2",
    out_name: str = "cv_boxplot.png",
) -> None:
    """
    Read all_eval_metrics.xlsx, select the best row per fold using `monitor_col`
    (e.g., val_f2), and plot a boxplot for Accuracy / Precision / Recall / F2.
    Robust to truncated column names like 'val_precisio'.
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] Boxplot skipped: not found → {all_eval_excel}")
        return

    df = pd.read_excel(all_eval_excel)
    if df.empty:
        print("[Viz] Boxplot skipped: empty dataframe.")
        return

    # normalize col names
    def norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")
    df.columns = [norm(c) for c in df.columns]

    def find_col(cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        # prefix fallback (handles 'val_precisio' truncation)
        for c in cands:
            for cc in df.columns:
                if cc.startswith(c):
                    return cc
        return None

    col_fold   = find_col(["fold"])
    col_acc    = find_col(["val_acc", "val_accuracy"])
    col_prec   = find_col(["val_precision", "val_precisio", "val_prec"])
    col_recall = find_col(["val_recall"])
    col_f2     = find_col(["val_f2"])
    col_mon    = find_col([monitor_col]) or col_f2

    needed = [col_fold, col_acc, col_prec, col_recall, col_f2, col_mon]
    if any(c is None for c in needed):
        print(f"[Viz] Boxplot skipped: required columns missing. Have: {df.columns.tolist()}")
        return

    # best epoch per fold by monitor
    best_idx = df.groupby(col_fold)[col_mon].idxmax()
    best = df.loc[best_idx].sort_values(by=col_fold)

    metric_dict = {
        "Accuracy":  best[col_acc].astype(float).tolist(),
        "Precision": best[col_prec].astype(float).tolist(),
        "Recall":    best[col_recall].astype(float).tolist(),
        "F2-score":  best[col_f2].astype(float).tolist(),
    }

    out_png = os.path.join(out_dir, out_name)
    plot_cv_boxplot(metric_dict, out_png)

# ---------------- driver ----------------
def _run_cv_plots(all_eval_path: str, out_dir: str) -> None:
    """Run all CV plots if the Excel source exists."""
    if not os.path.exists(all_eval_path):
        print(f"[Viz] Skipped CV curves: not found → {all_eval_path}")
        return

    plot_cv_mean_std_curves(all_eval_path, out_dir)
    plot_per_fold_curves(all_eval_path, out_dir, metric_col="val_f2")
    plot_per_fold_curves(all_eval_path, out_dir, metric_col="val_loss")
    plot_cv_f2_and_loss_combined(all_eval_path, out_dir)
    plot_cv_f2_and_loss_mean_only(all_eval_path, out_dir)
    plot_cv_mean_loss(all_eval_path, out_dir)
    plot_cv_mean_f2(all_eval_path, out_dir)
    plot_cv_boxplot_from_excel(all_eval_path, out_dir, monitor_col="val_f2", out_name="cv_boxplot.png")
