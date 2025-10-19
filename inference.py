#!/usr/bin/env python
"""Inference and evaluation utilities."""

import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from typing import Optional
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
import glob
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import train_test_split
import config
from data import PatchClassificationDataset
from utils import load_obj
from viz import save_confusion_matrix

def evaluate_model(model, csv_path: str, cfg, stage: str) -> None:
    """
    Evaluate on a validation split, save a confusion matrix PNG via viz.save_confusion_matrix,
    and print a short report.

    PNG path: <config.BASE_SAVE_DIR>/eval/confusion_matrix_<stage or best-fold>.png
    """
    # ---- build validation split ----
    df = pd.read_csv(csv_path)
    _, valid_df = train_test_split(
        df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=df[cfg.data.label_col],
    )

    # ---- transforms & loader ----
    tf = A.Compose([
        load_obj(aug["class_name"])(**aug["params"])
        for aug in cfg.augmentation.valid.augs
    ])
    ds = PatchClassificationDataset(
        valid_df,
        cfg.data.folder_path,
        transforms=tf,
        image_col=getattr(cfg.data, "image_col", "filename"),
        label_col=cfg.data.label_col,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    # ---- forward pass ----
    device = next(model.parameters()).device
    model.eval()

    probs_pos, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbs = lbs.to(device, non_blocking=True)

            logits = model(imgs)
            if logits.shape[-1] == 1:           # sigmoid head
                p1 = torch.sigmoid(logits.flatten())
            else:                                # softmax (2-class)
                p1 = torch.softmax(logits, dim=1)[:, 1]

            probs_pos.append(p1.detach().cpu().numpy())
            labels.append(lbs.detach().cpu().numpy())

    probs_pos = np.concatenate(probs_pos, axis=0) if probs_pos else np.array([])
    labels    = np.concatenate(labels, axis=0)    if labels    else np.array([])

    # ---- choose a threshold (fixed 0.5 here) & preds ----
    thr   = 0.5
    preds = (probs_pos >= thr).astype(int)

    # ---- save CM via viz helper ----
    eval_dir = os.path.join(config.BASE_SAVE_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    is_best_fold = bool(getattr(model, "best_ckpt_path", None))
    title_tag = "Best fold" if is_best_fold else str(stage)
    cm_title = f"Confusion Matrix: {title_tag}"
    cm_name  = f"confusion_matrix_{'best-fold' if is_best_fold else stage}.png"
    cm_path  = os.path.join(eval_dir, cm_name)

    save_confusion_matrix(
        y_true=labels,
        y_pred=preds,
        out_path=cm_path,
        title=cm_title,
        labels=("0", "1"),
        cmap="Blues",
    )
    print(f"[Evaluate] Saved confusion matrix → {cm_path}")

    # ---- console report ----
    print(classification_report(labels, preds, digits=4, zero_division=0))
    f2 = fbeta_score(labels, preds, beta=2, average="weighted", zero_division=0)
    print(f"[Evaluate] Weighted F2 @ {thr:.2f}: {f2:.4f}")

def predict_test_folder(
    model: torch.nn.Module,
    test_folder: str,
    transform,
    output_excel: str,
    print_results: bool = True,
    model_path: Optional[str] = None
) -> None:
    """
    Predict on all images in a test folder and save results in an Excel file.

    Args:
        model (torch.nn.Module): The trained model.
        test_folder (str): Path to the test images folder.
        transform (Any): Albumentations transform for inference.
        output_excel (str): Path to save the Excel file with predictions.
        print_results (bool): Whether to print predictions.
        model_path (str): Optional path to a model checkpoint to load.
    """
    if not test_folder or test_folder.lower() == "none":
        print("No test folder provided. Skipping test predictions.")
        return

    if model_path and model_path.lower() != "none":
        print(f"Loading model checkpoint from {model_path}")
        state = torch.load(model_path, map_location=torch.device("cpu"))
        # Handle both plain state_dict and Lightning checkpoints with "state_dict"
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("model."):
                new_key = k[len("model."):]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

    image_files = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No image files found in test folder. Skipping test predictions.")
        return

    predictions = []
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for file in image_files:
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read {file}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            augmented = transform(image=image)
            image_tensor = augmented["image"]
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.tensor(image_tensor)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).item()
            prob = torch.softmax(logits, dim=1)[0, 1].item()

            predictions.append({"filename": file, "predicted_label": pred, "probability": prob})
            if print_results:
                print(f"File: {file} -> Predicted Label: {pred}, Prob: {prob:.8f}")

    predictions = sorted(predictions, key=lambda x: x["filename"])
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Predictions"
    ws.append(["Filename", "Predicted Label", "Probability", "Image"])

    row_num = 2
    for pred in predictions:
        ws.append([pred["filename"], pred["predicted_label"], pred["probability"]])
        try:
            img = XLImage(pred["filename"])
            img.width = 100
            img.height = 100
            cell_ref = f"D{row_num}"
            ws.add_image(img, cell_ref)
        except Exception as e:
            print(f"Could not insert image for {pred['filename']}: {e}")
        row_num += 1

    wb.save(output_excel)
    print(f"Saved test predictions with images to {output_excel}")

    pred_df = pd.DataFrame(predictions)
    print("\nFinal Test Predictions:")
    print(pred_df.to_string(index=False))


def apply_compose(augs, x):
    """
    Helper to apply a list of Albumentations transforms sequentially (manually),
    for times when you cannot directly create a Compose object.

    Args:
        augs (list): List of Albumentations transforms.
        x (dict): Dict with "image": <image>.

    Returns:
        dict: Dict with transformed "image".
    """
    image = x["image"]
    for aug in augs:
        image = aug(image=image)["image"]
    return {"image": image}
    
def _short_sheet(name: str, maxlen: int = 28) -> str:
    """Excel sheet-name safe: strip illegal chars and clamp length."""
    safe = "".join(c for c in str(name) if c not in r'[]:*?/\\')
    return safe[:maxlen] if len(safe) > maxlen else safe

@torch.no_grad()
def predict_folder_to_df(
    model: torch.nn.Module,
    folder: str,
    transform: A.Compose,
    *,
    device: str | torch.device = None,
) -> pd.DataFrame:
    """
    Run model on all images in a folder and return a DataFrame:
    columns: ['filename', 'predicted_label', 'probability']
    - Works for binary or multi-class; for multi-class, 'probability' = max-softmax.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for pat in exts:
        files.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    files = sorted(set(files))

    rows = []
    for f in files:
        img_bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug = transform(image=img)
        x = aug["image"]

        # Accept both numpy HWC and already-to-tensor cases
        if isinstance(x, np.ndarray):
            # HWC -> CHW, float32
            x = np.transpose(x, (0,1,2)) if x.ndim == 3 else x
            x = np.transpose(x, (2,0,1)).astype("float32")
            xt = torch.from_numpy(x)
        else:
            xt = x  # assume tensor

        if xt.ndim == 3:
            xt = xt.unsqueeze(0)  # NCHW
        xt = xt.to(device)

        logits = model(xt)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        rows.append({
            "filename": os.path.basename(f),
            "predicted_label": int(pred.item()),
            "probability": float(conf.item()),
        })

    return pd.DataFrame(rows, columns=["filename", "predicted_label", "probability"])


def save_multi_sheet_workbook(results_by_folder: dict[str, pd.DataFrame], out_xlsx: str) -> None:
    """
    One workbook, one sheet per folder + a Summary sheet.
    results_by_folder: {folder_name: DataFrame}
    """
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # detail sheets
        used_names = set()
        for folder_name, df in results_by_folder.items():
            sheet = _short_sheet(folder_name)
            base = sheet
            suffix = 1
            while sheet in used_names:
                sheet = _short_sheet(f"{base}_{suffix}")
                suffix += 1
            used_names.add(sheet)
            (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=sheet, index=False)

        # summary sheet
        summary_rows = []
        for folder_name, df in results_by_folder.items():
            if df is None or df.empty:
                c1 = c0 = n = 0
            else:
                n = int(len(df))
                c1 = int((df["predicted_label"] == 1).sum()) if "predicted_label" in df.columns else 0
                c0 = int((df["predicted_label"] == 0).sum()) if "predicted_label" in df.columns else 0
            summary_rows.append({"folder": folder_name, "n_images": n, "pred_1": c1, "pred_0": c0})
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)


def save_multi_ckpt_comparison(all_results: dict[str, dict[str, pd.DataFrame]], out_xlsx: str) -> None:
    """
    all_results: {ckpt_tag: {folder_name: df(...)}}
    Creates a sheet per folder that merges predictions by filename across ckpts,
    plus a summary sheet of 0/1 counts per (ckpt, folder).
    """
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    summary = []

    # union of all folders
    all_folders = set()
    for d in all_results.values():
        all_folders.update(d.keys())
    all_folders = sorted(all_folders)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        used_names = set()
        for folder in all_folders:
            merged = None
            for ckpt_tag, by_folder in all_results.items():
                df = by_folder.get(folder, pd.DataFrame(columns=["filename", "predicted_label", "probability"]))
                df = df[["filename", "predicted_label", "probability"]].copy()
                df = df.rename(columns={
                    "predicted_label": f"pred_{ckpt_tag}",
                    "probability": f"prob_{ckpt_tag}"
                })
                merged = df if merged is None else merged.merge(df, on="filename", how="outer")

                src = by_folder.get(folder, pd.DataFrame())
                if not src.empty and "predicted_label" in src.columns:
                    c1 = int((src["predicted_label"] == 1).sum())
                    c0 = int((src["predicted_label"] == 0).sum())
                else:
                    c1 = c0 = 0
                summary.append({"folder": folder, "ckpt": ckpt_tag, "pred_1": c1, "pred_0": c0})

            sheet = _short_sheet(folder + "_cmp")
            base = sheet
            suffix = 1
            while sheet in used_names:
                sheet = _short_sheet(f"{base}_{suffix}")
                suffix += 1
            used_names.add(sheet)

            (merged if merged is not None else pd.DataFrame()).to_excel(writer, sheet_name=sheet, index=False)

        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary_All", index=False)


def predict_folders_to_combined_workbook(
    model: torch.nn.Module,
    folders: list[str],
    transform: A.Compose,
    combined_xlsx_path: str,
    *,
    ckpt_print_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """
    For a given model/ckpt:
      - Builds a dict {folder_name: df}
      - Writes a single combined workbook (one sheet per folder)
      - Prints only a tidy line per folder
    Returns the dict (for cross-ckpt comparison).
    """
    results_by_folder: dict[str, pd.DataFrame] = {}

    for folder in folders:
        folder_name = Path(folder).name
        print(f"{ckpt_print_prefix}→ Predicting on '{folder_name}'")
        df = predict_folder_to_df(model, folder, transform)
        results_by_folder[folder_name] = df

    save_multi_sheet_workbook(results_by_folder, combined_xlsx_path)
    print(f"[Pred] Wrote multi-sheet workbook → {combined_xlsx_path}")
    return results_by_folder
