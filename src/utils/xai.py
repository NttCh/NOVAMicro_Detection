#!/usr/bin/env python
"""Grad-CAM++ utilities for CNN classifiers: generate heatmaps, overlays, and panels for single images or folders."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ALLOWED_EXTS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".webp", ".gif", ".jfif", ".ppm", ".pgm",
)

# ---------------- Path helpers ----------------
def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _shorten_with_hash(s: str, maxlen: int = 48) -> str:
    """Clamp to maxlen with a short md5 suffix (filesystem-safe)."""
    s = _slugify(s)
    if len(s) <= maxlen:
        return s
    tail = hashlib.md5(s.encode("utf-8")).hexdigest()[:6]
    return f"{s[: maxlen - 7]}_{tail}"


def _win_long(path: str) -> str:
    """Windows extended-length path if on Windows; otherwise unchanged."""
    if os.name == "nt":
        ap = os.path.abspath(path)
        if not ap.startswith("\\\\?\\"):
            ap = "\\\\?\\" + ap
        return ap
    return path


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------- CAM helpers ----------------
def _generate_cam_fused(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
    layer_paths: List[str],
    cam_cls: type,
) -> np.ndarray:
    """Average CAMs from multiple layers (e.g., ['layer2','layer3','layer4'])."""
    cams = []
    for layer in layer_paths:
        gc = cam_cls(model, target_layer=layer, device=device)
        cam = gc.generate(x, class_idx=None)
        gc.close()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cams.append(cam)
    fused = np.mean(cams, axis=0)
    return (fused - fused.min()) / (fused.max() + 1e-8)


def _tta_cam(
    generate_fn: Callable[[torch.Tensor], np.ndarray],
    x: torch.Tensor,
    H: int,
    W: int,
) -> np.ndarray:
    """Flip/rotate TTA for CAMs; returns (H, W) CAM."""
    cams = []

    def _nr(c: np.ndarray) -> np.ndarray:
        c = (c - c.min()) / (c.max() + 1e-8)
        return cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR)

    cams.append(_nr(generate_fn(x)))  # identity

    xr = torch.flip(x, dims=[3])
    cams.append(_nr(np.flip(generate_fn(xr), axis=1)))  # hflip

    xr = torch.flip(x, dims=[2])
    cams.append(_nr(np.flip(generate_fn(xr), axis=0)))  # vflip

    xr = torch.rot90(x, k=1, dims=[2, 3])
    cams.append(_nr(np.rot90(generate_fn(xr), 3)))  # rot90

    out = np.mean(cams, axis=0)
    return (out - out.min()) / (out.max() + 1e-8)


def _smooth_cam(
    generate_fn: Callable[[torch.Tensor], np.ndarray],
    x: torch.Tensor,
    H: int,
    W: int,
    n: int = 6,
    sigma: float = 0.05,
) -> np.ndarray:
    """SmoothGrad-style averaging of CAMs over noisy inputs."""
    cams = []
    for _ in range(n):
        noise = torch.randn_like(x) * sigma
        c = generate_fn((x + noise).clamp(0, 1))
        c = (c - c.min()) / (c.max() + 1e-8)
        cams.append(cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR))
    out = np.mean(cams, axis=0)
    return (out - out.min()) / (out.max() + 1e-8)


def _mask_scale_bar_inplace(rgb: np.ndarray, w_frac: float = 0.18, h_frac: float = 0.08) -> None:
    """White-out a bottom-right rectangle to remove a scale bar (in place)."""
    h, w = rgb.shape[:2]
    bw, bh = int(w_frac * w), int(h_frac * h)
    rgb[h - bh : h, w - bw : w, :] = 255


def _resolve_root(m: nn.Module) -> nn.Module:
    for name in ["backbone", "feature_extractor", "encoder", "model", "net", "network"]:
        if hasattr(m, name) and isinstance(getattr(m, name), nn.Module):
            return getattr(m, name)
    return m


def _get_module_by_path(root: nn.Module, dotted: str) -> nn.Module:
    m = root
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def _find_last_conv(root: nn.Module) -> nn.Module:
    last = None
    for _, mod in root.named_modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM++.")
    return last


def _draw_caption(
    img: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    top_offset: int = 14,
    left_offset: int = 8,
) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x0 = left_offset
    y_top = top_offset
    y0 = y_top + th + pad
    cv2.rectangle(out, (x0 - pad, y_top), (x0 + tw + pad, y0 + pad), (0, 0, 0), -1)
    cv2.putText(out, text, (x0, y0), font, scale, color, thickness, cv2.LINE_AA)
    return out


def _maybe_caption(img: np.ndarray, text: str, enabled: bool) -> np.ndarray:
    return _draw_caption(img, text) if enabled else img


# ---------------- Grad-CAM++ (self-contained) ----------------
class GradCAMPP:
    """Minimal Grad-CAM++ implementation."""

    def __init__(self, model: nn.Module, target_layer: str, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device).eval()

        root = _resolve_root(self.model)
        layer = _find_last_conv(root) if target_layer in (None, "", "auto") else _get_module_by_path(root, target_layer)

        self._feats, self._grads = None, None

        def f_hook(_, __, out):  # noqa: ANN001
            self._feats = out.detach()

        def b_hook(_, grad_in, grad_out):  # noqa: ANN001
            self._grads = grad_out[0].detach()

        self._h1 = layer.register_forward_hook(f_hook)
        self._h2 = layer.register_full_backward_hook(b_hook)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        logits = self.model(x.to(self.device))
        probs = torch.softmax(logits, dim=1)
        p, c = probs.max(dim=1)
        return int(c.item()), float(p.item()), logits

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        x = x.to(self.device).requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        feats, grads = self._feats, self._grads
        grads2, grads3 = grads**2, grads**3
        denom = 2 * grads2 + (feats * grads3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        alpha = grads2 / denom
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = torch.relu((weights * feats).sum(dim=1, keepdim=True))[0, 0]
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        H, W = x.shape[-2:]
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        return cam

    def close(self) -> None:
        self._h1.remove()
        self._h2.remove()


# ---------------- Public API ----------------
def explain_image_path(
    img_path: str,
    transform: A.Compose,
    gradcam: GradCAMPP,
    k_patches: int = 3,
    patch_size: int = 96,
    save_dir: str = ".",
    file_prefix: Optional[str] = None,
    *,
    save_heatmap: bool = True,
    save_overlay: bool = True,
    panel: bool = True,
    output_dirs: Optional[dict] = None,
    class_names: Optional[List[str]] = None,
    fuse_layers: Optional[List[str]] = None,
    tta: bool = True,
    smooth: bool = True,
    smooth_n: int = 6,
    smooth_sigma: float = 0.05,
    mask_scale_bar: bool = True,
    mask_rect_wh: Tuple[float, float] = (0.18, 0.08),
    save_original_overlay: bool = False,
    caption_heatmap: bool = True,
    caption_overlay: bool = True,
    caption_panel: bool = True,
    combo_caption_mode: str = "pred_only",  # "none" | "pred_only" | "full"
    pred_prob_digits: int = 6,
    save_triple_panel: bool = True,
) -> dict:
    """Generate CAM visualizations for one image."""
    os.makedirs(save_dir, exist_ok=True)
    heat_dir = (output_dirs or {}).get("heatmaps", save_dir)
    overlay_dir = (output_dirs or {}).get("overlays", save_dir)
    panel_dir = (output_dirs or {}).get("panels", save_dir)
    evid_dir = (output_dirs or {}).get("evidence", save_dir)
    combo_dir = (output_dirs or {}).get("overlay_with_original", save_dir)
    triple_dir = (output_dirs or {}).get("original_overlay_heatmap", os.path.join(save_dir, "original_overlay_heatmap"))

    for d in (heat_dir, overlay_dir, panel_dir, evid_dir, combo_dir, triple_dir):
        os.makedirs(d, exist_ok=True)

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if mask_scale_bar:
        _mask_scale_bar_inplace(rgb, *mask_rect_wh)
    h0, w0 = rgb.shape[:2]

    aug = transform(image=rgb)
    x = aug["image"]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    elif torch.is_tensor(x):
        x = x.unsqueeze(0)
    else:
        raise TypeError("Unexpected transform output type for 'image'.")

    pred_cls, pred_prob, _ = gradcam.predict(x)
    label_txt = class_names[pred_cls] if (class_names and 0 <= pred_cls < len(class_names)) else str(pred_cls)

    stem_for_caption = (
        file_prefix.strip() if isinstance(file_prefix, str) and file_prefix.strip() else os.path.splitext(os.path.basename(img_path))[0]
    )
    full_caption = f"{stem_for_caption} | pred={label_txt}  p={pred_prob:.{pred_prob_digits}f}"
    pred_only_caption = f"pred={label_txt}  p={pred_prob:.{pred_prob_digits}f}"

    if combo_caption_mode == "none":
        combo_caption_enabled, combo_caption_text = False, ""
    elif combo_caption_mode == "full":
        combo_caption_enabled, combo_caption_text = True, full_caption
    else:
        combo_caption_enabled, combo_caption_text = True, pred_only_caption

    H, W = h0, w0

    def _single(z: torch.Tensor) -> np.ndarray:
        if fuse_layers:
            return _generate_cam_fused(gradcam.model, z, gradcam.device, fuse_layers, type(gradcam))
        return gradcam.generate(z, class_idx=pred_cls)

    base_cam = cv2.resize(_single(x), (W, H), interpolation=cv2.INTER_LINEAR)
    cams = [(base_cam - base_cam.min()) / (base_cam.max() + 1e-8)]
    if tta:
        cams.append(_tta_cam(_single, x, H, W))
    if smooth:
        cams.append(_smooth_cam(_single, x, H, W, n=smooth_n, sigma=smooth_sigma))
    cam = np.mean(cams, axis=0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    stem_raw = file_prefix if file_prefix else os.path.splitext(os.path.basename(img_path))[0]
    stem = _shorten_with_hash(stem_raw, maxlen=60)

    heat_path = overlay_path = panel_path = combo_path = triple_path = ""

    heat_rgb = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]

    if save_heatmap:
        heat_to_save = _maybe_caption(heat_rgb, full_caption, caption_heatmap)
        heat_path = os.path.join(heat_dir, f"{stem}.png")
        _ensure_parent_dir(heat_path)
        Image.fromarray(heat_to_save).save(_win_long(heat_path))

    overlay_img = None
    if save_overlay or panel or save_original_overlay or save_triple_panel:
        overlay_img = (0.5 * heat_rgb + 0.5 * rgb).astype(np.uint8)

    if save_overlay and overlay_img is not None:
        overlay_path = os.path.join(overlay_dir, f"{stem}.png")
        _ensure_parent_dir(overlay_path)
        Image.fromarray(_maybe_caption(overlay_img, full_caption, caption_overlay)).save(_win_long(overlay_path))

    if save_original_overlay and overlay_img is not None:
        left, right = rgb, overlay_img
        if left.shape[0] != right.shape[0]:
            h = min(left.shape[0], right.shape[0])
            lw = int(left.shape[1] * h / max(1, left.shape[0]))
            rw = int(right.shape[1] * h / max(1, right.shape[0]))
            left = cv2.resize(left, (lw, h), interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, (rw, h), interpolation=cv2.INTER_LINEAR)
        side = np.hstack([left, right])
        if combo_caption_enabled:
            side = _draw_caption(side, combo_caption_text)
        combo_path = os.path.join(combo_dir, f"{stem}.png")
        _ensure_parent_dir(combo_path)
        Image.fromarray(side).save(_win_long(combo_path))

    if save_triple_panel and overlay_img is not None:
        left, middle, right = rgb, overlay_img, heat_rgb
        h = min(left.shape[0], middle.shape[0], right.shape[0])

        def _rh(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (int(img.shape[1] * h / max(1, img.shape[0])), h), interpolation=cv2.INTER_LINEAR)

        triple = np.hstack([_rh(left), _rh(middle), _rh(right)])
        if combo_caption_enabled:
            triple = _draw_caption(triple, combo_caption_text)
        triple_path = os.path.join(triple_dir, f"{stem}.png")
        _ensure_parent_dir(triple_path)
        Image.fromarray(triple).save(_win_long(triple_path))

    if panel and overlay_img is not None:
        panel_img = overlay_img.copy()
        ph, pw = patch_size, patch_size
        scores, boxes = [], []
        for y in range(0, h0 - ph + 1, patch_size):
            for x0 in range(0, w0 - pw + 1, patch_size):
                patch = cam[y : y + ph, x0 : x0 + pw]
                scores.append(float(patch.mean()))
                boxes.append((x0, y, x0 + pw, y + ph))
        order = np.argsort(scores)[::-1][:k_patches]
        for i in order:
            x0, y0, x1, y1 = boxes[i]
            cv2.rectangle(panel_img, (x0, y0), (x1, y1), (255, 255, 255), 2)
        panel_img = _maybe_caption(panel_img, full_caption, caption_panel)
        panel_path = os.path.join(panel_dir, f"{stem}.png")
        _ensure_parent_dir(panel_path)
        Image.fromarray(panel_img).save(_win_long(panel_path))

    return {
        "file": os.path.basename(img_path),
        "pred_class": int(pred_cls),
        "pred_label": label_txt,
        "pred_prob": float(pred_prob),
        "heatmap_path": heat_path,
        "overlay_path": overlay_path,
        "panel_path": panel_path,
        "overlay_with_original_path": combo_path,
        "original_overlay_heatmap_path": triple_path,
    }


def explain_folder(
    model: nn.Module,
    transform: A.Compose,
    folder: str,
    target_layer: Union[str, List[str]],
    save_root: str,
    k_patches: int = 3,
    patch_size: int = 96,
    device: Optional[str] = None,
    *,
    save_heatmap: bool = True,
    save_overlay: bool = True,
    panel: bool = True,
    class_names: Optional[List[str]] = None,
    tta: bool = True,
    smooth: bool = True,
    smooth_n: int = 6,
    smooth_sigma: float = 0.05,
    mask_scale_bar: bool = True,
    mask_rect_wh: Tuple[float, float] = (0.18, 0.08),
    save_original_overlay: bool = False,
    caption_heatmap: bool = True,
    caption_overlay: bool = True,
    caption_panel: bool = True,
    combo_caption_mode: str = "pred_only",
    pred_prob_digits: int = 6,
) -> str:
    """Run Grad-CAM++ on all images in a folder (recursively)."""
    if not folder or not os.path.isdir(str(folder)):
        print(f"[XAI] Folder missing or invalid: {folder}. Skipping.")
        return ""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    fuse_layers, first_layer = None, target_layer
    if isinstance(target_layer, Sequence) and not isinstance(target_layer, (str, bytes)):
        fuse_layers = [str(x) for x in list(target_layer)]
        first_layer = fuse_layers[0] if fuse_layers else "auto"
    else:
        first_layer = str(target_layer)

    gc = GradCAMPP(model, target_layer=first_layer, device=device)

    os.makedirs(save_root, exist_ok=True)
    output_dirs = {
        "heatmaps": os.path.join(save_root, "heatmaps"),
        "overlays": os.path.join(save_root, "overlays"),
        "panels": os.path.join(save_root, "panels"),
        "overlay_with_original": os.path.join(save_root, "overlay_with_original"),
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(ALLOWED_EXTS):
                paths.append(os.path.join(root, f))
    paths.sort()
    if len(paths) == 0:
        print(f"[XAI] No image files in '{folder}'. Skipping.")
        gc.close()
        return ""

    records = []
    for p in paths:
        rec = explain_image_path(
            img_path=p,
            transform=transform,
            gradcam=gc,
            k_patches=k_patches,
            patch_size=patch_size,
            save_dir=save_root,
            save_heatmap=save_heatmap,
            save_overlay=save_overlay,
            panel=panel,
            output_dirs=output_dirs,
            class_names=class_names,
            fuse_layers=fuse_layers,
            tta=tta,
            smooth=smooth,
            smooth_n=smooth_n,
            smooth_sigma=smooth_sigma,
            mask_scale_bar=mask_scale_bar,
            mask_rect_wh=mask_rect_wh,
            save_original_overlay=save_original_overlay,
            caption_heatmap=caption_heatmap,
            caption_overlay=caption_overlay,
            caption_panel=caption_panel,
            combo_caption_mode=combo_caption_mode,
            pred_prob_digits=pred_prob_digits,
        )
        records.append(rec)

    csv_path = os.path.join(save_root, "xai_summary.csv")
    _ensure_parent_dir(csv_path)
    pd.DataFrame.from_records(records).to_csv(_win_long(csv_path), index=False)
    gc.close()
    return csv_path


def _maybe_run_xai(
    model: nn.Module,
    transform: A.Compose,
    folder: str,
    dirs: dict,
    cfg_xai: object,
    ckpt_tag: str | None = None,
) -> None:
    """
    Run XAI with Grad-CAM++ if enabled.

    Output path:
      <eval>/xai_by_ckpt/<ckpt_tag>/<folder_name>/
    """
    if not getattr(cfg_xai, "enabled", False):
        return
    if not folder or not os.path.isdir(str(folder)):
        return

    target_layer = getattr(cfg_xai, "target_layer", "auto") or "auto"
    k_patches = int(getattr(cfg_xai, "k_patches", 3))
    patch_size = int(getattr(cfg_xai, "patch_size", 96))
    save_heatmap = bool(getattr(cfg_xai, "save_heatmap", True))
    save_overlay = bool(getattr(cfg_xai, "save_overlay", True))
    save_original_overlay = bool(getattr(cfg_xai, "save_original_overlay", False))
    panel = bool(getattr(cfg_xai, "panel", True))
    class_names = getattr(cfg_xai, "class_names", None)
    tta = bool(getattr(cfg_xai, "tta", True))
    smooth = bool(getattr(cfg_xai, "smooth", True))
    smooth_n = int(getattr(cfg_xai, "smooth_n", 6))
    smooth_sigma = float(getattr(cfg_xai, "smooth_sigma", 0.05))
    mask_scale_bar = bool(getattr(cfg_xai, "mask_scale_bar", True))
    mask_rect_wh = tuple(getattr(cfg_xai, "mask_rect_wh", (0.18, 0.08)))

    caption_heatmap = bool(getattr(cfg_xai, "caption_heatmap", True))
    caption_overlay = bool(getattr(cfg_xai, "caption_overlay", True))
    caption_panel = bool(getattr(cfg_xai, "caption_panel", True))
    combo_caption_mode = str(getattr(cfg_xai, "combo_caption_mode", "pred_only")).lower()
    pred_prob_digits = int(getattr(cfg_xai, "pred_prob_digits", 6))

    folder_name = Path(folder).name
    folder_tag = _shorten_with_hash(folder_name, maxlen=48)

    if ckpt_tag:
        ckpt_tag_s = _shorten_with_hash(_slugify(ckpt_tag), maxlen=32)
        base_out = os.path.join(dirs["eval"], "xai_by_ckpt", ckpt_tag_s)
    else:
        base_out = os.path.join(dirs["eval"], "xai_by_ckpt", "default")

    xai_out = os.path.join(base_out, folder_tag)
    os.makedirs(xai_out, exist_ok=True)

    print(f"→ XAI [Grad-CAM++] for '{folder_name}' (layer={target_layer}) → '{xai_out}'")

    explain_folder(
        model=model,
        transform=transform,
        folder=folder,
        target_layer=target_layer,
        save_root=xai_out,
        k_patches=k_patches,
        patch_size=patch_size,
        save_heatmap=save_heatmap,
        save_overlay=save_overlay,
        panel=panel,
        class_names=class_names,
        tta=tta,
        smooth=smooth,
        smooth_n=smooth_n,
        smooth_sigma=smooth_sigma,
        mask_scale_bar=mask_scale_bar,
        mask_rect_wh=mask_rect_wh,
        save_original_overlay=save_original_overlay,
        caption_heatmap=caption_heatmap,
        caption_overlay=caption_overlay,
        caption_panel=caption_panel,
        combo_caption_mode=combo_caption_mode,
        pred_prob_digits=pred_prob_digits,
    )
