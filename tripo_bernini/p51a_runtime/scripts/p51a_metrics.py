"""Deterministic RGB video metrics for P51A validation outputs."""
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def _frames(directory: Path):
    return [np.asarray(Image.open(directory / f'F{i:02d}.png').convert('RGB'), dtype=np.float32) / 255.0 for i in range(72)]


def _ssim_frame(a, b):
    # Global SSIM (window-free fallback, no third-party dependency).
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float((2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def compare_dirs(pred_dir: Path, target_dir: Path) -> Dict[str, float]:
    pred, target = _frames(pred_dir), _frames(target_dir)
    mse = float(np.mean([(a - b) ** 2 for a, b in zip(pred, target)]))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    ssim = float(np.mean([_ssim_frame(a, b) for a, b in zip(pred, target)]))
    pred_delta = [pred[i + 1] - pred[i] for i in range(71)]
    target_delta = [target[i + 1] - target[i] for i in range(71)]
    temporal_delta_error = float(np.mean([(a - b) ** 2 for a, b in zip(pred_delta, target_delta)]))
    flicker_proxy = float(np.mean([np.abs(a.mean((0, 1)) - b.mean((0, 1))).mean() for a, b in zip(pred_delta, target_delta)]))
    # Exposure match by a least-squares affine scalar per frame.
    emse_values, essim_values = [], []
    for a, b in zip(pred, target):
        x = a.reshape(-1)
        y = b.reshape(-1)
        scale = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        aa = np.clip(a * scale, 0.0, 1.0)
        emse_values.append(np.mean((aa - b) ** 2))
        essim_values.append(_ssim_frame(aa, b))
    return {
        'mse': mse,
        'psnr_db': psnr,
        'ssim_global': ssim,
        'exposure_matched_mse': float(np.mean(emse_values)),
        'exposure_matched_ssim_global': float(np.mean(essim_values)),
        'temporal_delta_error': temporal_delta_error,
        'flicker_proxy': flicker_proxy,
        'frame_count': 72,
    }


def write_metrics(pred_dir: str, target_dir: str, output_json: str):
    result = compare_dirs(Path(pred_dir), Path(target_dir))
    Path(output_json).write_text(json.dumps(result, indent=2) + '\n')
    return result
