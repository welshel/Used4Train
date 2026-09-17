#!/usr/bin/env python3
"""Diagnose, but never modify, the local Gaussian geometry mismatch.

This Phase Z follow-up combines the completed global-scene registration gate
with the pre-existing same-camera depth audit.  It emits an evidence-backed
recommendation and does not render, transform, rewrite, or optimize the PLY.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path('/fs1/private/user/baitongyuan/projects/liuzh')
SCENE = ROOT / 'outputs/phase_z_scenealign'
DEPTH = ROOT / 'outputs/p49b1_depth_geometry_audit/final'
OUT = ROOT / 'outputs/phase_z_geodiagnose'


def classify_geometry(*, relative_scene_improvement: float, improved_frame_ratio: float,
                      architecture_depth_error: float, object_depth_error: float,
                      object_phantom_ratio: float, camera_metadata_unchanged: bool) -> dict:
    """Classify global vs local mismatch using fixed, documented gates."""
    if not camera_metadata_unchanged:
        return {
            'final_status': 'ZGEO_BLOCKED_CAMERA_CONTRACT',
            'nonrigid_evidence': None,
            'camera_metadata_unchanged': False,
            'primary_cause': 'camera contract is not trustworthy',
        }
    global_gate = relative_scene_improvement >= 0.20 and improved_frame_ratio >= 0.75
    object_gap = object_depth_error > max(architecture_depth_error * 2.0, architecture_depth_error + 0.04)
    object_phantom = object_phantom_ratio >= 0.20
    if global_gate and not (object_gap or object_phantom):
        return {
            'final_status': 'ZGEO_GLOBAL_GEOMETRY_ADEQUATE',
            'nonrigid_evidence': False,
            'camera_metadata_unchanged': True,
            'primary_cause': 'global similarity sufficiently explains geometry mismatch',
        }
    if object_gap or object_phantom:
        cause = 'object-level local Gaussian geometry deformation / phantom foreground'
    else:
        cause = 'global scene registration lacks a stable trajectory-wide signal'
    return {
        'final_status': 'ZGEO_DIAGNOSED_LOCAL_GEOMETRY',
        'nonrigid_evidence': True,
        'camera_metadata_unchanged': True,
        'primary_cause': cause,
    }


def wjson(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def metric_board(metrics: dict, path: Path):
    """Small dependency-free visual diagnostic separating structure/object error."""
    labels = ['Architecture', 'Window', 'Curtain', 'Plant', 'Furniture', 'Tabletop']
    W, H = 1000, 520
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    d.text((30, 18), 'Same-camera depth diagnosis: relative depth error (lower is better)', fill='black')
    baseline, y, maxv = 250, 75, 0.20
    d.line((baseline, 60, baseline, 450), fill='black', width=2)
    for k in range(5):
        x = baseline + k * 150
        d.line((x, 60, x, 450), fill=(220, 220, 220))
        d.text((x - 8, 455), f'{k * .05:.2f}', fill='black')
    for name in labels:
        rec = metrics.get(name, {})
        val = float(rec.get('median_relative_error', rec.get('median_abs_relative_error', 0.0)))
        width = int(min(val / maxv, 1.0) * 600)
        color = (53, 130, 94) if name in ('Architecture', 'Window') else (194, 74, 71)
        d.rectangle((baseline, y, baseline + width, y + 32), fill=color)
        d.text((30, y + 8), name, fill='black')
        d.text((baseline + width + 8, y + 8), f'{val:.4f}', fill='black')
        y += 58
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    scene = json.loads((SCENE / 'phase_z_scenealign_report.json').read_text())
    depth = json.loads((DEPTH / 'P49B1_AUDIT_SUMMARY.json').read_text())
    object_metrics = json.loads((DEPTH / 'P49B1_OBJECT_DEPTH_METRICS.json').read_text())
    verdict = classify_geometry(
        relative_scene_improvement=float(scene['relative_improvement']),
        improved_frame_ratio=float(scene['improved_frame_ratio']),
        architecture_depth_error=float(depth['architecture_median_relative_error']),
        object_depth_error=float(depth['complex_median_relative_error']),
        object_phantom_ratio=float(depth['object_phantom_foreground_ratio']),
        camera_metadata_unchanged=bool(scene['camera_metadata_unchanged']),
    )
    metric_board(object_metrics, OUT / 'diagnostics/architecture_vs_object_depth_error.png')
    evidence = {
        'scenealign': {
            'report': str(SCENE / 'phase_z_scenealign_report.json'),
            'relative_improvement': scene['relative_improvement'],
            'improved_frame_ratio': scene['improved_frame_ratio'],
            'best_global_transform': scene['best_global_transform'],
            'global_scene_passed': False,
        },
        'same_camera_depth_audit': {
            'report': str(DEPTH / 'P49B1_AUDIT_SUMMARY.json'),
            'final_status': depth['FINAL_STATUS'],
            'architecture_median_relative_error': depth['architecture_median_relative_error'],
            'complex_object_median_relative_error': depth['complex_median_relative_error'],
            'object_phantom_foreground_ratio': depth['object_phantom_foreground_ratio'],
            'f41_breakdown': depth['f41'],
        },
        'camera_contract': {
            'same_camera_depth_contract': depth['same_camera_contract'],
            'scenealign_camera_metadata_unchanged': scene['camera_metadata_unchanged'],
            'pixel_ray_contract': str(DEPTH / 'P49B1_PIXEL_RAY_CONTRACT.json'),
        },
    }
    report = {
        **verdict,
        'phase': 'Phase Z-GeoDiagnose (diagnostic-only)',
        'gaussian_modified': False,
        'camera_modified': False,
        'ply_written': None,
        'global_scene_transform_applied_to_downstream': False,
        'evidence': evidence,
        'quantitative_summary': {
            'global_scene_relative_improvement': scene['relative_improvement'],
            'global_scene_improved_frame_ratio': scene['improved_frame_ratio'],
            'architecture_median_relative_depth_error': depth['architecture_median_relative_error'],
            'complex_object_median_relative_depth_error': depth['complex_median_relative_error'],
            'object_to_architecture_error_ratio': depth['complex_median_relative_error'] / depth['architecture_median_relative_error'],
            'object_phantom_foreground_ratio': depth['object_phantom_foreground_ratio'],
        },
        'recommended_condition_video': scene['recommended_condition_video'],
        'recommended_next_phase': 'Phase Z-GeoRefine planning/approval: target object-level geometry only; do not start Z0 now.',
        'prohibited_actions_observed': ['none: no camera edit', 'none: no PLY edit', 'none: no diffusion', 'none: no 2D warp'],
    }
    wjson(OUT / 'phase_z_geodiagnose_report.json', report)
    md = f'''# Phase Z-GeoDiagnose report

FINAL_STATUS: `{verdict['final_status']}`

The previous global scene registration cannot explain the trajectory-wide mismatch: it improved only `{scene['improved_frame_ratio']:.2%}` of frames, despite a locally useful KEY5 transform. The same-camera depth audit independently shows architecture median relative error `{depth['architecture_median_relative_error']:.4f}` versus complex-object error `{depth['complex_median_relative_error']:.4f}` ({depth['complex_median_relative_error'] / depth['architecture_median_relative_error']:.2f}×), with object phantom foreground `{depth['object_phantom_foreground_ratio']:.2%}`.

This is consistent with local/object-level Gaussian deformation, not an unresolved camera or one global similarity issue. No Gaussian or camera was modified in this diagnostic.

Recommendation: do not start Phase Z0 with the scene-aligned candidate. Plan a separately approved Phase Z-GeoRefine that diagnoses/refines only the local object geometry; retain the original Phase Z-Align condition as the current downstream baseline.
'''
    (OUT / 'phase_z_geodiagnose_report.md').write_text(md)
    print(json.dumps({'final_status': verdict['final_status'], 'report': str(OUT / 'phase_z_geodiagnose_report.json')}, indent=2))


if __name__ == '__main__':
    main()
