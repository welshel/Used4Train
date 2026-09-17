#!/usr/bin/env python3
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

root = Path('/fs1/private/user/baitongyuan/projects/liuzh/outputs/phase_z_scenealign')
cond = Path('/fs1/private/user/baitongyuan/projects/liuzh/outputs/p48_tripo_clean_aligned/condition_rgb')
clean = Path('/fs1/private/user/baitongyuan/projects/liuzh/outputs/p48_tripo_clean_aligned/clean_rgb')
scene = root / 'full72/condition_rgb'

def edges(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.morphologyEx(cv2.Canny(g, 40, 110), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

def score(a, b):
    ea, eb = edges(a), edges(b)
    if not ea.any() or not eb.any():
        return 0.5, bool(not ea.any())
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    return min(0.5, float(.5 * (db[ea].mean() + da[eb].mean()) / 896)), False

rows = []
for i in range(72):
    a = np.asarray(Image.open(cond / f'F{i:02d}.png').convert('RGB'))
    b = np.asarray(Image.open(clean / f'F{i:02d}.png').convert('RGB'))
    s = np.asarray(Image.open(scene / f'F{i:02d}.png').convert('RGB'))
    x, empty_a = score(a, b)
    y, empty_s = score(s, b)
    rows.append({'frame': i, 'identity_edge_error': x, 'global_edge_error': y,
                 'improved': bool(y < x), 'delta': float(x - y),
                 'identity_empty_edge': empty_a, 'global_empty_edge': empty_s})

mi = float(np.mean([r['identity_edge_error'] for r in rows]))
mg = float(np.mean([r['global_edge_error'] for r in rows]))
rel = float((mi - mg) / max(mi, 1e-9))
ratio = float(np.mean([r['improved'] for r in rows]))
report_path = root / 'phase_z_scenealign_report.json'
obj = json.loads(report_path.read_text())
metric_def = 'blurred Canny symmetric edge Chamfer, normalized by 896 and capped at 0.5; empty edge map receives finite 0.5 missing-structure penalty'
obj['metrics_identity'] = {'edge_error': mi, 'metric_definition': metric_def}
obj['metrics_global'] = {'edge_error': mg, 'metric_definition': metric_def}
obj['relative_improvement'] = rel
obj['improved_frame_ratio'] = ratio
obj['improved_frames'] = int(sum(r['improved'] for r in rows))
obj['worsened_frames'] = int(sum(not r['improved'] for r in rows))
obj['failure_reason'] = 'Single global scene transform improves only 26/72 frames (36.11%); it is not trajectory-wide and fails the >=75% global alignment gate.'
report_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
full = json.loads((root / 'full72/metrics.json').read_text())
full.update({'identity_mean_edge_error': mi, 'global_mean_edge_error': mg, 'relative_improvement': rel,
             'improved_frame_ratio': ratio, 'improved_frames': int(sum(r['improved'] for r in rows)),
             'worsened_frames': int(sum(not r['improved'] for r in rows)),
             'metric_definition': metric_def, 'rows': rows})
(root / 'full72/metrics.json').write_text(json.dumps(full, indent=2) + '\n')
md = f'''# Phase Z-SceneAlign report

FINAL_STATUS: `ZSCENE_FAIL_NO_GLOBAL_ALIGNMENT_SIGNAL`

## Transform

- Scene up axis: Z (`ssl_z_up`). Cameras were not changed.
- Best global sim5 (KEY5 objective): yaw `8.633914` deg; translation `[tx=-0.422291, ty=0.161205, tz=-0.106256]` m; scale `1.03541020`.
- Transform acts on Gaussian world coordinates at runtime; source `splat.ply` was not overwritten and no transformed PLY was emitted.

## Metrics

- Identity mean structural edge error: `{mi:.8f}`.
- Global sim5 mean structural edge error: `{mg:.8f}`.
- Relative improvement: `{rel:.4%}`.
- Improved frame ratio: `{ratio:.4%}` ({sum(r['improved'] for r in rows)}/72); {sum(not r['improved'] for r in rows)} frames worsened.
- KEY5 identity / yaw-only / planar-sim4 / sim5: `0.08469920` / `0.06358119` / `0.06517829` / `0.05277020`.
- Metric: {metric_def}.

## Interpretation

The global transform helps selected views but is not stable across the trajectory. Diagnostic per-frame yaw optima are inconsistent (`-6°, -4°, 0°, +4°` across KEY5), supporting non-rigid/object-level or view-dependent geometry mismatch. Per-frame transforms were diagnostic only and were not used for final output.

`NONRIGID_EVIDENCE = true` as a diagnosis; no local deformation was applied. Do not enter Z0 with the scene-aligned candidate without resolving geometry mismatch or choosing a restricted-view experiment.

Source topdown found: `True`. Camera metadata hashes are unchanged before/after. No 2D warp, crop, diffusion, or camera modification was used.

Outputs: `{root/'full72/aligned_scene_global.mp4'}`, `{root/'comparison_scenealign.mp4'}`, `{root/'edge_comparison.mp4'}`, `{root/'transforms/best_global_sim5.json'}`.
'''
(root / 'phase_z_scenealign_report.md').write_text(md)
print(json.dumps({'identity_mean': mi, 'global_mean': mg, 'relative_improvement': rel,
                  'improved_frame_ratio': ratio, 'status': obj['final_status']}, indent=2))
