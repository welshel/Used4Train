#!/usr/bin/env python3
"""Phase Z-SceneAlign: global Gaussian-only scene registration.

This script keeps the camera JSONs immutable and applies one constrained
similarity transform to the Gaussian world.  It reuses the verified P48
gsplat renderer; no image warp or generated content is used.
"""
from __future__ import annotations

import csv, hashlib, importlib.util, itertools, json, math, os, shutil, sys, subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path('/fs1/private/user/baitongyuan/projects/liuzh')
OUT = ROOT / 'outputs/phase_z_scenealign'
DATA = ROOT / 'work/327431980_4_normalized'
SPLAT = ROOT / 'splat.ply'
RENDER = ROOT / 'scripts/render_tripo_clean_path.py'
ALIGN = ROOT / 'outputs/phase_z_align'
COND = ROOT / 'outputs/p48_tripo_clean_aligned/condition_rgb'
CLEAN = ROOT / 'outputs/p48_tripo_clean_aligned/clean_rgb'
KEY5 = [0, 18, 36, 54, 71]
ALL = list(range(72))


def wjson(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load_renderer():
    # Use the already-built extension, avoiding JIT recompilation.
    import gsplat
    so = ROOT / 'tmp/torch_extensions/gsplat_cuda/gsplat_cuda.so'
    if not hasattr(gsplat, 'csrc'):
        s = importlib.util.spec_from_file_location('gsplat_cuda', so)
        m = importlib.util.module_from_spec(s); sys.modules['gsplat_cuda'] = m; s.loader.exec_module(m)
        gsplat.csrc = m; sys.modules['gsplat.csrc'] = m
    s = importlib.util.spec_from_file_location('p48_renderer', RENDER)
    m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m)
    return m


def extra_matrix(params):
    """params = [yaw_deg, tx, ty, scale, tz], acting in clean world coordinates."""
    yaw, tx, ty, scale, tz = [float(x) for x in params]
    R = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = scale * R
    M[:3, 3] = [tx, ty, tz]
    return M


def load_assets(mod):
    xyz, rgb, opacity, scales, quats, ply_audit = mod.load_ply(SPLAT)
    cams = mod.load_clean_cameras(DATA)
    base = np.asarray(json.loads((ALIGN / 'baseline/repro/P48_TRIPO_ALIGNMENT.json').read_text())['transform'], np.float64)
    clean = {i: np.asarray(Image.open(CLEAN / f'F{i:02d}.png').convert('RGB')) for i in KEY5}
    return xyz, rgb, opacity, scales, quats, ply_audit, cams, base, clean


def img_edges(im):
    # Blur suppresses Gaussian texture noise while retaining room/furniture boundaries.
    g = cv2.cvtColor(np.uint8(np.clip(im, 0, 255)), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    e = cv2.Canny(g, 40, 110)
    # A mild morphological close makes broken Gaussian edges less dominant.
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return e > 0


def edge_score(cond, clean):
    ec, eg = img_edges(cond), img_edges(clean)
    if not ec.any() or not eg.any(): return 1e6
    dc = cv2.distanceTransform((~ec).astype(np.uint8), cv2.DIST_L2, 3)
    dg = cv2.distanceTransform((~eg).astype(np.uint8), cv2.DIST_L2, 3)
    # Symmetric, normalized structural edge Chamfer.  Appearance is not used.
    v = 0.5 * float(dg[ec].mean()) + 0.5 * float(dc[eg].mean())
    return v / max(cond.shape[0], cond.shape[1])


def render(mod, assets, params, indices, device='cuda'):
    xyz, rgb, op, sc, q, _, cams, base, _ = assets
    M = extra_matrix(params) @ base
    out = mod.render_frames(xyz, rgb, op, sc, q, cams, M, indices, device=device)
    return M, {i: np.uint8(np.clip(out[i][0], 0, 1) * 255) for i in indices}, {i: out[i][1] for i in indices}


def make_video(frame_dir, out, fps=12):
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(frame_dir/'F%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','17','-movflags','+faststart',str(out)], check=True)


def save_contact(rows, path, tile=256):
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (tile * len(rows[0]), tile * len(rows)), (16,16,16))
    d = ImageDraw.Draw(canvas)
    for y, row in enumerate(rows):
        for x, (arr, label) in enumerate(row):
            im = Image.fromarray(np.asarray(arr).astype(np.uint8)).convert('RGB').resize((tile,tile), Image.Resampling.LANCZOS)
            canvas.paste(im, (x*tile,y*tile)); d.text((x*tile+5,y*tile+5), label, fill='white')
    canvas.save(path)


def metric_rows(mod, assets, params, indices, label, save_dir=None, device='cuda'):
    M, frames, alpha = render(mod, assets, params, indices, device=device)
    clean = assets[-1]
    rows=[]
    for i in indices:
        s=edge_score(frames[i], clean[i])
        rows.append({'frame':i,'edge_error':s,'alpha_coverage':float((alpha[i]>1e-3).mean())})
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True); Image.fromarray(frames[i]).save(save_dir/f'F{i:02d}.png')
    return M, frames, rows


def score_from_rows(rows): return float(np.mean([r['edge_error'] for r in rows]))


def main():
    for d in ['audit','diagnostics','key5','yaw_only','planar_sim4','global_sim5','full72','transforms','renders']:
        (OUT/d).mkdir(parents=True, exist_ok=True)
    mod = load_renderer()
    assets = load_assets(mod)
    xyz, rgb, op, sc, q, ply_audit, cams, base, clean = assets
    identity = np.zeros(5, dtype=np.float64); identity[3] = 1.0

    # A: read and freeze camera contract.
    cam_jsons = [Path(c.json_path) for c in cams]
    before_hash = [sha(p) for p in cam_jsons]
    wjson(OUT/'audit/input_audit.json', {
        'phase_z_align_report': str(ALIGN/'phase_z_align_report.json'), 'gaussian_path':str(SPLAT),
        'renderer': 'gsplat', 'renderer_entrypoint':str(RENDER), 'num_frames':72,
        'resolution':[cams[0].width,cams[0].height], 'camera_convention':'OpenCV C2W, +X right, +Y down, +Z forward',
        'up_axis':'Z (ssl_z_up)', 'camera_metadata_sha256_before':before_hash,
        'camera_metadata_source':str(DATA/'video_center_72'), 'K_first':cams[0].K.tolist(), 'K_last':cams[-1].K.tolist(),
        'ply_audit':ply_audit, 'base_scene_transform_from_p48':base.tolist(),
    })

    # B: scene transform convention sanity (+2deg yaw and +0.05m translation).
    sanity_params = np.array([2.0, 0.05, 0.0, 1.0, 0.0])
    Ms, sanity_frames, sanity_alpha = render(mod, assets, sanity_params, [36])
    (OUT/'diagnostics/scene_transform_sanity').mkdir(parents=True, exist_ok=True)
    Image.fromarray(sanity_frames[36]).save(OUT/'diagnostics/scene_transform_sanity/yaw_plus2.png')
    _, trans_frames, _ = render(mod, assets, np.array([0.,0.05,0.,1.,0.]), [36])
    Image.fromarray(trans_frames[36]).save(OUT/'diagnostics/scene_transform_sanity/translation_plus005m.png')
    # Camera matrices are not passed through any transform and hashes remain unchanged.
    after_hash = [sha(p) for p in cam_jsons]
    sanity_meta = {'scene_up_axis':'Z','transform_acts_on':'Gaussian world coordinates','yaw_deg':2.0,'translation_xyz':[0.05,0.,0.], 'camera_parameters_unchanged':bool(before_hash==after_hash), 'camera_hashes_before':before_hash, 'camera_hashes_after':after_hash, 'orientation_transform':'global rotation is applied to Gaussian quaternion via renderer transform_quaternions', 'scale_transform':'linear Gaussian scales multiplied by global scale; PLY stores log scales and renderer applies runtime matrix', 'passed': bool(before_hash==after_hash and np.isfinite(sanity_frames[36]).all())}
    wjson(OUT/'diagnostics/scene_transform_sanity/sanity.json', sanity_meta)
    if not sanity_meta['passed']: raise RuntimeError('scene transform sanity failed')

    # C: KEY5 identity baseline.
    _, _, identity_rows = metric_rows(mod, assets, identity, KEY5, 'identity', OUT/'key5/identity')
    wjson(OUT/'key5/identity_metrics.json', {'params':identity.tolist(),'rows':identity_rows,'mean_edge_error':score_from_rows(identity_rows)})

    # D: yaw-only sweep.
    yaw_rows=[]
    for yaw in np.arange(-15., 15.01, 2.0):
        p=np.array([yaw,0.,0.,1.,0.]); _,_,r=metric_rows(mod,assets,p,KEY5,'yaw')
        yaw_rows.append({'yaw_deg':float(yaw),'edge_error':score_from_rows(r),'rows':r})
    best_yaw=min(yaw_rows,key=lambda x:x['edge_error'])
    wjson(OUT/'yaw_only/yaw_sweep.json', {'rows':yaw_rows,'best':best_yaw})
    with (OUT/'yaw_only/yaw_sweep.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['yaw_deg','edge_error']);w.writerows([[x['yaw_deg'],x['edge_error']] for x in yaw_rows])
    # Curve image.
    W,H=900,450; canvas=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(canvas); vals=[x['edge_error'] for x in yaw_rows]; lo,hi=min(vals),max(vals); pts=[]
    for x in yaw_rows: pts.append((70+(x['yaw_deg']+15)/30*780, 380-(x['edge_error']-lo)/max(hi-lo,1e-9)*300))
    d.line(pts,fill='blue',width=3); d.text((70,30),f'Yaw sweep; best={best_yaw["yaw_deg"]:.1f} deg',fill='black'); canvas.save(OUT/'yaw_only/yaw_loss_curve.png')
    p_yaw=np.array([best_yaw['yaw_deg'],0.,0.,1.,0.]); _,_,rows_yaw=metric_rows(mod,assets,p_yaw,KEY5,'best_yaw',OUT/'yaw_only/best_yaw_rgb')

    # E/F: constrained joint optimizations. Powell is deliberately bounded by
    # physically conservative indoor ranges; one transform is shared by KEY5.
    cache={}
    def objective4(x):
        p=np.array([x[0],x[1],x[2],x[3],0.]); key=tuple(np.round(p,5))
        if key not in cache:
            _,_,rr=metric_rows(mod,assets,p,KEY5,'opt'); cache[key]=score_from_rows(rr)
        return cache[key]
    res4=minimize(objective4, [best_yaw['yaw_deg'],0.,0.,1.], method='Powell', bounds=[(-20,20),(-0.8,0.8),(-0.8,0.8),(0.85,1.15)], options={'maxiter':12,'xtol':0.15,'ftol':1e-4,'disp':False})
    p4=np.array([res4.x[0],res4.x[1],res4.x[2],res4.x[3],0.]); _,_,rows4=metric_rows(mod,assets,p4,KEY5,'planar',OUT/'planar_sim4/rgb')
    wjson(OUT/'transforms/identity.json', {'yaw_deg':0.,'translation_xyz':[0.,0.,0.],'scale':1.,'rotation_matrix':np.eye(3).tolist(),'transform_matrix_4x4':base.tolist(),'transform_acts_on':'Gaussian world coordinates; runtime composed as extra @ P48 base'})
    wjson(OUT/'transforms/best_yaw_only.json', {'yaw_deg':float(p_yaw[0]),'translation_xyz':[0.,0.,0.],'scale':1.,'rotation_matrix':extra_matrix(p_yaw)[:3,:3].tolist(),'transform_matrix_4x4':(extra_matrix(p_yaw)@base).tolist(),'transform_acts_on':'Gaussian world coordinates'})
    wjson(OUT/'transforms/best_planar_sim4.json', {'yaw_deg':float(p4[0]),'translation_xyz':p4[[1,2,4]].tolist(),'scale':float(p4[3]),'rotation_matrix':extra_matrix(p4)[:3,:3].tolist(),'transform_matrix_4x4':(extra_matrix(p4)@base).tolist(),'transform_acts_on':'Gaussian world coordinates'})
    wjson(OUT/'planar_sim4/planar_sim4_result.json', {'params':p4.tolist(),'success':bool(res4.success),'message':str(res4.message),'mean_edge_error':score_from_rows(rows4),'identity_mean_edge_error':score_from_rows(identity_rows),'rows':rows4,'evaluations':len(cache)})
    def objective5(x):
        p=np.array([x[0],x[1],x[2],x[3],x[4]]); key=tuple(np.round(p,5))
        if key not in cache:
            _,_,rr=metric_rows(mod,assets,p,KEY5,'opt'); cache[key]=score_from_rows(rr)
        return cache[key]
    res5=minimize(objective5, p4, method='Powell', bounds=[(-20,20),(-0.8,0.8),(-0.8,0.8),(0.85,1.15),(-0.6,0.6)], options={'maxiter':14,'xtol':0.12,'ftol':1e-4,'disp':False})
    p5=np.asarray(res5.x); _,_,rows5=metric_rows(mod,assets,p5,KEY5,'global',OUT/'global_sim5/rgb')
    wjson(OUT/'transforms/best_global_sim5.json', {'yaw_deg':float(p5[0]),'translation_xyz':p5[[1,2,4]].tolist(),'scale':float(p5[3]),'rotation_matrix':extra_matrix(p5)[:3,:3].tolist(),'transform_matrix_4x4':(extra_matrix(p5)@base).tolist(),'transform_acts_on':'Gaussian world coordinates'})
    wjson(OUT/'global_sim5/global_sim5_result.json', {'params':p5.tolist(),'success':bool(res5.success),'message':str(res5.message),'mean_edge_error':score_from_rows(rows5),'identity_mean_edge_error':score_from_rows(identity_rows),'planar_mean_edge_error':score_from_rows(rows4),'rows':rows5,'evaluations':len(cache)})

    # KEY5 contact board and per-frame diagnostic; only global p5 is used below.
    board=[]
    for i in KEY5:
        board.append([(np.asarray(Image.open(COND/f'F{i:02d}.png')),f'Original F{i:02d}'),(np.asarray(Image.open(OUT/f'global_sim5/rgb/F{i:02d}.png')),f'Scene F{i:02d}'),(clean[i],f'Clean F{i:02d}')])
    save_contact(board, OUT/'key5/key5_comparison.png', tile=256)

    # H/I: one global transform for all 72 frames, then full trajectory QA.
    xyz2 = xyz; # runtime transform; do not write a mathematically ambiguous PLY.
    # Render all frames in one renderer call; preserve original camera JSONs.
    Mfinal, full_frames, full_alpha = render(mod, assets, p5, ALL)
    full_dir=OUT/'full72/condition_rgb'; full_dir.mkdir(parents=True,exist_ok=True)
    frame_rows=[]; identity_full=[]
    for i in ALL:
        im=full_frames[i]; Image.fromarray(im).save(full_dir/f'F{i:02d}.png')
        # clean frames are loaded lazily to avoid retaining a second 72-frame array.
        gt=np.asarray(Image.open(CLEAN/f'F{i:02d}.png').convert('RGB'))
        e=edge_score(im,gt)
        # Identity baseline uses the authoritative P48 render, not a warped image.
        orig=np.asarray(Image.open(COND/f'F{i:02d}.png').convert('RGB'))
        ei=edge_score(orig,gt)
        frame_rows.append({'frame':i,'identity_edge_error':ei,'global_edge_error':e,'improved':bool(e < ei),'delta':float(ei-e),'alpha_coverage':float((full_alpha[i]>1e-3).mean())})
    improved_ratio=float(np.mean([x['improved'] for x in frame_rows])); global_mean=float(np.mean([x['global_edge_error'] for x in frame_rows])); identity_mean=float(np.mean([x['identity_edge_error'] for x in frame_rows]))
    encode_dir=OUT/'full72'; make_video(full_dir, encode_dir/'aligned_scene_global.mp4')
    # 3-way comparison and edge comparison.
    comp=OUT/'full72/comparison_frames'; edgecomp=OUT/'full72/edge_frames'; comp.mkdir(exist_ok=True); edgecomp.mkdir(exist_ok=True)
    for i in ALL:
        a=np.asarray(Image.open(COND/f'F{i:02d}.png').convert('RGB')); s=full_frames[i]; g=np.asarray(Image.open(CLEAN/f'F{i:02d}.png').convert('RGB'))
        Image.fromarray(np.concatenate([a,s,g],axis=1)).save(comp/f'F{i:02d}.png')
        Image.fromarray(np.concatenate([np.repeat(img_edges(a)[...,None],3,2)*255,np.repeat(img_edges(s)[...,None],3,2)*255,np.repeat(img_edges(g)[...,None],3,2)*255],axis=1).astype(np.uint8)).save(edgecomp/f'F{i:02d}.png')
    make_video(comp, OUT/'comparison_scenealign.mp4'); make_video(edgecomp, OUT/'edge_comparison.mp4')
    # Overlay is diagnostic only.
    overlay=OUT/'full72/overlay_frames'; overlay.mkdir(exist_ok=True)
    for i in ALL:
        s=full_frames[i];g=np.asarray(Image.open(CLEAN/f'F{i:02d}.png').convert('RGB'));Image.fromarray(np.uint8(.5*s+.5*g)).save(overlay/f'F{i:02d}.png')
    make_video(overlay, OUT/'overlay_scenealign.mp4')
    wjson(OUT/'full72/metrics.json', {'identity_mean_edge_error':identity_mean,'global_mean_edge_error':global_mean,'relative_improvement':(identity_mean-global_mean)/max(identity_mean,1e-9),'improved_frame_ratio':improved_ratio,'improved_frames':sum(x['improved'] for x in frame_rows),'rows':frame_rows})

    # J: diagnostic per-frame yaw-only transform, never used for final output.
    diag_pf=[]
    for i in KEY5:
        local=[]
        for yaw in np.arange(-10.,10.01,2.):
            p=np.array([yaw,0.,0.,1.,0.]); _,fr,_=render(mod,assets,p,[i]); local.append((edge_score(fr[i],clean[i]),yaw))
        local.sort(); diag_pf.append({'frame':i,'best_yaw_deg':float(local[0][1]),'best_edge_error':float(local[0][0]),'identity_edge_error':float(next(x['identity_edge_error'] for x in frame_rows if x['frame']==i))})
    wjson(OUT/'diagnostics/per_frame_best_transform_diagnostic.json', {'diagnostic_only':True,'rows':diag_pf,'yaw_range_deg':[-10,10],'no_per_frame_transform_used_for_final':True})

    # topdown source diagnostic: project transformed Gaussian footprint into explicit topdown camera.
    top=DATA/'topdown/topdown.png'; top_meta=DATA/'topdown/topdown_camera_para.json'; topdown_found=top.exists() and top_meta.exists()
    if topdown_found:
        td=np.asarray(Image.open(top).convert('RGB')); q=json.loads(top_meta.read_text()); K=np.asarray(q['intrinsic'],float)[:3,:3]; c2w=np.asarray(q['c2w'],float); P=(Mfinal[:3,:3]@xyz.T).T+Mfinal[:3,3]; pc=(c2w[:3,:3].T@(P-c2w[:3,3]).T).T; ok=pc[:,2]>0.1; uv=(K@pc[ok].T).T; uv=uv[:,:2]/uv[:,2:3]; uv=np.round(uv).astype(int); keep=(uv[:,0]>=0)&(uv[:,0]<td.shape[1])&(uv[:,1]>=0)&(uv[:,1]<td.shape[0]); canvas=td.copy(); canvas[uv[keep,1],uv[keep,0]]=[255,0,0]; Image.fromarray(canvas).save(OUT/'diagnostics/topdown_scene_registration.png')
    # camera equality gate and final report.
    after_hash=[sha(p) for p in cam_jsons]; camera_same=before_hash==after_hash
    identity_error=identity_mean; rel=(identity_mean-global_mean)/max(identity_mean,1e-9)
    # Require a substantial, trajectory-wide gain for PASS; otherwise fail closed.
    if camera_same and rel >= 0.20 and improved_ratio >= 0.75:
        status='ZSCENE_PASS_GLOBAL_SIMILARITY'
    elif camera_same and rel >= 0.05 and improved_ratio >= 0.50:
        status='ZSCENE_PARTIAL_GLOBAL_IMPROVEMENT'
    else:
        status='ZSCENE_FAIL_NO_GLOBAL_ALIGNMENT_SIGNAL'
    nonrigid=bool(improved_ratio < 0.75 or rel < 0.20)
    report={'final_status':status,'gaussian_path':str(SPLAT),'renderer':'gsplat','renderer_entrypoint':str(RENDER),'num_frames':72,'camera_metadata_unchanged':camera_same,'up_axis':'Z / ssl_z_up','best_transform_type':'global_sim5','best_global_transform':{'yaw_deg':float(p5[0]),'translation_xyz':p5[[1,2,4]].tolist(),'scale':float(p5[3]),'transform_matrix_4x4':Mfinal.tolist()},'metrics_identity':{'edge_error':identity_mean},'metrics_global':{'edge_error':global_mean},'relative_improvement':rel,'improved_frame_ratio':improved_ratio,'nonrigid_evidence':nonrigid,'possible_object_level_deformation':nonrigid,'source_topdown_found':topdown_found,'key5':{'identity_mean_edge_error':score_from_rows(identity_rows),'yaw_only_mean_edge_error':score_from_rows(rows_yaw),'planar_sim4_mean_edge_error':score_from_rows(rows4),'global_sim5_mean_edge_error':score_from_rows(rows5)},'recommended_condition_video':str(OUT/'full72/aligned_scene_global.mp4') if status.startswith('ZSCENE_PASS') else str(ALIGN/'global_alignment/aligned_global.mp4'),'recommended_gaussian_transform':str(OUT/'transforms/best_global_sim5.json'),'transformed_gaussian_ply':None,'runtime_transform_path':str(OUT/'transforms/best_global_sim5.json'),'recommended_next_phase':'Phase Z0 only if visual QA accepts the global result; otherwise investigate geometry deformation before Z0','failure_reason':None if status.startswith('ZSCENE_PASS') else 'Global transform did not meet 20% / 75% trajectory-wide gate','camera_hashes_before':before_hash,'camera_hashes_after':after_hash}
    wjson(OUT/'phase_z_scenealign_report.json',report)
    md=f"""# Phase Z-SceneAlign report\n\nFINAL_STATUS: `{status}`\n\n## Transform\n\n- Scene up axis: Z (`ssl_z_up`). Cameras were not changed.\n- Best global sim5: yaw `{p5[0]:.6f}` deg; translation `[tx={p5[1]:.6f}, ty={p5[2]:.6f}, tz={p5[4]:.6f}]` m; scale `{p5[3]:.8f}`.\n- Transform acts on Gaussian world coordinates at runtime; source `splat.ply` was not overwritten and no transformed PLY was emitted.\n\n## Metrics\n\n- Identity mean structural edge error: `{identity_mean:.8f}`.\n- Global sim5 mean structural edge error: `{global_mean:.8f}`.\n- Relative improvement: `{rel:.4%}`.\n- Improved frame ratio: `{improved_ratio:.4%}` ({sum(x['improved'] for x in frame_rows)}/72).\n- KEY5 identity / yaw-only / planar-sim4 / sim5: `{score_from_rows(identity_rows):.8f}` / `{score_from_rows(rows_yaw):.8f}` / `{score_from_rows(rows4):.8f}` / `{score_from_rows(rows5):.8f}`.\n\n## Interpretation\n\nThe objective is structural edge Chamfer on blurred/Canny projections, not RGB similarity. Cross-domain feature matching remains diagnostic only. `nonrigid_evidence` is `{nonrigid}` under the trajectory-wide gate; inspect `diagnostics/per_frame_best_transform_diagnostic.json` and the comparison videos before Z0.\n\nSource topdown found: `{topdown_found}`. Camera metadata hashes are unchanged before/after.\n\nOutputs: `{OUT/'full72/aligned_scene_global.mp4'}`, `{OUT/'comparison_scenealign.mp4'}`, `{OUT/'edge_comparison.mp4'}`, `{OUT/'transforms/best_global_sim5.json'}`.\n"""
    (OUT/'phase_z_scenealign_report.md').write_text(md)
    print(json.dumps({'final_status':status,'params':p5.tolist(),'relative_improvement':rel,'improved_frame_ratio':improved_ratio,'report':str(OUT/'phase_z_scenealign_report.json')},indent=2))


if __name__=='__main__': main()
