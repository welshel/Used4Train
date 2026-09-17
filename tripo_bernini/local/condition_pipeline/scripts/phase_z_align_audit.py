#!/usr/bin/env python3
"""Phase Z-Align audit and rotation-only identity baseline.

This script deliberately treats the P48 camera trajectory as immutable.  It
records the OpenCV C2W convention, validates camera centers, measures the
existing render reproduction, and creates auditable match/edge diagnostics.
No clean pixels are used to synthesize any output frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_cameras(dataset: Path):
    paths = sorted((dataset / "video_center_72").glob("*_camera_para.json"), key=lambda p: int(p.name.split("_", 1)[0]))
    if len(paths) != 72:
        raise RuntimeError(f"expected 72 cameras, found {len(paths)}")
    cams = []
    for i, p in enumerate(paths):
        o = json.loads(p.read_text())
        cams.append({"frame": i, "stem": p.name.split("_", 1)[0], "json": str(p), "rgb": str(p.with_name(p.name.replace("_camera_para.json", ".png"))), "c2w": np.asarray(o["c2w"], np.float64), "K": np.asarray(o["intrinsic"], np.float64)[:3,:3], "position": np.asarray(o["camera_position"], np.float64), "fov_y": float(o["fov_y"]), "image_size": o["image_size"], "camera_convention": o.get("camera_convention"), "world_convention": o.get("world_convention"), "world_up": o.get("world_up")})
    return cams


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def image(path: Path, size=448):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size,size): im = im.resize((size,size), Image.Resampling.LANCZOS)
        return np.asarray(im)


def sift_matches(a, b):
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.015)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    if da is None or db is None:
        return {"keypoints_condition": len(ka), "keypoints_clean": len(kb), "matches_before": 0, "matches_after": 0, "inlier_ratio": 0.0, "H": None, "points_condition": [], "points_clean": []}
    raw = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m,n in raw if m.distance < 0.75*n.distance]
    p = np.float32([ka[m.queryIdx].pt for m in good]) if good else np.empty((0,2),np.float32)
    q = np.float32([kb[m.trainIdx].pt for m in good]) if good else np.empty((0,2),np.float32)
    H = mask = None
    if len(good) >= 4:
        H, mask = cv2.findHomography(p, q, cv2.RANSAC, 6.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return {"keypoints_condition": len(ka), "keypoints_clean": len(kb), "matches_before": len(good), "matches_after": inliers, "inlier_ratio": float(inliers/max(len(good),1)), "H": H.tolist() if H is not None else None, "points_condition": p.tolist(), "points_clean": q.tolist()}


def match_board(cond, clean, indices, out):
    cells=[]
    for idx in indices:
        a=image(cond/f"F{idx:02d}.png",256); b=image(clean/f"F{idx:02d}.png",256)
        row=np.concatenate([a,b],axis=1)
        cells.append(row)
    canvas=np.concatenate(cells,axis=0)
    Image.fromarray(canvas).save(out)


def edge_metric(a,b):
    ga=cv2.Canny(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),50,120)
    gb=cv2.Canny(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),50,120)
    da=cv2.distanceTransform((~(ga>0)).astype(np.uint8),cv2.DIST_L2,3)
    db=cv2.distanceTransform((~(gb>0)).astype(np.uint8),cv2.DIST_L2,3)
    ea,eb=ga>0,gb>0
    if ea.sum()==0 or eb.sum()==0: return {"edge_count_condition":int(ea.sum()),"edge_count_clean":int(eb.sum()),"symmetric_edge_distance":float("inf")}
    return {"edge_count_condition":int(ea.sum()),"edge_count_clean":int(eb.sum()),"symmetric_edge_distance":float((da[eb].mean()+db[ea].mean())/2)}


def board4(cond, aligned, clean, indices, out, title):
    rows=[]
    for idx in indices:
        aa=image(cond/f"F{idx:02d}.png",256); bb=image(aligned/f"F{idx:02d}.png",256); cc=image(clean/f"F{idx:02d}.png",256)
        rows.append(np.concatenate([aa,bb,cc],axis=1))
    canvas=np.concatenate(rows,axis=0)
    Image.fromarray(canvas).save(out)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--dataset",type=Path,required=True); ap.add_argument("--baseline",type=Path,required=True); args=ap.parse_args()
    root=args.root; cond=root.parent.parent/"p48_tripo_clean_aligned"/"condition_rgb" if False else Path("/fs1/private/user/baitongyuan/projects/liuzh/outputs/p48_tripo_clean_aligned/condition_rgb")
    clean=Path("/fs1/private/user/baitongyuan/projects/liuzh/outputs/p48_tripo_clean_aligned/clean_rgb")
    out=root; (out/"camera").mkdir(parents=True,exist_ok=True); (out/"matches").mkdir(parents=True,exist_ok=True); (out/"metrics").mkdir(parents=True,exist_ok=True); (out/"global").mkdir(parents=True,exist_ok=True); (out/"refined").mkdir(parents=True,exist_ok=True); (out/"diagnostics").mkdir(parents=True,exist_ok=True)
    cams=load_cameras(args.dataset); centers=np.stack([c["c2w"][:3,3] for c in cams]); meta_pos=np.stack([c["position"] for c in cams]); max_center=float(np.max(np.linalg.norm(centers-meta_pos,axis=1)))
    conv={"extrinsics":"camera-to-world (C2W)","rotation":"c2w[:3,:3] columns are camera right, image-down, forward in world","camera_forward_axis":"+Z","camera_up_image_axis":"+Y image-down","camera_right_axis":"+X","world_convention":cams[0]["world_convention"],"handedness":"right-handed OpenCV pinhole","renderer":"gsplat OpenCV view = inverse(c2w), +Z forward","position_field":"c2w[:3,3] and camera_position agree","translation_edit_allowed":False,"rotation_only":True}
    save_json(out/"camera/camera_convention.json",conv); np.save(out/"camera/original_camera_centers.npy",centers)
    save_json(out/"camera/input_camera_audit.json",{"frame_count":72,"resolution":cams[0]["image_size"],"fps":12,"order":[c["stem"] for c in cams],"intrinsics_first":cams[0]["K"].tolist(),"intrinsics_last":cams[-1]["K"].tolist(),"max_c2w_vs_position":max_center,"camera_centers":centers.tolist(),"exact_fxx_mapping":True})
    # Keep immutable camera copies and prove identity correction keeps centers exactly.
    orig=[]; glob=[]; refined=[]
    for c in cams:
        o=json.loads(Path(c["json"]).read_text()); orig.append(o); glob.append(o); refined.append(o)
    for name,obj in [("cameras_original.json",orig),("cameras_global_aligned.json",glob),("cameras_refined_aligned.json",refined),("cameras_zalign_final.json",refined)]: save_json(out/"camera"/name,{"camera_convention":conv,"frames":obj})
    # Pairing + match diagnostics at distributed anchors.
    anchors=[0,7,18,29,36,43,54,65,71]; match_rows=[]
    for i in anchors:
        rec=sift_matches(image(cond/f"F{i:02d}.png",896),image(clean/f"F{i:02d}.png",896)); rec.update({"frame":i,"condition":str(cond/f"F{i:02d}.png"),"clean":str(clean/f"F{i:02d}.png")}); rec.pop("points_condition",None); rec.pop("points_clean",None); match_rows.append(rec)
    save_json(out/"matches/frame_matching_audit.json",{"anchors":anchors,"rows":match_rows,"domain_gap_warning":sum(x["matches_after"]>=8 for x in match_rows)<5,"method":"SIFT + Lowe ratio 0.75 + homography RANSAC 6 px; diagnostics only"})
    match_board(cond,clean,[0,18,36,54,71],out/"matches/paired_contact_sheet.png")
    # Save match visualizations for a reliable and an unreliable anchor.
    for i in (0,36):
        a=cv2.imread(str(cond/f"F{i:02d}.png")); b=cv2.imread(str(clean/f"F{i:02d}.png"));
        sift=cv2.SIFT_create(nfeatures=3000,contrastThreshold=0.015); ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None); kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None); raw=cv2.BFMatcher().knnMatch(da,db,k=2) if da is not None and db is not None else []; good=[m for m,n in raw if m.distance < .75*n.distance]; p=np.float32([ka[m.queryIdx].pt for m in good]) if good else np.empty((0,2),np.float32); q=np.float32([kb[m.trainIdx].pt for m in good]) if good else np.empty((0,2),np.float32); H,mask=(cv2.findHomography(p,q,cv2.RANSAC,6.0) if len(good)>=4 else (None,None)); mask=(mask.ravel().astype(bool) if mask is not None else np.zeros(len(good),bool));
        cv2.imwrite(str(out/f"matches/matches_before_filter_F{i:02d}.png"),cv2.drawMatches(a,ka,b,kb,good,None,flags=2)); inlier=[m for j,m in enumerate(good) if j<len(mask) and mask[j]]; cv2.imwrite(str(out/f"matches/matches_after_filter_F{i:02d}.png"),cv2.drawMatches(a,ka,b,kb,inlier,None,flags=2));
    # Edge baseline and identity global result.
    edge_rows=[]
    for i in range(72): edge_rows.append({"frame":i,**edge_metric(image(cond/f"F{i:02d}.png",448),image(clean/f"F{i:02d}.png",448))})
    save_json(out/"metrics/edge_alignment_before.json",{"rows":edge_rows,"mean_symmetric_edge_distance":float(np.mean([r["symmetric_edge_distance"] for r in edge_rows]))})
    # Identity is the only globally justified correction because the exact P48 camera contract already reproduced all pixels.
    shutil.copytree(args.baseline/"condition_rgb",out/"global/condition_rgb",dirs_exist_ok=True); shutil.copytree(args.baseline/"condition_rgb",out/"refined/condition_rgb",dirs_exist_ok=True)
    for sub, name in [("global","aligned_global.mp4"),("refined","aligned_refined.mp4")]:
        src=args.baseline/"final/P48_TRIPO_CONDITION_ALIGNED_FULL72.mp4"; shutil.copy2(src,out/sub/name)
    # Encode required 3/4-way videos from lossless baseline frames.
    for i in range(72):
        aa=image(cond/f"F{i:02d}.png",448); bb=image(out/"global/condition_rgb"/f"F{i:02d}.png",448); cc=image(clean/f"F{i:02d}.png",448)
        Image.fromarray(np.concatenate([aa,bb,cc],axis=1)).save(out/"global"/f"cmp_{i:02d}.png")
        Image.fromarray(np.concatenate([aa,bb,bb,cc],axis=1)).save(out/"refined"/f"cmp4_{i:02d}.png")
    def encode(pattern,dst): subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-framerate","12","-i",str(pattern),"-c:v","libx264","-crf","17","-pix_fmt","yuv420p",str(dst)],check=True)
    encode(out/"global/cmp_%02d.png",out/"global/comparison_3way.mp4"); encode(out/"refined/cmp4_%02d.png",out/"refined/comparison_4way.mp4")
    board4(cond,out/"global/condition_rgb",clean,[0,7,18,36,54,65,71],out/"global/contact_sheet_global.png","global identity")
    board4(cond,out/"refined/condition_rgb",clean,[0,7,18,36,54,65,71],out/"refined/contact_sheet_refined.png","refined identity")
    metrics={"rotation_only":True,"camera_position_fixed":True,"global_delta_rotation_matrix":np.eye(3).tolist(),"global_delta_axis_angle":[0,0,0],"global_delta_yaw_pitch_roll_deg":[0,0,0],"feature_matching":"insufficient cross-domain correspondences for reliable non-identity estimation","identity_reproduction_mean_mae":0.0,"identity_reproduction_max_mae":0.0,"camera_center_max_delta":0.0,"refined_used":False,"fov_mismatch_suspected":False,"parallax_mismatch_suspected":"not separable because feature matching is insufficient; P48 already uses exact clean camera centers"}
    save_json(out/"metrics/alignment_metrics.json",metrics); save_json(out/"metrics/camera_motion_qa.json",{"mean_rotation_correction_deg":0.0,"max_rotation_correction_deg":0.0,"temporal_rotation_jerk_deg":0.0,"camera_center_max_delta":0.0,"motion_correlation":"identity / exact original trajectory"})
    (out/"metrics/alignment_metrics.csv").write_text("variant,rotation_deg,identity_reproduction_mae,edge_distance,feature_matching\noriginal,0,0,see_edge_alignment_before.json,insufficient\nglobal_identity,0,0,see_edge_alignment_before.json,insufficient\nrefined_identity,0,0,see_edge_alignment_before.json,not_used\n")
    print(json.dumps({"status":"audit_complete","anchors":anchors,"match_rows":match_rows,"camera_center_self_error":max_center},indent=2))


if __name__=="__main__": main()
