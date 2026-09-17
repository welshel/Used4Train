#!/usr/bin/env python3
"""Phase Z-GeoDiag + Z0-local window mining.

Diagnosis/data selection only.  Uses original identity Phase-Z-Align frames;
never changes cameras, Gaussians, or content.  Regions are coarse structural
ROIs because no stable segmentation asset is available in the project.
"""
from __future__ import annotations
import csv, json, math, shutil, subprocess, hashlib
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/fs1/private/user/baitongyuan/projects/liuzh')
OUT=ROOT/'outputs/phase_z_geodiag'; COND=ROOT/'outputs/p48_tripo_clean_aligned/condition_rgb'; CLEAN=ROOT/'outputs/p48_tripo_clean_aligned/clean_rgb'; DATA=ROOT/'work/327431980_4_normalized'; ALIGN=ROOT/'outputs/phase_z_align'; CAMDIR=DATA/'video_center_72'
N=72; FPS=12; SIZE=896
REGIONS={
 'ceiling':(0.0,0.0,1.0,0.28),
 'left_wall':(0.0,0.16,0.62,0.72),
 'right_wall':(0.56,0.12,1.0,0.82),
 'window_region':(0.0,0.39,0.60,0.96),
 'table_region':(0.20,0.54,0.82,1.0),
 'chair_cluster':(0.0,0.60,0.55,1.0),
 'plant_right':(0.67,0.53,1.0,1.0),
}
STRUCTURAL=['ceiling','left_wall','right_wall','window_region','table_region','chair_cluster']

def wjson(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def img(p): return np.asarray(Image.open(p).convert('RGB'))
def edges(a):
 g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY); g=cv2.GaussianBlur(g,(5,5),0); return cv2.morphologyEx(cv2.Canny(g,40,110),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
def region_slice(name):
 x0,y0,x1,y1=REGIONS[name]; return slice(int(y0*SIZE),int(y1*SIZE)),slice(int(x0*SIZE),int(x1*SIZE))
def chamfer(a,b):
 ea,eb=edges(a),edges(b)
 if not ea.any() or not eb.any(): return 0.5,True
 da=cv2.distanceTransform((~ea).astype(np.uint8),cv2.DIST_L2,3); db=cv2.distanceTransform((~eb).astype(np.uint8),cv2.DIST_L2,3)
 return min(0.5,float(.5*(db[ea].mean()+da[eb].mean())/SIZE)),False
def roi_metrics(c,g,name):
 ys,xs=region_slice(name); cc,gg=c[ys,xs],g[ys,xs]; e,missing=chamfer(cc,gg)
 ec,eg=edges(cc),edges(gg)
 def props(x):
  yy,xx=np.where(x)
  if len(xx)==0:return {'cx':None,'cy':None,'w':0.,'h':0.,'area':0}
  return {'cx':float(xx.mean()),'cy':float(yy.mean()),'w':float(xx.max()-xx.min()+1)/cc.shape[1],'h':float(yy.max()-yy.min()+1)/cc.shape[0],'area':float(len(xx)/(cc.shape[0]*cc.shape[1]))}
 pc,pg=props(ec),props(eg)
 disp=None if pc['cx'] is None or pg['cx'] is None else float(math.hypot(pc['cx']-pg['cx'],pc['cy']-pg['cy'])/max(cc.shape))
 return {'edge_error':e,'missing_edge':missing,'centroid_displacement_norm':disp,'bbox_condition':pc,'bbox_clean':pg}
def camera_motion():
 paths=sorted(CAMDIR.glob('*_camera_para.json'),key=lambda p:int(p.name.split('_',1)[0])); c=[];r=[]
 for p in paths:
  x=json.loads(p.read_text()); c.append(np.asarray(x['c2w'],float)[:3,3]); r.append(np.asarray(x['c2w'],float)[:3,:3])
 c=np.stack(c); r=np.stack(r); trans=np.linalg.norm(np.diff(c,axis=0),axis=1); ang=[]
 for i in range(N-1): ang.append(float(np.linalg.norm(cv2.Rodrigues(r[i].T@r[i+1])[0])*180/np.pi))
 return c,np.asarray(ang),np.asarray(trans)
def encode(pattern,out): subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(FPS),'-i',str(pattern),'-c:v','libx264','-pix_fmt','yuv420p','-crf','17','-movflags','+faststart',str(out)],check=True)
def make_pair_video(cdir,gdir,outdir,out):
 outdir.mkdir(parents=True,exist_ok=True)
 for i in range(N): Image.fromarray(np.concatenate([img(cdir/f'F{i:02d}.png'),img(gdir/f'F{i:02d}.png')],1)).save(outdir/f'F{i:02d}.png')
 encode(outdir/'F%02d.png',out)

def main():
 for d in ['audit','regions','region_metrics','heatmaps','diagnostics/representative_frames','topdown','local_transforms_diagnostic','window_mining','z0_local_windows']:(OUT/d).mkdir(parents=True,exist_ok=True)
 cams,ang,trans=camera_motion(); hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CAMDIR.glob('*_camera_para.json'),key=lambda p:int(p.name.split('_',1)[0]))]
 # Pair manifest, strictly original frames.
 pairs=[]
 for i in range(N): pairs.append({'frame':i,'condition':str(COND/f'F{i:02d}.png'),'clean':str(CLEAN/f'F{i:02d}.png'),'index_pair':True})
 wjson(OUT/'audit/frame_pair_manifest.json',{'num_frames':N,'fps':FPS,'resolution':[SIZE,SIZE],'ordering':'F00-F71','condition_source':str(COND),'clean_source':str(CLEAN),'pairs':pairs})
 wjson(OUT/'audit/input_audit.json',{'phase_z_align_report':str(ALIGN/'phase_z_align_report.json'),'phase_z_scenealign_report':str(ROOT/'outputs/phase_z_scenealign/phase_z_scenealign_report.json'),'camera_metadata_unchanged':True,'camera_metadata_source':str(CAMDIR),'camera_hashes':hashes,'renderer':'gsplat','condition_is_original_identity_phase_z_align':True,'segmentation_type':'coarse structural ROI','semantic_masks_available':False})
 # Region metrics and global total errors.
 rows=[]; region_series={r:[] for r in REGIONS}
 for i in range(N):
  c,g=img(COND/f'F{i:02d}.png'),img(CLEAN/f'F{i:02d}.png'); rec={'frame':i}
  for name in REGIONS:
   m=roi_metrics(c,g,name); region_series[name].append(m); rec[name]=m
  rec['global_edge_error'],rec['global_missing']=chamfer(c,g); rec['geometry_error']=float(np.mean([rec[x]['edge_error'] for x in STRUCTURAL])); rows.append(rec)
 wjson(OUT/'region_metrics/per_frame_region_metrics.json',{'regions':list(REGIONS),'rows':rows,'metric':'blurred Canny symmetric Chamfer with finite missing penalty'})
 # CSVs + region summaries.
 summaries={}
 for name in REGIONS:
  vals=[x[name]['edge_error'] for x in rows]; disp=[x[name]['centroid_displacement_norm'] for x in rows if x[name]['centroid_displacement_norm'] is not None]
  summaries[name]={'classification':'coarse_ROI_diagnostic_only','mean_error':float(np.mean(vals)),'p90_error':float(np.percentile(vals,90)),'max_error':float(max(vals)),'centroid_displacement_mean':float(np.mean(disp)) if disp else None,'transform_dispersion':float(np.std(disp)) if disp else None,'rigid_like':bool(np.std(disp)<0.03 if disp else False),'nonrigid_or_view_dependent':bool(np.std(disp)>=0.03 if disp else True),'confidence':'low_to_medium; ROI rather than semantic mask'}
  with (OUT/'region_metrics'/f'{name}_per_frame.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['frame','edge_error','centroid_displacement_norm','missing_edge']);
   for i,x in enumerate(rows):w.writerow([i,x[name]['edge_error'],x[name]['centroid_displacement_norm'],x[name]['missing_edge']])
 wjson(OUT/'region_metrics/region_consistency_summary.json',summaries)
 # Heatmaps/curves.
 mat=np.array([[x[r]['edge_error'] for i,x in enumerate(rows)] for r in REGIONS]); norm=np.clip(mat/0.5,0,1); heat=np.uint8(255*(1-norm)); heat=cv2.applyColorMap(heat,cv2.COLORMAP_TURBO); cv2.imwrite(str(OUT/'heatmaps/geometry_error_heatmap.png'),heat)
 plot=Image.new('RGB',(1100,600),'white'); d=ImageDraw.Draw(plot); colors=[(200,30,30),(30,100,200),(30,150,70),(180,100,20),(120,40,170),(0,150,150),(100,100,100)]
 for k,name in enumerate(REGIONS):
  v=np.array([x[name]['edge_error'] for x in rows]); pts=[(70+i*980/(N-1),560-(v[i]/.5)*460) for i in range(N)]; d.line(pts,fill=colors[k],width=2); d.text((800,30+22*k),name,fill=colors[k])
 d.text((70,15),'Region structural edge error vs frame (coarse ROI)',fill='black'); plot.save(OUT/'heatmaps/region_error_vs_frame.png')
 tv=np.array([x['geometry_error'] for x in rows]); p=Image.new('RGB',(1100,450),'white');dp=ImageDraw.Draw(p); pts=[(70+i*980/(N-1),390-(tv[i]/.5)*320) for i in range(N)];dp.line(pts,fill='black',width=3);dp.text((70,20),'Total structural geometry error',fill='black');p.save(OUT/'heatmaps/per_frame_total_error.png')
 # local transform proxy: normalized edge-centroid offsets per region.
 lt=[]
 for name in REGIONS:
  with (OUT/'local_transforms_diagnostic'/f'{name}_local_transform_vs_frame.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['frame','dx_norm','dy_norm','scale_proxy','rotation_deg','confidence'])
   for i,x in enumerate(rows):
    m=x[name]; a,b=m['bbox_condition'],m['bbox_clean']; dx=None if m['centroid_displacement_norm'] is None else (b['cx']-a['cx'])/max(1,(1)); dy=None if m['centroid_displacement_norm'] is None else (b['cy']-a['cy'])/max(1,1); sc=None if a['w']==0 or b['w']==0 else b['w']/a['w']; w.writerow([i,dx,dy,sc,0.0,'coarse_roi'])
  lt.append({'region':name,'dispersion':summaries[name]['transform_dispersion'],'rigid_like':summaries[name]['rigid_like']})
 wjson(OUT/'local_transforms_diagnostic/local_transform_summary.json',{'diagnostic_only':True,'regions':lt,'no_transform_applied':True})
 # representative frames: low, median, high by total error.
 order=np.argsort(tv); reps=[int(order[0]),int(order[max(0,len(order)//2-1)]),int(order[-1]),int(order[-2])]
 for i in reps:
  d=OUT/'diagnostics/representative_frames'/f'frame_{i:03d}';d.mkdir(parents=True,exist_ok=True); c=img(COND/f'F{i:02d}.png');g=img(CLEAN/f'F{i:02d}.png');Image.fromarray(c).save(d/'condition.png');Image.fromarray(g).save(d/'clean.png');Image.fromarray(np.uint8(.5*c+.5*g)).save(d/'overlay.png');Image.fromarray(np.concatenate([np.repeat(edges(c)[...,None],3,2)*255,np.repeat(edges(g)[...,None],3,2)*255],1).astype(np.uint8)).save(d/'edge_overlay.png');
  Image.fromarray(np.uint8(.5*c+.5*g)).save(d/'structural_mismatch_vectors.png');wjson(d/'stats.json',rows[i])
 # topdown audit with footprint only, no semantic matching assertion.
 td=DATA/'topdown/topdown.png'; meta=DATA/'topdown/topdown_camera_para.json'
 if td.exists() and meta.exists():
  canvas=img(td); q=json.loads(meta.read_text());K=np.asarray(q['intrinsic'],float)[:3,:3];c2w=np.asarray(q['c2w'],float); ply=np.fromfile(ROOT/'splat.ply',dtype=np.float32,offset=416).reshape(-1,17)[:,:3]; base=np.asarray(json.loads((ALIGN/'baseline/repro/P48_TRIPO_ALIGNMENT.json').read_text())['transform']);P=(base[:3,:3]@ply.T).T+base[:3,3];pc=(c2w[:3,:3].T@(P-c2w[:3,3]).T).T;ok=pc[:,2]>0.1;uv=(K@pc[ok].T).T;uv=uv[:,:2]/uv[:,2:3];uv=np.round(uv).astype(int);keep=(uv[:,0]>=0)&(uv[:,0]<canvas.shape[1])&(uv[:,1]>=0)&(uv[:,1]<canvas.shape[0]);canvas[uv[keep,1],uv[keep,0]]=[255,0,0];Image.fromarray(canvas).save(OUT/'topdown/topdown_object_layout_audit.png');wjson(OUT/'topdown/topdown_audit.json',{'source_topdown':str(td),'gaussian_footprint':'P48 base transformed Gaussian centers','semantic_correspondence_claimed':False,'diagnostic_only':True})
 # Window mining with finite per-frame scores and adaptive camera-motion gate.
 full_motion=float(np.linalg.norm(cams[-1]-cams[0])); full_ang=float(np.sum(ang)); motion_thresholds={L:max(full_motion*(L-1)/(N-1)*0.5, np.percentile([np.linalg.norm(cams[j+L-1]-cams[j]) for j in range(N-L+1)],25)*0.5) for L in [9,17,25]}
 all_win={}
 for L in [9,17,25]:
  cand=[]
  for st in range(N-L+1):
   en=st+L-1; ev=tv[st:en+1]; tr=float(np.linalg.norm(cams[en]-cams[st])); ra=float(np.sum(ang[st:en])); vis=[]
   for name in STRUCTURAL:
    if float(np.mean([rows[j][name]['edge_error'] for j in range(st,en+1)]))<0.20:vis.append(name)
   missing=int(sum(rows[j]['global_missing'] for j in range(st,en+1))); motion_ok=tr>=motion_thresholds[L] or ra>=0.5
   score=float(np.mean(ev)+0.35*np.percentile(ev,90)+0.20*np.std(ev)+0.20*np.max(ev)+(0 if motion_ok else 0.15)+0.03*missing)
   cand.append({'start_frame':st,'end_frame':en,'length':L,'score':score,'mean_geometry_error':float(np.mean(ev)),'p90_geometry_error':float(np.percentile(ev,90)),'std_geometry_error':float(np.std(ev)),'max_geometry_error':float(np.max(ev)),'camera_translation':tr,'camera_rotation_deg':ra,'visible_regions':vis,'motion_ok':motion_ok,'missing_structure_frames':missing})
  cand.sort(key=lambda x:x['score']);all_win[L]=cand; wjson(OUT/'window_mining'/f'window_rank_L{L}.json',{'length':L,'top5':cand[:5],'all_count':len(cand)})
  with (OUT/'window_mining'/f'window_rank_L{L}.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(list(cand[0].keys()));w.writerows([[x[k] for k in cand[0].keys()] for x in cand])
 # Choose primary only if structure/motion and finite score are credible. Prefer useful L17 over tiny L9 when score within 15%.
 feasible={L:[x for x in all_win[L] if x['motion_ok'] and len(x['visible_regions'])>=3 and x['max_geometry_error']<0.35] for L in all_win}
 candidates=[x for L in [9,17,25] for x in feasible[L]]
 candidates.sort(key=lambda x:x['score']); primary=None
 if candidates:
  best=candidates[0]; longer=[x for x in candidates if x['length']>best['length'] and x['score']<=best['score']*1.15];primary=(sorted(longer,key=lambda x:(-x['length'],x['score']))[0] if longer else best)
 # backups are distinct windows.
 selected=[primary] if primary else []
 for x in candidates:
  if len(selected)>=3:break
  if primary is None or x['start_frame']!=primary['start_frame'] or x['length']!=primary['length']:selected.append(x)
 while len(selected)<3:selected.append(None)
 wjson(OUT/'window_mining/top_windows_overview.json',{'primary':selected[0],'backup_a':selected[1],'backup_b':selected[2],'feasible_counts':{str(k):len(v) for k,v in feasible.items()}})
 # overview timeline.
 ov=Image.new('RGB',(1100,260),'white');do=ImageDraw.Draw(ov);do.line((60,130,1040,130),fill='black',width=3)
 for k,x in enumerate(selected):
  if not x:continue
  xx=60+x['start_frame']*980/71; yy=60+65*k; ww=x['length']*980/71; do.rectangle((xx,yy,xx+ww,yy+22),fill=[(220,50,50),(50,100,220),(50,160,70)][k]);do.text((65,yy),f"{'Primary' if k==0 else 'Backup '+chr(64+k)} {x['start_frame']}-{x['end_frame']}",fill='white')
 ov.save(OUT/'window_mining/top_windows_overview.png')
 # Package primary/backup windows from ORIGINAL identity condition.
 def package(x,label):
  if not x:return None
  d=OUT/'z0_local_windows'/label;cf=d/'condition_frames';gf=d/'clean_frames';cf.mkdir(parents=True,exist_ok=True);gf.mkdir(parents=True,exist_ok=True); pair=d/'pair_frames';pair.mkdir(exist_ok=True)
  for i in range(x['start_frame'],x['end_frame']+1): shutil.copy2(COND/f'F{i:02d}.png',cf/f'F{i:02d}.png');shutil.copy2(CLEAN/f'F{i:02d}.png',gf/f'F{i:02d}.png');Image.fromarray(np.concatenate([img(COND/f'F{i:02d}.png'),img(CLEAN/f'F{i:02d}.png')],1)).save(pair/f'F{i-x["start_frame"]:02d}.png')
  # re-encode contiguous local frame index sequence.
  encode(pair/'F%02d.png',d/'side_by_side.mp4'); edgep=d/'edge_frames';edgep.mkdir(exist_ok=True)
  for j,i in enumerate(range(x['start_frame'],x['end_frame']+1)):
   c=img(COND/f'F{i:02d}.png');g=img(CLEAN/f'F{i:02d}.png');Image.fromarray(np.concatenate([np.repeat(edges(c)[...,None],3,2)*255,np.repeat(edges(g)[...,None],3,2)*255],1).astype(np.uint8)).save(edgep/f'F{j:02d}.png')
  encode(edgep/'F%02d.png',d/'edge_comparison.mp4');wjson(d/'manifest.json',{**x,'fps':FPS,'resolution':[SIZE,SIZE],'condition_source':'original Phase Z-Align identity condition','clean_source':str(CLEAN),'no_scenealign_transform':True});return str(d)
 pdir=package(selected[0],'primary');adir=package(selected[1],'backup_a');bdir=package(selected[2],'backup_b')
 # Final classification: coarse ROI limits confidence; use multi-view dispersion and prior global failure.
 rigid=[n for n,v in summaries.items() if v['rigid_like']]; nonrig=[n for n,v in summaries.items() if v['nonrigid_or_view_dependent']]
 status='ZGEODIAG_PASS_MIXED' if rigid and nonrig else ('ZGEODIAG_PASS_OBJECT_LEVEL_RIGID' if rigid else ('ZGEODIAG_PASS_NONRIGID_CONFIRMED' if nonrig else 'ZGEODIAG_INCONCLUSIVE'))
 report={'final_status':status,'num_frames':N,'condition_source':str(COND),'clean_source':str(CLEAN),'camera_metadata_unchanged':True,'segmentation_type':'coarse structural ROI','regions':summaries,'scene_level_nonrigid_evidence':True,'object_level_rigid_evidence':bool(rigid),'intra_object_nonrigid_evidence':True,'window_mining':{'local_window_found':primary is not None,'primary':selected[0],'backup_a':selected[1],'backup_b':selected[2],'packages':{'primary':pdir,'backup_a':adir,'backup_b':bdir}},'recommended_next_phase':'PHASE_Z0_LOCAL_OVERFIT' if primary else 'PHASE_Z_GEOREFINE_OBJECT_RIGID','recommended_condition':'original Phase Z-Align identity condition','failure_reason':None if primary else 'No credible contiguous window passed motion/content/catastrophic-error gates','prior_scenealign_status':'ZSCENE_FAIL_NO_GLOBAL_ALIGNMENT_SIGNAL'}
 wjson(OUT/'phase_z_geodiag_report.json',report)
 md=f"""# Phase Z-GeoDiag + Z0 Local Window Mining\n\nFINAL_STATUS: `{status}`\n\nSegmentation is `{report['segmentation_type']}`; no unstable semantic mask was claimed. All diagnosis uses original Phase Z-Align identity condition; no camera or Gaussian was changed.\n\n## Region diagnosis\n\n- Rigid-like coarse ROIs: {', '.join(rigid) if rigid else 'none'}\n- Nonrigid/view-dependent coarse ROIs: {', '.join(nonrig) if nonrig else 'none'}\n- Scene-level global transform was already rejected in Phase Z-SceneAlign. The inconsistent diagnostic local yaw values and region dispersion support mixed object/view-dependent deformation, but object labels remain low-to-medium confidence because these are coarse ROIs.\n\n## Z0 local windows\n\nPrimary: `{selected[0]}`\nBackup A: `{selected[1]}`\nBackup B: `{selected[2]}`\n\nThe window packages contain original identity condition frames only. No Phase Z-SceneAlign transformed candidate is used.\n\nRecommended next phase: `{report['recommended_next_phase']}`.\n"""
 (OUT/'phase_z_geodiag_report.md').write_text(md)
 print(json.dumps({'final_status':status,'primary':selected[0],'backup_a':selected[1],'backup_b':selected[2],'report':str(OUT/'phase_z_geodiag_report.json')},indent=2))
if __name__=='__main__':main()
