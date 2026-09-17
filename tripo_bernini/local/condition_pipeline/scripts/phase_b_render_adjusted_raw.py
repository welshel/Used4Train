#!/usr/bin/env python3
"""Phase B exact-camera render of adjusted and raw Gaussian conditions."""
from pathlib import Path
import argparse, importlib.util, json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
PA=load('phase_a_render',ROOT/'scripts/phase_a_clean_adjusted_gaussian.py'); PA.load_p48(); P48=PA.P48

def write_json(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def png(p,a):
 p.parent.mkdir(parents=True,exist_ok=True); a=np.asarray(a); Image.fromarray(np.uint8(np.clip(a,0,1)*255) if a.dtype!=np.uint8 else a).save(p)
def depth_vis(z,m):
 z=np.asarray(z,np.float32); m=np.asarray(m,bool); out=np.zeros((*z.shape,3),np.uint8)
 if m.any():
  v=z[m]; q=np.clip((z-np.percentile(v,2))/max(np.percentile(v,98)-np.percentile(v,2),1e-6),0,1); out=np.repeat(np.uint8(q*255)[...,None],3,2); out[~m]=0
 return out
def encode(pat,out,fps=12):
 out.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(pat),'-c:v','libx264','-pix_fmt','yuv420p','-crf','17','-movflags','+faststart',str(out)],check=True)
def dec(rec):
 xyz=np.column_stack([rec[x] for x in ('x','y','z')]).astype(np.float32)
 rgb=np.clip(.5+P48.SH_C0*np.column_stack([rec[x] for x in ('f_dc_0','f_dc_1','f_dc_2')]),0,1).astype(np.float32)
 op=(1/(1+np.exp(-rec['opacity']))).astype(np.float32); sc=np.exp(np.column_stack([rec[x] for x in ('scale_0','scale_1','scale_2')])).astype(np.float32); q=np.column_stack([rec[x] for x in ('rot_0','rot_1','rot_2','rot_3')]).astype(np.float32); q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8); return xyz,rgb,op,sc,q
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--adjusted',type=Path,required=True); ap.add_argument('--raw',type=Path,default=ROOT/'outputs/p48_1_tripo_cleanup/final/splat_cleaned.ply'); ap.add_argument('--dataset',type=Path,default=ROOT/'work/327431980_4_normalized'); ap.add_argument('--alignment',type=Path,default=ROOT/'outputs/p48_tripo_clean_aligned/P48_TRIPO_ALIGNMENT.json'); ap.add_argument('--output',type=Path,default=ROOT/'outputs/phase_b_adjusted_raw'); ap.add_argument('--device',default='cuda'); ap.add_argument('--only',choices=['both','adjusted','raw'],default='both'); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=72); args=ap.parse_args()
 out=args.output; out.mkdir(parents=True,exist_ok=True)
 dirs={}
 for name in ('adjusted_rgb','adjusted_depth','adjusted_lines','raw_rgb','raw_depth','raw_lines','comparison_rgb','comparison_depth','comparison_lines'):
  dirs[name]=out/f'condition_{name}' if name.startswith(('adjusted','raw')) else out/name; dirs[name].mkdir(parents=True,exist_ok=True)
 cams=P48.load_clean_cameras(args.dataset); M=np.asarray(json.loads(args.alignment.read_text())['transform'],float)
 records={}; payload={}
 requested=('adjusted','raw') if args.only=='both' else (args.only,)
 paths={'adjusted':args.adjusted,'raw':args.raw}
 for label in requested:
  path=paths[label]
  rec,header,hb=PA.load_raw(path); xyz,rgb,op,sc,q=dec(rec); sim=float(np.cbrt(np.linalg.det(M[:3,:3]))); world=P48.apply_similarity(xyz,M).astype(np.float32); ws=sc*abs(sim); wq=P48.transform_quaternions(q,M); records[label]=(world,rgb,op,ws,wq,len(rec),path)
  payload[label]={'path':str(path),'sha256':PA.sha256(path),'gaussian_count':int(len(rec))}
 if len(payload)==2 and payload['adjusted']['gaussian_count']!=payload['raw']['gaussian_count']: raise RuntimeError('Gaussian count mismatch')
 # Clean target is copied only for synchronized QA/comparison; never used to
 # produce either inference condition.
 clean=[np.asarray(Image.open(c.rgb_path).convert('RGB')) for c in cams]
 rows=[]
 for i,cam in enumerate(cams):
  if i < args.start or i >= args.end: continue
  ims={}; mets={}
  for label in requested:
   world,rgb,op,ws,wq,_,_=records[label]; im,a,d=PA.render_one(world,rgb,op,ws,wq,cam,args.device); ims[label]=(im,a,d)
   valid=(a>PA.ALPHA_HIT)&np.isfinite(d)&(d>0); e=PA.depth_edges(d,valid)
   png(dirs[f'{label}_rgb']/f'F{i:02d}.png',im); np.save(dirs[f'{label}_depth']/f'F{i:02d}.npy',d); png(dirs[f'{label}_lines']/f'F{i:02d}.png',e.astype(np.uint8)); png(dirs[f'{label}_depth']/f'F{i:02d}.png',depth_vis(d,valid))
   mets[label]={'coverage':float(valid.mean()),'alpha_mean':float(a.mean()),'valid_ratio':float(valid.mean()),'depth_median':float(np.median(d[valid])) if valid.any() else None,'line_pixels':int(e.sum())}
  if args.only != 'both':
   rows.append({'frame':i, args.only:mets[args.only], 'pairing':{f'{args.only}_rgb':f'F{i:02d}.png'}})
   continue
  # 3-way synchronized visual QA; Clean appears only in a comparison artifact.
  a=np.uint8(np.clip(ims['adjusted'][0],0,1)*255); r=np.uint8(np.clip(ims['raw'][0],0,1)*255); c=clean[i]
  Image.fromarray(np.concatenate([a,r,c],axis=1)).save(dirs['comparison_rgb']/f'F{i:02d}.png')
  da=depth_vis(ims['adjusted'][2],ims['adjusted'][1]>PA.ALPHA_HIT); dr=depth_vis(ims['raw'][2],ims['raw'][1]>PA.ALPHA_HIT); Image.fromarray(np.concatenate([da,dr],axis=1)).save(dirs['comparison_depth']/f'F{i:02d}.png')
  la=np.uint8(PA.depth_edges(ims['adjusted'][2],ims['adjusted'][1]>PA.ALPHA_HIT))*255; lr=np.uint8(PA.depth_edges(ims['raw'][2],ims['raw'][1]>PA.ALPHA_HIT))*255; Image.fromarray(np.concatenate([la,lr],axis=1)).save(dirs['comparison_lines']/f'F{i:02d}.png')
  rows.append({'frame':i,'adjusted':mets['adjusted'],'raw':mets['raw'],'pairing':{'adjusted_rgb':f'F{i:02d}.png','raw_rgb':f'F{i:02d}.png','clean_rgb':str(cams[i].rgb_path)}})
 manifest={'status':'PASS','frame_count':len(rows),'frames':rows,'frame_order':'F00-F71 exact numeric camera timestamp order','resolution':[896,896],'camera_contract':'P48 immutable OpenCV C2W','alignment':str(args.alignment),'clean_not_in_condition':True,'requested':requested,'adjusted':payload.get('adjusted'),'raw':payload.get('raw')}
 metrics={'frame_count':len(rows),'requested':requested}
 for lab in requested: metrics[f'{lab}_mean_coverage']=float(np.mean([r[lab]['coverage'] for r in rows]))
 write_json(out/'PHASE_B_PAIRING_MANIFEST.json',manifest); write_json(out/'PHASE_B_RENDER_METRICS.json',metrics)
 if 'adjusted' in requested: encode(dirs['adjusted_rgb']/'F%02d.png',out/'final_ADJUSTED_CONDITION_FULL72.mp4')
 if 'raw' in requested: encode(dirs['raw_rgb']/'F%02d.png',out/'final_RAW_CONDITION_FULL72.mp4')
 if args.only=='both': encode(dirs['comparison_rgb']/'F%02d.png',out/'final_ADJUSTED_RAW_CLEAN_FULL72.mp4')
 (out/'HANDOFF_PHASE_B.md').write_text('# HANDOFF Phase B\n\n'+json.dumps(manifest,indent=2)+'\n\nClean RGB/depth/lines are QA-only; adjusted/raw conditions are rendered from fixed-count Gaussian PLYs with exact P48 cameras and alignment.\n')
 print(json.dumps({'status':'PASS','output':str(out),'adjusted_mean_coverage':np.mean([r['adjusted']['coverage'] for r in rows]),'raw_mean_coverage':np.mean([r['raw']['coverage'] for r in rows])},indent=2))
if __name__=='__main__': main()
