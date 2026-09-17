#!/usr/bin/env python3
"""P2.0 frozen normal-aware rasterizer for an official InfiniSplat PLY.

The PLY is the only inference geometry.  GLB is deliberately not opened here;
post-write GLB checks are a separate script.
"""
import argparse, json, math, shutil, subprocess, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from numba import njit
from scipy import ndimage
from plyfile import PlyData

@njit(cache=True)
def raster(points, normals, colors, radius, camera, R, K, W, H, tau):
    owner=np.full((H,W),-1,np.int32); zbuf=np.full((H,W),np.inf,np.float32)
    for i in range(points.shape[0]):
        qx=points[i,0]-camera[0]; qy=points[i,1]-camera[1]; qz=points[i,2]-camera[2]
        X=R[0,0]*qx+R[1,0]*qy+R[2,0]*qz; Y=R[0,1]*qx+R[1,1]*qy+R[2,1]*qz; Z=R[0,2]*qx+R[1,2]*qy+R[2,2]*qz
        if Z<=1e-4: continue
        u=K[0,0]*X/Z+K[0,2]; v=K[1,1]*Y/Z+K[1,2]
        rr=max(0.75,min(10.0,1.6*K[0,0]*radius[i]/Z))
        if u < -rr or u >= W+rr or v < -rr or v >= H+rr: continue
        x0=max(0,int(math.floor(u-rr-1))); x1=min(W-1,int(math.ceil(u+rr+1)))
        y0=max(0,int(math.floor(v-rr-1))); y1=min(H-1,int(math.ceil(v+rr+1)))
        for yy in range(y0,y1+1):
            for xx in range(x0,x1+1):
                dx=(xx-u)/rr; dy=(yy-v)/rr
                if dx*dx+dy*dy>1.0: continue
                if Z<zbuf[yy,xx]: zbuf[yy,xx]=Z; owner[yy,xx]=i
    return owner,zbuf

def sha(p):
 h=hashlib.sha256(); f=open(p,'rb')
 for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 f.close(); return h.hexdigest()

def decode_ply(path, reg_path):
 d=PlyData.read(path); v=d['vertex'].data
 xyz=np.stack([v['x'],v['y'],v['z']],1).astype(np.float32)
 dc=np.stack([v['f_dc_0'],v['f_dc_1'],v['f_dc_2']],1).astype(np.float32)
 col=np.clip(0.5+0.2820947918*dc,0,1)*255
 op=1/(1+np.exp(-np.asarray(v['opacity'],np.float32)))
 sc=np.exp(np.stack([v['scale_0'],v['scale_1'],v['scale_2']],1).astype(np.float32))
 keep=np.isfinite(xyz).all(1)&(op>0.02)&np.isfinite(sc).all(1)
 xyz=xyz[keep]; col=col[keep].astype(np.uint8); sc=sc[keep]
 # Covariance-derived normals: smallest Gaussian axis, oriented toward source view.
 q=np.stack([v[f'rot_{i}'] for i in range(4)],1).astype(np.float32)[keep]
 q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
 ax=np.argmin(sc,1); n=np.zeros_like(xyz)
 for j in range(3):
  sel=ax==j; n[sel,j]=1
 # Quaternion rotation (w,x,y,z), vectorized for selected canonical axes.
 w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]
 # R columns applied to canonical axis j.
 R0=np.stack([1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)],1)
 R1=np.stack([2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)],1)
 R2=np.stack([2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)],1)
 n=np.where((ax==0)[:,None],R0,np.where((ax==1)[:,None],R1,R2)).astype(np.float32)
 # Source camera is origin in InfiniSplat coordinates: face it when ambiguous.
 toward=-xyz; sgn=np.sum(n*toward,1)<0; n[sgn]*=-1
 T=np.asarray(json.loads(Path(reg_path).read_text())['row_affine_4x4'],np.float32); L=T[:3,:3]
 xyz=(np.c_[xyz,np.ones(len(xyz),np.float32)]@T)[:,:3].astype(np.float32)
 n=(n@np.linalg.inv(L).T); n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
 # conservative Gaussian splat radius after registration
 radius=(np.median(sc,1)*np.linalg.norm(L,axis=0).mean()).astype(np.float32)
 radius=np.clip(radius,0.002,0.12)
 # Keep render tractable while retaining spatial coverage.
 if len(xyz)>220000:
  step=int(math.ceil(len(xyz)/220000)); xyz, n, col, radius=xyz[::step],n[::step],col[::step],radius[::step]
 return xyz.astype(np.float32),n.astype(np.float32),col,radius

def tax(owner,points,normals,camera,tau=0.10):
 g=owner>=0; valid=np.zeros_like(g); grazing=np.zeros_like(g); back=np.zeros_like(g)
 yy,xx=np.nonzero(g)
 if len(yy):
  ids=owner[yy,xx]; view=camera[None,:]-points[ids]; view/=np.maximum(np.linalg.norm(view,axis=1,keepdims=True),1e-8); d=np.sum(normals[ids]*view,1)
  f=d>tau; gr=(d>0)&~f; b=~f&~gr; valid[yy[f],xx[f]]=1; grazing[yy[gr],xx[gr]]=1; back[yy[b],xx[b]]=1
 no=~g; lab,num=ndimage.label(no); thin=np.zeros_like(no); large=np.zeros_like(no)
 for k in range(1,num+1):
  c=lab==k; ys,xs=np.nonzero(c); area=len(ys); hh=ys.max()-ys.min()+1; ww=xs.max()-xs.min()+1
  (thin if area<=96 or min(hh,ww)<=2 else large)[c]=1
 return {'geometry_hit_mask':g,'appearance_valid_mask':valid,'invalid_but_occluding_mask':g&~valid,'no_hit_mask':no,'thin_hole_mask':thin,'large_no_hit_mask':large,'grazing_mask':grazing,'backside_mask':back}

def vis_depth(z):
 ok=np.isfinite(z)&(z>0); o=np.zeros((*z.shape,3),np.uint8)
 if ok.any():
  lo,hi=np.percentile(z[ok],[2,98]); q=np.clip((z-lo)/max(hi-lo,1e-6),0,1); o[...,0]=np.clip(255*(1.5*q-.5),0,255); o[...,1]=np.clip(255*(1.5-np.abs(2*q-1)),0,255); o[...,2]=np.clip(255*(1-q),0,255)
 return o
def vis_tax(t):
 o=np.zeros((*t['geometry_hit_mask'].shape,3),np.uint8); o[t['no_hit_mask']]=[35,85,220]; o[t['invalid_but_occluding_mask']]=[255,0,255]; o[t['appearance_valid_mask']]=[35,220,85]; o[t['grazing_mask']]=[255,210,35]; return o
def enc(frames,suff,out,fps,indices):
 tmp=frames.parent/('_enc_'+suff); tmp.mkdir(exist_ok=True)
 for p in tmp.glob('*.png'): p.unlink()
 for j,i in enumerate(indices): shutil.copy2(frames/f'{i:03d}_{suff}.png',tmp/f'{j:03d}.png')
 subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(tmp/'%03d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(out)],check=True)
def board(frames,items,out,cols=4,panel=(320,240)):
 rows=math.ceil(len(items)/cols); c=Image.new('RGB',(cols*panel[0],rows*(panel[1]+22)),(20,20,25)); d=ImageDraw.Draw(c)
 for j,(i,s,title) in enumerate(items):
  im=Image.open(frames/f'{i:03d}_{s}.png').convert('RGB').resize(panel); x=(j%cols)*panel[0]; y=(j//cols)*(panel[1]+22); c.paste(im,(x,y+22)); d.text((x+3,y+4),title,fill='white')
 c.save(out)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--ply',required=True); ap.add_argument('--loop',required=True); ap.add_argument('--registration',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
 out=Path(args.out); out.mkdir(parents=True,exist_ok=True); frames=out/'frames'; frames.mkdir(exist_ok=True)
 p=json.loads(Path(args.loop).read_text()); W=int(p['intrinsics']['width']); H=int(p['intrinsics']['height']); K=np.array([[p['intrinsics']['fx'],0,p['intrinsics']['cx']],[0,p['intrinsics']['fy'],p['intrinsics']['cy']],[0,0,1]],np.float32)
 pts,norm,col,rad=decode_ply(args.ply,args.registration); meta=[]; rendered=[]
 for i,pose in enumerate(p['poses']):
  c2w=np.asarray(pose['c2w_opencv_pointmap'],np.float32); cam=c2w[:3,3]; R=c2w[:3,:3]; owner,z=raster(pts,norm,col,rad,cam,R,K,W,H,0.1); t=tax(owner,pts,norm,cam); rgb=np.zeros((H,W,3),np.uint8); yy,xx=np.nonzero(t['appearance_valid_mask']); rgb[yy,xx]=col[owner[yy,xx]]
  Image.fromarray(rgb).save(frames/f'{i:03d}_condition_rgb.png'); Image.fromarray(vis_tax(t)).save(frames/f'{i:03d}_taxonomy.png'); Image.fromarray(vis_depth(np.where(t['appearance_valid_mask'],z,np.inf))).save(frames/f'{i:03d}_depth.png'); Image.fromarray(vis_depth(z)).save(frames/f'{i:03d}_occlusion_depth.png')
  for k,v in t.items(): Image.fromarray(v.astype(np.uint8)*255).save(frames/f'{i:03d}_{k}.png'); np.save(frames/f'{i:03d}_{k}.npy',v)
  np.save(frames/f'{i:03d}_occlusion_depth.npy',z); np.save(frames/f'{i:03d}_appearance_depth.npy',np.where(t['appearance_valid_mask'],z,np.inf)); np.save(frames/f'{i:03d}_owner.npy',owner)
  meta.append({'video_frame_id':i,'geometry_hit_ratio':float(t['geometry_hit_mask'].mean()),'appearance_valid_ratio':float(t['appearance_valid_mask'].mean()),'invalid_but_occluding_ratio':float(t['invalid_but_occluding_mask'].mean()),'no_hit_ratio':float(t['no_hit_mask'].mean()),'thin_hole_ratio':float(t['thin_hole_mask'].mean()),'large_no_hit_ratio':float(t['large_no_hit_mask'].mean()),'grazing_ratio':float(t['grazing_mask'].mean()),'backside_ratio':float(t['backside_mask'].mean())}); rendered.append(t)
 qa={'frame_count':len(meta),'adjacent_geometry_iou_median':float(np.median([np.mean(a['geometry_hit_mask']&b['geometry_hit_mask'])/max(np.mean(a['geometry_hit_mask']|b['geometry_hit_mask']),1e-8) for a,b in zip(rendered[:-1],rendered[1:])])), 'adjacent_appearance_iou_median':float(np.median([np.mean(a['appearance_valid_mask']&b['appearance_valid_mask'])/max(np.mean(a['appearance_valid_mask']|b['appearance_valid_mask']),1e-8) for a,b in zip(rendered[:-1],rendered[1:])])), 'loop_closure_geometry_iou':float(np.mean(rendered[0]['geometry_hit_mask']&rendered[-1]['geometry_hit_mask'])/max(np.mean(rendered[0]['geometry_hit_mask']|rendered[-1]['geometry_hit_mask']),1e-8)), 'loop_closure_appearance_iou':float(np.mean(rendered[0]['appearance_valid_mask']&rendered[-1]['appearance_valid_mask'])/max(np.mean(rendered[0]['appearance_valid_mask']|rendered[-1]['appearance_valid_mask']),1e-8))}
 (out/'render_metadata.json').write_text(json.dumps({'representation':'official InfiniSplat Gaussian PLY','ply':str(args.ply),'ply_sha256':sha(args.ply),'glb_used_for_inference':False,'frames':meta},indent=2)); (out/'temporal_qa.json').write_text(json.dumps(qa,indent=2))
 idx=list(range(len(meta))); rnd=[(17+i)%len(meta) for i in idx]; win=rnd[:61]; fps=int(p['intrinsics']['fps'])
 for s,nm in [('condition_rgb','condition_loop_full.mp4'),('taxonomy','condition_taxonomy.mp4'),('depth','condition_depth.mp4'),('occlusion_depth','geometry_hit.mp4'),('appearance_valid_mask','appearance_valid.mp4'),('invalid_but_occluding_mask','invalid_occluding.mp4'),('no_hit_mask','no_hit.mp4')]: enc(frames,s,out/nm,fps,idx)
 enc(frames,'condition_rgb',out/'condition_loop_random_start.mp4',fps,rnd); enc(frames,'condition_rgb',out/'condition_window_61.mp4',fps,win)
 board(frames,[(i,'condition_rgb',f'F{i:02d}') for i in [0,10,20,30,40,50,60,71]],out/'CONDITION_QUALITY_BOARD.png')
 board(frames,[(i,s,f'{i:02d} {name}') for i in [0,10,20,30,40,50,60] for s,name in [('condition_rgb','RGB'),('taxonomy','tax'),('depth','depth'),('no_hit_mask','no-hit')]],out/'BC_CONDITION_BOARD.png')
 board(frames,[(i,'condition_rgb',f'RGB {i}') for i in [0,1,10,11,20,21,30,31,40,41,50,51,60,61]],out/'TEMPORAL_QA_BOARD.png',cols=7,panel=(180,135))
 board(frames,[(0,'condition_rgb','first'),(71,'condition_rgb','last'),(0,'taxonomy','first tax'),(71,'taxonomy','last tax')],out/'LOOP_CLOSURE_BOARD.png',cols=4,panel=(320,240))
 (out/'random_start_metadata.json').write_text(json.dumps({'seed':20250905,'start_idx':17,'canonical_frame_count':len(meta),'random_order':rnd,'window_61_order':win,'is_cyclic_shift':True},indent=2))
 print(json.dumps({'points':len(pts),'qa':qa,'median_geometry':float(np.median([m['geometry_hit_ratio'] for m in meta])),'median_appearance':float(np.median([m['appearance_valid_ratio'] for m in meta]))},indent=2))
if __name__=='__main__': main()
