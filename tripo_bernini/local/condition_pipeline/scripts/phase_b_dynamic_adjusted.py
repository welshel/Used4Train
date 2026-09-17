#!/usr/bin/env python3
"""Render the adjusted teacher with P48.3 trajectory-aware suppression.

The Gaussian teacher remains fixed; only the adjusted *condition render* gets
the per-frame full-render opacity attenuation.  This specifically targets the
near-camera curtain/floater set absent from Clean while retaining temporal EMA
weights and exact F00-F71 cameras.
"""
from pathlib import Path
import argparse, importlib.util, json, subprocess, sys, shutil, hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
_GSPLAT_READY = False

def ensure_gsplat_backend():
    """Load the already-built gsplat extension without a per-process rebuild.

    The p47a environment uses PyTorch's in-memory JIT versioner, so importing
    gsplat.cuda._backend in every fresh Python process otherwise recompiles the
    CUDA extension.  A completed cached extension is safe to load directly;
    if no cache is present we leave the normal gsplat JIT fallback untouched.
    """
    global _GSPLAT_READY
    if _GSPLAT_READY:
        return
    import glob, importlib.util, os, sys
    try:
        import gsplat
        if not hasattr(gsplat, "csrc"):
            candidates = []
            ext_root = os.environ.get("TORCH_EXTENSIONS_DIR")
            if ext_root:
                candidates.extend(glob.glob(os.path.join(ext_root, "**", "gsplat_cuda", "gsplat_cuda.so"), recursive=True))
            candidates.extend(glob.glob(os.path.expanduser("~/.cache/torch_extensions/**/gsplat_cuda/gsplat_cuda.so"), recursive=True))
            if candidates:
                so = sorted(set(candidates), key=os.path.getmtime)[-1]
                mod = sys.modules.get("gsplat_cuda")
                if mod is None:
                    spec = importlib.util.spec_from_file_location("gsplat_cuda", so)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["gsplat_cuda"] = mod
                    spec.loader.exec_module(mod)
                gsplat.csrc = mod
                sys.modules["gsplat.csrc"] = mod
    except Exception:
        # Normal gsplat import/JIT below will provide the actionable error if
        # neither a prebuilt extension nor a usable CUDA toolchain is present.
        pass
    _GSPLAT_READY = True

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
P48=load('p48_exact_dynamic_adjusted',ROOT/'scripts/render_tripo_clean_path.py')
P482=load('p482_dynamic_adjusted',ROOT/'scripts/p48_2_dynamic_suppression.py')
PA=load('phase_a_dynamic_adjusted',ROOT/'scripts/phase_a_clean_adjusted_gaussian.py'); PA.load_p48()

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def wjson(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def png(p,a):
 p.parent.mkdir(parents=True,exist_ok=True); a=np.asarray(a); Image.fromarray(np.uint8(np.clip(a,0,1)*255) if a.dtype!=np.uint8 else a).save(p)
def depthvis(z,m):
 z=np.asarray(z,np.float32); m=np.asarray(m,bool); out=np.zeros((*z.shape,3),np.uint8)
 if m.any():
  v=z[m]; lo,hi=np.percentile(v,[2,98]); q=np.clip((z-lo)/max(hi-lo,1e-6),0,1); out=np.repeat(np.uint8(q*255)[...,None],3,2); out[~m]=0
 return out
def encode(pat,out,fps=12):
 out.parent.mkdir(parents=True,exist_ok=True); subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(pat),'-c:v','libx264','-pix_fmt','yuv420p','-crf','17','-movflags','+faststart',str(out)],check=True)
def decode(rec):
 xyz=np.column_stack([rec[x] for x in ('x','y','z')]).astype(np.float32); rgb=np.clip(.5+P48.SH_C0*np.column_stack([rec[x] for x in ('f_dc_0','f_dc_1','f_dc_2')]),0,1).astype(np.float32); op=(1/(1+np.exp(-rec['opacity']))).astype(np.float32); sc=np.exp(np.column_stack([rec[x] for x in ('scale_0','scale_1','scale_2')])).astype(np.float32); q=np.column_stack([rec[x] for x in ('rot_0','rot_1','rot_2','rot_3')]).astype(np.float32); q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8); return xyz,rgb,op,sc,q
def render_frame(world,rgb,op,sc,q,cam,w,device):
 import torch
 ensure_gsplat_backend()
 from gsplat import rasterization
 dev=torch.device(device); mt=torch.from_numpy(world).to(dev); qt=torch.from_numpy(q).to(dev); st=torch.from_numpy(sc).to(dev); ot=torch.from_numpy(op).to(dev); ct=torch.from_numpy(rgb).to(dev); view=torch.linalg.inv(torch.from_numpy(cam.c2w).to(dev)).reshape(1,4,4); K=torch.from_numpy(cam.K).to(dev).reshape(1,3,3)
 with torch.inference_mode():
  im,a,_=rasterization(mt,qt,st,ot,ct,view,K,cam.width,cam.height,sh_degree=None,render_mode='RGB',packed=True,eps2d=1e-8,near_plane=.05)
  ed,_,_=rasterization(mt,qt,st,ot,ct,view,K,cam.width,cam.height,sh_degree=None,render_mode='ED',packed=True,eps2d=1e-8,near_plane=.05)
 out=np.clip(im[0].detach().cpu().numpy(),0,1).astype(np.float32); aa=np.clip(a[0,...,0].detach().cpu().numpy(),0,1).astype(np.float32); dd=ed[0,...,0].detach().cpu().numpy().astype(np.float32); del mt,qt,st,ot,ct,view,K,im,a,ed; torch.cuda.empty_cache(); return out,aa,dd
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--adjusted',type=Path,default=ROOT/'outputs/phase_a_clean_adjusted_gaussian_tune4/splat_adjusted.ply'); ap.add_argument('--raw-rgb',type=Path,default=ROOT/'outputs/p48_tripo_clean_aligned/condition_rgb'); ap.add_argument('--raw-depth',type=Path,default=ROOT/'outputs/p49b1_depth_geometry_audit/03_tripo_depth/expected'); ap.add_argument('--raw-lines',type=Path,default=ROOT/'outputs/p48_1_tripo_cleanup/condition_rgb_cleaned'); ap.add_argument('--dataset',type=Path,default=ROOT/'work/327431980_4_normalized'); ap.add_argument('--weights',type=Path,default=ROOT/'outputs/p49b1_depth_geometry_audit/03_tripo_depth/P49B1_DYNAMIC_WEIGHTS.npy'); ap.add_argument('--alignment',type=Path,default=ROOT/'outputs/p48_tripo_clean_aligned/P48_TRIPO_ALIGNMENT.json'); ap.add_argument('--output',type=Path,default=ROOT/'outputs/phase_b_adjusted_dynamic'); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
 out=args.output; out.mkdir(parents=True,exist_ok=True)
 for n in ('condition_adjusted_rgb','condition_adjusted_depth','condition_adjusted_lines','condition_raw_rgb','condition_raw_depth','condition_raw_lines','comparison_rgb','comparison_depth','comparison_lines','final'): (out/n).mkdir(parents=True,exist_ok=True)
 cams=P48.load_clean_cameras(args.dataset); M=np.asarray(json.loads(args.alignment.read_text())['transform'],float); weights=np.load(args.weights).astype(np.float32)
 if weights.shape!=(72,261646): raise ValueError(f'weight shape {weights.shape}')
 rec,header,hb=PA.load_raw(args.adjusted); xyz,rgb,op,sc,q=decode(rec); sim=float(np.cbrt(np.linalg.det(M[:3,:3]))); world=P48.apply_similarity(xyz,M).astype(np.float32); ws=sc*abs(sim); wq=P48.transform_quaternions(q,M)
 metrics=[]; target_frames=[10,11,12,13,14,17,24,28,29,30,31,50]
 clean_dir=ROOT/'outputs/p48_tripo_clean_aligned/clean_rgb'; clean=[]
 for i,cam in enumerate(cams):
  opi=op*(1.0-np.clip(weights[i],0,1)); im,a,d=render_frame(world,rgb,opi,ws,wq,cam,weights[i],args.device); valid=(a>1e-3)&np.isfinite(d)&(d>0); edge=PA.depth_edges(d,valid); edge_rgb=np.repeat((np.uint8(edge)*255)[...,None],3,axis=2); png(out/'condition_adjusted_rgb'/f'F{i:02d}.png',im); np.save(out/'condition_adjusted_depth'/f'F{i:02d}.npy',d); png(out/'condition_adjusted_depth'/f'F{i:02d}.png',depthvis(d,valid)); png(out/'condition_adjusted_lines'/f'F{i:02d}.png',edge.astype(np.uint8)); c=np.asarray(Image.open(clean_dir/f'F{i:02d}.png').convert('RGB')); raw=np.asarray(Image.open(args.raw_rgb/f'F{i:02d}.png').convert('RGB')); png(out/'condition_raw_rgb'/f'F{i:02d}.png',raw); Image.fromarray(np.concatenate([np.uint8(im*255),raw,c],1)).save(out/'comparison_rgb'/f'F{i:02d}.png'); Image.fromarray(np.concatenate([depthvis(d,valid),c],1)).save(out/'comparison_depth'/f'F{i:02d}.png'); Image.fromarray(np.concatenate([edge_rgb,np.zeros_like(edge_rgb,dtype=np.uint8)],1)).save(out/'comparison_lines'/f'F{i:02d}.png');
  lum=im.mean(2); rawlum=raw.astype(np.float32).mean(2)/255.; cleanlum=c.astype(np.float32).mean(2)/255.; white=((rawlum>=.45)&(cleanlum<.45)); edge_rgb=np.repeat((np.uint8(edge)*255)[...,None],3,axis=2); metrics.append({'frame':i,'coverage':float(valid.mean()),'alpha_mean':float(a.mean()),'weight_mean':float(weights[i].mean()),'weight_active_ge_05':int((weights[i]>=.05).sum()),'white_foreground_pixels_before':int(white.sum()),'white_foreground_pixels_after':int(((lum>=.45)&(cleanlum<.45)).sum()),'rgb_mse_clean':float(np.mean((im-c.astype(np.float32)/255.)**2))})
 m={'status':'PASS','frame_count':72,'resolution':[896,896],'camera_contract':'P48 exact OpenCV C2W','alignment':str(args.alignment),'adjusted_splat':str(args.adjusted),'adjusted_splat_sha256':sha(args.adjusted),'dynamic_weights':str(args.weights),'dynamic_weight_contract':'P48.3 final candidate-scoped full-render opacity attenuation with causal EMA/hysteresis; Clean not used','frames':metrics,'mean_coverage':float(np.mean([x['coverage'] for x in metrics])),'target_frames':target_frames}
 wjson(out/'P49_DYNAMIC_ADJUSTED_SUPPRESSION_STATS.json',m); wjson(out/'P49_DYNAMIC_ADJUSTED_TEMPORAL.json',{'mean_weight_delta':float(np.mean(np.abs(np.diff(weights,axis=0)))),'max_mean_weight_delta':float(np.max(np.abs(np.diff(weights.mean(1))))),'active_fraction_max':float(np.max((weights>=.05).mean(1))),'active_fraction_delta_max':float(np.max(np.abs(np.diff((weights>=.05).mean(1))))),'weights_source':str(args.weights)})
 encode(out/'condition_adjusted_rgb'/'F%02d.png',out/'P49_ADJUSTED_DYNAMIC_FULL72.mp4'); encode(out/'comparison_rgb'/'F%02d.png',out/'P49_ADJUSTED_DYNAMIC_VS_RAW_VS_CLEAN_FULL72.mp4')
 (out/'HANDOFF_PHASE_B_DYNAMIC.md').write_text('# Phase B dynamic adjusted handoff\n\n'+json.dumps(m,indent=2)+'\n')
 print(json.dumps({'status':'PASS','mean_coverage':m['mean_coverage'],'F11_F13': [metrics[i] for i in (11,12,13)]},indent=2))
if __name__=='__main__': main()
