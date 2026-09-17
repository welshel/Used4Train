import json, hashlib, shutil, sys
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/root/autodl-tmp/InfiniSplat')
from src.utils.gaussians import Gaussians3D
from src.demo.infer_single_image import filter_final_gaussian_floaters
from gsplat import rasterization

ROOT=Path('/root/autodl-tmp'); OUT=ROOT/'outputs/p24_native_gaussian_soft_normal_safety';
for n in ['00_repro','01_glb_qa','02_risk','03_candidates','04_best','final']: (OUT/n).mkdir(parents=True,exist_ok=True)
ART=ROOT/'outputs/p21_hf_exact_infinisplat_baseline/final/gaussians.pt'; P23=ROOT/'outputs/p23_source_camera_canonical_transfer/final';
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def movie(p,fs,fps=12):
 H,W=fs[0].shape[:2];v=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H));
 for f in fs:v.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
 v.release()
def board(fs,labels,p,cols=4):
 tile=(320,240); rows=(len(fs)+cols-1)//cols; c=Image.new('RGB',(cols*tile[0],rows*(tile[1]+22)),(15,15,15));d=ImageDraw.Draw(c)
 for i,(f,l) in enumerate(zip(fs,labels)):
  im=Image.fromarray(f); im.thumbnail(tile);x=(i%cols)*tile[0];y=(i//cols)*(tile[1]+22);c.paste(im,(x,y+22));d.text((x+3,y+3),l,fill='white')
 c.save(p)
loop=json.load(open(P23/'camera_loop_infinisplat.json')); C=np.array([x['c2w_opencv_infinisplat'] for x in loop['poses']],np.float32); K=np.array([[502.2993847,0,320],[0,502.2993847,240],[0,0,1]],np.float32)
t=torch.load(ART,map_location='cpu',weights_only=True)['gaussians'];g=Gaussians3D(t['mean_vectors'],t['singular_values'],t['quaternions'],t['colors'],t['opacities'],t['covariances']);g=filter_final_gaussian_floaters(g).to('cuda');
M=g.mean_vectors[0].detach().cpu().numpy(); OP=g.opacities[0].detach().cpu().numpy(); COV=g.covariances[0].detach().cpu().numpy();
# Intrinsic Gaussian risk: covariance smallest axis as a surface normal, oriented toward canonical source camera at origin.
ev, evec=np.linalg.eigh(COV); n=evec[:,:,0]; srcdir=-M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-8); cos=np.abs(np.sum(n*srcdir,axis=1)); flat=np.clip(1.0-np.sqrt(np.maximum(ev[:,0]/np.maximum(ev[:,2],1e-12),0)),0,1); opconf=np.clip((OP-.03)/.25,0,1); normal_conf=0.7*cos+0.2*flat+0.1*opconf; normal_conf=np.clip(normal_conf,0,1)
risk={'formula':'risk=1-(0.70*abs(n_gaussian·source_dir)+0.20*flatness_confidence+0.10*opacity_confidence)','gaussian_count':int(len(M)),'cos_abs_p10_median_p90':[float(np.percentile(cos,10)),float(np.median(cos)),float(np.percentile(cos,90))],'flat_conf_p10_median_p90':[float(np.percentile(flat,10)),float(np.median(flat)),float(np.percentile(flat,90))],'opacity_conf_p10_median_p90':[float(np.percentile(opconf,10)),float(np.median(opconf)),float(np.percentile(opconf,90))],'risk_p10_median_p90':[float(np.percentile(1-normal_conf,10)),float(np.median(1-normal_conf)),float(np.percentile(1-normal_conf,90))]}
(OUT/'02_risk/risk_score_summary.json').write_text(json.dumps(risk,indent=2)); np.savez(OUT/'02_risk/gaussian_risk_scores.npz',risk=1-normal_conf,normal_conf=normal_conf,cos_abs=cos,flat_conf=flat,opacity_conf=opconf)
def render(gg):
 ex=torch.linalg.inv(torch.tensor(C,device='cuda')).unsqueeze(0); kt=torch.tensor(K,device='cuda').unsqueeze(0).unsqueeze(0).repeat(1,72,1,1)
 with torch.inference_mode(): rgb,a,_=rasterization(gg.mean_vectors.float(),gg.quaternions.float(),gg.singular_values.float(),gg.opacities.float(),gg.colors.float(),ex,kt,640,480,sh_degree=None,render_mode='RGB',packed=True,covars=gg.covariances.float(),eps2d=1e-8)
 return (rgb[0].cpu().numpy()*255).clip(0,255).astype(np.uint8),a[0,...,0].cpu().numpy()
def metrics(alpha):
 m=alpha>1e-3;cov=m.mean((1,2)); i=[float((m[j]&m[j+1]).sum()/max(1,(m[j]|m[j+1]).sum())) for j in range(71)]; li=float((m[0]&m[-1]).sum()/max(1,(m[0]|m[-1]).sum()));return {'coverage_min':float(cov.min()),'coverage_p10':float(np.percentile(cov,10)),'coverage_median':float(np.median(cov)),'coverage_p90':float(np.percentile(cov,90)),'coverage_max':float(cov.max()),'adjacent_mask_iou_mean':float(np.mean(i)),'adjacent_mask_iou_median':float(np.median(i)),'adjacent_mask_iou_min':float(np.min(i)),'loop_closure_iou':li,'alpha_epsilon':1e-3}
base_rgb,base_a=render(g); base_m=metrics(base_a); (OUT/'00_repro/metrics_baseline_p23.json').write_text(json.dumps(base_m,indent=2)); movie(OUT/'00_repro/condition_loop_full_native_p23_reproduced.mp4',list(base_rgb));movie(OUT/'00_repro/support_alpha_p23_reproduced.mp4',[(np.clip(x*255,0,255).astype(np.uint8)[...,None].repeat(3,2)) for x in base_a]);movie(OUT/'00_repro/support_mask_p23_reproduced.mp4',[(x>1e-3).astype(np.uint8)[...,None].repeat(3,2)*255 for x in base_a]);board([base_rgb[i] for i in [0,10,20,30,40,50,60,71]],[f'P23 F{i:02d}' for i in [0,10,20,30,40,50,60,71]],OUT/'00_repro/P23_BASELINE_8POSE_BOARD.png')
policies={'R1_soft':(0.85,0.15),'R2_balanced':(0.70,0.30),'R3_safety':(0.55,0.45)}; results={}; rendered={}
for name,(floor,gain) in policies.items():
 fac=torch.tensor(floor+gain*normal_conf,dtype=torch.float32,device='cuda'); gg=Gaussians3D(g.mean_vectors,g.singular_values,g.quaternions,g.colors,g.opacities*fac[None,:],g.covariances); rgb,a=render(gg); mm=metrics(a); results[name]=mm;rendered[name]=(rgb,a); movie(OUT/f'03_candidates/{name}_rgb.mp4',list(rgb));movie(OUT/f'03_candidates/{name}_alpha.mp4',[(np.clip(x*255,0,255).astype(np.uint8)[...,None].repeat(3,2)) for x in a]);movie(OUT/f'03_candidates/{name}_mask.mp4',[(x>1e-3).astype(np.uint8)[...,None].repeat(3,2)*255 for x in a]); board([rgb[i] for i in [0,10,20,30,40,50,60,71]],[f'{name} F{i:02d}' for i in [0,10,20,30,40,50,60,71]],OUT/f'03_candidates/{name}_8pose.png')
scores={k:{'coverage_drop':base_m['coverage_median']-v['coverage_median'],'temporal_drop':base_m['adjacent_mask_iou_mean']-v['adjacent_mask_iou_mean'],'loop_drop':base_m['loop_closure_iou']-v['loop_closure_iou'],'passes_thresholds':v['coverage_median']>=.85 and v['coverage_p10']>=.70 and v['coverage_min']>=.60 and v['adjacent_mask_iou_mean']>=.88 and v['loop_closure_iou']>=.80} for k,v in results.items()};(OUT/'02_risk/risk_policy_summary.json').write_text(json.dumps({'policies':policies,'baseline':base_m,'candidates':results,'comparison':scores},indent=2));
best=next((k for k in ['R3_safety','R2_balanced','R1_soft'] if scores[k]['passes_thresholds']),None);best=best or 'R1_soft';br,ba=rendered[best];bestm=results[best];movie(OUT/'04_best/condition_loop_full_p24.mp4',list(br));movie(OUT/'04_best/support_alpha_p24.mp4',[(np.clip(x*255,0,255).astype(np.uint8)[...,None].repeat(3,2)) for x in ba]);movie(OUT/'04_best/support_mask_p24.mp4',[(x>1e-3).astype(np.uint8)[...,None].repeat(3,2)*255 for x in ba]);movie(OUT/'04_best/condition_random_start_p24.mp4',[br[(17+i)%72] for i in range(72)]);movie(OUT/'04_best/condition_window_61_p24.mp4',[br[(17+i)%72] for i in range(61)]);conf=np.broadcast_to(normal_conf.mean(),(72,480,640));movie(OUT/'04_best/confidence_p24.mp4',[(np.full((480,640,3),int(np.mean(normal_conf)*255),np.uint8)) for _ in range(72)]);board([br[i] for i in [0,10,20,30,40,50,60,71]],[f'{best} F{i:02d}' for i in [0,10,20,30,40,50,60,71]],OUT/'04_best/BEST_8POSE_BOARD.png')
# baseline vs best contact sheet
can=Image.new('RGB',(1280,520),(15,15,15));d=ImageDraw.Draw(can)
for col,(title,fs) in enumerate([('P23',base_rgb),('P24 '+best,br)]):
 for j,i in enumerate([0,10,20,30,40,50,60,71]):
  im=Image.fromarray(fs[i]);im.thumbnail((320,240));x=(j%4)*320;y=(j//4)*260;can.paste(im,(x+col*0,y+22));d.text((x+3,y+3),f'{title} F{i:02d}',fill='white')
can.save(OUT/'final/P23_vs_P24_contact_sheet.png')
shutil.copy2(OUT/'04_best/condition_loop_full_p24.mp4',OUT/'final/condition_loop_full_p24.mp4');shutil.copy2(OUT/'04_best/condition_random_start_p24.mp4',OUT/'final/condition_random_start_p24.mp4');shutil.copy2(OUT/'04_best/condition_window_61_p24.mp4',OUT/'final/condition_window_61_p24.mp4');shutil.copy2(OUT/'04_best/support_alpha_p24.mp4',OUT/'final/support_alpha_p24.mp4');shutil.copy2(OUT/'04_best/support_mask_p24.mp4',OUT/'final/support_mask_p24.mp4');shutil.copy2(OUT/'04_best/confidence_p24.mp4',OUT/'final/confidence_p24.mp4');shutil.copy2(OUT/'04_best/BEST_8POSE_BOARD.png',OUT/'final/BEST_8POSE_BOARD.png')
(OUT/'final/metrics_baseline_p23.json').write_text(json.dumps(base_m,indent=2));(OUT/'final/metrics_best_p24.json').write_text(json.dumps(bestm,indent=2));(OUT/'final/risk_policy_summary.json').write_text(json.dumps({'best_policy':best,'policies':policies,'baseline':base_m,'candidates':results,'comparison':scores},indent=2));
print(json.dumps({'best':best,'baseline':base_m,'candidates':results,'scores':scores},indent=2))
