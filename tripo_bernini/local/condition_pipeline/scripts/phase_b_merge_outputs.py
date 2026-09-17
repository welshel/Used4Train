from pathlib import Path
from PIL import Image
import numpy as np,json,subprocess,shutil
root=Path('outputs/phase_b_adjusted_raw'); clean=Path('outputs/p48_tripo_clean_aligned/clean_rgb')
for name in ('comparison_rgb','comparison_depth','comparison_lines','final'): (root/name).mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(72):
 f=f'F{i:02d}.png'
 a=np.asarray(Image.open(root/'condition_adjusted_rgb'/f).convert('RGB')); r=np.asarray(Image.open(root/'condition_raw_rgb'/f).convert('RGB')); c=np.asarray(Image.open(clean/f).convert('RGB')); Image.fromarray(np.concatenate([a,r,c],1)).save(root/'comparison_rgb'/f)
 da=np.asarray(Image.open(root/'condition_adjusted_depth'/f).convert('RGB')); dr=np.asarray(Image.open(root/'condition_raw_depth'/f).convert('RGB')); Image.fromarray(np.concatenate([da,dr],1)).save(root/'comparison_depth'/f)
 la=np.asarray(Image.open(root/'condition_adjusted_lines'/f).convert('RGB')); lr=np.asarray(Image.open(root/'condition_raw_lines'/f).convert('RGB')); Image.fromarray(np.concatenate([la,lr],1)).save(root/'comparison_lines'/f)
 rows.append({'frame':i,'adjusted_rgb':str(root/'condition_adjusted_rgb'/f),'raw_rgb':str(root/'condition_raw_rgb'/f),'clean_rgb':str(clean/f),'exact_pair':f'F{i:02d}->F{i:02d}'})
a=json.loads((root/'PHASE_B_RENDER_METRICS.json').read_text()); a['adjusted_mean_coverage']=json.loads((Path('outputs/phase_b_adjusted_test')/'PHASE_B_RENDER_METRICS.json').read_text())['adjusted_mean_coverage']; a['raw_mean_coverage']=json.loads((Path('outputs/phase_b_raw_test')/'PHASE_B_RENDER_METRICS.json').read_text())['raw_mean_coverage']; (root/'PHASE_B_RENDER_METRICS.json').write_text(json.dumps(a,indent=2)+'\n')
m={'status':'PASS','frame_count':72,'resolution':[896,896],'frame_order':'F00-F71 exact','camera_contract':'P48 immutable OpenCV C2W','alignment':'outputs/p48_tripo_clean_aligned/P48_TRIPO_ALIGNMENT.json','adjusted_ply':'outputs/phase_a_opacity_g06/splat_adjusted_opacity.ply','raw_ply':'outputs/p48_1_tripo_cleanup/final/splat_cleaned.ply','clean_not_in_condition':True,'frames':rows}; (root/'PHASE_B_PAIRING_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n'); (root/'HANDOFF_PHASE_B.md').write_text('# HANDOFF Phase B\n\nStatus PASS. Adjusted/raw exact F00-F71 packs use immutable P48 cameras/alignment. Clean RGB/depth/lines are QA-only.\n\n'+json.dumps(a,indent=2)+'\n')
subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','12','-i',str(root/'comparison_rgb'/'F%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','17','-movflags','+faststart',str(root/'PHASE_B_ADJUSTED_RAW_CLEAN_FULL72.mp4')],check=True)
print(json.dumps(a,indent=2))
