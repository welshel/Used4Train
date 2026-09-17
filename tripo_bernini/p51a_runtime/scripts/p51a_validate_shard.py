#!/usr/bin/env python3
"""Recoverable P51A per-start validation shard and CPU finalizer."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
import torch
from bernini.pipeline import BerniniRendererPipeline
from bernini.weights import HIGH_NOISE_PREFIXES,load_transformer_state_dict
import p51a_validate_checkpoint as common

def args_():
 p=argparse.ArgumentParser()
 for k in ("base","checkpoint","output-dir","adjusted-clips","clean-clips","comparison-script"):p.add_argument("--"+k,required=True)
 p.add_argument("--start",type=int);p.add_argument("--finalize-only",action="store_true")
 p.add_argument("--device",default="cuda:0");p.add_argument("--num-inference-steps",type=int,default=40);p.add_argument("--seed",type=int,default=42)
 a=p.parse_args()
 if a.finalize_only == (a.start is not None):p.error("specify exactly one of --start or --finalize-only")
 if a.start is not None and a.start not in common.FIXED_STARTS:p.error(f"--start must be one of {common.FIXED_STARTS}")
 return a
def atomic(path,value):
 t=path.with_name(path.name+".tmp");t.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(t,path)
def art(out,start):
 n=f"start{start:02d}"
 return n,out/f"{n}.mp4",out/f"comparison_{n}.mp4",out/f"{n}.metrics.json",out/f"P51A_START_COMPLETE_{n}.json"
def record(out,start):
 n,p,c,m,k=art(out,start)
 if not all(x.is_file() for x in (p,c,m,k)):return None
 try: marker=json.loads(k.read_text());row=json.loads(m.read_text())
 except (OSError,json.JSONDecodeError):return None
 if marker.get("status")!="complete" or marker.get("start")!=start or marker.get("clean_used_as_inference_input") is not False:return None
 if row.get("clean_used_as_inference_input") is not False:return None
 return row
def run(a,provenance):
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 if record(out,a.start) is not None:print(f"start{a.start:02d} already complete");return
 n,p,c,m,k=art(out,a.start);condition=Path(a.adjusted_clips)/f"cyclic_start{a.start:02d}.mkv";clean=Path(a.clean_clips)/f"cyclic_start{a.start:02d}.mkv"
 if not condition.is_file() or not clean.is_file():raise FileNotFoundError(f"missing validation clip for {n}")
 device=torch.device(a.device);torch.cuda.set_device(device)
 pipe=BerniniRendererPipeline.from_pretrained(a.base,device=str(device),load_ckpt_weights=False)
 state,prefix=load_transformer_state_dict(a.checkpoint,HIGH_NOISE_PREFIXES);missing,unexpected=pipe.model.diff_dec.transformer.load_state_dict(state,strict=False)
 if missing or unexpected:raise RuntimeError(f"checkpoint/model mismatch: missing={len(missing)} unexpected={len(unexpected)}")
 pipe.model.eval().to(device)
 pp=p.with_name(p.stem+".partial.mp4");cp=c.with_name(c.stem+".partial.mp4");pp.unlink(missing_ok=True);cp.unlink(missing_ok=True)
 # video=condition is the only visual inference argument; clean is QA-only below.
 pipe(common.PROMPT,video=str(condition),output_path=str(pp),num_frames=33,max_image_size=672,num_inference_steps=a.num_inference_steps,guidance_mode="v2v",seed=a.seed,fps=16)
 pipe.model.to(device).eval();os.replace(pp,p)
 subprocess.run([a.comparison_script,str(condition),str(p),str(clean),str(cp)],check=True);os.replace(cp,c)
 with tempfile.TemporaryDirectory(prefix="p51a_validate_",dir=out) as t:
  root=Path(t)
  prediction_frames=common.load_rgb_frames(p,root/"prediction")
  clean_frames=common.load_rgb_frames(clean,root/"clean")
  condition_frames=common.load_rgb_frames(condition,root/"condition")
  pm=common.metrics(prediction_frames,clean_frames)
  cm=common.metrics(condition_frames,clean_frames)
 row={"start":a.start,"prediction_vs_clean":pm,"condition_vs_clean":cm,"condition_input":str(condition),"clean_target_for_qa_only":str(clean),"prediction":str(p),"comparison_for_qa_only":str(c),"clean_used_as_inference_input":False,"inference":{"base":a.base,"guidance_mode":"v2v","num_frames":33,"max_image_size":672,"num_inference_steps":a.num_inference_steps,"seed":a.seed,"transformer_prefix":prefix}}
 atomic(m,row);atomic(k,{"status":"complete","start":a.start,"metrics":str(m),"clean_used_as_inference_input":False,"checkpoint_provenance":provenance});print(json.dumps(pm,indent=2))
def finalize(a,provenance):
 out=Path(a.output_dir);rows=[]
 for s in common.FIXED_STARTS:
  r=record(out,s)
  if r is None:raise RuntimeError(f"start{s:02d} lacks a complete, provenance-safe shard")
  rows.append(r)
 result={"checkpoint":str(Path(a.checkpoint)),"checkpoint_provenance":provenance,"fixed_starts":list(common.FIXED_STARTS),"inference":rows[0]["inference"]|{"clean_used_as_inference_input":False},"per_start":{r["prediction"].rsplit("/",1)[-1].removesuffix(".mp4"):{"prediction_vs_clean":r["prediction_vs_clean"],"condition_vs_clean":r["condition_vs_clean"],"condition_input":r["condition_input"],"clean_target_for_qa_only":r["clean_target_for_qa_only"],"prediction":r["prediction"]} for r in rows},"mean_prediction_vs_clean":common.mean_metrics([r["prediction_vs_clean"] for r in rows]),"mean_condition_vs_clean":common.mean_metrics([r["condition_vs_clean"] for r in rows]),"not_implemented":["hf_edge_f1","flow_warp_residual","roi_metrics"]}
 atomic(out/"metrics.json",result);atomic(out/"P51A_VALIDATION_COMPLETE.json",{"status":"complete","metrics":str(out/"metrics.json")});print(json.dumps(result["mean_prediction_vs_clean"],indent=2))
def main():
 a=args_();provenance=common.check_checkpoint(Path(a.checkpoint));finalize(a,provenance) if a.finalize_only else run(a,provenance)
if __name__=="__main__":main()
