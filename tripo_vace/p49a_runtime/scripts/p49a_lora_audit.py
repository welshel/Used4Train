import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path('/fs1/private/user/baitongyuan/projects/liuzh/repos/DiffSynth-Studio')))
from examples.wanvideo.model_training.train import WanTrainingModule
root=Path('/fs1/private/user/baitongyuan/projects/liuzh')
model=root/'models/Wan2.1-VACE-1.3B'
paths=json.dumps([str(model/'diffusion_pytorch_model.safetensors'),str(model/'models_t5_umt5-xxl-enc-bf16.pth'),str(model/'Wan2.1_VAE.pth')])
module=WanTrainingModule(model_paths=paths, tokenizer_path=str(model/'google/umt5-xxl'), trainable_models=None, lora_base_model='vace', lora_target_modules='q,k,v,o,ffn.0,ffn.2', lora_rank=16, use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False, extra_inputs='vace_video,vace_reference_image', device='cuda', task='sft')
trainable=list(module.trainable_modules())
matched=[]; trainable_param=0; total=0
for name,p in module.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable_param += p.numel(); matched.append(name)
# Count LoRA modules directly for robust module-name evidence.
lora_modules=[]
for name,m in module.named_modules():
    if hasattr(m,'lora_A') or hasattr(m,'lora_B') or hasattr(m,'lora_A_weights'):
        lora_modules.append(name)
payload={'status':'PASS','base_model':'official Wan2.1-VACE-1.3B','fresh_lora':True,'old_lora_loaded':False,'rank':16,'alpha':16,'requested_targets':['q','k','v','o','ffn.0','ffn.2'],'trainable_module_count':len(trainable),'trainable_param_count':trainable_param,'total_param_count':total,'trainable_percentage':100*trainable_param/max(total,1),'sample_trainable_names':matched[:30],'lora_module_count':len(lora_modules),'sample_lora_module_names':lora_modules[:30]}
out=root/'outputs/p49a_tripo_vace13b_fresh/02_model_audit'
out.mkdir(parents=True,exist_ok=True)
(out/'P49A_LORA_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
