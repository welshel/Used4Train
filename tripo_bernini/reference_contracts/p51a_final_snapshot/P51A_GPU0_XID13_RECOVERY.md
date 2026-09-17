# P51A GPU0 Xid 13 recovery

- Original fault: GPU0 (0000:8e:00; GPU-d779afbc-7f7d-b7d0-07c1-a1b79a1078bb) recorded Xid 13 (Out Of Range Address / Multiple Warp Errors) during the former four-GPU run.
- Hardware reset was unavailable without administrator privileges.
- Corrective change: persistent training now exports MODELING_BACKEND=veomni; official VeOmni resolves FA2 to veomni_flash_attention_2_with_sp for ulysses_size=4.
- Verification: an isolated four-GPU official recovery smoke loaded only the complete formal global_step_500 DCP, executed one real optimization step (loss 0.0461; grad norm 0.5559; 46.19 s), and atomically published isolated global_step_501.
- Smoke log contains the official replacement, and contains neither the prior missing-SP warning nor CUDA illegal-memory-access. No new Xid was recorded during the smoke.
- Integrity: formal training resumes from the original complete global_step_500; the isolated smoke checkpoint is never merged into or used to resume the formal run. Clean remains target/QA only.
- Durability: one continuous supervisor runs all remaining 20-step units and validations without intentional half-hour pauses. A scheduler kill releases its flock; cron resumes only from verified DCP checkpoints and atomic validation-shard markers.
- Fail-closed durability: the cron guard also refuses to relaunch whenever the atomic orchestrator status has state fail_closed; a restart then requires an explicit reviewed recovery.
