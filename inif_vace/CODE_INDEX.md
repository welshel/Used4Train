# Code index

| Stage | Organized entry | Provenance / role |
|---|---|---|
| Topdown inference | `01_infinisplat/infer_topdown.py` | Official `InfiniSplat/src/demo/hf_runtime.py` API; explicit topdown intrinsics override |
| Camera transfer | `01_infinisplat/build_condition_cameras.py` | P32 `condition_camera_payload.json` contract; source-relative C2W and audited lambda |
| 72-frame render | `01_infinisplat/render_condition_72.py` | Official Gaussian artifact + `gsplat.rasterization`; no clean target reads |
| Pair manifest | `01_infinisplat/make_pair_manifest.py` | 72 condition/clean RGB pairs, same stems and exact camera flag |
| Condition-only manifest | `01_infinisplat/make_condition_only_manifest.py` | Strict schema consumed by P39 full72 |
| Pair QA | `01_infinisplat/validate_condition_pair.py` | Fail-closed 72-frame, 896², path and camera audit |
| 33-frame dataset | `02_pair_dataset/p38_dataset_builder.py` | Original P39/P38 builder; LANCZOS, cyclic 33f, 68/4 split |
| Dataset contract | `02_pair_dataset/p38_contract.py` | 4n+1 temporal, square resolution, no auxiliary clean fields |
| Trainer | `03_training/p38_train.py` | Original exact-step Wan VACE LoRA trainer |
| Training commands | `03_training/train_*.sh` | Reproducible P39 H672/H704 and smoke settings; overwrite guard |
| Validation | `04_validation/p38_validate.py` | Original four-start validation implementation |
| Validation commands | `04_validation/validate_*.sh` | Fixed starts 0,17,31,50 and seed 1337 |
| Metrics | `05_metrics/p38_evaluate.py`, `p38_compare.py`, `p39_compare.py` | Existing PSNR/SSIM/LPIPS/temporal and comparison code |
| Full72 | `05_metrics/p39_full72.py` | direct73/overlap33 generation; clean-free condition-only input |
| Full72 QA | `05_metrics/p39_full72_qa.py` | 72-frame video contract, transitions, loop and visual boards |
| Tests | `tests/test_*.py` | 40 original P38/P39 tests |
| Audit | `audit_pipeline.py` | Organized package read-only audit |

The `legacy/` folders preserve the exact scripts used in the original P39 output. `01_infinisplat/legacy_reference/` contains P20/P23/P24 historical render code and is explicitly not the exact P32 producer.
