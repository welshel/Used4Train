# P49A Data Contract

Condition is P48.3 lossless PNGs at outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic.
Target is exact Clean RGB PNGs at outputs/p48_tripo_clean_aligned/clean_rgb.
Both sets contain F00 through F71 RGB 896x896 PNGs.
Training reads ordered PNG lists directly and resizes in the training operator to 672x672.
Each 33-frame training clip is cyclic and pairs condition Fxx only with target Fxx.
Validation starts are fixed at 0, 17, 31, and 50.
No MP4 is decoded for training.
