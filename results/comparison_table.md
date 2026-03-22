# Experiment Comparison Table

_Last updated: 2026-03-21 10:16_

| Date | Experiment | Task | Mode | Arch | Pretrained | Accuracy | F1-Macro | F1-Weighted | Precision | Recall | Duration(s) |
|------|------------|------|------|------|------------|----------|----------|-------------|-----------|--------|-------------|
| 2026-03-20 23:52 | 01_rgb_baseline_binary | binary | rgb | rgb_resnet50 | yes | 0.8331 | 0.6990 | 0.8329 | 0.8328 | 0.8331 | 29406 |
| 2026-03-20 23:51 | 01_rgb_baseline_binary_w_pretrainedF | binary | rgb | rgb_resnet50 | no | 0.8319 | 0.6976 | 0.8320 | 0.8320 | 0.8319 | 29406 |
| 2026-03-20 23:52 | 02_rgb_baseline_multiclass | multiclass | rgb | rgb_resnet50 | yes | 0.4712 | 0.4708 | 0.4708 | 0.4729 | 0.4712 | 18084 |
| 2026-03-20 23:52 | 04_fft_perchannel_multiclass | multiclass | fft_perchannel | fft_resnet50 | yes | 0.4138 | 0.4137 | 0.4137 | 0.4160 | 0.4138 | 16307 |
| 2026-03-20 23:52 | 07_clip_multiclass | multiclass | clip-vit-b32 | vlm_mlp | frozen | 0.4356 | 0.4326 | 0.4326 | 0.4386 | 0.4356 | 308 |
| 2026-03-20 23:52 | 08_dinov2_multiclass | multiclass | dinov2-l | vlm_mlp | frozen | 0.4240 | 0.4235 | 0.4235 | 0.4255 | 0.4240 | 940 |
| 2026-03-20 23:52 | 11_rgb_augmented_multiclass | multiclass | rgb | rgb_resnet50 | yes | 0.4428 | 0.4419 | 0.4419 | 0.4470 | 0.4428 | 30392 |
| 2026-03-21 10:16 | 13_late_fusion_perchannel_binary | binary | fusion_late | late_fusion | yes | 0.8312 | 0.6971 | 0.8315 | 0.8317 | 0.8312 | 36937 |
