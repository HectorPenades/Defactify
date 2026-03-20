# Experiment Comparison Table

_Last updated: 2026-03-20 09:24_

| Date | Experiment | Task | Mode | Arch | Accuracy | F1-Macro | F1-Weighted | Precision | Recall | Duration(s) |
|------|------------|------|------|------|----------|----------|-------------|-----------|--------|-------------|
| 2026-03-19 16:34 | 01_rgb_baseline_binary | binary | rgb | rgb_resnet50 | 0.8331 | 0.6990 | 0.8329 | 0.8328 | 0.8331 | 9190 |
| 2026-03-19 23:59 | 02_rgb_baseline_multiclass | multiclass | rgb | rgb_resnet50 | 0.4712 | 0.4708 | 0.4708 | 0.4729 | 0.4712 | 18084 |
| 2026-03-20 00:12 | 07_clip_multiclass | multiclass | clip-vit-b32 | vlm_mlp | 0.4356 | 0.4326 | 0.4326 | 0.4386 | 0.4356 | 308 |
| 2026-03-20 00:39 | 08_dinov2_multiclass | multiclass | dinov2-l | vlm_mlp | 0.4240 | 0.4235 | 0.4235 | 0.4255 | 0.4240 | 940 |
| 2026-03-20 09:24 | 11_rgb_augmented_multiclass | multiclass | rgb | rgb_resnet50 | 0.4428 | 0.4419 | 0.4419 | 0.4470 | 0.4428 | 30398 |
