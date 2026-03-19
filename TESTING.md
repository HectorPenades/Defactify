# Defactify — Testing & Execution Guide

Referencia completa de experimentos, métricas esperadas y validación.

---

## Estado de Implementación

### ✅ Fase 1 — RGB Baseline
- ResNet50 pretrained en ImageNet
- Experimentos 01 (binario) y 02 (multiclase)
- Training loop con early stopping y checkpoint

### ✅ Fase 2 — FFT y Late Fusion
- FFT grayscale y per-channel con cache en disco
- FFTResNet50 (Conv1 adaptado a 1 canal)
- LateFusionModel (RGB + FFT → MLP)
- Experimentos 03, 04, 05, 06

### ⏳ Fase 3 — VLM Embeddings
- CLIP, DINOv2, SigLIP

### ⏳ Fase 4 — Tests automatizados

### ⏳ Fase 5 — Variaciones de resolución

---

## Experimentos — Fase 1

### 01 · RGB Baseline Binary

**Task**: Detectar si una imagen es real o generada por IA (Label_A).

**Config**: `configs/experiments/01_rgb_baseline_binary.yaml`

```bash
python scripts/run_experiments.py --config configs/experiments/01_rgb_baseline_binary.yaml
```

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | ResNet50 pretrained |
| num_classes | 2 (real=0, ai=1) |
| Loss | weighted CE — pesos [5.0, 1.0] por desbalance 1:5 |
| Epochs | 50 (early stopping patience=10) |
| Batch size | 32 |
| LR | 0.001 |

**Salida en consola**:
```
Epoch 1/50 | Train Loss: 0.0850 | Val Loss: 0.1100 | Val Acc: 0.9726 | Val BalAcc: 0.9501 | Val F1-macro: 0.9489
```

**Métricas a observar** (por desbalance 1:5):
- `Val BalAcc` y `Val F1-macro` son las principales — no `Val Acc`
- En `final_metrics.json`, revisar `per_class.real.recall` — si es bajo (<0.70), el modelo ignora los reales

**Archivos generados**:
```
results/experiments/01_rgb_baseline_binary/
├── config.json
├── metrics.json              # train/val por época
├── final_metrics.json        # test + best_epoch + per_class
├── predictions.csv
├── checkpoint_best.pth       # mejor val_loss
├── checkpoint_last.pth       # última época (para resume)
└── experiment.log
```

**Tiempos estimados**:
- RTX 3090: ~20-30 min
- RTX 2080: ~30-45 min

---

### 02 · RGB Baseline Multiclass

**Task**: Identificar qué generador AI produjo la imagen (Label_B, 6 clases).

**Config**: `configs/experiments/02_rgb_baseline_multiclass.yaml`

```bash
python scripts/run_experiments.py --config configs/experiments/02_rgb_baseline_multiclass.yaml
```

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | ResNet50 pretrained |
| num_classes | 6 |
| Loss | CE estándar (dataset balanceado: 7000 por clase) |
| Epochs | 100 (early stopping patience=10) |
| Batch size | 32 |
| LR | 0.001 |

**Clases**:
```
0: real        — imagen COCO original
1: sd21        — Stable Diffusion 2.1
2: sdxl        — Stable Diffusion XL
3: sd3         — Stable Diffusion 3
4: dalle3      — DALL-E 3
5: midjourney  — Midjourney
```

**Tiempos estimados**:
- RTX 3090: ~40-60 min

---

## Experimentos — Fase 2

### 03 · FFT Grayscale Multiclass

**Task**: Multiclase sobre espectro FFT en escala de grises.

```bash
python scripts/run_experiments.py --config configs/experiments/03_fft_grayscale_multiclass.yaml
```

- Input: (1, 224, 224) — magnitud log-normalizada [0,1]
- La primera ejecución calcula y cachea los FFTs en `data/cache/fft/fft_grayscale/`
- Las siguientes ejecuciones cargan desde cache (mucho más rápido)

---

### 04 · FFT Per-Channel Multiclass

```bash
python scripts/run_experiments.py --config configs/experiments/04_fft_perchannel_multiclass.yaml
```

- Input: (3, 224, 224) — FFT independiente por canal R/G/B

---

### 05 · Late Fusion Grayscale

```bash
python scripts/run_experiments.py --config configs/experiments/05_late_fusion_grayscale.yaml
```

- Dos ramas ResNet50: una para RGB (3ch) y otra para FFT grayscale (1ch)
- Features concatenadas: 2048 + 2048 → MLP → 6 clases
- batch_size=16 (usa más memoria que modelos de una rama)

---

### 06 · Late Fusion Per-Channel

```bash
python scripts/run_experiments.py --config configs/experiments/06_late_fusion_perchannel.yaml
```

- Igual que 05 pero FFT branch usa 3 canales

---

## Flujo Completo de un Experimento

```
run_experiments.py
│
├── Carga train/val/test desde HuggingFace (splits nativos)
├── Train loop:
│   ├── Cada época → train_epoch() + validate()
│   ├── Guarda checkpoint_best.pth si mejora val_loss
│   ├── Guarda checkpoint_last.pth cada época (para resume)
│   └── Early stopping si patience se agota
│
├── Carga checkpoint_best.pth
├── Evaluación en test set completo
├── Guarda final_metrics.json + predictions.csv
└── Actualiza results/comparison_table.csv + .md
```

---

## Re-evaluar sin Reentrenar

```bash
# Default: usa checkpoint_best.pth del experimento
python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml

# Con checkpoint específico
python scripts/run_test.py \
  --config configs/experiments/01_rgb_baseline_binary.yaml \
  --checkpoint results/experiments/01_rgb_baseline_binary/checkpoint_best.pth

# Sin actualizar tabla comparativa
python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml --no-record
```

---

## Reanudar Entrenamiento

```bash
# Continuar desde donde se interrumpió
python scripts/run_experiments.py \
  --config configs/experiments/01_rgb_baseline_binary.yaml \
  --resume

# Ampliar épocas (ej: de 50 a 100):
# 1. Editar num_epochs: 100 en el YAML
# 2. Ejecutar con --resume
```

---

## Tabla Comparativa

Se actualiza automáticamente al finalizar cada experimento o `run_test.py`:

```bash
cat results/comparison_table.md
```

Columnas: `date`, `experiment`, `task`, `mode`, `architecture`, `accuracy`, `f1_macro`, `f1_weighted`, `precision`, `recall`, `duration_s`

---

## Checklist Antes de Ejecutar

```bash
# 1. GPU disponible
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# 2. Entorno activo
conda activate defactify

# 3. Imports correctos
python validate_framework.py

# 4. Config válida
python scripts/run_experiments.py --config configs/experiments/<exp>.yaml --dry-run

# 5. Dataset accesible
python -c "
from datasets import load_dataset
ds = load_dataset('Rajarshi-Roy-research/Defactify_Image_Dataset', split='train', keep_in_memory=False)
print('Train OK:', len(ds), 'rows | Cols:', ds.column_names)
"
```

---

## Interpretar `final_metrics.json`

```json
{
  "test": {
    "accuracy": 0.9726,
    "balanced_accuracy": 0.9501,
    "f1_macro": 0.9489,
    "f1_weighted": 0.9712,
    "roc_auc": 0.9923,
    "per_class": {
      "real":         {"precision": 0.91, "recall": 0.88, "f1": 0.89, "support": 7500},
      "ai_generated": {"precision": 0.98, "recall": 0.99, "f1": 0.98, "support": 37500}
    },
    "confusion_matrix": [[...]]
  },
  "best_epoch": 12,
  "total_epochs_trained": 32,
  "experiment_duration_seconds": 1823.4
}
```

**Qué mirar primero**:
- `f1_macro` y `balanced_accuracy` > `accuracy` (especialmente en binario)
- `per_class.real.recall` en binario — si es bajo, el modelo falla en detectar reales
- `best_epoch` vs `total_epochs_trained` — si best_epoch es muy temprano, puede haber overfitting

---

Ver `PLAN.md` para el roadmap completo.
