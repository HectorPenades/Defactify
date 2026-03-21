# Defactify — AI Image Detection Framework

Framework para entrenar y comparar múltiples arquitecturas en detección de imágenes generadas por IA, basado en el dataset [Rajarshi-Roy-research/Defactify_Image_Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset).

---

## Estructura del Proyecto

```
Defactify/
├── configs/
│   ├── base_config.yaml           # Parámetros por defecto
│   └── experiments/               # Un YAML por experimento
├── src/
│   ├── config.py                  # Cargador YAML con herencia
│   ├── data/                      # Dataset, transforms, splits
│   ├── models/                    # rgb_resnet, fft_resnet, fusion
│   ├── processing/                # FFT computation
│   ├── training/                  # Trainer, losses, metrics
│   ├── inference/                 # Evaluator
│   └── utils/                     # Logging, cache, results tracker
├── scripts/
│   ├── run_experiments.py         # Lanzar experimento ResNet/fusion
│   ├── run_test.py                # Re-evaluar con checkpoint guardado
│   ├── compute_embeddings.py      # Pre-computar embeddings VLM
│   └── run_vlm_experiments.py     # Lanzar experimento VLM + classifier sweep
├── results/
│   ├── comparison_table.csv       # Tabla comparativa acumulativa
│   ├── comparison_table.md        # Tabla comparativa en Markdown
│   └── experiments/
│       └── <exp_name>/            # Resultados por experimento
└── data/
    └── cache/
        ├── fft/                   # FFT arrays (regenerable, no en git)
        └── embeddings/            # VLM embeddings (regenerable, no en git)
```

---

## Dataset

**`Rajarshi-Roy-research/Defactify_Image_Dataset`** — descargado automáticamente desde HuggingFace.

| Split      | Filas  | Notas |
|------------|--------|-------|
| train      | 42 000 | Entrenamiento |
| validation |  9 000 | Hyperparameter tuning / early stopping |
| test       | 45 000 | Evaluación final (no se toca durante entrenamiento) |

**Columnas**: `Caption` (str), `Image` (PIL), `Label_A` (int), `Label_B` (int)

| Label | Task A (binaria) | Task B (multiclase) |
|-------|-----------------|---------------------|
| 0 | Real | Real |
| 1 | AI-Generated | Stable Diffusion 2.1 |
| 2 | — | Stable Diffusion XL |
| 3 | — | Stable Diffusion 3 |
| 4 | — | DALL-E 3 |
| 5 | — | Midjourney |

**Imbalance**: Task A tiene ratio **1:5** (real:AI). El framework aplica `weighted_ce` automáticamente.
**Task B**: perfectamente balanceada (7 000 muestras por clase).

---

## Instalación

```bash
# Crear entorno conda
conda env create -f environment.yml -n defactify
conda activate defactify
pip install -e .

# Verificar instalación
python validate_framework.py
```

---

## Uso Rápido

```bash
# Entrenar (incluye validación por época + test al finalizar)
python scripts/run_experiments.py --config configs/experiments/01_rgb_baseline_binary.yaml

# Reanudar entrenamiento interrumpido
python scripts/run_experiments.py --config configs/experiments/01_rgb_baseline_binary.yaml --resume

# Re-evaluar en test con el mejor checkpoint (sin reentrenar)
python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml

# Ver tabla comparativa de todos los experimentos
cat results/comparison_table.md
```

---

## Resultados

> Leaderboard de referencia — Task A: **0.8334** · Task B: **0.4986**

### Task A — Binaria (real vs AI-generated)

#### Backbone preentrenado (ImageNet)

| # | Experimento | Modelo | Pretrained | Accuracy | F1-macro | Estado |
|---|------------|--------|------------|----------|----------|--------|
| 01 | RGB Baseline | ResNet50 | yes | 0.8331 | 0.6990 | ✅ |
| 12 | Late Fusion Grayscale | ResNet50 + FFT | yes | — | — | ⏳ |
| 13 | Late Fusion Perchannel | ResNet50 + FFT | yes | — | — | ⏳ |

#### Sin preentrenamiento

| # | Experimento | Modelo | Pretrained | Accuracy | F1-macro | Estado |
|---|------------|--------|------------|----------|----------|--------|
| 01b | RGB Baseline | ResNet50 | no | 0.8319 | 0.6976 | ✅ |

---

### Task B — Multiclase (6 generadores)

#### Backbone preentrenado (ImageNet)

| # | Experimento | Modelo | Pretrained | Accuracy | F1-macro | Estado |
|---|------------|--------|------------|----------|----------|--------|
| 02 | RGB Baseline | ResNet50 | yes | 0.4712 | 0.4708 | ✅ |
| 11 | RGB + Aug. agresivas | ResNet50 | yes | 0.4428 | 0.4419 | ✅ |
| 03 | FFT Grayscale | FFTResNet50 | yes | — | — | ⏳ |
| 04 | FFT Perchannel | FFTResNet50 | yes | 0.4138 | 0.4137 | ✅ |
| 05 | Late Fusion Grayscale | ResNet50 + FFT | yes | — | — | ⏳ |
| 06 | Late Fusion Perchannel | ResNet50 + FFT | yes | — | — | ⏳ |

#### VLM features congelados (frozen backbone)

| # | Experimento | Backbone | Embed dim | F1-macro | Mejor clf | Estado |
|---|------------|---------|-----------|----------|-----------|--------|
| 07 | CLIP | ViT-B/32 | 512 | 0.4326 | MLP | ✅ |
| 08 | DINOv2 | Large | 1024 | 0.4235 | MLP | ✅ |
| 09 | SigLIP | SO400M | 1152 | — | — | ⏳ |
| 10 | VLM Ensemble | CLIP+DINOv2+SigLIP | 2688 | — | — | ⏳ |

---

## Experimentos Disponibles

### Fase 1 — RGB Baseline

```bash
python scripts/run_experiments.py --config configs/experiments/01_rgb_baseline_binary.yaml
python scripts/run_experiments.py --config configs/experiments/02_rgb_baseline_multiclass.yaml
```

### Fase 2 — FFT y Late Fusion

```bash
python scripts/run_experiments.py --config configs/experiments/03_fft_grayscale_multiclass.yaml
python scripts/run_experiments.py --config configs/experiments/04_fft_perchannel_multiclass.yaml
python scripts/run_experiments.py --config configs/experiments/05_late_fusion_grayscale.yaml
python scripts/run_experiments.py --config configs/experiments/06_late_fusion_perchannel.yaml
# Versiones binarias:
python scripts/run_experiments.py --config configs/experiments/12_late_fusion_grayscale_binary.yaml
python scripts/run_experiments.py --config configs/experiments/13_late_fusion_perchannel_binary.yaml
```

### Fase 3 — VLM Embeddings

```bash
# Paso 1: pre-computar embeddings (una sola vez por modelo)
python scripts/compute_embeddings.py --model all --splits all

# Paso 2: ejecutar experimentos
python scripts/run_vlm_experiments.py --config configs/experiments/07_clip_multiclass.yaml
python scripts/run_vlm_experiments.py --config configs/experiments/08_dinov2_multiclass.yaml
python scripts/run_vlm_experiments.py --config configs/experiments/09_siglip_multiclass.yaml
python scripts/run_vlm_experiments.py --config configs/experiments/10_vlm_ensemble_multiclass.yaml
```

Cada experimento VLM prueba automáticamente: LR, RandomForest, XGBoost, LinearSVM, MLP.

---

## Métricas

Dado el desbalance de Task A, las métricas principales son:

| Métrica | Por qué usarla |
|---------|----------------|
| `f1_macro` | Trata todas las clases igual — **métrica principal** |
| `balanced_accuracy` | Media del recall por clase |
| `roc_auc` | Solo binaria — independiente del threshold |
| `accuracy` | Referencia, pero engañosa con desbalance |

El log por época muestra:
```
Epoch 5/50 | Train Loss: 0.1234 | Val Loss: 0.1456 | Val Acc: 0.9712 | Val BalAcc: 0.9501 | Val F1-macro: 0.9489
```

---

## Resultados por Experimento

```
results/experiments/01_rgb_baseline_binary/
├── config.json           # Config exacta usada
├── metrics.json          # Métricas train/val por época
├── final_metrics.json    # Métricas test + best_epoch + total_epochs_trained + per_class
├── predictions.csv       # Predicción por imagen
├── checkpoint_best.pth   # Mejor modelo (no en git)
├── checkpoint_last.pth   # Último checkpoint para resume (no en git)
└── experiment.log        # Log timestamped
```

`final_metrics.json` incluye:
- Métricas globales: `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `roc_auc`
- `per_class`: precision / recall / f1 / support por clase
- `best_epoch`: época del mejor checkpoint (contando desde 1)
- `total_epochs_trained`: épocas totales antes de parar

---

## Tabla Comparativa

Al finalizar cada experimento se actualiza automáticamente:
- `results/comparison_table.csv` — para análisis en pandas/Excel
- `results/comparison_table.md` — para visualizar en GitHub

```bash
cat results/comparison_table.md
```

---

## Reproducibilidad

- Seed global configurable (`seed: 42` por defecto)
- CUDA deterministic mode habilitado
- Config completa guardada en cada resultado

---

## Próximos Pasos

- ✅ Fase 1: RGB baseline
- ✅ Fase 2: FFT y late fusion
- ✅ Fase 3: VLM embeddings (CLIP, DINOv2, SigLIP + ensemble)
- ⏳ Fase 4: Tests exhaustivos
- ⏳ Fase 5: Variaciones de resolución (256×256, 512×512)

Ver `PLAN.md` para el roadmap completo.
