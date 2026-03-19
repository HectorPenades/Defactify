# Plan de Desarrollo - Defactify Image Detection Framework

**Estado**: Fase 1 y 2 completadas, Fase 3-5 pendientes

---

## Fase 1: Infraestructura Base ✅ COMPLETADO

### Componentes Implementados:

#### Data Layer
- [x] `src/data/base.py` - BaseDataset (abstract)
- [x] `src/data/dataset.py` - DefactifyDataset (RGB mode)
- [x] `src/data/transforms.py` - ImagePreprocessor (resize + augmentations)
  - Aspect ratio + padding a 224x224
  - Normalización ImageNet
  - Augmentations: flip, color jitter, blur

#### Models
- [x] `src/models/base.py` - BaseModel (abstract)
- [x] `src/models/rgb_resnet.py` - ResNet50 para RGB
  - Pretrained en ImageNet
  - Dropout configurable
  - Heads para binaria (2 clases) y multiclase (6 clases)

#### Training
- [x] `src/training/base.py` - BaseTrainer (abstract)
- [x] `src/training/trainer.py` - DefaultTrainer
  - Training loop con early stopping
  - Validation cada epoch
  - Learning rate scheduling (ReduceLROnPlateau)
  - Checkpoint management

- [x] `src/training/losses.py` - Loss functions
  - CrossEntropyLoss
  - WeightedCrossEntropyLoss

- [x] `src/training/metrics.py` - MetricsComputer
  - Accuracy, F1 (macro/weighted)
  - Precision, Recall
  - ROC-AUC para binaria
  - Per-class metrics para multiclase

#### Inference & Evaluation
- [x] `src/inference/evaluation.py` - Evaluator
  - Predict on batches
  - Full split evaluation
  - Save predictions to CSV

#### Utils
- [x] `src/utils/logging.py` - ExperimentLogger
  - JSON metrics logging
  - CSV predictions export
  - **Weights & Biases (W&B) integration**
  - Console + file logging

- [x] `src/utils/reproducibility.py` - ReproducibilityManager
  - Seed management (numpy, torch, random, CUDA)
  - Deterministic CUDA mode
  - Data leakage verification

#### Configuration
- [x] `src/config.py` - Config loader
  - YAML parsing
  - Config inheritance support

- [x] `configs/base_config.yaml` - Template base
- [x] `configs/experiments/01_rgb_baseline_binary.yaml` - Exp 01
- [x] `configs/experiments/02_rgb_baseline_multiclass.yaml` - Exp 02

#### Orchestration
- [x] `scripts/run_experiments.py` - ExperimentRunner
  - Dataset loading
  - Model building
  - Training loop execution
  - Evaluation on test set
  - Results persisting

#### Documentation & Setup
- [x] `environment.yml` - Conda environment spec
- [x] `requirements.txt` - Pip dependencies
- [x] `QUICKSTART.md` - Setup guide
- [x] `TESTING.md` - Execution guide
- [x] `README.md` - Project overview
- [x] `setup.py` - Package installer
- [x] `validate_framework.py` - Validation script

### Experimentos Listos (Fase 1)

| ID | Nombre | Task | Status |
|----|--------|------|--------|
| 01 | RGB Baseline Binary | Binary (real/AI) | ✅ Ready |
| 02 | RGB Baseline Multiclass | Multiclass (6 gen) | ✅ Ready |

### Cómo Ejecutar Fase 1

```bash
# Setup única vez
conda env create -f environment.yml -n defactify
conda activate defactify
pip install -e .

# Validar
python validate_framework.py

# Ejecutar experimentos (train + val cada epoch + test al final)
python scripts/run_experiments.py --config configs/experiments/01_rgb_baseline_binary.yaml
python scripts/run_experiments.py --config configs/experiments/02_rgb_baseline_multiclass.yaml

# Re-evaluar test con pesos guardados (sin reentrenar)
python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml
python scripts/run_test.py --config configs/experiments/02_rgb_baseline_multiclass.yaml

# Ver tabla comparativa de todos los experimentos
cat results/comparison_table.md
```

---

## Fase 2: FFT y Late Fusion ✅ COMPLETADO

### Componentes Implementados

#### Processing
- [x] `src/processing/fft.py`
  - `compute_fft_grayscale(image)` → log magnitude (H,W), normalizado [0,1]
  - `compute_fft_perchannel(image)` → log magnitude por canal (H,W,3), normalizado [0,1]
  - Usa numpy.fft (sin scipy)

#### Models
- [x] `src/models/fft_resnet.py` — FFTResNet50
  - ResNet50 con Conv1 adaptado a 1 canal (pesos averaged de RGB pretrained)
  - Soporta 1 canal (grayscale) o 3 canales (perchannel)
  - Método `get_features()` para extraer 2048-d features

- [x] `src/models/fusion.py` — LateFusionModel
  - RGB branch: ResNet50 → 2048-d features
  - FFT branch: ResNet50 (1-ch o 3-ch) → 2048-d features
  - MLP head: 4096 → 512 → num_classes
  - `forward(x_rgb, x_fft)` — recibe dos tensores

#### Dataset Updates
- [x] `dataset.py` actualizado con 4 modos:
  - `rgb` → batch['image'] (3,H,W) — sin cambios
  - `fft_grayscale` → batch['image'] (1,H,W)
  - `fft_perchannel` → batch['image'] (3,H,W)
  - `fusion_late` → batch['image'] (3,H,W) + batch['fft'] (C,H,W)
- [x] FFT calculado on-demand y cacheado en `data/cache/fft/`

#### Utils
- [x] `src/utils/cache_manager.py` — FFTCacheManager
  - Save/load .npy arrays por imagen
  - Layout: `<cache_dir>/<mode>/<split>/<image_id>.npy`

- [x] `src/utils/results_tracker.py` — ResultsTracker
  - Tabla acumulativa CSV: `results/comparison_table.csv`
  - Tabla Markdown: `results/comparison_table.md`
  - Actualizada automáticamente al acabar cada experimento
  - También actualizable con `scripts/run_test.py`

#### Training & Evaluation Updates
- [x] `trainer.py`: `_model_forward(batch)` helper — pasa `fft` si existe en el batch
- [x] `evaluation.py`: `predict_batch` — soporta modelos de fusión

#### Scripts
- [x] `scripts/run_experiments.py` — añadidos FFTResNet50, LateFusionModel, results_tracker
- [x] `scripts/run_test.py` — script standalone para re-evaluar con pesos guardados

#### Configs
- [x] `03_fft_grayscale_multiclass.yaml`
- [x] `04_fft_perchannel_multiclass.yaml`
- [x] `05_late_fusion_grayscale.yaml`
- [x] `06_late_fusion_perchannel.yaml`

### Experimentos Fase 2

| ID | Nombre | Mode | Status |
|----|--------|------|--------|
| 03 | FFT Grayscale | fft_grayscale | ✅ Ready |
| 04 | FFT PerChannel | fft_perchannel | ✅ Ready |
| 05 | Late Fusion Grayscale | fusion_late | ✅ Ready |
| 06 | Late Fusion PerChannel | fusion_late | ✅ Ready |

### Cómo Ejecutar Fase 2

```bash
# Experimentos individuales (train + val cada epoch + test al final)
python scripts/run_experiments.py --config configs/experiments/03_fft_grayscale_multiclass.yaml
python scripts/run_experiments.py --config configs/experiments/04_fft_perchannel_multiclass.yaml
python scripts/run_experiments.py --config configs/experiments/05_late_fusion_grayscale.yaml
python scripts/run_experiments.py --config configs/experiments/06_late_fusion_perchannel.yaml

# Re-evaluar con pesos guardados (sin reentrenar)
python scripts/run_test.py --config configs/experiments/03_fft_grayscale_multiclass.yaml
# Con checkpoint personalizado:
python scripts/run_test.py --config configs/experiments/03_fft_grayscale_multiclass.yaml \
    --checkpoint results/experiments/03_fft_grayscale_multiclass/checkpoint_best.pth
# Solo métricas, sin actualizar tabla comparativa:
python scripts/run_test.py --config configs/experiments/03_fft_grayscale_multiclass.yaml --no-record

# Ver tabla comparativa de todos los experimentos
cat results/comparison_table.md
```

### Nota sobre FFT cache

La primera ejecución calcula los FFTs y los guarda en `data/cache/fft/`.
Las ejecuciones posteriores los cargan directamente (mucho más rápido).
Para forzar recálculo: borrar `data/cache/fft/`.

---

## Fase 3: VLM Embeddings ⏳ PENDIENTE

### Componentes a Implementar

#### Processing
- [ ] `src/processing/embeddings.py`
  - CLIP extractor (OpenAI clip-vit-b32)
  - DINOv2 extractor (Meta dinov2-l)
  - SigLIP extractor (Google siglip-so400m)
  - Frozen embeddings (no gradients)
  - Embeddings cache

#### Models
- [ ] `src/models/vlm.py`
  - VLMClassifier
  - MLP decoder head
  - Logistic Regression option

- [ ] `src/models/ensemble.py` (opcional)
  - EnsembleVLMModel
  - Promediado de embeddings CLIP+DINOv2+SigLIP

#### Dataset Updates
- [ ] Actualizar `dataset.py`
  - Modo "vlm"
  - Cargar embeddings pre-computados
  - Cache de embeddings

#### Configs
- [ ] `07_clip_embeddings.yaml`
- [ ] `08_dinov2_embeddings.yaml`
- [ ] `09_siglip_embeddings.yaml`
- [ ] `10_vlm_ensemble.yaml`

### Experimentos Fase 3

| ID | Nombre | VLM | Status |
|----|--------|-----|--------|
| 07 | CLIP Embeddings | clip-vit-b32 | ⏳ Pending |
| 08 | DINOv2 Embeddings | dinov2-l | ⏳ Pending |
| 09 | SigLIP Embeddings | siglip-so400m | ⏳ Pending |
| 10 | VLM Ensemble | CLIP+DINOv2+SigLIP | ⏳ Pending |

### Orden de Implementación Fase 3

1. **Implementar `src/processing/embeddings.py`**
   - 3 extractores (no training)
   - Caching logic

2. **Actualizar `dataset.py`**
   - Modo VLM
   - Lazy load embeddings

3. **Implementar `src/models/vlm.py`**
   - VLMClassifier simple
   - MLP + LogReg options

4. **Crear configs**
   - 4 archivos YAML

5. **Test Fase 3**
   - Correr 4 experimentos
   - Rápidos (solo training MLP)

---

## Fase 4: Tests y Validación ⏳ PENDIENTE

### Tests a Implementar

- [ ] `tests/test_dataset.py`
  - Test dataset length
  - Test getitem shapes
  - Test label distribution
  - Test no NaN/Inf

- [ ] `tests/test_models.py`
  - Test model forward pass
  - Test output shapes
  - Test parameter count

- [ ] `tests/test_cache.py`
  - Test FFT cache save/load
  - Test embeddings cache
  - Test metadata integrity

- [ ] `tests/test_reproducibility.py`
  - Test same seed = same results
  - Test no data leakage
  - Test determinism

### Validación E2E

- [ ] `scripts/validate_reproducibility.py`
  - Run 2 experiments with same seed
  - Compare loss values (deben ser idénticos)

---

## Fase 5: Variaciones de Tamaño ⏳ PENDIENTE

### Configs a Crear

- [ ] `11_rgb_256_multiclass.yaml` - ResNet50 @ 256x256
- [ ] `12_rgb_512_multiclass.yaml` - ResNet50 @ 512x512

### Experimentos

| ID | Nombre | Size | Status |
|----|--------|------|--------|
| 11 | RGB 256x256 | 256x256 | ⏳ Pending |
| 12 | RGB 512x512 | 512x512 | ⏳ Pending |

### Objetivo

- Comparar accuracy vs input size
- Determinar optimal input resolution
- Trade-off speed vs accuracy

---

## Weights & Biases Integration ✅ IMPLEMENTADO

### Features:
- [x] Auto-logging de época -> W&B
- [x] Config guardado en W&B
- [x] Métricas finales loguadas
- [x] Project: "real-vs-synthetic"
- [x] Graceful fallback si W&B no disponible

### Usar W&B:

```bash
# Setup (una sola vez)
wandb login
# Copiar API key desde https://wandb.ai

# Ejecutar experiments (se logguean automáticamente)
python scripts/run_experiments.py --config configs/experiments/02_rgb_baseline_multiclass.yaml

# Ver resultados en https://wandb.ai/your-username/real-vs-synthetic
```

---

## Estructura de Resultados

```
results/experiments/
├── 01_rgb_baseline_binary/
│   ├── config.json
│   ├── metrics.json              # {train.loss, train.acc, val.loss, ...} per epoch
│   ├── final_metrics.json        # {test.accuracy, test.f1_macro, ...}
│   ├── predictions.csv           # image_id, true_label, pred_label, prob_*
│   ├── checkpoint_best.pth       # Model weights
│   └── experiment.log            # Timestamped logs
├── 02_rgb_baseline_multiclass/
│   └── [similar structure]
├── ...
└── [future experiments]
```

---

## File Dependencies

```
scripts/run_experiments.py
├── src/config.py (load_config)
├── src/data/dataset.py (DefactifyDataset)
├── src/data/transforms.py (ImagePreprocessor)
├── src/models/rgb_resnet.py (RGBResNet50)
│   └── src/models/base.py
├── src/training/trainer.py (DefaultTrainer)
│   ├── src/training/losses.py
│   ├── src/training/metrics.py
│   └── src/utils/logging.py (ExperimentLogger with W&B)
├── src/inference/evaluation.py (Evaluator)
└── src/utils/reproducibility.py (ReproducibilityManager)
```

---

## Roadmap Timeline (Estimado)

- **Fase 1**: ✅ Completado
- **Fase 2**: ✅ Completado (FFT + fusion + results tracker + run_test.py)
- **Fase 3**: 1-2 días (VLM + embeddings)
- **Fase 4**: 1 día (tests exhaustivos)
- **Fase 5**: 1 día (tamaño experiments)

---

## Métricas a Reportar (Final)

| Exp | Type | Accuracy | F1-Macro | F1-Weighted | Time (min) | Notes |
|-----|------|----------|----------|-------------|-----------|-------|
| 01 | Binary RGB | TBD | TBD | TBD | ~30 | Baseline |
| 02 | Multi RGB | TBD | TBD | TBD | ~60 | Baseline |
| 03 | Multi FFT-GS | TBD | TBD | TBD | ~60 | Grayscale FFT |
| 04 | Multi FFT-PC | TBD | TBD | TBD | ~60 | Per-channel FFT |
| 05 | Multi Fusion-GS | TBD | TBD | TBD | ~90 | Late fusion |
| 06 | Multi Fusion-PC | TBD | TBD | TBD | ~90 | Late fusion |
| 07 | Multi CLIP | TBD | TBD | TBD | ~10 | VLM CLIP |
| 08 | Multi DINOv2 | TBD | TBD | TBD | ~10 | VLM DINOv2 |
| 09 | Multi SigLIP | TBD | TBD | TBD | ~10 | VLM SigLIP |
| 10 | Multi Ensemble | TBD | TBD | TBD | ~10 | VLM Ensemble |
| 11 | Multi RGB-256 | TBD | TBD | TBD | ~60 | 256x256 |
| 12 | Multi RGB-512 | TBD | TBD | TBD | ~150 | 512x512 |

---

## Regla de Documentación

> **La documentación debe mantenerse actualizada en cada PR.**

Al completar cualquier cambio significativo (nueva fase, nuevo script, cambio de comportamiento):

1. **`README.md`** — actualizar lista de experimentos, comandos y estructura si cambia
2. **`QUICKSTART.md`** — actualizar comandos si cambian flags, scripts o flujo
3. **`TESTING.md`** — actualizar parámetros, salida esperada y checklist del experimento nuevo
4. **`PLAN.md`** — marcar como ✅ lo completado, añadir próximos pasos

**Qué NO necesita actualización de doc**:
- Bugfixes internos que no cambian el comportamiento externo
- Refactors sin cambio de interfaz
- Ajustes de hiperparámetros en YAML

---

## Estado de Este Plan

**Última actualización**: Marzo 2026

**Completado por**: Fase 2

**Próximo paso**: Fase 3 (VLM Embeddings)

**Revisión**: Actualizar tras completar cada fase
