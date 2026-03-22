# Plan de Automatización de Experimentos — Task A (Binary)

**Objetivo**: Encontrar la mejor configuración para Task A (real vs AI-generated)
de forma sistemática, documentando cada decisión para el artículo.

**Leaderboard referencia — Task A**: F1-macro = 0.8334

---

## Preguntas de Investigación (RQ)

| RQ | Pregunta | Variable | Fijado |
|----|----------|----------|--------|
| RQ1 | ¿Cómo tratar el desbalance 1:5? | loss + datos | arch=ResNet50, res=224, mode=RGB |
| RQ2 | ¿Qué resolución de entrada es óptima? | image_size | arch=ResNet50, loss=best_RQ1, mode=RGB |
| RQ3 | ¿CNN vs Transformer? | architecture | res=best_RQ2, loss=best_RQ1, mode=RGB |
| RQ4 | ¿Aporta información de frecuencia? | input_mode | arch=best_RQ3, res=best_RQ2, loss=best_RQ1 |

Cada RQ es una tabla independiente en el artículo.
Las fases son secuenciales: RQ2 depende de RQ1, RQ3 de RQ2, RQ4 de RQ3.

---

## RQ1 — Estrategia anti-desbalance

**Condición fijada**: ResNet50 pretrained, 224×224, RGB, datos sin modificar (salvo undersample).

| ID | Config | Loss | Datos | Estado | Notas |
|----|--------|------|-------|--------|-------|
| 01 | `01_rgb_baseline_binary` | CE (sin pesos) | full 1:5 | ✅ F1=0.6990 | baseline absoluto |
| 21 | `21_weighted_ce_binary` | weighted_CE(5,1) | full 1:5 | ❌ config falta | pesos estáticos |
| 19 | `19_focal_binary` | focal(γ=2) | full 1:5 | ⏳ listo | pesos dinámicos |
| 22 | `22_undersample_ce_binary` | CE | undersample 1:1 | ❌ config falta | dato balanceado |
| 15 | `15_binary_undersample` | weighted_CE(5,1) | undersample 1:1 | ⏳ listo | combinado |

**Resultado esperado**: tabla comparativa de F1-macro, balanced_accuracy y recall de clase real.
**Ganador** → `best_loss` (se usa en todas las fases siguientes).

**Nota importante**: exp 01b (ResNet50 scratch) usa weighted_CE pero exp 01 usa CE plain.
Para RQ3 se crearán comparaciones con la misma loss.

---

## RQ2 — Resolución de entrada

**Condición fijada**: ResNet50 pretrained, `best_loss` de RQ1, RGB.

| ID | Config | Resolución | Batch | Estado | Notas |
|----|--------|-----------|-------|--------|-------|
| — | (mejor de RQ1) | 224×224 | 32 | de RQ1 | reutilizar |
| 23 | `23_rgb_512_binary` | 512×512 | 8 | ❌ config falta | más detalle, más lento |

**Consideraciones 512×512**:
- Memoria GPU: ~5× más que 224. Batch_size=8 para GPU ≥ 8 GB.
- ResNet50 soporta cualquier resolución (adaptive pooling).
- ViT NO se estudia a 512 (diseñado para 224; requiere interpolación de pos. embeddings).

**Resultado esperado**: ¿Merece la pena el coste computacional del 512?

---

## RQ3 — Arquitectura

**Condición fijada**: `best_res` de RQ2, `best_loss` de RQ1, RGB.

| ID | Config | Arquitectura | Pretrained | Estado | Notas |
|----|--------|-------------|-----------|--------|-------|
| — | mejor RQ2 | ResNet50 | yes | de RQ2 | referencia |
| 24 | `24_resnet_scratch_binary` | ResNet50 | no | ❌ config falta | comparación justa con ViT scratch |
| 16 | `16_vit_binary` | ViT-B/16 | yes | ⏳ listo | weighted_CE — actualizar si best_loss ≠ weighted_CE |
| 18 | `18_vit_binary_scratch` | ViT-B/16 | no | ⏳ listo | idem |

**Nota**: exps 16 y 18 usan `weighted_CE`. Si `best_loss` de RQ1 resulta ser `focal`,
crear versiones ViT con focal antes de ejecutar RQ3.

**Resultado esperado**: ¿ViT supera a ResNet con los mismos datos de entrenamiento?

---

## RQ4 — Modalidad de entrada (frecuencia)

**Condición fijada**: `best_arch` de RQ3, `best_res` de RQ2, `best_loss` de RQ1.

| ID | Config | Modalidad | Estado | Notas |
|----|--------|----------|--------|-------|
| — | mejor RQ3 | RGB | de RQ3 | referencia |
| 25 | `25_fft_grayscale_binary` | FFT grayscale | ❌ config falta | solo espectro |
| 12 | `12_late_fusion_grayscale_binary` | RGB + FFT gray | ⏳ listo | fusion |
| 13 | `13_late_fusion_perchannel_binary` | RGB + FFT 3ch | ✅ F1=0.6971 | fusion (con weighted_CE) |

**Nota**: exp 13 ya está ejecutado pero con weighted_CE. Si best_loss cambia,
re-ejecutar con la config actualizada.

---

## Configs que faltan crear

| # | Nombre | Diferencia respecto a base |
|---|--------|---------------------------|
| 21 | `21_weighted_ce_binary` | = exp 01 + `loss: weighted_ce` |
| 22 | `22_undersample_ce_binary` | = exp 01 + `undersample: true` |
| 23 | `23_rgb_512_binary` | = best_RQ1 + `image_size: 512, batch_size: 8` |
| 24 | `24_resnet_scratch_binary` | = best_RQ1+2 + `pretrained: false` |
| 25 | `25_fft_grayscale_binary` | = exp 03 adaptado a task binary |

Nota: configs 23 y 24 se crean **después** de conocer el ganador de RQ1 y RQ2.
Config 25 se crea **después** de conocer el ganador de RQ3.

---

## Orden de ejecución

### Fase 1 — RQ1 (pueden ejecutarse en paralelo si hay múltiples GPUs)

```bash
# Lanzar en orden de prioridad (o en paralelo):
python scripts/run_experiments.py --config configs/experiments/21_weighted_ce_binary.yaml
python scripts/run_experiments.py --config configs/experiments/19_focal_binary.yaml
python scripts/run_experiments.py --config configs/experiments/22_undersample_ce_binary.yaml
python scripts/run_experiments.py --config configs/experiments/15_binary_undersample.yaml
```

**Antes de continuar**: comparar F1-macro y recall de clase real. Elegir `best_loss`.

### Fase 2 — RQ2 (necesita best_loss de Fase 1)

```bash
# Actualizar 23_rgb_512_binary.yaml con best_loss, luego:
python scripts/run_experiments.py --config configs/experiments/23_rgb_512_binary.yaml
```

**Antes de continuar**: elegir `best_res` (224 o 512).

### Fase 3 — RQ3 (necesita best_loss + best_res de Fases 1+2)

```bash
# Si best_loss == weighted_ce, 16 y 18 ya están listos.
# Si best_loss == focal, crear 16b y 18b con focal antes de lanzar.
python scripts/run_experiments.py --config configs/experiments/24_resnet_scratch_binary.yaml
python scripts/run_experiments.py --config configs/experiments/16_vit_binary.yaml
python scripts/run_experiments.py --config configs/experiments/18_vit_binary_scratch.yaml
```

**Antes de continuar**: elegir `best_arch` (ResNet50 o ViT).

### Fase 4 — RQ4 (necesita best_arch + best_loss + best_res)

```bash
# Crear 25_fft_grayscale_binary con best_arch + best_loss.
# Si best_arch == vit, los experimentos de fusión necesitan adaptarse.
# (FFT fusion con ViT no está implementado — evaluar si merece la pena.)
python scripts/run_experiments.py --config configs/experiments/25_fft_grayscale_binary.yaml
python scripts/run_experiments.py --config configs/experiments/12_late_fusion_grayscale_binary.yaml
```

---

## Estimación de tiempos

| Fase | Experimentos | Tiempo/exp | Total (secuencial) |
|------|-------------|-----------|-------------------|
| RQ1 | 4 nuevos | ~10h | ~40h |
| RQ2 | 1 | ~25h (512×512) | ~25h |
| RQ3 | 3 | ~10-18h | ~45h |
| RQ4 | 2-3 | ~15h | ~30-45h |
| **Total** | **~10-11** | — | **~140-155h** |

Con 1 GPU: ~6-7 días continuos.
Con 2 GPUs en paralelo por fase: ~3-4 días.

---

## Métrica principal para selección del ganador

**F1-macro** (trata todas las clases igual — métrica oficial del leaderboard).

Como métricas secundarias para análisis:
- `balanced_accuracy` (media del recall por clase)
- `recall` de clase real (clase 0) — el talón de Aquiles actual
- `roc_auc` (independiente del threshold)

---

## Tabla resumen de resultados (rellenar al ejecutar)

### RQ1 — Ablación de imbalance

| Exp | Loss | Datos | F1-macro | BalAcc | Recall-real | roc_auc |
|-----|------|-------|----------|--------|-------------|---------|
| 01 | CE | full 1:5 | 0.6990 | 0.6977 | 0.4973 | — |
| 21 | weighted_CE | full 1:5 | — | — | — | — |
| 19 | focal(γ=2) | full 1:5 | — | — | — | — |
| 22 | CE | 1:1 | — | — | — | — |
| 15 | weighted_CE | 1:1 | — | — | — | — |

### RQ2 — Resolución

| Exp | Resolución | F1-macro | BalAcc | Tiempo/época |
|-----|-----------|----------|--------|-------------|
| best_RQ1 | 224×224 | — | — | — |
| 23 | 512×512 | — | — | — |

### RQ3 — Arquitectura

| Exp | Arch | Pretrained | F1-macro | BalAcc | Params |
|-----|------|-----------|----------|--------|--------|
| best_RQ2 | ResNet50 | yes | — | — | 25M |
| 24 | ResNet50 | no | — | — | 25M |
| 16 | ViT-B/16 | yes | — | — | 86M |
| 18 | ViT-B/16 | no | — | — | 86M |

### RQ4 — Modalidad

| Exp | Input | F1-macro | BalAcc |
|-----|-------|----------|--------|
| best_RQ3 | RGB | — | — |
| 25 | FFT gray | — | — |
| 12 | RGB+FFT gray | — | — |
| 13 | RGB+FFT 3ch | 0.6971 | — |

---

## Nota sobre Task B

Una vez cerrado el análisis de Task A, aplicar la mejor configuración encontrada
a Task B (multiclass) para verificar que generaliza:
- Sustituir `task: binary` → `task: multiclass`, `num_classes: 2` → `6`
- Task B es balanceada → no necesita estrategias anti-imbalance
- Añadir label_smoothing (exp 20) como única variación de loss relevante para Task B
