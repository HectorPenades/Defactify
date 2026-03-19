# Defactify — Quick Start Guide

---

## 1. Setup del Entorno

```bash
cd /path/to/Defactify

# Crear entorno conda
conda env create -f environment.yml -n defactify
conda activate defactify
pip install -e .

# Verificar
python validate_framework.py
```

Si el entorno ya existe:
```bash
conda activate defactify
conda env update -f environment.yml --prune
```

---

## 2. Weights & Biases (opcional pero recomendado)

```bash
pip install wandb
wandb login   # Pide API key de https://wandb.ai
```

Los experimentos se loguean al proyecto `real-vs-synthetic` automáticamente.

---

## 3. Verificar Setup

```bash
# Validar imports
python validate_framework.py

# Validar config sin ejecutar
python scripts/run_experiments.py \
  --config configs/experiments/01_rgb_baseline_binary.yaml \
  --dry-run
```

---

## 4. Ejecutar Experimentos

### Flujo completo (train + val cada época + test al final)

```bash
# Experimento 01: Binary (real vs AI)
python scripts/run_experiments.py --config configs/experiments/01_rgb_baseline_binary.yaml

# Experimento 02: Multiclass (6 generadores)
python scripts/run_experiments.py --config configs/experiments/02_rgb_baseline_multiclass.yaml

# Fase 2 — FFT
python scripts/run_experiments.py --config configs/experiments/03_fft_grayscale_multiclass.yaml
python scripts/run_experiments.py --config configs/experiments/04_fft_perchannel_multiclass.yaml

# Fase 2 — Late Fusion
python scripts/run_experiments.py --config configs/experiments/05_late_fusion_grayscale.yaml
python scripts/run_experiments.py --config configs/experiments/06_late_fusion_perchannel.yaml
```

### Reanudar entrenamiento interrumpido

```bash
python scripts/run_experiments.py \
  --config configs/experiments/01_rgb_baseline_binary.yaml \
  --resume
```

Restaura modelo, optimizer, scheduler y patience counter desde `checkpoint_last.pth`.
Para ampliar épocas: cambia `num_epochs` en el YAML y usa `--resume`.

### Re-evaluar en test sin reentrenar

```bash
# Usa checkpoint_best.pth por defecto
python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml

# Con checkpoint personalizado
python scripts/run_test.py \
  --config configs/experiments/01_rgb_baseline_binary.yaml \
  --checkpoint results/experiments/01_rgb_baseline_binary/checkpoint_best.pth

# Solo métricas, sin actualizar la tabla comparativa
python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml --no-record
```

---

## 5. Monitorear en Tiempo Real

```bash
# Log en vivo
tail -f results/experiments/01_rgb_baseline_binary/experiment.log
```

Formato de log por época:
```
Epoch 5/50 | Train Loss: 0.0921 | Val Loss: 0.1123 | Val Acc: 0.9701 | Val BalAcc: 0.9489 | Val F1-macro: 0.9476
```

O en W&B: `https://wandb.ai/<usuario>/real-vs-synthetic`

---

## 6. Ver Resultados

```bash
# Tabla comparativa de todos los experimentos
cat results/comparison_table.md

# Métricas finales de un experimento
cat results/experiments/01_rgb_baseline_binary/final_metrics.json | python -m json.tool

# Primeras predicciones
head -5 results/experiments/01_rgb_baseline_binary/predictions.csv
```

`final_metrics.json` incluye:
- `test.accuracy`, `test.balanced_accuracy`, `test.f1_macro`, `test.f1_weighted`, `test.roc_auc` (binario)
- `test.per_class`: precision / recall / f1 / support por clase con nombre (`real`, `ai_generated`, `sd21`, ...)
- `best_epoch`: época del mejor val loss (desde 1)
- `total_epochs_trained`: épocas totales antes de parar

---

## 7. Troubleshooting

### CUDA out of memory
```bash
# Reducir batch_size en el YAML del experimento
# training:
#   batch_size: 16
```

### Dataset no descarga
```bash
# Verificar conexión y descargar manualmente
python -c "
from datasets import load_dataset
ds = load_dataset('Rajarshi-Roy-research/Defactify_Image_Dataset', split='train')
print('OK:', len(ds), 'rows')
"
```

### Training lento / sin GPU
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Module not found
```bash
cd /path/to/Defactify
pip install -e .
```

---

## 8. Resumen de Comandos

```bash
# Setup (una sola vez)
conda env create -f environment.yml -n defactify && conda activate defactify && pip install -e .

# Ejecutar experimento
python scripts/run_experiments.py --config configs/experiments/<exp>.yaml

# Reanudar
python scripts/run_experiments.py --config configs/experiments/<exp>.yaml --resume

# Re-evaluar test
python scripts/run_test.py --config configs/experiments/<exp>.yaml

# Comparar todos
cat results/comparison_table.md
```

---

Ver `TESTING.md` para detalles de cada experimento. Ver `PLAN.md` para el roadmap.
