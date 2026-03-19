#!/bin/bash
# ==============================================================
# DEFACTIFY - SETUP COMMANDS
# ==============================================================
# Copiar y ejecutar estos comandos en terminal para configurar
# el proyecto y correr los experimentos
# ==============================================================

# ============ PASO 1: SETUP INICIAL (SOLO PRIMERA VEZ) ============

# Navegar a directorio del proyecto
cd /media/hector/Hector/Defactify

# Crear ambiente conda desde environment.yml
conda env create -f environment.yml -n defactify

# Activar ambiente
conda activate defactify

# Instalar paquete en modo desarrollo
pip install -e .

# Verificar instalación completa
python validate_framework.py
# Deberías ver: "✅ All imports successful!"

# =========== PASO 2: SETUP W&B (OPCIONAL - RECOMENDADO) ===========

# Login en Weights & Biases
wandb login
# Sigue el prompt:
#   1. Ir a https://wandb.ai
#   2. Login o crear cuenta
#   3. Ir a Settings -> API keys
#   4. Copiar la API key
#   5. Pegar en terminal

# ============= PASO 3: EJECUTAR EXPERIMENTOS FASE 1 =============

# Validar config sin ejecutar (quick check)
python scripts/run_experiments.py \
  --config configs/experiments/02_rgb_baseline_multiclass.yaml \
  --dry-run
# Deberías ver: "✅ Config valid: 02_rgb_baseline_multiclass"

# -------- EXPERIMENTO 1: RGB Binary (30 minutos) --------
python scripts/run_experiments.py \
  --config configs/experiments/01_rgb_baseline_binary.yaml

# Resultados en: results/experiments/01_rgb_baseline_binary/

# -------- EXPERIMENTO 2: RGB Multiclass (60 minutos) --------
python scripts/run_experiments.py \
  --config configs/experiments/02_rgb_baseline_multiclass.yaml

# Resultados en: results/experiments/02_rgb_baseline_multiclass/

# ============= PASO 4: REVISAR RESULTADOS =============

# Ver métricas finales (prettified JSON)
cat results/experiments/02_rgb_baseline_multiclass/final_metrics.json | python -m json.tool

# Ver primeras 10 predicciones
head -10 results/experiments/02_rgb_baseline_multiclass/predictions.csv

# Ver histórico de training
cat results/experiments/02_rgb_baseline_multiclass/metrics.json | python -m json.tool

# Ver logs completos
cat results/experiments/02_rgb_baseline_multiclass/experiment.log

# ============= PASO 5: COMPARAR RESULTADOS =============

# Script Python para comparar experimentos
python << 'EOF'
import json
import pandas as pd

# Cargar métricas finales
with open('results/experiments/01_rgb_baseline_binary/final_metrics.json') as f:
    exp1 = json.load(f)
with open('results/experiments/02_rgb_baseline_multiclass/final_metrics.json') as f:
    exp2 = json.load(f)

print("\n" + "="*60)
print("COMPARACIÓN DE EXPERIMENTOS FASE 1")
print("="*60)
print(f"\nExp 1 (Binary):     Accuracy = {exp1['test']['accuracy']:.4f}")
print(f"Exp 2 (Multiclass): Accuracy = {exp2['test']['accuracy']:.4f}")
print(f"\nExp 1 (Binary):     F1 Macro = {exp1['test']['f1_macro']:.4f}")
print(f"Exp 2 (Multiclass): F1 Macro = {exp2['test']['f1_macro']:.4f}")
print("="*60 + "\n")
EOF

# ============= MONITOREO EN TIEMPO REAL =============

# Durante ejecución, para ver logs en vivo
tail -f results/experiments/02_rgb_baseline_multiclass/experiment.log

# Ver uso de GPU durante training
watch -n 1 nvidia-smi
# (Presionar Ctrl+C para salir)

# Ver uso de RAM
watch -n 1 'free -h'

# ============= TROUBLESHOOTING =============

# Si da error "CUDA out of memory":
# Editar configs/base_config.yaml y cambiar:
#   training:
#     batch_size: 16  # (en vez de 32)
# Luego reintentar

# Si no se descarga el dataset (sin internet o lento):
# Descargar manualmente:
python << 'PYEOF'
from datasets import load_dataset
ds = load_dataset('NasrinImp/Defactify4_Train', split='train')
print(f"Dataset loaded: {len(ds)} examples")
PYEOF

# Si no se ve GPU:
python -c "import torch; print('GPU:', torch.cuda.is_available())"
# Si muestra "GPU: False", revisar instalación CUDA

# ============= MANTENER EN SYNC =============

# Actualizar el plan con resultados
# Editar PLAN.md y cambiar "TBD" por valores reales en tabla de métricas

# ============= SIGUIENTE PASO =============

# Después de completar Phase 1:
# 1. Revisar STATUS.md para entender qué sigue
# 2. Leer PLAN.md para ver Fase 2-5 (FFT, VLM, etc)
# 3. Esperar a que se implemente Fase 2 o comenzar si quieres

# ==============================================================
# FIN SETUP COMMANDS
# ==============================================================
# Documentación: Ver QUICKSTART.md, TESTING.md, PLAN.md
# ==============================================================
