# Git Workflow — Defactify

Guía de cómo trabajar con git y GitHub en este proyecto.

---

## Estructura de Ramas

```
main          ← código estable y revisado (lo que está en producción)
  └── dev     ← rama de integración (trabajo en curso)
        ├── feat/phase-3-vlm        ← nueva funcionalidad
        ├── fix/dataset-loader      ← bug fix
        └── exp/03-fft-grayscale    ← resultados de experimento
```

**Reglas**:
- Nunca commitear directamente a `main`
- `dev` recibe PRs de ramas de trabajo
- `main` solo recibe PRs desde `dev` cuando todo está estable

---

## Setup Inicial (una sola vez)

```bash
cd /media/hector/Hector/Defactify

# Inicializar git
git init
git checkout -b main

# Conectar con GitHub (crea el repo en GitHub primero en github.com)
git remote add origin https://github.com/TU_USUARIO/Defactify.git

# Primer commit con todo el código
git add .
git commit -m "feat: initial framework - Phase 1 & 2 complete"
git push -u origin main

# Crear rama dev
git checkout -b dev
git push -u origin dev
```

---

## Flujo de Trabajo Diario

### Caso 1: Implementar nueva funcionalidad (ej. Fase 3 VLM)

```bash
# 1. Partir siempre desde dev actualizado
git checkout dev
git pull origin dev

# 2. Crear rama de trabajo
git checkout -b feat/phase-3-vlm

# 3. Trabajar... hacer cambios en src/, configs/, etc.

# 4. Commitear los cambios
git add src/processing/embeddings.py
git add src/models/vlm.py
git add configs/experiments/07_clip_embeddings.yaml
git commit -m "feat: add CLIP embeddings extractor and classifier"

# 5. Continuar trabajando, más commits
git add src/models/ensemble.py
git commit -m "feat: add VLM ensemble model"

# 6. Cuando esté listo, subir la rama
git push origin feat/phase-3-vlm

# 7. Crear PR hacia dev
gh pr create \
  --base dev \
  --head feat/phase-3-vlm \
  --title "feat: Phase 3 VLM embeddings" \
  --body "Adds CLIP, DINOv2 and SigLIP extractors with caching. See PLAN.md Phase 3."

# 8. Mergear el PR (en GitHub o desde terminal)
gh pr merge --squash    # squash = un solo commit limpio en dev
```

---

### Caso 2: Arreglar un bug

```bash
git checkout dev
git pull origin dev
git checkout -b fix/weighted-loss-binary

# ... corregir el bug ...

git add src/training/trainer.py
git commit -m "fix: apply weighted CE loss for binary imbalance 1:5"
git push origin fix/weighted-loss-binary

gh pr create --base dev --head fix/weighted-loss-binary \
  --title "fix: weighted loss for binary task"
```

---

### Caso 3: Subir resultados de un experimento

Los resultados NO necesitan PR — se commitean directamente a `dev` porque son datos, no código.

```bash
git checkout dev
git pull origin dev

# Añadir solo métricas (nunca pesos .pth — están en .gitignore)
git add results/comparison_table.csv
git add results/comparison_table.md
git add results/experiments/01_rgb_baseline_binary/final_metrics.json
git add results/experiments/01_rgb_baseline_binary/config.json
git add results/experiments/01_rgb_baseline_binary/experiment.log

git commit -m "exp(01): binary baseline - f1_macro=0.9489 bal_acc=0.9501 best_epoch=12"
git push origin dev
```

---

### Caso 4: Estabilizar dev → main (al acabar una fase)

```bash
# Cuando dev está estable y todos los experimentos de la fase están hechos:
gh pr create \
  --base main \
  --head dev \
  --title "release: Phase 2 complete - FFT and Late Fusion" \
  --body "All 4 FFT/fusion experiments done. See results/comparison_table.md"

gh pr merge --merge   # merge commit (no squash, para mantener historial)
```

---

## Cuándo Hacer PR y Cuándo No

| Situación | ¿PR? | Rama destino |
|-----------|------|--------------|
| Nueva funcionalidad (nueva fase, nuevo modelo) | ✅ Sí | `dev` |
| Bug fix en código | ✅ Sí | `dev` |
| Actualización de documentación relevante | ✅ Sí | `dev` |
| Resultados de experimento (métricas, logs) | ❌ No — commit directo | `dev` |
| Cambio menor en YAML de config | ❌ No — commit directo | `dev` |
| Estabilizar fase completa | ✅ Sí | `main` |

---

## Comandos de Consulta Útiles

```bash
# Ver estado actual
git status

# Ver historial de commits
git log --oneline -10

# Ver en qué rama estás
git branch

# Ver todas las ramas (local + remoto)
git branch -a

# Ver PRs abiertos
gh pr list

# Ver PR específico
gh pr view 5

# Ver diff antes de commitear
git diff

# Ver diff de staged
git diff --staged
```

---

## Mensajes de Commit

Formato: `tipo(scope): descripción corta`

| Tipo | Cuándo usarlo |
|------|---------------|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `exp` | Resultados de experimento |
| `docs` | Solo documentación |
| `refactor` | Refactor sin cambio funcional |
| `chore` | Cambios menores (deps, config) |

**Ejemplos**:
```
feat: add FFT per-channel computation
fix: cast lr to float to avoid YAML string parsing
exp(03): fft grayscale multiclass - f1_macro=0.8234
docs: update TESTING.md with Phase 2 experiments
chore: fix lr notation in all experiment configs
```

---

## .gitignore — Qué NO va a Git

```
results/**/*.pth       ← pesos del modelo (grandes)
data/cache/            ← FFT cache (regenerable)
data/processed/        ← splits procesados
wandb/                 ← logs de W&B (están en la nube)
__pycache__/
```

Los pesos del modelo se guardan localmente. Para compartirlos usa HuggingFace Hub o un bucket S3.
