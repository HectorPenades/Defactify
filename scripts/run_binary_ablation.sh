#!/usr/bin/env bash
# =============================================================================
# run_binary_ablation.sh — Ablación completa Task A (binary)
#
# Ejecuta todos los experimentos del artículo en orden de RQ.
# Cada experimento termina antes de lanzar el siguiente.
# Los fallos se registran pero NO detienen el script.
#
# Uso:
#   chmod +x scripts/run_binary_ablation.sh
#   ./scripts/run_binary_ablation.sh                    # todo
#   ./scripts/run_binary_ablation.sh --phase 1          # solo RQ1
#   ./scripts/run_binary_ablation.sh --phase 2          # solo RQ2
#   ./scripts/run_binary_ablation.sh --phase 3          # solo RQ3
#   ./scripts/run_binary_ablation.sh --phase 4          # solo RQ4
#
# Lanzar en background y redirigir salida:
#   nohup ./scripts/run_binary_ablation.sh > logs/ablation.log 2>&1 &
#   tail -f logs/ablation.log
# =============================================================================

set -uo pipefail

# ---------- configuración -----------------------------------------------
PYTHON="${PYTHON:-python}"          # sobreescribir con: PYTHON=python3.9 ./run...
SCRIPT="scripts/run_experiments.py"
LOG_DIR="logs"
PHASE_FILTER="${2:-all}"            # all | 1 | 2 | 3 | 4

# Parsear --phase N
if [[ "${1:-}" == "--phase" && -n "${2:-}" ]]; then
    PHASE_FILTER="$2"
fi

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/ablation_$(date +%Y%m%d_%H%M%S).log"

# ---------- contadores ---------------------------------------------------
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
declare -A RESULTS   # exp_name -> OK | FAIL | SKIP

# ---------- helpers ------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
sep() { log "$(printf '=%.0s' {1..70})"; }

run_exp() {
    local config="$1"
    local exp_name
    exp_name=$(grep 'experiment_name:' "$config" | head -1 | awk -F'"' '{print $2}')
    local exp_log="$LOG_DIR/${exp_name}.log"

    TOTAL=$((TOTAL + 1))
    log ""
    log ">>> Iniciando: $exp_name"
    log "    Config : $config"
    log "    Log    : $exp_log"
    local t_start
    t_start=$(date +%s)

    if $PYTHON "$SCRIPT" --config "$config" 2>&1 | tee -a "$exp_log" "$MASTER_LOG"; then
        local elapsed=$(( $(date +%s) - t_start ))
        log "<<< OK: $exp_name  (${elapsed}s / $(( elapsed/3600 ))h$(( (elapsed%3600)/60 ))m)"
        RESULTS["$exp_name"]="OK"
        PASSED=$((PASSED + 1))
    else
        local elapsed=$(( $(date +%s) - t_start ))
        log "<<< FALLO: $exp_name  (${elapsed}s)"
        log "    Ver detalles en: $exp_log"
        RESULTS["$exp_name"]="FAIL"
        FAILED=$((FAILED + 1))
    fi
}

should_run() {
    local phase="$1"
    [[ "$PHASE_FILTER" == "all" || "$PHASE_FILTER" == "$phase" ]]
}

# =============================================================================
sep
log "ABLACIÓN COMPLETA TASK A — $(date)"
log "Fase(s): ${PHASE_FILTER}"
log "Log maestro: $MASTER_LOG"
sep

# =============================================================================
# FASE 1 — RQ1: estrategia anti-desbalance
# Todos independientes entre sí — comparación con exp 01 (ya ejecutado, F1=0.6990)
# =============================================================================
if should_run 1; then
    sep
    log "FASE 1 — RQ1: Estrategia anti-desbalance (ResNet50 pretrained, 224, RGB)"
    log "  exp 01  CE plain           full 1:5  [REFERENCIA — ya ejecutado]"
    log "  exp 21  weighted_CE(5,1)   full 1:5"
    log "  exp 19  focal(gamma=2)     full 1:5"
    log "  exp 22  CE                 1:1 undersample"
    log "  exp 15  weighted_CE(5,1)   1:1 undersample"
    sep

    run_exp "configs/experiments/21_weighted_ce_binary.yaml"
    run_exp "configs/experiments/19_focal_binary.yaml"
    run_exp "configs/experiments/22_undersample_ce_binary.yaml"
    run_exp "configs/experiments/15_binary_undersample.yaml"

    sep
    log "FASE 1 completada. Revisar results/comparison_table.md para elegir best_loss."
    log "NOTA: Si best_loss != weighted_CE, actualizar configs 23, 24, 25 antes de continuar."
    sep
fi

# =============================================================================
# FASE 2 — RQ2: resolución de entrada
# Pre-configurado con weighted_CE. Actualizar si RQ1 da otro ganador.
# =============================================================================
if should_run 2; then
    sep
    log "FASE 2 — RQ2: Resolución de entrada (ResNet50 pretrained, RGB, best_loss≈weighted_CE)"
    log "  mejor_RQ1  224×224  [REFERENCIA — ya ejecutado en Fase 1]"
    log "  exp 23     512×512"
    sep

    run_exp "configs/experiments/23_rgb_512_binary.yaml"

    sep
    log "FASE 2 completada. Comparar 224 vs 512. Elegir best_res."
    sep
fi

# =============================================================================
# FASE 3 — RQ3: arquitectura (CNN vs Transformer)
# Pre-configurado con weighted_CE + 224. Actualizar si best_loss o best_res cambia.
# =============================================================================
if should_run 3; then
    sep
    log "FASE 3 — RQ3: Arquitectura (best_loss≈weighted_CE, best_res≈224, RGB)"
    log "  mejor_RQ2  ResNet50  pretrained    [REFERENCIA — ya ejecutado]"
    log "  exp 24     ResNet50  scratch"
    log "  exp 16     ViT-B/16  pretrained"
    log "  exp 18     ViT-B/16  scratch"
    sep

    run_exp "configs/experiments/24_resnet_scratch_binary.yaml"
    run_exp "configs/experiments/16_vit_binary.yaml"
    run_exp "configs/experiments/18_vit_binary_scratch.yaml"

    sep
    log "FASE 3 completada. Comparar ResNet50 vs ViT, pretrained vs scratch. Elegir best_arch."
    sep
fi

# =============================================================================
# FASE 4 — RQ4: modalidad de entrada (¿aporta FFT?)
# Usa best_arch de RQ3. Pre-configurado con ResNet50 (si ViT gana, adaptar manualmente).
# =============================================================================
if should_run 4; then
    sep
    log "FASE 4 — RQ4: Modalidad de entrada (best_arch≈ResNet50, best_loss, best_res)"
    log "  mejor_RQ3  RGB solo               [REFERENCIA — ya ejecutado]"
    log "  exp 25     FFT grayscale solo"
    log "  exp 12     RGB + FFT gray (fusion)"
    log "  exp 13     RGB + FFT 3ch  (fusion) [ya ejecutado F1=0.6971]"
    sep

    run_exp "configs/experiments/25_fft_grayscale_binary.yaml"
    run_exp "configs/experiments/12_late_fusion_grayscale_binary.yaml"

    sep
    log "FASE 4 completada. Comparar RGB vs FFT vs Fusion."
    sep
fi

# =============================================================================
# RESUMEN FINAL
# =============================================================================
sep
log "RESUMEN FINAL"
sep
log "  Total ejecutados : $TOTAL"
log "  OK               : $PASSED"
log "  Fallidos         : $FAILED"
log ""
log "  Detalle:"
for exp_name in "${!RESULTS[@]}"; do
    status="${RESULTS[$exp_name]}"
    icon="✅"
    [[ "$status" == "FAIL" ]] && icon="❌"
    log "    $icon  $exp_name  →  $status"
done | sort
log ""
log "  Tabla de resultados: results/comparison_table.md"
log "  Log completo      : $MASTER_LOG"
sep
log "Fin: $(date)"
sep
