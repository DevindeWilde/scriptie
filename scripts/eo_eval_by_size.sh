#!/bin/bash
# ============================================================================
# EO Military-UAV — Size-stratified evaluation (eval_by_size.py)
# ============================================================================
# Runs eval_by_size.py for every baseline at Stage 1 and Stage 3.
# Stage 1 = baseline (all classes are "new"), Stage 3 = after incremental
# learning (measures forgetting on old classes by size).
#
# Output JSON paths match the CONFIG in make_military_tables.py.
#
# Run from: /home/jovyan/thesis_cl/ednet/Thesis
# Usage:    bash scripts/eo_eval_by_size.sh
# ============================================================================

set -euo pipefail

ROOT=/home/jovyan/thesis_cl/ednet/Thesis
RUNS=$ROOT/runs
YAML=$ROOT/yaml
OUTDIR=$RUNS/eo_val_by_size

run_eval() {
    local weights=$1
    local data=$2
    local out=$3
    echo "────────────────────────────────────────────────────────────────"
    echo "  weights : $weights"
    echo "  data    : $data"
    echo "  out     : $out"
    echo "────────────────────────────────────────────────────────────────"
    python scripts/eval_by_size.py \
        --weights "$weights" \
        --data "$data" \
        --split test \
        --imgsz 640 \
        --out "$out"
}

# ── 1. Fine-tuning (naive) ──────────────────────────────────────────────────
run_eval $RUNS/eo_finetune/stage1-finetune/weights/best.pt \
         $YAML/EO-stage1.yaml \
         $OUTDIR/finetune/stage1.json

run_eval $RUNS/eo_finetune/stage3-finetune/weights/best.pt \
         $YAML/EO-stage3.yaml \
         $OUTDIR/finetune/stage3.json

# ── 2. Replay ───────────────────────────────────────────────────────────────
run_eval $RUNS/eo_replay/stage1-replay/weights/best.pt \
         $YAML/EO-stage1.yaml \
         $OUTDIR/replay/stage1.json

run_eval $RUNS/eo_replay/stage3-replay/weights/best.pt \
         $YAML/EO-stage3.yaml \
         $OUTDIR/replay/stage3.json

# ── 3. KD (Knowledge Distillation + memory) ─────────────────────────────────
run_eval $RUNS/eo_kd/stage1-kd/weights/best.pt \
         $YAML/EO-stage1.yaml \
         $OUTDIR/kd/stage1.json

run_eval $RUNS/eo_kd/stage3-kd/weights/best.pt \
         $YAML/EO-stage3.yaml \
         $OUTDIR/kd/stage3.json

# ── 4. Pseudo-labeling ──────────────────────────────────────────────────────
run_eval $RUNS/eo_pseudo/stage1-pseudo/weights/best.pt \
         $YAML/EO-stage1.yaml \
         $OUTDIR/pseudo/stage1.json

run_eval $RUNS/eo_pseudo/stage3-pseudo/weights/best.pt \
         $YAML/EO-stage3.yaml \
         $OUTDIR/pseudo/stage3.json

# ── 5. SA-AB (cls+box prototypes + memory) ───────────────────────────────────
run_eval $RUNS/eo_saab/stage1-saab/weights/best.pt \
         $YAML/EO-stage1.yaml \
         $OUTDIR/saab/stage1.json

run_eval $RUNS/eo_saab/stage3-saab/weights/best.pt \
         $YAML/EO-stage3.yaml \
         $OUTDIR/saab/stage3.json

# ── 6. KD + Pseudo ──────────────────────────────────────────────────────────
run_eval $RUNS/eo_kd_pseudo/stage1-kd-pseudo/weights/best.pt \
         $YAML/EO-stage1.yaml \
         $OUTDIR/kd_pseudo/stage1.json

run_eval $RUNS/eo_kd_pseudo/stage3-kd-pseudo/weights/best.pt \
         $YAML/EO-stage3.yaml \
         $OUTDIR/kd_pseudo/stage3.json

# ── 7. SA-AB + Pseudo ───────────────────────────────────────────────────────
run_eval $RUNS/eo_saab_pseudo/stage1-saab-pseudo/weights/best.pt \
         $YAML/EO-stage1.yaml \
         $OUTDIR/saab_pseudo/stage1.json

run_eval $RUNS/eo_saab_pseudo/stage3-saab-pseudo/weights/best.pt \
         $YAML/EO-stage3.yaml \
         $OUTDIR/saab_pseudo/stage3.json

# ── 8. Joint training (upper bound) ──────────────────────────────────────────
run_eval $RUNS/eo_upper/upper/weights/best.pt \
         $YAML/EO-all.yaml \
         $OUTDIR/upper/stage1.json

run_eval $RUNS/eo_upper/upper/weights/best.pt \
         $YAML/EO-all.yaml \
         $OUTDIR/upper/stage3.json

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All size-stratified evals complete.  Results in: $OUTDIR/"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Update make_military_tables.py CONFIG['size_jsons'] paths to:"
echo "    \$OUTDIR/<method>/stage{1,3}.json"
