#!/bin/bash
# ============================================================================
# EO Military-UAV — Standard validation (val.py) for all baselines × stages
# ============================================================================
# Runs val.py sequentially for every (method, stage) combination.
# Evaluates each stage model against its cumulative YAML so that
# per-class mAP reveals old-class vs new-class performance.
#
# Run from: /home/jovyan/thesis_cl/ednet/Thesis
# Usage:    bash scripts/eo_val.sh
# ============================================================================

set -euo pipefail

ROOT=/home/jovyan/thesis_cl/ednet/Thesis
RUNS=$ROOT/runs
YAML=$ROOT/yaml
VALDIR=$RUNS/eo_val

run_val() {
    local weights=$1
    local data=$2
    local outdir=$3
    echo "────────────────────────────────────────────────────────────────"
    echo "  weights : $weights"
    echo "  data    : $data"
    echo "  outdir  : $outdir"
    echo "────────────────────────────────────────────────────────────────"
    python val.py \
        --weights "$weights" \
        --data "$data" \
        --imgsz 640 \
        --split test \
        --outdir "$outdir"
}

# ── 1. Fine-tuning (naive) ──────────────────────────────────────────────────
run_val $RUNS/eo_finetune/stage1-finetune/weights/best.pt \
        $YAML/EO-stage1.yaml \
        $VALDIR/finetune/stage1

run_val $RUNS/eo_finetune/stage2-finetune/weights/best.pt \
        $YAML/EO-stage2.yaml \
        $VALDIR/finetune/stage2

run_val $RUNS/eo_finetune/stage3-finetune/weights/best.pt \
        $YAML/EO-stage3.yaml \
        $VALDIR/finetune/stage3

# ── 2. Replay ───────────────────────────────────────────────────────────────
run_val $RUNS/eo_replay/stage1-replay/weights/best.pt \
        $YAML/EO-stage1.yaml \
        $VALDIR/replay/stage1

run_val $RUNS/eo_replay/stage2-replay/weights/best.pt \
        $YAML/EO-stage2.yaml \
        $VALDIR/replay/stage2

run_val $RUNS/eo_replay/stage3-replay/weights/best.pt \
        $YAML/EO-stage3.yaml \
        $VALDIR/replay/stage3

# ── 3. KD (Knowledge Distillation + memory) ─────────────────────────────────
run_val $RUNS/eo_kd/stage1-kd/weights/best.pt \
        $YAML/EO-stage1.yaml \
        $VALDIR/kd/stage1

run_val $RUNS/eo_kd/stage2-kd/weights/best.pt \
        $YAML/EO-stage2.yaml \
        $VALDIR/kd/stage2

run_val $RUNS/eo_kd/stage3-kd/weights/best.pt \
        $YAML/EO-stage3.yaml \
        $VALDIR/kd/stage3

# ── 4. Pseudo-labeling ──────────────────────────────────────────────────────
run_val $RUNS/eo_pseudo/stage1-pseudo/weights/best.pt \
        $YAML/EO-stage1.yaml \
        $VALDIR/pseudo/stage1

run_val $RUNS/eo_pseudo/stage2-pseudo/weights/best.pt \
        $YAML/EO-stage2.yaml \
        $VALDIR/pseudo/stage2

run_val $RUNS/eo_pseudo/stage3-pseudo/weights/best.pt \
        $YAML/EO-stage3.yaml \
        $VALDIR/pseudo/stage3

# ── 5. SA-AB (cls+box prototypes + memory) ───────────────────────────────────
run_val $RUNS/eo_saab/stage1-saab/weights/best.pt \
        $YAML/EO-stage1.yaml \
        $VALDIR/saab/stage1

run_val $RUNS/eo_saab/stage2-saab/weights/best.pt \
        $YAML/EO-stage2.yaml \
        $VALDIR/saab/stage2

run_val $RUNS/eo_saab/stage3-saab/weights/best.pt \
        $YAML/EO-stage3.yaml \
        $VALDIR/saab/stage3

# ── 6. KD + Pseudo ──────────────────────────────────────────────────────────
run_val $RUNS/eo_kd_pseudo/stage1-kd-pseudo/weights/best.pt \
        $YAML/EO-stage1.yaml \
        $VALDIR/kd_pseudo/stage1

run_val $RUNS/eo_kd_pseudo/stage2-kd-pseudo/weights/best.pt \
        $YAML/EO-stage2.yaml \
        $VALDIR/kd_pseudo/stage2

run_val $RUNS/eo_kd_pseudo/stage3-kd-pseudo/weights/best.pt \
        $YAML/EO-stage3.yaml \
        $VALDIR/kd_pseudo/stage3

# ── 7. SA-AB + Pseudo ───────────────────────────────────────────────────────
run_val $RUNS/eo_saab_pseudo/stage1-saab-pseudo/weights/best.pt \
        $YAML/EO-stage1.yaml \
        $VALDIR/saab_pseudo/stage1

run_val $RUNS/eo_saab_pseudo/stage2-saab-pseudo/weights/best.pt \
        $YAML/EO-stage2.yaml \
        $VALDIR/saab_pseudo/stage2

run_val $RUNS/eo_saab_pseudo/stage3-saab-pseudo/weights/best.pt \
        $YAML/EO-stage3.yaml \
        $VALDIR/saab_pseudo/stage3

# ── 8. Joint training (upper bound) ──────────────────────────────────────────
run_val $RUNS/eo_upper/upper/weights/best.pt \
        $YAML/EO-all.yaml \
        $VALDIR/upper/final

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All validation runs complete.  Results in: $VALDIR/"
echo "════════════════════════════════════════════════════════════════"
