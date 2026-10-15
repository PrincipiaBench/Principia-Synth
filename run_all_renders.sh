#!/usr/bin/env bash
set -uo pipefail

# Render the Principia-Synth scenarios in parallel, spread round-robin over
# the GPUs in RENDER_GPUS. Each scenario writes to OUTPUT_ROOT/<Scenario> and
# logs to LOG_DIR/render_<scenario>.log.
#
# Usage (from anywhere):
#   ISAAC_PYTHON=/path/to/isaac-sim/python.sh bash run_all_renders.sh
#   RENDER_GPUS="0 1 2" NUM_SAMPLES=10 bash run_all_renders.sh
#   SCENARIOS="moi pendulum" bash run_all_renders.sh   # only some scenarios
#   EXTRA_ARGS="--seed 0 --spp 32" bash run_all_renders.sh
#
# GPU indices are the RTX renderer's device order (--/renderer/activeGpu),
# which usually matches nvidia-smi but not necessarily CUDA's order. The log's
# GPU table marks the active device.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Isaac Sim's python.sh, or `python` from an environment with the isaacsim pip package
ISAAC_PYTHON="${ISAAC_PYTHON:-python.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

RENDER_GPUS=(${RENDER_GPUS:-0 1})
NUM_SAMPLES="${NUM_SAMPLES:-100}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

SCENARIOS=(${SCENARIOS:-friction com moi pendulum projectile restitution spring})

declare -A OUT_NAMES=(
    [friction]=Friction
    [com]=COM
    [moi]=MOI
    [pendulum]=Pendulum
    [projectile]=Projectile
    [restitution]=Restitution
    [spring]=Spring
)

if ! command -v "$ISAAC_PYTHON" > /dev/null 2>&1; then
    echo "Isaac Sim python not found: '$ISAAC_PYTHON'. Set ISAAC_PYTHON to the path of python.sh." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

declare -A PIDS

for i in "${!SCENARIOS[@]}"; do
    s="${SCENARIOS[$i]}"
    if [[ -z "${OUT_NAMES[$s]:-}" ]]; then
        echo "unknown scenario: $s (choose from: ${!OUT_NAMES[*]})" >&2
        exit 1
    fi
    gpu="${RENDER_GPUS[$(( i % ${#RENDER_GPUS[@]} ))]}"
    log="$LOG_DIR/render_${s}.log"

    "$ISAAC_PYTHON" "$REPO_ROOT/scenarios/render_${s}.py" --headless \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_ROOT/${OUT_NAMES[$s]}" \
        "${EXTRA_ARGS[@]}" \
        --/renderer/activeGpu="$gpu" \
        --/renderer/multiGpu/enabled=false \
        > "$log" 2>&1 &

    PIDS[$s]=$!
    echo "started $s on renderer GPU $gpu (pid ${PIDS[$s]}) -> $log"
done

echo "waiting for ${#PIDS[@]} renders..."

status=0
for s in "${SCENARIOS[@]}"; do
    if wait "${PIDS[$s]}"; then
        echo "done:   $s"
    else
        echo "FAILED: $s (exit $?) - see $LOG_DIR/render_${s}.log"
        status=1
    fi
done

exit $status
