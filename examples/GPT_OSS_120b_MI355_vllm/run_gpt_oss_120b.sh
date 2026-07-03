#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# GPT-OSS-120B Eagle3 Draft Model Distillation (vLLM + Mooncake TCP) — MI355
#
# Single-pass training on combined UltraChat + Magpie dataset (~503K samples).
# Data must be prepared first via lumenrl.data.make_dataset (see run_docker.sh Step 0).
#
# GPU split (8x MI355):
#   GPUs 0-3: torchrun FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM teacher inference (TP=4, BF16, FP8 KV cache)
#
# vLLM captures raw hidden states at natural scale (~1-10), matching
# NVIDIA's approach. No capture normalization — Eagle3's hidden_norm
# provides the single learned normalization.
#
# Usage:
#   bash examples/GPT_OSS_120b_MI355_vllm/run_gpt_oss_120b.sh
#   bash examples/GPT_OSS_120b_MI355_vllm/run_gpt_oss_120b.sh --smoke-test
#   MODEL_PATH=/path/to/model bash examples/GPT_OSS_120b_MI355_vllm/run_gpt_oss_120b.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="gpt-oss-120b-eagle3-vllm-mi355"
OUTPUT_DIR="${REPO_ROOT}/output/GPT_OSS_120b_SDDD/LumenRL"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

MODEL_PATH="${MODEL_PATH:-/dev/shm/gpt-oss-120b}"
if [ "${SMOKE_TEST}" = true ]; then
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/gpt_oss_120b_smoke_test_vllm}"
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 5-step Eagle3 validation (vLLM+Mooncake TCP, gpt-oss-120b)"
else
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/gpt_oss_120b_eagle3_vllm}"
    CONFIG="${SCRIPT_DIR}/configs/train.yaml"
fi

# Environment
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200
export GLOG_minloglevel="${GLOG_minloglevel:-3}"
export GLOG_v="${GLOG_v:-0}"
export GLOG_logtostderr="${GLOG_logtostderr:-1}"
export MOONCAKE_LOG_LEVEL="${MOONCAKE_LOG_LEVEL:-FATAL}"

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

# Clear pyc caches to pick up volume-mounted code changes
find /root/lumenrl -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Apply vLLM patches (volume-mounted, idempotent)
PATCH_DIR="${SCRIPT_DIR}/patch"
if [ -d "${PATCH_DIR}" ]; then
    for p in "${PATCH_DIR}"/patch_*.py; do
        [ -f "$p" ] && python3 "$p" 2>&1 || echo ">>> WARNING: patch $p failed"
    done
fi

cleanup_orphans() {
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
    pkill -9 -f "EngineCore"  2>/dev/null || true
    pkill -9 -f "AsyncLLMEngine" 2>/dev/null || true
    pkill -9 -f "mooncake_master" 2>/dev/null || true
}
trap cleanup_orphans EXIT

echo "═══════════════════════════════════════════════════════════════"
echo "  GPT-OSS-120B Eagle3 Draft Model Distillation (vLLM) — MI355"
echo "  Teacher:     ${MODEL_PATH} (vLLM, GPUs 4-7, TP=4, BF16)"
echo "  Draft:       Eagle3 (1 Transformer block, BF16)"
echo "  Train GPUs:  ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} GPUs, FSDP2+aiter)"
echo "  Transfer:    Mooncake TCP"
echo "  Config:      ${CONFIG}"
echo "  Output:      ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

OVERRIDES=(
    "policy.model_name=${MODEL_PATH}"
    "algorithm.teacher.model_name=${MODEL_PATH}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)

MOONCAKE_NOISE_RE='scoped_vlog_timer|MasterClient::(Ping|FetchTasks)|transfer_task\.cpp|client_service\.cpp|BatchGet completed|Transfer (completed|engine operation)|Setting transfer result'

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        2>&1 \
    | grep --line-buffered -v -E "^[IWEF][0-9]{4} [0-9:.]+\s+[0-9]+ \S+\.(cpp|cc|h):" \
    | grep --line-buffered -vE "${MOONCAKE_NOISE_RE}" \
    | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -eq 0 ] && grep -qE \
    '(Traceback \(most recent call last\)|HfHubHTTPError|Training failed|FAILED|exitcode\s*:\s*-[0-9]+|MEMORY_APERTURE_VIOLATION|out of memory)' \
    "${LOG_FILE}" 2>/dev/null; then
    echo ">>> Crash detected in log despite torchrun exit code 0." >&2
    EXIT_CODE=1
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> GPT-OSS-120B Eagle3 distillation (vLLM, MI355) completed successfully."
else
    echo ">>> GPT-OSS-120B Eagle3 distillation (vLLM, MI355) failed with exit code ${EXIT_CODE}." >&2
fi
echo ">>> Log: ${LOG_FILE}"
exit ${EXIT_CODE}
