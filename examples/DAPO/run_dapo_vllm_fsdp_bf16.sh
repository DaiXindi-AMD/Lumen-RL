#!/usr/bin/env bash
# Native LumenRL DAPO — vLLM rollout + Lumen FSDP (BF16), no verl dependency.
#
# Reproduces the verl recipe/dapo math RL run (dapo-lumen-bf16-fp8-runbook.md)
# inside LumenRL's native RLTrainer: DAPO dynamic sampling (filter_groups),
# clip-higher + dual-clip token-mean loss, soft overlong-buffer reward, and
# token-level TIS rollout correction, with vanilla vLLM generation + FSDP2.
#
# Qwen3-8B-Base on 8×MI300X/MI350X. Override MODEL_DIR / DATA_DIR / CKPT_DIR.
set -uo pipefail

SMOKE_TEST=false
DRY_RUN=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
        --dry-run)    DRY_RUN=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="dapo-vllm-fsdp-bf16"
CONFIG="${SCRIPT_DIR}/configs/dapo_qwen3_8b_vllm_fsdp_bf16.yaml"
OUTPUT_DIR="${REPO_ROOT}/output/DAPO/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"
NUM_GPUS="${NUM_GPUS:-8}"

MODEL_DIR="${MODEL_DIR:-/home/danyzhan/model}"
DATA_DIR="${DATA_DIR:-/home/danyzhan/data}"
CKPT_DIR="${CKPT_DIR:-/home/danyzhan/ckpts/lumenrl-dapo/${EXP_NAME}}"

if [ ! -d "${MODEL_DIR}/qwen3-8b-base" ]; then
    echo "ERROR: Model not found at ${MODEL_DIR}/qwen3-8b-base" >&2
    exit 1
fi
if [ ! -f "${DATA_DIR}/dapo-math-17k.parquet" ]; then
    echo "ERROR: Training data not found at ${DATA_DIR}/dapo-math-17k.parquet" >&2
    exit 1
fi

# ROCm + vLLM environment
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# If ATOM is installed in the image, its vLLM model plugin auto-loads and can
# crash vanilla vLLM on aiter API drift. Disable it for this native vLLM path.
export ATOM_DISABLE_VLLM_PLUGIN=1
export LUMENRL_LOG_LEVEL=INFO
export LUMENRL_WEIGHT_SYNC_DIR=/dev/shm/lumenrl_weight_sync
export NCCL_TIMEOUT=7200

# vLLM rollout attention backend (CK/ASM FA first, unified-attn fallback).
ATTN_BACKEND="${VLLM_ROCM_ATTN_BACKEND:-ROCM_AITER_FA}"
ATTN_BACKEND_FALLBACK="${VLLM_ROCM_ATTN_BACKEND_FALLBACK:-ROCM_AITER_UNIFIED_ATTN}"
export VLLM_ROCM_ATTN_BACKEND="${ATTN_BACKEND}"

pkill -f "lumenrl.trainer.main" 2>/dev/null || true
pkill -f "pt_elastic" 2>/dev/null || true
sleep 2

mkdir -p "${OUTPUT_DIR}" /dev/shm/lumenrl_weight_sync "${CKPT_DIR}"

echo ""
echo "=============================================================="
echo " LumenRL DAPO — vLLM rollout + Lumen FSDP (BF16), no verl"
echo "   Model:  ${MODEL_DIR}/qwen3-8b-base"
echo "   GPUs:   ${NUM_GPUS}    Attn: ${VLLM_ROCM_ATTN_BACKEND}"
echo "   Config: ${CONFIG}"
echo "   Log:    ${LOG_FILE}"
echo "   Smoke:  ${SMOKE_TEST}   Dry-run: ${DRY_RUN}"
echo "=============================================================="
echo ""

OVERRIDES=()
OVERRIDES+=("policy.model_name=${MODEL_DIR}/qwen3-8b-base")
OVERRIDES+=("reward.dataset=${DATA_DIR}/dapo-math-17k.parquet")
OVERRIDES+=("checkpointing.checkpoint_dir=${CKPT_DIR}")
if [ -n "${TOTAL_STEPS:-}" ]; then
    OVERRIDES+=("num_training_steps=${TOTAL_STEPS}")
fi

# Smoke test: tiny 1-step pipeline validation (gen 24 prompts → keep 8).
if [ "${SMOKE_TEST}" = true ]; then
    EXP_NAME="dapo-vllm-fsdp-bf16-smoke"
    OUTPUT_DIR="${REPO_ROOT}/output/DAPO/${EXP_NAME}"
    LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"
    mkdir -p "${OUTPUT_DIR}"
    OVERRIDES+=("num_training_steps=1")
    OVERRIDES+=("policy.train_global_batch_size=64")     # 8 prompts × 8 gens
    OVERRIDES+=("policy.gen_batch_size=24")
    OVERRIDES+=("policy.max_response_length=2048")
    OVERRIDES+=("policy.max_total_sequence_length=3072")
    OVERRIDES+=("policy.max_token_len_per_gpu=3072")
    OVERRIDES+=("policy.generation.vllm_cfg.max_model_len=3072")
    OVERRIDES+=("algorithm.dapo.num_generations=8")
    OVERRIDES+=("algorithm.dapo.max_resp_len=2048")
    OVERRIDES+=("algorithm.dapo.overlong_buffer.len=512")
    echo "*** SMOKE-TEST MODE: 1 step, train 8p×8g, gen 24p, max_resp=2048 ***"
fi

if [ "${DRY_RUN}" = true ]; then
    export LUMENRL_DRY_RUN=1
    echo "*** DRY-RUN MODE: mock generation (no vLLM) ***"
fi

MAX_RETRIES="${MAX_RETRIES:-50}"
RETRY_DELAY="${RETRY_DELAY:-10}"
_fell_back=false

for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "=== Attempt ${attempt}/${MAX_RETRIES} ($(date)) — VLLM_ROCM_ATTN_BACKEND=${VLLM_ROCM_ATTN_BACKEND} ==="
    pkill -9 -f "python.*vllm" 2>/dev/null || true
    sleep 2

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT:-29500}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        "${OVERRIDES[@]}" \
        >> "${LOG_FILE}" 2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "Training completed successfully on attempt ${attempt}."
        exit 0
    fi

    echo "*** Training crashed (exit=${EXIT_CODE}). Cleaning up GPU state... ***"
    pkill -9 -f "python.*lumenrl" 2>/dev/null || true
    pkill -9 -f "python.*vllm" 2>/dev/null || true

    if [ "${_fell_back}" = false ] && [ "${VLLM_ROCM_ATTN_BACKEND}" = "ROCM_AITER_FA" ]; then
        if grep -q "fmha_fwd_v3\|undefined symbol.*aiter.*mha_fwd\|ROCM_AITER_FA.*fail" "${LOG_FILE}" 2>/dev/null; then
            echo "*** ROCM_AITER_FA failed — falling back to ${ATTN_BACKEND_FALLBACK} ***"
            export VLLM_ROCM_ATTN_BACKEND="${ATTN_BACKEND_FALLBACK}"
            _fell_back=true
        fi
    fi
    sleep "${RETRY_DELAY}"
done

echo "ERROR: Training failed after ${MAX_RETRIES} attempts." >&2
exit 1
