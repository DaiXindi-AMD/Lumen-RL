# GPT-OSS-120B Eagle3 SDDD — vLLM Teacher (MI355)

Off-policy speculative decoding draft distillation for GPT-OSS-120B using
**vLLM** for teacher inference. Replaces ATOM teacher to eliminate the
double-normalization problem caused by ATOM's fused kernel residual scale.

## Architecture

```
GPUs 0-3: FSDP2 Eagle3 draft model training (BF16, LumenRL + aiter)
GPUs 4-7: vLLM teacher inference (TP=4, BF16, FP8 KV cache)
Transfer: Mooncake TCP (hidden states)
```

## Why vLLM instead of ATOM

vLLM captures raw `hidden_states + residual` at natural scale (~1-10),
matching NVIDIA's approach. Eagle3's `hidden_norm` provides the single
learned normalization — no double-norm.

ATOM's MXFP4 fused kernels accumulate residuals to 10^35+ scale, forcing
capture normalization that conflicts with Eagle3's `hidden_norm`.

## Quick Start

```bash
# Build Docker image
docker buildx build \
    -f examples/GPT_OSS_120b_MI355_vllm/Dockerfile.train \
    -t gpt_oss_eagle3_vllm_train:latest .

# Smoke test (5 steps)
bash examples/GPT_OSS_120b_MI355_vllm/run_docker.sh --smoke-test

# Full training with auto-retry
bash examples/GPT_OSS_120b_MI355_vllm/run_with_retry.sh
```

## vLLM Patches

Patches for vLLM 0.19.1+rocm721 are in `patch/`. They are applied
automatically at container startup by `run_gpt_oss_120b.sh` (idempotent).

| Patch | Description |
|-------|-------------|
| `patch_vllm_kv_cache_grouping.py` | Fix `HiddenStatesCacheSpec` being merged with normal attention layers in KV cache group init. `HiddenStatesCacheSpec` inherits `FullAttentionSpec`, so `UniformTypeKVCacheSpecs.from_specs()` incorrectly treats it as uniform. The patch strips hidden-state specs before uniformity checks and reattaches them as singleton groups. |

To apply manually (e.g. in a running container):

```bash
docker exec <container> python /root/lumenrl/examples/GPT_OSS_120b_MI355_vllm/patch/patch_vllm_kv_cache_grouping.py
```

## Prerequisites

- 8x MI355 GPUs (288GB each)
- Model weights at `/dev/shm/gpt-oss-120b`
- Dataset at `/dev/shm/gpt_oss_120b_dataset/train.jsonl`
