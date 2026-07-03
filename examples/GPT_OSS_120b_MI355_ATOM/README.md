# GPT-OSS-120B Eagle3 Draft Distillation (ATOM + Mooncake TCP) — MI355

Train Eagle3 speculative decoding draft model using OpenAI's `gpt-oss-120b` (117B-param MoE, 5.1B active) teacher hidden states on **8x MI355 GPUs** with ATOM inference and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 — Eagle3 draft model training (BF16, LumenRL + aiter)
- **GPUs 4-7**: ATOM — gpt-oss-120b teacher (TP=4, native MXFP4 MoE)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2 + aiter          <---   ATOM AsyncLLMEngine (RLHFModelRunner)
  Eagle3 draft model, BF16              TP=4, native MXFP4 MoE
       ^                                       |
  Mooncake TCP  <---------------------  hidden_states via
  EagleMooncakeStore                   configure_hidden_states()
```

## Docker image

Built from `rocm/atom:latest` with training dependencies:

```bash
docker build -f examples/GPT_OSS_120b_MI355_ATOM/Dockerfile.train \
    -t gpt_oss_eagle3_train:latest /home/danyzhan/Lumen-RL/
```

## Quick Start

### 1. Download model

```bash
huggingface-cli download openai/gpt-oss-120b --local-dir /dev/shm/gpt-oss-120b
```
(~196 GB MXFP4 distribution)

### 2. Smoke test (5 steps, synthetic prompts)

```bash
bash examples/GPT_OSS_120b_MI355_ATOM/run_docker.sh --smoke-test
```

### 3. Full training (~15,870 steps, combined UltraChat + Magpie)

```bash
# Prepare combined dataset first
# python3 -m lumenrl.data.make_dataset

HF_TOKEN=hf_xxx bash examples/GPT_OSS_120b_MI355_ATOM/run_docker.sh
```

### 4. Auto-restart wrapper (long runs)

```bash
bash examples/GPT_OSS_120b_MI355_ATOM/run_with_retry.sh
```

Wraps `run_docker.sh` with auto-restart on HSA aperture faults and idle-log watchdog. `resume:true` in the config picks up the latest checkpoint.

### Recipe alignment with [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context)

| | This config | NVIDIA recipe |
|---|---|---|
| Draft arch | 1-layer Llama-style, head_dim=64, ffn=17280, GQA 8:1, llama3 RoPE (factor=8) | identical |
| `eagle_aux_hidden_state_layer_ids` | `[1, 17, 32]` | `[1, 17, 32]` |
| Max seq len (train) | 2048 | 8192 |
| Training data | Combined UltraChat + Magpie (~503K prompts) | ultrachat + Magpie |
| Learning rate | 1e-4 | 1e-4 |
| Loss decay | 0.9 | 0.9 |

Override env vars as needed:
```bash
MODEL_PATH=/some/other/path \
CKPT_DIR=/some/checkpoint/dir \
DOCKER_IMAGE=gpt_oss_eagle3_train:my-tag \
HF_TOKEN=hf_xxx \
bash examples/GPT_OSS_120b_MI355_ATOM/run_docker.sh
```

## Model facts (gpt-oss-120b)

| | |
|---|---|
| Total params | 117B |
| Active params / token | 5.1B (MoE: 128 experts, 4 per token) |
| Layers | 36 |
| Hidden | 2880 |
| Heads / KV heads | 64 / 8 (GQA) |
| Vocab | 201088 (`o200k_harmony` tokenizer) |
| RoPE | θ=150000 + YaRN(factor=32, original_max=4096) |
| Native quantization | MXFP4 for MoE weights (~196 GB on disk) |

## File structure

```
configs/
  train.yaml                    # Single-pass training (~15,870 steps, ATOM teacher)
  smoke_test.yaml               # 5-step e2e pipeline validation (synthetic prompts)
Dockerfile.train                # Training image based on rocm/atom:latest
Dockerfile.benchmark            # Benchmark image based on rocm/atom:latest
patches/                        # ATOM patches (embed_tokens fix, prepare_mtp_decode, mtp_stats)
run_gpt_oss_120b.sh             # In-container entrypoint (torchrun + overrides)
run_docker.sh                   # Host-side wrapper, launches container
run_with_retry.sh               # Auto-restart wrapper (HSA-fault loop + hang watchdog)
bench_eagle3_atom.py            # ATOM benchmark script
run_benchmark_atom.sh           # ATOM benchmark runner
benchmark_results/              # Benchmark output
../../lumenrl/data/make_dataset.py        # Combine UltraChat + Magpie into single JSONL
../../lumenrl/data/generate_synthetic.py  # Generate synthetic conversations from base model
```

## Reference

- [openai/gpt-oss-120b on Hugging Face](https://huggingface.co/openai/gpt-oss-120b)
- [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context)
- Sibling examples: `examples/Kimi_K25_SDDD_MI350_ATOM/`
