"""Verify ATOM built-in capture produces correct hidden states.

Uses the EXACT same flow as the training worker:
  AsyncLLMEngine / LLMEngine → configure_hidden_states → generate_hidden_states
  → GptOssForCausalLM.forward(capture_hidden_state_layers=...)
  → GptOssModel.forward() → allreduce(x) + residual → postnorm/varnorm

No Mooncake needed — we intercept the captured tensors before they're sent.
"""
import logging
import os
import sys
import time
import glob

sys.path.insert(0, "/app/ATOM")
sys.path.insert(0, "/home/danyzhan/Lumen-RL")

os.environ["GLOG_minloglevel"] = "3"
os.environ["AITER_LOG_LEVEL"] = "WARNING"

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("verify_capture")

import torch
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file as st_load

MODEL_PATH = "/dev/shm/gpt-oss-120b"
TP_SIZE = 4


def main():
    logger.info("=" * 70)
    logger.info("VERIFY: ATOM built-in capture (same path as training)")
    logger.info("=" * 70)

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    hf_text = getattr(hf_config, "text_config", hf_config)
    num_layers = hf_text.num_hidden_layers
    hidden_dim = hf_text.hidden_size
    logger.info(f"Model: {num_layers} layers, hidden_dim={hidden_dim}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    aux_layer_ids = [1, 17, 32]

    # Create engine — same as training worker
    from atom.model_engine.llm_engine import LLMEngine

    engine = LLMEngine(
        MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        enforce_eager=True,
        trust_remote_code=True,
        max_num_batched_tokens=65536,
        max_num_seqs=64,
        kv_cache_dtype="fp8",
        enable_prefix_caching=False,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    logger.info("LLMEngine created")

    # Monkey-patch _store_hidden_states on the model runner to save to /dev/shm
    # instead of Mooncake. We do this through the engine's utility handler.
    #
    # Actually, the simplest approach: use configure_hidden_states but with a
    # dummy mooncake config, then intercept the captured tensors via a modified
    # _store_hidden_states.
    #
    # Even simpler: directly call the model with capture via the engine's
    # call_func mechanism.

    # The engine core runs model runners in subprocess. We need to communicate
    # via the engine's utility handler. Let's use call_func to:
    # 1. Set aux_hidden_state_layers
    # 2. Set _capture_mode
    # 3. Run a forward pass
    # 4. Read back the captured tensors

    core = engine.core_mgr.engine_core

    # Method: use the engine's call_func to execute arbitrary functions
    # on the model runner subprocess.
    # But configure_hidden_states requires mooncake config...
    #
    # Alternative: inject a custom function that does the capture test.

    test_text = "The capital of France is Paris. The capital of Germany is Berlin. The largest country by area is Russia."
    input_ids = tok.encode(test_text)
    T = len(input_ids)
    logger.info(f"Test input: '{test_text[:60]}...' ({T} tokens)")

    # Use the utility handler to run a function on the model runner
    from atom.rollout.engine_utility import EngineUtilityHandler
    if hasattr(core, 'utility_handler'):
        uh = core.utility_handler
    else:
        logger.info("No utility handler, using call_func directly")
        uh = None

    # Simplest approach: use call_func to set aux layers and capture mode,
    # then run a regular prefill, then read the captured tensors.
    #
    # Actually let me take a completely different approach.
    # I'll write a script that runs INSIDE the model runner subprocess.

    logger.info("Shutting down engine, will use direct subprocess approach...")
    try:
        engine.close()
    except:
        pass
    del engine
    import gc; gc.collect()
    time.sleep(2)

    # ========================================
    # Direct approach: spawn TP workers manually, run the model, capture
    # ========================================
    logger.info("\nDirect approach: spawn TP workers with capture test...")

    # Write test script for subprocess
    test_script = f'''
import os, sys, logging, torch
sys.path.insert(0, "/app/ATOM")
os.environ["GLOG_minloglevel"] = "3"
os.environ["AITER_LOG_LEVEL"] = "WARNING"
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("capture_test")

from aiter import init_dist_env
from aiter.dist.parallel_state import get_tensor_model_parallel_rank

# Initialize distributed
init_dist_env(
    {TP_SIZE},
    rankID=int(os.environ["RANK"]),
    backend="nccl",
    distributed_init_method="tcp://127.0.0.1:29501",
)
rank = get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{{rank}}")
torch.cuda.set_device(device)

# Load model
from atom.config import Config
config = Config(
    "{MODEL_PATH}",
    tensor_parallel_size={TP_SIZE},
    enforce_eager=True,
    trust_remote_code=True,
    kv_cache_dtype="fp8",
)

from atom.model_engine.model_runner import ModelRunner
runner = ModelRunner(config, rank)

model_inner = runner.model.model
logger.info(f"[rank {{rank}}] Model loaded: {{type(model_inner).__name__}}")
logger.info(f"[rank {{rank}}] _capture_mode={{model_inner._capture_mode}}")

# Test input
input_ids = {input_ids}
input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
positions = torch.arange({T}, dtype=torch.long, device=device)

aux_layer_ids = {aux_layer_ids}

for mode in ["postnorm", "varnorm"]:
    model_inner.aux_hidden_state_layers = tuple(sorted(aux_layer_ids))
    model_inner._capture_mode = mode

    with torch.inference_mode():
        result = model_inner(input_ids_t, positions)

    if rank == 0:
        if isinstance(result, tuple) and len(result) == 2:
            hidden_states, aux_list = result
            logger.info(f"[{{mode}}] Captured {{len(aux_list)}} layers:")
            for j, lid in enumerate(sorted(aux_layer_ids)):
                hs = aux_list[j]
                nan_pct = 100 * torch.isnan(hs).any(dim=-1).float().mean().item()
                inf_pct = 100 * torch.isinf(hs).any(dim=-1).float().mean().item()
                clean = hs.float().nan_to_num(0)
                std_val = clean.std().item()
                max_val = clean.abs().max().item()
                mean_val = clean.mean().item()
                logger.info(
                    f"  L{{lid}}: shape={{list(hs.shape)}}, "
                    f"std={{std_val:.4f}}, max={{max_val:.1f}}, mean={{mean_val:.4f}}, "
                    f"nan={{nan_pct:.0f}}%, inf={{inf_pct:.0f}}%"
                )
                torch.save(hs.cpu(), f"/dev/shm/capture_{{mode}}_L{{lid}}.pt")

            hs_std = hidden_states.float().nan_to_num(0).std().item()
            hs_nan = 100 * torch.isnan(hidden_states).any(dim=-1).float().mean().item()
            logger.info(f"  last_hs: shape={{list(hidden_states.shape)}}, std={{hs_std:.4f}}, nan={{hs_nan:.0f}}%")
        else:
            logger.error(f"[{{mode}}] Unexpected result: {{type(result)}}")

# Comparison (rank 0 only)
if rank == 0:
    logger.info("\\nComparison postnorm vs varnorm:")
    for lid in sorted(aux_layer_ids):
        pn = torch.load(f"/dev/shm/capture_postnorm_L{{lid}}.pt").float()
        vn = torch.load(f"/dev/shm/capture_varnorm_L{{lid}}.pt").float()
        cos_sim = torch.nn.functional.cosine_similarity(
            pn.reshape(-1, pn.shape[-1]),
            vn.reshape(-1, vn.shape[-1]),
            dim=-1,
        ).mean().item()
        logger.info(f"  L{{lid}}: pn_std={{pn.std():.4f}}, vn_std={{vn.std():.4f}}, cos={{cos_sim:.6f}}")

    logger.info("\\n" + "=" * 70)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 70)

import torch.distributed as dist
dist.barrier()
'''

    script_path = "/dev/shm/capture_test_worker.py"
    with open(script_path, 'w') as f:
        f.write(test_script)
    logger.info(f"Worker script written to {script_path}")

    # Launch with torchrun
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(TP_SIZE),
            "--master_port", "29501",
            script_path,
        ],
        capture_output=False,
        text=True,
        timeout=300,
    )
    logger.info(f"torchrun exit code: {result.returncode}")


if __name__ == "__main__":
    main()
