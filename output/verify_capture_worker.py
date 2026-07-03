"""Worker: verify ATOM capture code path via mock-attention forward (TP=4).

Monkey-patches the attention forward to return zeros instead of requiring KV
cache. All other ops (embedding, RMSNorm, MoE/MLP, allreduce, residual
accumulation, capture normalization) run exactly as in production.

This exercises the actual capture code in gpt_oss.py lines 400-414.
"""
import os, sys, logging, torch

sys.path.insert(0, "/app/ATOM")
os.environ["GLOG_minloglevel"] = "3"
os.environ["AITER_LOG_LEVEL"] = "WARNING"
logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
logger = logging.getLogger("capture_test")

MODEL_PATH = "/dev/shm/gpt-oss-120b"
TP_SIZE = 4

rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

from aiter import init_dist_env
from aiter.dist.utils import get_distributed_init_method

init_dist_env(
    TP_SIZE,
    rankID=rank,
    backend="nccl",
    distributed_init_method=get_distributed_init_method("127.0.0.1", 29501),
)
from aiter.dist.parallel_state import get_tensor_model_parallel_rank

tp_rank = get_tensor_model_parallel_rank()
logger.info(f"[rank {tp_rank}] device={device}")

from atom.config import Config, set_current_atom_config

config = Config(
    MODEL_PATH,
    tensor_parallel_size=TP_SIZE,
    enforce_eager=True,
    trust_remote_code=True,
    kv_cache_dtype="fp8",
    max_model_len=4096,
)
set_current_atom_config(config)

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(device)

from atom.models.gpt_oss import GptOssForCausalLM

logger.info(f"[rank {tp_rank}] Creating model...")
model = GptOssForCausalLM(atom_config=config)
torch.set_default_device(None)

logger.info(f"[rank {tp_rank}] Loading weights...")
from atom.model_loader.loader import load_model
load_model(model, MODEL_PATH, config.hf_config)
model.eval()
logger.info(f"[rank {tp_rank}] Weights loaded")

model_inner = model.model

# Monkey-patch attention to return zeros (valid data, no KV cache needed).
# We patch at the PagedAttention level (the nn.Module wrapping attention ops).
from atom.model_ops.paged_attention import PagedAttention

_orig_attn_forward = PagedAttention.forward

def mock_attn_forward(self, q, k, v, position=None, qkv=None):
    return torch.zeros_like(q)

PagedAttention.forward = mock_attn_forward
logger.info(f"[rank {tp_rank}] Attention monkey-patched (zeros)")

# Set up minimal ForwardContext
from atom.utils.forward_context import Context, ForwardContext
import atom.utils.forward_context as _fctx

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
test_text = "The capital of France is Paris. The capital of Germany is Berlin."
input_ids = tok.encode(test_text)
T = len(input_ids)
input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
positions = torch.arange(T, dtype=torch.long, device=device)
aux_layer_ids = [1, 17, 32]

dummy_ctx = Context(positions=positions, is_prefill=True, is_dummy_run=False)
_fctx._forward_context = ForwardContext(context=dummy_ctx)

if tp_rank == 0:
    logger.info(f"Test: '{test_text}' ({T} tokens)")

for mode in ["postnorm", "varnorm"]:
    model_inner.aux_hidden_state_layers = tuple(sorted(aux_layer_ids))
    model_inner._capture_mode = mode

    torch.distributed.barrier()

    with torch.inference_mode():
        result = model_inner(input_ids_t, positions)

    if tp_rank == 0:
        if isinstance(result, tuple) and len(result) == 2:
            hidden_states, aux_list = result
            logger.info(f"\n[{mode}] Captured {len(aux_list)} layers:")
            for j, lid in enumerate(sorted(aux_layer_ids)):
                hs = aux_list[j]
                nan_pct = 100 * torch.isnan(hs).any(dim=-1).float().mean().item()
                inf_pct = 100 * torch.isinf(hs).any(dim=-1).float().mean().item()
                clean = hs.float().nan_to_num(0)
                std_val = clean.std().item()
                max_val = clean.abs().max().item()
                logger.info(
                    f"  L{lid}: shape={list(hs.shape)}, "
                    f"std={std_val:.4f}, max={max_val:.1f}, "
                    f"nan={nan_pct:.0f}%, inf={inf_pct:.0f}%"
                )
                torch.save(hs.cpu(), f"/dev/shm/capture_{mode}_L{lid}.pt")

            hs_std = hidden_states.float().nan_to_num(0).std().item()
            hs_nan = 100 * torch.isnan(hidden_states).any(dim=-1).float().mean().item()
            logger.info(f"  last_hs: std={hs_std:.4f}, nan={hs_nan:.0f}%")
        else:
            logger.error(f"[{mode}] Unexpected: {type(result)}")

# Comparison
if tp_rank == 0:
    logger.info("\npostnorm vs varnorm:")
    for lid in sorted(aux_layer_ids):
        pn = torch.load(f"/dev/shm/capture_postnorm_L{lid}.pt").float()
        vn = torch.load(f"/dev/shm/capture_varnorm_L{lid}.pt").float()
        cos_sim = torch.nn.functional.cosine_similarity(
            pn.reshape(-1, pn.shape[-1]),
            vn.reshape(-1, vn.shape[-1]),
            dim=-1,
        ).mean().item()
        logger.info(
            f"  L{lid}: pn_std={pn.std():.4f}, vn_std={vn.std():.4f}, cos={cos_sim:.6f}"
        )
    logger.info("\nVERIFICATION COMPLETE")

# Restore attention
PagedAttention.forward = _orig_attn_forward

torch.distributed.barrier()
torch.distributed.destroy_process_group()
