"""Patch: inject prepare_mtp_decode into AiterAttentionMetadataBuilder.

Eagle3 speculation calls attn_metadata_builder.prepare_mtp_decode() on every
draft step. Only MLA/GDN attention builders had this method; standard GQA
(AiterAttentionMetadataBuilder) did not, causing AttributeError for non-MLA
models like gpt-oss-120b.
"""
import re

TARGET = "/app/ATOM/atom/model_ops/attentions/aiter_attention.py"

PATCH = '''
    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: "torch.Tensor",
        only_update: bool = False,
        num_reject_tokens=None,
    ):
        var = self.model_runner.forward_vars
        kv_indptr = var["kv_indptr"].gpu[: bs + 1]
        kv_indices_generate_triton(
            var["block_tables"].gpu[:bs],
            var["kv_indices"].gpu,
            kv_indptr,
            self.block_ratio,
            max_seqlen_k,
        )

        result = {}
        if self.block_size == 1024:
            result = self.set_aiter_persistent_worker_buffers(bs)
        return result

'''

with open(TARGET) as f:
    src = f.read()

if "prepare_mtp_decode" in src:
    print("prepare_mtp_decode already exists, skipping patch")
else:
    # Insert before _prepare_ubatch_decode
    anchor = "    def _prepare_ubatch_decode("
    if anchor not in src:
        # Fallback: insert before the last method boundary before class end
        raise RuntimeError(f"Cannot find anchor '{anchor}' in {TARGET}")
    src = src.replace(anchor, PATCH + anchor)
    with open(TARGET, "w") as f:
        f.write(src)
    print(f"Patched {TARGET}: added prepare_mtp_decode")
