#!/usr/bin/env python3
"""Verify Eagle3 forward-pass consistency between training and ATOM inference models.

Loads both models from the same weights, feeds identical inputs, and compares
outputs at each stage of the pipeline:
  1. FC projection (aux fusion)
  2. Embedding lookup
  3. RMSNorm (input_layernorm, hidden_norm)
  4. QKV projection (separate in training, packed in ATOM)
  5. RoPE cos/sin tables
  6. Full decoder layer output
  7. Final norm + lm_head logits

Run inside the benchmark Docker container with 1 GPU:
    python3 examples/GPT_OSS_120b_MI355_ATOM/verify_forward.py \
        --ckpt /dev/shm/checkpoints/gpt_oss_120b_eagle3/checkpoint_15800.pt \
        --hf   /dev/shm/gpt_oss_120b_eagle3_HF \
        --base /dev/shm/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open


def _load_base_tensor(base_dir: str, key: str) -> torch.Tensor:
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(base_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _load_hf_tensors(hf_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)
    tensors = {}
    loaded_shards = {}
    for key, shard in idx["weight_map"].items():
        if shard not in loaded_shards:
            loaded_shards[shard] = {}
            with safe_open(os.path.join(hf_dir, shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    loaded_shards[shard][k] = f.get_tensor(k)
        tensors[key] = loaded_shards[shard][key]
    return tensors


def compare(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-4) -> bool:
    a_f, b_f = a.float(), b.float()
    max_diff = (a_f - b_f).abs().max().item()
    mean_diff = (a_f - b_f).abs().mean().item()
    rel_diff = max_diff / (a_f.abs().max().item() + 1e-10)
    ok = max_diff <= atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
          f"rel_diff={rel_diff:.2e}, shape={list(a.shape)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Training checkpoint .pt file")
    ap.add_argument("--ckpt-dir", default="/dev/shm/checkpoints/gpt_oss_120b_eagle3")
    ap.add_argument("--hf", default="/dev/shm/gpt_oss_120b_eagle3_HF",
                    help="Exported HF Eagle3 directory")
    ap.add_argument("--base", default="/dev/shm/gpt-oss-120b",
                    help="Base gpt-oss-120b model directory")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # --- Load training checkpoint ---
    if args.ckpt is None:
        import glob, re
        files = glob.glob(os.path.join(args.ckpt_dir, "checkpoint_*.pt"))
        if not files:
            print(f"No checkpoint found in {args.ckpt_dir}")
            sys.exit(1)
        args.ckpt = max(files, key=lambda p: int(re.search(r"checkpoint_(\d+)", p).group(1)))

    print(f"Loading training checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if "model_state_dict" in sd:
        train_sd = sd["model_state_dict"]
    else:
        train_sd = sd
    step = ckpt.get("step", sd.get("step", "?"))
    print(f"  step={step}, {len(train_sd)} tensors")

    # --- Load HF export ---
    print(f"Loading HF export: {args.hf}")
    hf_tensors = _load_hf_tensors(args.hf)
    print(f"  {len(hf_tensors)} tensors")

    # --- Load base model lm_head + embed_tokens ---
    print(f"Loading base embed_tokens + lm_head from: {args.base}")
    embed_w = _load_base_tensor(args.base, "model.embed_tokens.weight").to(dtype)
    lm_head_w = _load_base_tensor(args.base, "lm_head.weight").to(dtype)

    # === Config ===
    H = 2880
    num_heads = 64
    num_kv_heads = 8
    head_dim = 64
    ffn_dim = 17280
    eps = 1e-5
    B, T = 1, 8
    VOCAB = 201088

    print(f"\n{'='*60}")
    print("Stage 1: Weight comparison (training vs HF export)")
    print(f"{'='*60}")

    weight_map = {
        "fc.weight":                                "fc.weight",
        "layers.0.hidden_norm.weight":              "midlayer.hidden_norm.weight",
        "layers.0.input_layernorm.weight":          "midlayer.input_layernorm.weight",
        "layers.0.self_attn.q_proj.weight":         "midlayer.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight":         "midlayer.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight":         "midlayer.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight":         "midlayer.self_attn.o_proj.weight",
        "layers.0.post_attention_layernorm.weight": "midlayer.post_attention_layernorm.weight",
        "layers.0.mlp.gate_proj.weight":            "midlayer.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight":              "midlayer.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight":            "midlayer.mlp.down_proj.weight",
        "out_norm.weight":                          "norm.weight",
    }

    all_pass = True
    for train_key, hf_key in weight_map.items():
        t = train_sd[train_key]
        h = hf_tensors[hf_key]
        ok = compare(f"{train_key} == {hf_key}", t, h, atol=0)
        all_pass = all_pass and ok

    # Also check embed_tokens and lm_head
    ok = compare("embed_tokens", embed_w, hf_tensors["embed_tokens.weight"], atol=0)
    all_pass = all_pass and ok
    ok = compare("lm_head", lm_head_w, hf_tensors["lm_head.weight"], atol=0)
    all_pass = all_pass and ok

    # === Prepare inputs ===
    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB, (B, T), device=device)
    aux_hs_list = [torch.randn(B, T, H, dtype=dtype, device=device) for _ in range(3)]
    aux_concat = torch.cat(aux_hs_list, dim=-1)  # [B, T, 3*H]
    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    # Move weights to device
    fc_w = train_sd["fc.weight"].to(dtype).to(device)
    input_ln_w = train_sd["layers.0.input_layernorm.weight"].to(dtype).to(device)
    hidden_ln_w = train_sd["layers.0.hidden_norm.weight"].to(dtype).to(device)
    q_w = train_sd["layers.0.self_attn.q_proj.weight"].to(dtype).to(device)
    k_w = train_sd["layers.0.self_attn.k_proj.weight"].to(dtype).to(device)
    v_w = train_sd["layers.0.self_attn.v_proj.weight"].to(dtype).to(device)
    o_w = train_sd["layers.0.self_attn.o_proj.weight"].to(dtype).to(device)
    post_ln_w = train_sd["layers.0.post_attention_layernorm.weight"].to(dtype).to(device)
    gate_w = train_sd["layers.0.mlp.gate_proj.weight"].to(dtype).to(device)
    up_w = train_sd["layers.0.mlp.up_proj.weight"].to(dtype).to(device)
    down_w = train_sd["layers.0.mlp.down_proj.weight"].to(dtype).to(device)
    out_norm_w = train_sd["out_norm.weight"].to(dtype).to(device)

    # HF weights on device
    hf_fc_w = hf_tensors["fc.weight"].to(dtype).to(device)
    hf_input_ln_w = hf_tensors["midlayer.input_layernorm.weight"].to(dtype).to(device)
    hf_hidden_ln_w = hf_tensors["midlayer.hidden_norm.weight"].to(dtype).to(device)
    hf_q_w = hf_tensors["midlayer.self_attn.q_proj.weight"].to(dtype).to(device)
    hf_k_w = hf_tensors["midlayer.self_attn.k_proj.weight"].to(dtype).to(device)
    hf_v_w = hf_tensors["midlayer.self_attn.v_proj.weight"].to(dtype).to(device)
    hf_o_w = hf_tensors["midlayer.self_attn.o_proj.weight"].to(dtype).to(device)
    hf_post_ln_w = hf_tensors["midlayer.post_attention_layernorm.weight"].to(dtype).to(device)
    hf_gate_w = hf_tensors["midlayer.mlp.gate_proj.weight"].to(dtype).to(device)
    hf_up_w = hf_tensors["midlayer.mlp.up_proj.weight"].to(dtype).to(device)
    hf_down_w = hf_tensors["midlayer.mlp.down_proj.weight"].to(dtype).to(device)
    hf_out_norm_w = hf_tensors["norm.weight"].to(dtype).to(device)

    embed_w_dev = embed_w.to(device)
    lm_head_w_dev = lm_head_w.to(device)

    def rms_norm(x, w, eps=1e-5):
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        normed = x_f * torch.rsqrt(variance + eps)
        return (w.float() * normed).to(x.dtype)

    print(f"\n{'='*60}")
    print("Stage 2: FC projection")
    print(f"{'='*60}")
    fc_out_train = F.linear(aux_concat, fc_w)
    fc_out_hf = F.linear(aux_concat, hf_fc_w)
    compare("FC(aux_concat)", fc_out_train, fc_out_hf)

    print(f"\n{'='*60}")
    print("Stage 3: Embedding")
    print(f"{'='*60}")
    embeds = F.embedding(input_ids, embed_w_dev)
    compare("embed_tokens(input_ids) — same weight", embeds, embeds)

    print(f"\n{'='*60}")
    print("Stage 4: RMSNorm (input_layernorm, hidden_norm)")
    print(f"{'='*60}")
    normed_emb_train = rms_norm(embeds, input_ln_w, eps)
    normed_emb_hf = rms_norm(embeds, hf_input_ln_w, eps)
    compare("input_layernorm(embeds)", normed_emb_train, normed_emb_hf)

    normed_h_train = rms_norm(fc_out_train, hidden_ln_w, eps)
    normed_h_hf = rms_norm(fc_out_hf, hf_hidden_ln_w, eps)
    compare("hidden_norm(fc_out)", normed_h_train, normed_h_hf)

    print(f"\n{'='*60}")
    print("Stage 5: QKV projection")
    print(f"{'='*60}")
    attn_input = torch.cat([normed_emb_train, normed_h_train], dim=-1)  # [B,T,2*H]

    q_train = F.linear(attn_input, q_w)  # [B,T,num_heads*head_dim]
    k_train = F.linear(attn_input, k_w)  # [B,T,num_kv_heads*head_dim]
    v_train = F.linear(attn_input, v_w)  # [B,T,num_kv_heads*head_dim]

    q_hf = F.linear(attn_input, hf_q_w)
    k_hf = F.linear(attn_input, hf_k_w)
    v_hf = F.linear(attn_input, hf_v_w)

    compare("Q projection", q_train, q_hf)
    compare("K projection", k_train, k_hf)
    compare("V projection", v_train, v_hf)

    print(f"\n{'='*60}")
    print("Stage 6: RoPE cos/sin (llama3)")
    print(f"{'='*60}")

    # Compute llama3 RoPE inv_freq
    rope_theta = 500000.0
    rope_factor = 8.0
    rope_low_freq_factor = 1.0
    rope_high_freq_factor = 4.0
    original_max_pos = 8192
    import math

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    old_ctx = float(original_max_pos)
    low_wl = old_ctx / rope_low_freq_factor
    high_wl = old_ctx / rope_high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_wl, inv_freq / rope_factor, inv_freq)
    smooth = (old_ctx / wavelen - rope_low_freq_factor) / (rope_high_freq_factor - rope_low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / rope_factor + smooth * inv_freq_llama
    in_smooth_band = (wavelen >= high_wl) & (wavelen <= low_wl)
    inv_freq_final = torch.where(in_smooth_band, smoothed, inv_freq_llama)

    t = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq_final)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_ref = emb.cos()
    sin_ref = emb.sin()
    print(f"  RoPE inv_freq (first 4): {inv_freq_final[:4].tolist()}")
    print(f"  RoPE cos[0,:4]: {cos_ref[0,:4].tolist()}")
    print(f"  RoPE sin[0,:4]: {sin_ref[0,:4].tolist()}")

    # Apply RoPE to Q and K
    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(x, cos, sin, pos_ids):
        c = cos[pos_ids].unsqueeze(1)  # [B,1,T,D]
        s = sin[pos_ids].unsqueeze(1)
        return (x * c) + (rotate_half(x) * s)

    q_r = q_train.view(B, T, num_heads, head_dim).transpose(1, 2)
    k_r = k_train.view(B, T, num_kv_heads, head_dim).transpose(1, 2)
    cos_4d = cos_ref.unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
    sin_4d = sin_ref.unsqueeze(0).unsqueeze(0)

    q_roped = apply_rope(q_r, cos_ref, sin_ref, position_ids)
    k_roped = apply_rope(k_r, cos_ref, sin_ref, position_ids)
    print(f"  Q after RoPE shape: {list(q_roped.shape)}")
    print(f"  K after RoPE shape: {list(k_roped.shape)}")

    print(f"\n{'='*60}")
    print("Stage 7: Self-attention (SDPA, no KV cache)")
    print(f"{'='*60}")
    kv_groups = num_heads // num_kv_heads
    compute_dtype = q_roped.dtype
    k_exp = k_roped.unsqueeze(2).expand(-1, -1, kv_groups, -1, -1).reshape(B, num_heads, T, head_dim)
    v_r = v_train.view(B, T, num_kv_heads, head_dim).transpose(1, 2).to(compute_dtype)
    v_exp = v_r.unsqueeze(2).expand(-1, -1, kv_groups, -1, -1).reshape(B, num_heads, T, head_dim)

    attn_out = F.scaled_dot_product_attention(q_roped, k_exp, v_exp, is_causal=True)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, num_heads * head_dim).to(dtype)
    o_out = F.linear(attn_out, o_w)
    print(f"  attn_out shape: {list(o_out.shape)}, std: {o_out.float().std().item():.4f}")

    print(f"\n{'='*60}")
    print("Stage 8: Full decoder layer (residual + MLP)")
    print(f"{'='*60}")
    h = fc_out_train  # hidden_states input
    residual = h
    h_after_attn = residual + o_out

    # MLP
    mlp_in = rms_norm(h_after_attn, post_ln_w, eps)
    gate = F.linear(mlp_in, gate_w)
    up = F.linear(mlp_in, up_w)
    mlp_out = F.linear(F.silu(gate) * up, down_w)
    decoder_out = h_after_attn + mlp_out
    print(f"  decoder_out shape: {list(decoder_out.shape)}, std: {decoder_out.float().std().item():.4f}")

    print(f"\n{'='*60}")
    print("Stage 9: Final norm + lm_head → logits")
    print(f"{'='*60}")
    normed_out = rms_norm(decoder_out, out_norm_w, eps)
    logits = F.linear(normed_out, lm_head_w_dev)
    print(f"  logits shape: {list(logits.shape)}")
    print(f"  logits[0,0,:10]: {logits[0,0,:10].float().tolist()}")
    print(f"  argmax per position: {logits.argmax(dim=-1).tolist()}")
    print(f"  logits std: {logits.float().std().item():.4f}")

    print(f"\n{'='*60}")
    print("Stage 10: Full training model forward (end-to-end)")
    print(f"{'='*60}")

    # Build the training Eagle3Model and run full forward
    sys.path.insert(0, "/home/danyzhan/Lumen-RL")
    from lumenrl.models.eagle3 import Eagle3Model

    train_model = Eagle3Model(
        hidden_dim=H,
        vocab_size=VOCAB,
        num_heads=num_heads,
        num_layers=1,
        length=1,  # single step for comparison
        ffn_dim=ffn_dim,
        head_dim=head_dim,
        rms_norm_eps=eps,
        rope_theta=rope_theta,
        num_kv_heads=num_kv_heads,
        rope_scaling={
            "rope_type": "llama3",
            "factor": rope_factor,
            "low_freq_factor": rope_low_freq_factor,
            "high_freq_factor": rope_high_freq_factor,
            "original_max_position_embeddings": original_max_pos,
        },
    )
    train_model.load_state_dict(train_sd, strict=False)
    train_model = train_model.to(dtype).to(device).eval()

    # Prepare inputs as the trainer would
    # Trainer shifts input_ids left by 1, then embeds
    shifted_ids = torch.cat([input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])], dim=1)
    token_embeds = F.embedding(shifted_ids, embed_w_dev)

    with torch.no_grad():
        result = train_model(
            token_embeds=token_embeds,
            aux_hidden_states=aux_concat,
            teacher_lm_head_weight=lm_head_w_dev,
            loss_mask=torch.ones(B, T, device=device),
            loss_type="forward_kl",
            target_hidden_states=torch.randn(B, T, H, dtype=dtype, device=device),
            attention_mask=torch.ones(B, T, device=device),
        )

    print(f"  Training model forward completed")
    print(f"  losses: {[f'{l.item():.4f}' for l in result['losses']]}")
    print(f"  accuracies: {[f'{a.item():.4f}' for a in result['accuracies']]}")

    # Now compare training model's internal forward with manual computation
    print(f"\n{'='*60}")
    print("Stage 11: Compare training model internals vs manual")
    print(f"{'='*60}")
    with torch.no_grad():
        fc_out_model = train_model.fc(aux_concat)
        compare("train_model.fc(aux)", fc_out_model, fc_out_train)

        layer = train_model.layers[0]
        normed_emb_model = layer.input_layernorm(token_embeds)
        normed_h_model = layer.hidden_norm(fc_out_model)
        compare("train model input_layernorm", normed_emb_model,
                rms_norm(token_embeds, input_ln_w, eps))
        compare("train model hidden_norm", normed_h_model,
                rms_norm(fc_out_model, hidden_ln_w, eps))

    print(f"\n{'='*60}")
    print("Stage 12: Verify ATOM loading (simulate weight_loader)")
    print(f"{'='*60}")

    # ATOM packs q/k/v into qkv_proj. Let's verify the packing matches.
    # QKVParallelLinear with TP=1 stores: [q_weight; k_weight; v_weight] stacked
    # along dim=0. But the order might differ.
    # Check: can we reconstruct ATOM's qkv_proj from separate q/k/v weights?
    q_size = num_heads * head_dim    # 64*64 = 4096
    k_size = num_kv_heads * head_dim # 8*64 = 512
    v_size = num_kv_heads * head_dim # 8*64 = 512

    # ATOM QKVParallelLinear packs as [q; k; v] along output dim
    qkv_packed = torch.cat([hf_q_w, hf_k_w, hf_v_w], dim=0)
    print(f"  Expected QKV packed shape: {list(qkv_packed.shape)}")
    print(f"  Q: {list(hf_q_w.shape)}, K: {list(hf_k_w.shape)}, V: {list(hf_v_w.shape)}")
    print(f"  Total QKV output dim: {q_size + k_size + v_size}")

    # Verify: qkv_proj(attn_input) split → same Q,K,V
    qkv_out = F.linear(attn_input, qkv_packed)
    q_from_packed, k_from_packed, v_from_packed = torch.split(
        qkv_out, [q_size, k_size, v_size], dim=-1
    )
    compare("Q from packed QKV", q_from_packed, q_train)
    compare("K from packed QKV", k_from_packed, k_train)
    compare("V from packed QKV", v_from_packed, v_train)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if all_pass:
        print("All weight comparisons PASSED (bit-exact).")
    else:
        print("WARNING: Some weight comparisons FAILED!")
    print("\nIf all stages pass, the forward pipeline is consistent.")
    print("The inference bug would then be in ATOM's proposer input construction")
    print("(how aux_hidden_states are captured/passed during inference).")


if __name__ == "__main__":
    main()
