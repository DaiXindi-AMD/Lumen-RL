#!/usr/bin/env python3
"""Patch vLLM kv_cache_utils.py to fix HiddenStatesCacheSpec grouping bug.

Bug (vLLM 0.19.1+rocm721):
  HiddenStatesCacheSpec inherits FullAttentionSpec, so
  UniformTypeKVCacheSpecs.from_specs() treats it as uniform with normal
  attention layers. When merge() is later called on the mixed group, it
  raises AssertionError.

Fix:
  Strip HiddenStatesCacheSpec entries from the spec dict BEFORE the
  uniformity checks in get_kv_cache_groups(). Process the remaining
  attention-only layers through normal grouping, then reattach hidden-state
  layers as singleton KVCacheGroupSpec entries.

Usage:
  python patch/patch_vllm_kv_cache_grouping.py

  Or from Docker:
  docker exec <container> python /root/lumenrl/examples/GPT_OSS_120b_MI355_vllm/patch/patch_vllm_kv_cache_grouping.py
"""
import sys

KV_CACHE_UTILS = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"

with open(KV_CACHE_UTILS, "r") as f:
    content = f.read()

PATCH_MARKER = "[LumenRL patch] Strip HiddenStatesCacheSpec"
if PATCH_MARKER in content:
    print(f"Already patched: {KV_CACHE_UTILS}")
    sys.exit(0)

# --- Patch 1: Strip HiddenStatesCacheSpec before uniformity checks ---

OLD_BLOCK_1 = """\
    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)"""

NEW_BLOCK_1 = """\
    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    # [LumenRL patch] Strip HiddenStatesCacheSpec before uniformity checks.
    # HiddenStatesCacheSpec inherits FullAttentionSpec, so from_specs()
    # incorrectly routes it into the uniform-type branch.
    from vllm.model_executor.models.extract_hidden_states import (
        HiddenStatesCacheSpec as _HSCS,
    )
    _hidden_specs = {
        k: v for k, v in kv_cache_spec.items()
        if isinstance(v, _HSCS)
    }
    if _hidden_specs:
        kv_cache_spec = {
            k: v for k, v in kv_cache_spec.items()
            if not isinstance(v, _HSCS)
        }

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        _groups = _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        _groups = _get_kv_cache_groups_uniform_type(uniform_spec)
    else:
        _groups = None

    if _groups is not None:
        for _name, _spec in _hidden_specs.items():
            _groups.append(KVCacheGroupSpec([_name], _spec))
        return _groups"""

if OLD_BLOCK_1 not in content:
    print("ERROR: Could not find target code block 1 (uniformity checks). "
          "The vLLM version may have changed.", file=sys.stderr)
    sys.exit(1)

content = content.replace(OLD_BLOCK_1, NEW_BLOCK_1)

# --- Patch 2: Reattach hidden_specs in fallback branches ---

OLD_BLOCK_2 = """\
        return [
            KVCacheGroupSpec(
                layer_names=names,
                kv_cache_spec=kv_cache_spec[names[0]],
            )
            for names in by_type.values()
        ]

    # Model contains multiple attention types, but KV cache of all layers
    # have the same physical memory per block per layer. Split the layers
    # into groups with the same number of layers, and thus same total page
    # size.
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)"""

NEW_BLOCK_2 = """\
        _groups = [
            KVCacheGroupSpec(
                layer_names=names,
                kv_cache_spec=kv_cache_spec[names[0]],
            )
            for names in by_type.values()
        ]
        for _name, _spec in _hidden_specs.items():
            _groups.append(KVCacheGroupSpec([_name], _spec))
        return _groups

    # Model contains multiple attention types, but KV cache of all layers
    # have the same physical memory per block per layer. Split the layers
    # into groups with the same number of layers, and thus same total page
    # size.
    _groups = _get_kv_cache_groups_uniform_page_size(kv_cache_spec)
    for _name, _spec in _hidden_specs.items():
        _groups.append(KVCacheGroupSpec([_name], _spec))
    return _groups"""

if OLD_BLOCK_2 not in content:
    print("ERROR: Could not find target code block 2 (fallback branches). "
          "The vLLM version may have changed.", file=sys.stderr)
    sys.exit(1)

content = content.replace(OLD_BLOCK_2, NEW_BLOCK_2)

with open(KV_CACHE_UTILS, "w") as f:
    f.write(content)

print(f"Patched {KV_CACHE_UTILS} successfully.")
print("  - HiddenStatesCacheSpec stripped before uniformity checks")
print("  - Reattached as singleton groups in all return paths")
