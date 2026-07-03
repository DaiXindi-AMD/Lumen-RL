#!/usr/bin/env python3
"""Generate synthetic conversations using the base model for Eagle3 training.

Adapted from NVIDIA Model-Optimizer server_generate.py (Apache 2.0):
    https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/scripts/server_generate.py

Per the NVIDIA gpt-oss-120b-Eagle3 model card
(https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context):

    "only prompts from the datasets were used for data synthesis
     (the original responses from GPT were not used for data synthesis)"

This script takes prompt-only JSONL produced by make_dataset.py, sends each
prompt to a running OpenAI-compatible server (e.g. ATOM), and writes full
conversations (prompt + generated response) in the format expected by our
trainer (``chat_template: hf-generation``).

Usage:
    # 1. Start ATOM server with the base model (TP=8, no draft):
    #    python3 -m atom.entrypoints.openai.api_server \\
    #        --model /dev/shm/gpt-oss-120b --kv_cache_dtype fp8 -tp 8 \\
    #        --max-model-len 4096 --host 0.0.0.0 --port 8000 --level 0

    # 2. Generate synthetic data:
    python3 -m lumenrl.data.generate_synthetic \\
        --data-path /dev/shm/gpt_oss_120b_dataset/train.jsonl \\
        --output-path /dev/shm/gpt_oss_120b_dataset/train_synthetic.jsonl \\
        --url http://localhost:8000/v1

    # 3. Train with synthetic data:
    #    Update train.yaml: dataset: /dev/shm/gpt_oss_120b_dataset/train_synthetic.jsonl
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

write_lock = threading.Lock()


def generate_conversation(
    client: OpenAI,
    messages: list[dict],
    idx: int,
    model: str,
    max_tokens: int,
    temperature: float,
    output_path: str,
) -> None:
    """Generate a single conversation and append to output file.

    For single-turn: sends one user message, gets one assistant response.
    For multi-turn: sends user turns one at a time, appending each generated
    response before the next user turn (matching Model-Optimizer behavior).
    """
    try:
        user_turns = [m for m in messages if m.get("role") == "user"]
        if not user_turns:
            return

        output_messages: list[dict] = []

        for user_msg in user_turns:
            output_messages.append({"role": "user", "content": user_msg["content"]})
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=output_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if response.choices[0].finish_reason == "length":
                    break
                content = response.choices[0].message.content
                if content is None:
                    break
                output_messages.append({
                    "role": "assistant",
                    "content": content.strip(),
                })
            except Exception as e:
                logger.warning("idx=%d: API error: %s", idx, e)
                break

        if len(output_messages) < 2:
            return

        record = {"conversation_id": idx, "messages": output_messages}
        with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("idx=%d: unexpected error: %s", idx, e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversations for Eagle3 training"
    )
    parser.add_argument(
        "--data-path", required=True,
        help="Input JSONL with prompt-only messages (from make_dataset.py)",
    )
    parser.add_argument(
        "--output-path", required=True,
        help="Output JSONL with full conversations (prompt + generated response)",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--api-key", default="token-abc123", help="API key")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name for API")
    parser.add_argument(
        "--num-threads", type=int, default=256,
        help="Concurrent request threads (default: 256, matching Model-Optimizer)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 = greedy, matching Model-Optimizer)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens per generation (default: 2048, matching training_seq_len)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N prompts (for testing)",
    )
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = [json.loads(line) for line in f]
    logger.info("Loaded %d prompts from %s", len(data), args.data_path)

    if args.limit:
        data = data[:args.limit]
        logger.info("Limited to first %d prompts", args.limit)

    finished_ids: set[int] = set()
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    cid = record.get("conversation_id", -1)
                    if cid >= 0:
                        finished_ids.add(cid)
                except json.JSONDecodeError:
                    continue
        logger.info("Resuming: %d conversations already generated", len(finished_ids))

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    client = OpenAI(base_url=args.url, api_key=args.api_key)

    pending = [
        (idx, sample["messages"])
        for idx, sample in enumerate(data)
        if idx not in finished_ids
    ]
    logger.info(
        "Generating %d conversations (%d skipped) with %d threads",
        len(pending), len(finished_ids), args.num_threads,
    )

    import tqdm

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for idx, messages in pending:
            future = executor.submit(
                generate_conversation,
                client, messages, idx, args.model,
                args.max_tokens, args.temperature, args.output_path,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Generating",
        ):
            future.result()

    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            total = sum(1 for _ in f)
        logger.info("Done: %d conversations written to %s", total, args.output_path)
    else:
        logger.warning("No output generated")


if __name__ == "__main__":
    main()
