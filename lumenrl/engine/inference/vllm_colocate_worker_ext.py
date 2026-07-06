"""vLLM worker extension for colocated ZMQ CUDA-IPC weight sync.

Vendored + trimmed from verl's ``vLLMColocateWorkerExtension``
(``verl/workers/rollout/vllm_rollout/utils.py``). vLLM instantiates the worker
with this class mixed in via ``worker_extension_cls`` so that, no matter the
underlying worker class (V0/V1), the rollout worker exposes
``update_weights_from_ipc``. LoRA / FP8 / QAT branches are dropped; LumenRL's
verl-aligned BF16 path only needs standard ``model.load_weights``.

The ZMQ socket path is keyed by ``{ray_job_id, replica_rank, local_rank}`` so it
matches the training-side ``BucketedWeightSender`` regardless of per-worker
``CUDA_VISIBLE_DEVICES`` differences and is unique across replicas / jobs.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import signal

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "WARN"))


def set_death_signal() -> None:
    """Kill this process when its parent (the server actor) dies (Linux only)."""
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except Exception:
        pass


class vLLMColocateWorkerExtension:
    """Mixed into the vLLM worker; receives IPC weight buckets and loads them."""

    def __new__(cls, **kwargs):
        set_death_signal()
        return super().__new__(cls)

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("LUMEN_REPLICA_RANK", "0")
        job_id = os.environ.get("LUMEN_RAY_JOB_ID", "0")
        return f"ipc:///tmp/lumen-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"

    def update_weights_from_ipc(self, use_shm: bool = False) -> None:
        """Receive bucketed weights over ZMQ IPC and load them into the model."""
        from vllm.platforms import current_platform

        from lumenrl.engine.inference.bucketed_weight_transfer import BucketedWeightReceiver

        if getattr(self, "device", None) is None:
            dev_type = current_platform.device_type
            self.device = torch.device(f"{dev_type}:{self.local_rank}")

        model = self.model_runner.model
        model_config = self.model_runner.vllm_config.model_config

        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: model.load_weights(weights)
        )

        # Some post-load transforms are non-idempotent; run once after all buckets.
        try:
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            process_weights_after_loading(model, model_config, self.device)
        except Exception as exc:  # pragma: no cover - best effort parity with verl
            logger.warning("process_weights_after_loading skipped: %s", exc)
