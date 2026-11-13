"""GaudiRT device vocabulary sampler integrated with SGLang's Sampler."""

import random
from typing import List

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import (
    SGLANG_RETURN_ORIGINAL_LOGPROB,
    SYNC_TOKEN_IDS_ACROSS_TP,
    Sampler,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

_HOOKS_INSTALLED = False


def install_sampling_batch_hooks() -> None:
    """Populate per-request sampling_seed on GaudiRT even without enable_deterministic_inference."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    orig_from_schedule_batch = SamplingBatchInfo.adjusted_from_schedule_batch
    orig_merge_batch = SamplingBatchInfo.adjusted_merge_batch

    def _request_seed(req) -> int:
        seed = req.sampling_params.sampling_seed
        if seed is not None:
            return int(seed)
        cached = getattr(req, "_gaudirt_ephemeral_seed", None)
        if cached is None:
            cached = random.getrandbits(62)
            req._gaudirt_ephemeral_seed = cached
        return cached

    def _gaudirt_from_schedule_batch(self, batch, vocab_size: int):
        orig_from_schedule_batch(self, batch, vocab_size)
        if (
            self.sampling_seed is None
            and str(self.device).startswith("gaudirt")
            and any(r.sampling_params.sampling_seed is not None for r in batch.reqs)
        ):
            seeds = [_request_seed(r) for r in batch.reqs]
            self.sampling_seed = torch.tensor(seeds, dtype=torch.int64).to(self.device)

    def _gaudirt_merge_batch(self, other: SamplingBatchInfo):
        orig_merge_batch(self, other)
        if not str(self.device).startswith("gaudirt"):
            return
        if self.sampling_seed is not None and other.sampling_seed is None:
            other_len = len(other)
            pad = torch.tensor(
                [random.getrandbits(62) for _ in range(other_len)],
                dtype=torch.int64,
            ).to(self.device)
            self.sampling_seed = torch.cat([self.sampling_seed, pad])
        elif self.sampling_seed is None and other.sampling_seed is not None:
            other_len = len(other.sampling_seed)
            self_len = len(self) - other_len
            pad = torch.tensor(
                [random.getrandbits(62) for _ in range(self_len)],
                dtype=torch.int64,
            ).to(self.device)
            self.sampling_seed = torch.cat([pad, other.sampling_seed])

    SamplingBatchInfo.adjusted_from_schedule_batch = _gaudirt_from_schedule_batch
    SamplingBatchInfo.adjusted_merge_batch = _gaudirt_merge_batch
    _HOOKS_INSTALLED = True


class GaudiRTSampler(Sampler):
    def __init__(self):
        super().__init__()
        self._tp_coord = (
            get_parallel().attn_tp_group if is_dp_attention_enabled() else get_tp_group()
        )

    def _sync_token_ids_across_tp(
        self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo
    ):
        # Broadcast from TP rank 0 instead of int64 all_reduce(MIN):
        # 1) Process-local RNG differs across TP ranks when sampling_seed is None.
        # 2) Broadcast preserves rank-0's exact distribution (MIN skews independent draws)
        #    and works natively on int64 where Gaudi2 HCCL arithmetic reductions reject int64.
        if self._tp_coord.world_size > 1 and (
            not sampling_info.is_all_greedy
            or SYNC_TOKEN_IDS_ACROSS_TP
            or bool(sampling_info.grammars)
        ):
            self._tp_coord.broadcast(batch_next_token_ids, src=0)

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits_output.next_token_logits
        if logits.device.type != "gaudirt":
            return super().forward(
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
            )

        logits = self._preprocess_logits(logits, sampling_info)

        if sampling_info.is_all_greedy:
            batch_next_token_ids = torch.argmax(logits, -1)
            logprobs = (
                torch.nn.functional.log_softmax(logits, dim=-1)
                if return_logprob
                else None
            )
        else:
            logprobs = None
            if return_logprob:
                scaled_logits = (
                    logits
                    if SGLANG_RETURN_ORIGINAL_LOGPROB
                    else logits / sampling_info.temperatures
                )
                logprobs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)
            batch_next_token_ids = torch.ops.gaudirt.sample_from_logits.default(
                logits,
                sampling_info.temperatures,
                sampling_info.top_ps,
                sampling_info.top_ks,
                sampling_info.min_ps,
                sampling_info.sampling_seed,
                positions,
                True,
            )

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)

        if return_logprob:
            logprob_result = self.output_logprob_processor.compute_logprobs(
                logprobs,
                top_logprobs_nums,
                token_ids_logprobs,
                batch_next_token_ids,
            )
            logprob_result.write_output_to(logits_output)

        return batch_next_token_ids
