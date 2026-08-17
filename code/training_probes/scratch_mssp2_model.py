#!/usr/bin/env python3
"""Fully-scratch, GRU-free two-pass draft model on SpecForge's DFlash spine.

The draft transformer is initialized from a config, never from a checkpoint.
Pass 1 predicts every slot in parallel.  Pass 2 reuses the *same* transformer
weights and consumes a straight-through parallel candidate block.  There is no
recurrent module and no teacher forcing across candidate positions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDFlashModel,
    compute_accept_len,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    DynamicCache,
    extract_context_feature,
    sample,
)


class ScratchMSSP2DraftModel(DFlashDraftModel):
    """Randomly initialized shared-weight two-pass DFlash-family drafter."""

    def __init__(self, config) -> None:
        method = getattr(config, "dflash_config", None) or {}
        method["projector_type"] = "scratch_mssp2"
        config.dflash_config = method
        self.parallel_topk = int(method.get("parallel_topk", 32))
        self.parallel_temperature = float(method.get("parallel_temperature", 1.0))
        if self.parallel_topk <= 0 or self.parallel_temperature <= 0:
            raise ValueError("parallel_topk and parallel_temperature must be positive")
        super().__init__(config)

    def parallel_candidate_embeddings(
        self,
        logits: torch.Tensor,
        embed_tokens: nn.Module,
        *,
        straight_through: bool,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Return hard-forward candidate embeddings with soft top-k gradients."""

        topk = min(self.parallel_topk, logits.shape[-1])
        values, ids = torch.topk(logits, k=topk, dim=-1)
        hard_ids = ids[..., 0]
        hard_embeddings = embed_tokens(hard_ids)
        if not straight_through:
            return hard_embeddings, hard_ids
        probabilities = F.softmax(
            values.float() / self.parallel_temperature, dim=-1
        ).to(dtype=hard_embeddings.dtype)
        candidate_weights = embed_tokens.weight[ids]
        soft_embeddings = torch.sum(
            probabilities.unsqueeze(-1) * candidate_weights, dim=-2
        )
        # Forward value is exactly the argmax embedding used at inference;
        # gradients follow the local top-k probability simplex.
        embeddings = hard_embeddings + soft_embeddings - soft_embeddings.detach()
        return embeddings, hard_ids

    def make_second_pass_noise(
        self,
        first_logits4d: torch.Tensor,
        first_noise4d: torch.Tensor,
        embed_tokens: nn.Module,
        *,
        straight_through: bool,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Build [known anchor, parallel pass-1 candidates 0..B-2]."""

        candidates, candidate_ids = self.parallel_candidate_embeddings(
            first_logits4d, embed_tokens, straight_through=straight_through
        )
        second = first_noise4d.clone()
        second[:, :, 1:, :] = candidates[:, :, :-1, :]
        return second.flatten(1, 2), candidate_ids

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        """Inference-matched two-pass block generation without a GRU."""

        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)
        target_cache = DynamicCache()
        # The two passes share *weights*, not mutable KV state.  Each pass has
        # a different noise stream (pass 2 consumes pass-1 candidates), so it
        # must retain its own history across decode blocks.  Reusing one cache
        # here leaves pass-1 context keys in place before pass 2 and makes the
        # rotary-position span shorter than the concatenated K/V span.
        first_pass_cache = DynamicCache()
        second_pass_cache = DynamicCache()
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)
        acceptance_lengths = []
        start = num_input_tokens
        while start < max_length:
            block_ids = output_ids[:, start : start + block_size].clone()
            block_positions = position_ids[:, start : start + block_size]
            first_noise = target.model.embed_tokens(block_ids)
            draft_positions = position_ids[
                :, first_pass_cache.get_seq_length() : start + block_size
            ]
            first_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=first_noise,
                position_ids=draft_positions,
                past_key_values=first_pass_cache,
                use_cache=True,
                is_causal=False,
            )
            first_pass_cache.crop(start)
            if getattr(self, "inference_single_pass", False):
                # Pass1-only inference: draft directly from the parallel base.
                # Lossless SD correctness is preserved by target verification;
                # only the acceptance/latency trade changes (one backbone pass
                # and one lm_head decode per block instead of two).
                if getattr(self, "inference_pac_head", False) and (
                    getattr(self, "pac_head", None) is not None
                ):
                    # Additive-head inference: base argmax candidates feed the
                    # fp32 head; final = base + delta.  Costs one extra tiny
                    # head forward per block, no second backbone pass.
                    base_logits_blk = target.lm_head(first_hidden).unsqueeze(1)
                    candidates, _ = self.parallel_candidate_embeddings(
                        base_logits_blk,
                        target.model.embed_tokens,
                        straight_through=False,
                    )
                    prev4d = first_noise.unsqueeze(1).clone()
                    prev4d[:, :, 1:, :] = candidates[:, :, :-1, :]
                    # Optional Jacobi-style refinement: re-feed the head its own
                    # corrected argmax candidates for pac_iter_k passes.  Each
                    # pass is one tiny head forward; k=1 is the original
                    # single-shot behaviour.
                    pac_iter_k = max(1, int(getattr(self, "pac_iter_k", 1)))
                    with torch.autocast("cuda", enabled=False):
                        hidden_f = first_hidden.unsqueeze(1).float()
                        base_f = base_logits_blk.float()
                        for _ in range(pac_iter_k):
                            delta = self.pac_head(hidden_f, prev4d.float())
                            final_blk = base_f + delta
                            candidates, _ = self.parallel_candidate_embeddings(
                                final_blk.to(base_logits_blk.dtype),
                                target.model.embed_tokens,
                                straight_through=False,
                            )
                            prev4d = first_noise.unsqueeze(1).clone()
                            prev4d[:, :, 1:, :] = candidates[:, :, :-1, :]
                    block_ids[:, 1:] = sample(
                        final_blk[:, 0, : block_size - 1, :], temperature
                    )
                else:
                    block_ids[:, 1:] = sample(
                        target.lm_head(first_hidden[:, : block_size - 1, :]),
                        temperature,
                    )
                output = target(
                    block_ids,
                    position_ids=block_positions,
                    past_key_values=target_cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                posterior = sample(output.logits, temperature)
                accepted = (
                    (block_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )
                output_ids[:, start : start + accepted + 1] = block_ids[
                    :, : accepted + 1
                ]
                output_ids[:, start + accepted + 1] = posterior[:, accepted]
                start += accepted + 1
                target_cache.crop(start)
                target_hidden = extract_context_feature(
                    output.hidden_states, self.target_layer_ids
                )[:, : accepted + 1, :]
                acceptance_lengths.append(accepted + 1)
                if stop_token_ids is not None and any(
                    stop_id in output_ids[:, num_input_tokens:]
                    for stop_id in stop_token_ids
                ):
                    break
                continue
            first_logits = target.lm_head(first_hidden).unsqueeze(1)
            second_noise, _ = self.make_second_pass_noise(
                first_logits,
                first_noise.unsqueeze(1),
                target.model.embed_tokens,
                straight_through=False,
            )
            draft_positions = position_ids[
                :, second_pass_cache.get_seq_length() : start + block_size
            ]
            second_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=second_noise,
                position_ids=draft_positions,
                past_key_values=second_pass_cache,
                use_cache=True,
                is_causal=False,
            )
            second_pass_cache.crop(start)
            final_logits = target.lm_head(second_hidden[:, : block_size - 1, :])
            block_ids[:, 1:] = sample(final_logits, temperature)
            output = target(
                block_ids,
                position_ids=block_positions,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = sample(output.logits, temperature)
            accepted = (
                (block_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + accepted + 1] = block_ids[:, : accepted + 1]
            output_ids[:, start + accepted + 1] = posterior[:, accepted]
            start += accepted + 1
            target_cache.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : accepted + 1, :]
            acceptance_lengths.append(accepted + 1)
            if stop_token_ids is not None and any(
                stop_id in output_ids[:, num_input_tokens:] for stop_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stops = torch.tensor(stop_token_ids, device=output_ids.device)
            indices = torch.isin(output_ids[0][num_input_tokens:], stops).nonzero(
                as_tuple=True
            )[0]
            if indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + indices[0] + 1]
        return output_ids


class DominoGRUCorrector(nn.Module):
    """官方 Domino GRU 修正頭的忠實復刻(specforge domino.py 同構):
    prefix_gru(bias=False) + embed_proj(SiLU 瓶頸 256→V);
    GRU_SUFFIX_START(默認1)位之前 delta 置 0 = 官方 pure_draft_prefix_len 語義。"""

    def __init__(
        self,
        *,
        hidden_size: int,
        embed_size: int,
        vocab_size: int,
        gru_hidden: int = 1024,
        emb_dim: int = 256,
        suffix_start: int = 1,
        use_hidden_input: bool = False,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.suffix_start = suffix_start
        self.use_hidden_input = use_hidden_input
        self.prefix_gru = nn.GRU(
            input_size=embed_size + (hidden_size if use_hidden_input else 0),
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + gru_hidden, emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(emb_dim, vocab_size, bias=False),
        )

    def forward(self, hidden4d: torch.Tensor, prev4d: torch.Tensor) -> torch.Tensor:
        bsz, nb, L, _ = hidden4d.shape
        gru_src = (
            torch.cat([prev4d, hidden4d], dim=-1) if self.use_hidden_input else prev4d
        )
        gru_in = gru_src.reshape(bsz * nb, L, -1)
        gru_out = self.prefix_gru(gru_in)[0].reshape(bsz, nb, L, -1)
        delta = self.embed_proj(torch.cat([hidden4d, gru_out], dim=-1))
        if self.suffix_start > 0:
            delta = torch.cat(
                [torch.zeros_like(delta[:, :, : self.suffix_start, :]), delta[:, :, self.suffix_start :, :]],
                dim=2,
            )
        return delta


class CausalTransformerCorrector(nn.Module):
    """塊內因果 mini-Transformer 修正頭("更大的 GRU"):
    輸入 [骨幹 hidden 投影; 前一位 token 嵌入] → 2 層因果注意力 → rank 投影 delta。
    訓練並行因果(教師強制金鏈);推理逐位序列(部署期另行實現)。
    調用簽名與 ParallelAdditiveCorrectionHead 完全一致。"""

    def __init__(
        self,
        *,
        hidden_size: int,
        embed_size: int,
        vocab_size: int,
        d_model: int = 1024,
        n_layers: int = 2,
        num_heads: int = 8,
        rank: int = 512,
        mlp_mult: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.in_proj = nn.Linear(hidden_size + embed_size, d_model, bias=False)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.ModuleDict(
                dict(
                    ln1=nn.LayerNorm(d_model),
                    qkv=nn.Linear(d_model, 3 * d_model, bias=False),
                    attn_out=nn.Linear(d_model, d_model, bias=False),
                    ln2=nn.LayerNorm(d_model),
                    mlp_in=nn.Linear(d_model, mlp_mult * d_model, bias=False),
                    mlp_out=nn.Linear(mlp_mult * d_model, d_model, bias=False),
                )
            )
            self.layers.append(blk)
        self.out_ln = nn.LayerNorm(d_model)
        self.delta_in = nn.Linear(d_model, rank, bias=False)
        self.delta_out = nn.Linear(rank, vocab_size, bias=False)
        nn.init.zeros_(self.delta_out.weight)

    def forward(self, hidden4d: torch.Tensor, prev4d: torch.Tensor) -> torch.Tensor:
        bsz, nb, L, _ = hidden4d.shape
        x = self.in_proj(torch.cat([hidden4d, prev4d], dim=-1))
        x = x.view(bsz * nb, L, self.d_model)
        for blk in self.layers:
            h = blk["ln1"](x)
            qkv = blk["qkv"](h)
            q, k, v = qkv.chunk(3, dim=-1)
            hd = self.d_model // self.num_heads

            def _s(t):
                return t.view(bsz * nb, L, self.num_heads, hd).transpose(1, 2)

            a = F.scaled_dot_product_attention(_s(q), _s(k), _s(v), is_causal=True)
            a = a.transpose(1, 2).reshape(bsz * nb, L, self.d_model)
            x = x + blk["attn_out"](a)
            h2 = blk["ln2"](x)
            x = x + blk["mlp_out"](F.silu(blk["mlp_in"](h2)))
        x = self.out_ln(x).view(bsz, nb, L, self.d_model)
        return self.delta_out(F.silu(self.delta_in(x)))


class ParallelAdditiveCorrectionHead(nn.Module):
    """Domino-style additive logit correction without the GRU.

    One block-local causal self-attention layer (16x16, fully parallel)
    summarises the preceding draft-candidate prefix; a rank-256 projection
    emits a vocabulary delta added onto the protected base logits.  The
    delta output is zero-initialised so training starts exactly at the base
    distribution.  No recurrence anywhere.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        embed_size: int,
        vocab_size: int,
        rank: int = 256,
        attn_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must divide num_heads")
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.qkv_in = nn.Linear(hidden_size + embed_size, 3 * attn_dim, bias=False)
        self.attn_out = nn.Linear(attn_dim, attn_dim, bias=False)
        self.delta_in = nn.Linear(
            hidden_size + embed_size + attn_dim, rank, bias=False
        )
        self.delta_out = nn.Linear(rank, vocab_size, bias=False)
        nn.init.zeros_(self.delta_out.weight)

    def forward(
        self, hidden4d: torch.Tensor, prev_embed4d: torch.Tensor
    ) -> torch.Tensor:
        bsz, n_blocks, block, _hidden = hidden4d.shape
        features = torch.cat([hidden4d, prev_embed4d], dim=-1)
        # Parameter-free normalization: raw hidden/embedding magnitudes are
        # unbounded, and unnormalized QK products grow with the square of the
        # feature scale — the proven source of the 1e15 gradient spirals.
        # functional layer_norm adds no parameters, so checkpoints remain
        # loadable across this fix.
        features = F.layer_norm(features, (features.shape[-1],))
        head_dim = self.attn_dim // self.num_heads
        qkv = self.qkv_in(features).reshape(
            bsz * n_blocks, block, 3, self.num_heads, head_dim
        )
        query, key, value = (
            t.transpose(1, 2) for t in qkv.unbind(2)
        )
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        attended = attended.transpose(1, 2).reshape(
            bsz, n_blocks, block, self.attn_dim
        )
        attended = self.attn_out(attended)
        combined = torch.cat([features, attended], dim=-1)
        combined = F.layer_norm(combined, (combined.shape[-1],))
        delta = self.delta_out(F.silu(self.delta_in(combined)))
        return delta


def attach_pac_head(
    draft_model: nn.Module,
    *,
    hidden_size: int,
    embed_size: int,
    vocab_size: int,
    device,
    dtype,
    rank: int = 256,
    attn_dim: int = 512,
    num_heads: int = 8,
) -> ParallelAdditiveCorrectionHead:
    import os as _os_arch
    if _os_arch.environ.get("PAC_HEAD_ARCH", "") == "gru":
        head = DominoGRUCorrector(
            hidden_size=hidden_size,
            embed_size=embed_size,
            vocab_size=vocab_size,
            gru_hidden=int(_os_arch.environ.get("GRU_HIDDEN", "1024")),
            emb_dim=int(_os_arch.environ.get("GRU_EMB_DIM", "256")),
            suffix_start=int(_os_arch.environ.get("GRU_SUFFIX_START", "1")),
            use_hidden_input=_os_arch.environ.get("GRU_HIDDEN_INPUT", "0") == "1",
            num_layers=int(_os_arch.environ.get("GRU_LAYERS", "1")),
        ).to(device=device, dtype=dtype)
        draft_model.pac_head = head
        print(
            f"[pac] GRU head attached: params={sum(p.numel() for p in head.parameters())/1e6:.1f}M",
            flush=True,
        )
        return head
    if _os_arch.environ.get("PAC_HEAD_ARCH", "") == "ctc":
        head = CausalTransformerCorrector(
            hidden_size=hidden_size,
            embed_size=embed_size,
            vocab_size=vocab_size,
            d_model=int(_os_arch.environ.get("CTC_DMODEL", "1024")),
            n_layers=int(_os_arch.environ.get("CTC_LAYERS", "2")),
            num_heads=int(_os_arch.environ.get("CTC_HEADS", "8")),
            rank=int(_os_arch.environ.get("CTC_RANK", "512")),
        ).to(device=device, dtype=dtype)
        _init_ckpt = _os_arch.environ.get("CTC_INIT_CKPT", "")
        if _init_ckpt:
            _sd = torch.load(_init_ckpt, map_location=device, weights_only=True)
            head.load_state_dict({k: v.to(dtype) for k, v in _sd.items()}, strict=True)
            print(f"[pac] CTC head WARM-STARTED from {_init_ckpt}", flush=True)
        draft_model.pac_head = head
        print(
            f"[pac] CTC head attached: params={sum(p.numel() for p in head.parameters())/1e6:.1f}M",
            flush=True,
        )
        return head
    head = ParallelAdditiveCorrectionHead(
        hidden_size=hidden_size,
        embed_size=embed_size,
        vocab_size=vocab_size,
        rank=rank,
        attn_dim=attn_dim,
        num_heads=num_heads,
    ).to(device=device, dtype=dtype)
    draft_model.add_module("pac_head", head)
    return head


def pac_head_dims_from_state(state) -> dict:
    """Infer head geometry from checkpoint shapes (head_dim is fixed at 64)."""
    attn_dim = state["pac_head.qkv_in.weight"].shape[0] // 3
    return {
        "attn_dim": attn_dim,
        "num_heads": max(1, attn_dim // 64),
        "rank": state["pac_head.delta_out.weight"].shape[1],
    }


class ScratchMSSP2TrainingModel(OnlineDFlashModel):
    """Training wrapper with exactly the same two passes as inference."""

    draft_model: ScratchMSSP2DraftModel

    def _two_pass(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchors, keep = self._sample_anchor_positions(seq_len, loss_mask, device)
        first_noise = self._create_noise_embed(input_ids, anchors, keep)
        context_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            bsz, -1
        )
        draft_positions = self._create_position_ids(anchors)
        full_positions = torch.cat([context_positions, draft_positions], dim=1)
        if self.attention_backend == "flex_attention":
            if not FLEX_ATTENTION_AVAILABLE:
                raise ValueError("flex_attention is unavailable")
            attention_mask = create_dflash_block_mask(
                anchors, keep, seq_len, self.block_size, device
            )
        else:
            attention_mask = create_dflash_sdpa_mask(
                anchors, keep, seq_len, self.block_size, device
            )
        first_hidden = self.draft_model(
            position_ids=full_positions,
            noise_embedding=first_noise,
            target_hidden=hidden_states,
            attention_mask=attention_mask,
        )
        n_blocks = anchors.shape[1]
        first_logits4d = self.lm_head(first_hidden).reshape(
            bsz, n_blocks, self.block_size, -1
        )
        pac_head = getattr(self.draft_model, "pac_head", None)
        if pac_head is not None:
            # B2: single backbone pass; the additive head refines base logits
            # from the preceding-candidate prefix.  Training may teacher-force
            # the prefix (base stays protected additively, so the final CE
            # keeps informing the backbone); eval consumes base argmax —
            # inference-consistent.
            first_noise4d = first_noise.reshape(
                bsz, n_blocks, self.block_size, -1
            )
            hidden4d = first_hidden.reshape(bsz, n_blocks, self.block_size, -1)
            if self.training and getattr(
                self, "second_pass_teacher_forcing", False
            ):
                offsets = torch.arange(
                    1, self.block_size, device=device
                ).view(1, 1, -1)
                prefix_indices = (anchors.unsqueeze(-1) + offsets).clamp(
                    max=seq_len - 1
                )
                prefix_tokens = torch.gather(
                    input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
                    2,
                    prefix_indices,
                )
                prev4d = first_noise4d.clone()
                prev4d[:, :, 1:, :] = self.embed_tokens(prefix_tokens)
                candidate_ids = first_logits4d.argmax(dim=-1)
            else:
                candidates, candidate_ids = (
                    self.draft_model.parallel_candidate_embeddings(
                        first_logits4d,
                        self.embed_tokens,
                        straight_through=False,
                    )
                )
                prev4d = first_noise4d.clone()
                prev4d[:, :, 1:, :] = candidates[:, :, :-1, :]
            jacobi_k = int(getattr(self, "pac_jacobi_k", 1))
            # Full FP32 island: the head runs outside autocast on fp32 inputs
            # with fp32 weights, and the additive sum stays fp32.  Base logits
            # live at |x|~30 where one bf16 ULP is ~0.25, which swallows a
            # sub-ULP delta in the forward pass while gradients keep flowing;
            # both CE streams return fp32 so the calibers stay symmetric.
            with torch.autocast("cuda", enabled=False):
                if getattr(self, "pac_detached_base", False):
                    # TRUE decoupling: the head must see a frozen view of the
                    # trunk hidden states too.  Detaching only the logits sum
                    # left a side door — the head's vocab-scale CE gradients
                    # flowed through hidden4d into the backbone unclipped,
                    # which was the actual source of the 1e10-1e15 backbone
                    # gradient bombs across attempts 1-5.
                    head_hidden = hidden4d.detach().float()
                else:
                    head_hidden = hidden4d.float()
                delta = pac_head(head_hidden, prev4d.float())
                first_logits4d = first_logits4d.float()
                # Jacobi 級聯訓練:用第 r-1 輪修正後 argmax 重建鏈,再修一輪。
                # 訓推一致:推理側 spec_generate 的 pac_iter_k 走同構迭代。
                chain_topk = int(getattr(self, "pac_chain_topk", 1))
                chain_mix = float(getattr(self, "pac_chain_mix", 0.0))
                for _r in range(1, jacobi_k):
                    corr = first_logits4d + delta
                    am = corr.argmax(dim=-1)
                    if self.training and chain_topk > 1 and chain_mix > 0.0:
                        # 樹現實化鏈:以 chain_mix 概率把上一位換成 top-k 內隨機候選,
                        # 教頭在"非貪心父節點"(樹分支的真實輸入)上修正;label 不變,
                        # 故無自訓練漂移;推理側樹分支恰好就是這種輸入分佈。
                        tk = corr.topk(chain_topk, dim=-1).indices
                        pick = torch.randint(
                            0, chain_topk, am.shape + (1,), device=am.device
                        )
                        am_samp = tk.gather(-1, pick).squeeze(-1)
                        mix = (
                            torch.rand(am.shape, device=am.device) < chain_mix
                        )
                        am = torch.where(mix, am_samp, am)
                    prev_r = first_noise4d.clone()
                    prev_r[:, :, 1:, :] = self.embed_tokens(am[:, :, :-1])
                    delta = pac_head(head_hidden, prev_r.float())
                if getattr(self, "pac_detached_base", False):
                    # Decoupled training: the head learns its correction on a
                    # frozen view of the base, so the final CE cannot siphon
                    # gradient away from the backbone (B2r3 co-adaptation RCA).
                    final_logits4d = first_logits4d.detach() + delta
                else:
                    final_logits4d = first_logits4d + delta
            return anchors, keep, first_logits4d, final_logits4d, candidate_ids
        if self.training and getattr(self, "second_pass_teacher_forcing", False):
            # Domino-style base-anchored teacher forcing: the refinement pass
            # trains on ground-truth prefix embeddings (training-only inputs,
            # same legality class as the hard labels).  Inference and eval-mode
            # forwards keep the unchanged hard pass-1 argmax chain, accepting
            # the same train/infer mismatch as the official recipe.
            offsets = torch.arange(1, self.block_size, device=device).view(1, 1, -1)
            prefix_indices = (anchors.unsqueeze(-1) + offsets).clamp(
                max=seq_len - 1
            )
            prefix_tokens = torch.gather(
                input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, prefix_indices
            )
            first_noise4d = first_noise.reshape(bsz, n_blocks, self.block_size, -1)
            second4d = first_noise4d.clone()
            second4d[:, :, 1:, :] = self.embed_tokens(prefix_tokens)
            second_noise = second4d.flatten(1, 2)
            candidate_ids = first_logits4d.argmax(dim=-1)
        else:
            second_noise, candidate_ids = self.draft_model.make_second_pass_noise(
                first_logits4d,
                first_noise.reshape(bsz, n_blocks, self.block_size, -1),
                self.embed_tokens,
                straight_through=self.training,
            )
        second_hidden = self.draft_model(
            position_ids=full_positions,
            noise_embedding=second_noise,
            target_hidden=hidden_states,
            attention_mask=attention_mask,
        )
        final_logits4d = self.lm_head(second_hidden).reshape(
            bsz, n_blocks, self.block_size, -1
        )
        return anchors, keep, first_logits4d, final_logits4d, candidate_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchors, keep, first_logits, final_logits, candidates = self._two_pass(
            input_ids, hidden_states, loss_mask
        )
        offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        indices = anchors.unsqueeze(-1) + offsets
        valid = indices < seq_len
        safe = indices.clamp(max=seq_len - 1)
        targets = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchors.size(1), -1), 2, safe
        )
        weights = keep.unsqueeze(-1).expand_as(targets).float() * valid.float()
        weights *= torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchors.size(1), -1), 2, safe
        )
        eval_weights = weights.clone()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weights *= torch.exp(-k.float() / self.loss_decay_gamma)
        denom = weights.sum().clamp_min(1e-6)
        first_ce = F.cross_entropy(
            first_logits.flatten(0, 2), targets.flatten(), reduction="none"
        ).reshape_as(targets)
        final_ce = F.cross_entropy(
            final_logits.flatten(0, 2), targets.flatten(), reduction="none"
        ).reshape_as(targets)
        first_loss = (first_ce * weights).sum() / denom
        final_loss = (final_ce * weights).sum() / denom
        if getattr(self, "pac_detached_base", False):
            # Decoupled arm: both objectives at full strength; the detach in
            # the final stream already separates their gradient territories.
            loss = first_loss + final_loss
        else:
            loss = float(lambda_base) * first_loss + (1.0 - float(lambda_base)) * final_loss
        with torch.no_grad():
            predicted = final_logits.argmax(dim=-1)
            valid_eval = eval_weights.bool()
            accuracy_denom = valid_eval.sum()
            accuracy = ((predicted == targets) & valid_eval).sum().float() / accuracy_denom.clamp_min(1)
            accepted = compute_accept_len(predicted, targets, valid_eval)
            valid_blocks = valid_eval.any(dim=-1)
            accept_len = ((accepted + 1.0) * valid_blocks).sum() / valid_blocks.sum().clamp_min(1)
        metrics = {
            "final_loss": final_loss.detach(),
            "base_loss": first_loss.detach(),
            "accuracy_denom": accuracy_denom.detach(),
            "accept_len": accept_len.detach(),
            "lambda_base": torch.tensor(float(lambda_base), device=device),
            "pass1_unique_candidates": torch.tensor(
                float(candidates.unique().numel()), device=device
            ),
        }
        return loss, accuracy, metrics


def assert_no_recurrence(module: nn.Module) -> None:
    forbidden = (nn.RNNBase, nn.RNN, nn.GRU, nn.LSTM, nn.RNNCell, nn.GRUCell, nn.LSTMCell)
    found = [name for name, child in module.named_modules() if isinstance(child, forbidden)]
    if found:
        raise RuntimeError(f"scratch MSSP2 contains recurrent modules: {found}")


__all__ = [
    "ScratchMSSP2DraftModel",
    "ScratchMSSP2TrainingModel",
    "assert_no_recurrence",
]
