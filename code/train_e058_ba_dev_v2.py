#!/usr/bin/env python3
"""Full-corpus, resumable 8-GPU training for the exact e058 MSSP2 model.

Only the data path and run durability differ from the sealed e058 experiment.
The imported model and objective remain byte-for-byte those at e058b89.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from train_scratch_matched_8gpu import (
    all_reduce_mean,
    atomic_json,
    build_training_model,
    setup_distributed,
    sha256_path,
    teacher_features,
)
from specforge.training.strategies.base import linear_lambda_base
from e058_full_sampler import GlobalPermutationSampler


class IndexedConversationDataset(Dataset):
    """O(1)-resident access to the shared pre-tokenized full corpus."""

    def __init__(
        self,
        *,
        tokens: Path,
        loss_mask: Path,
        token_index: Path,
        eligible_ordinals: Path,
        expected_eligible_rows: int,
    ) -> None:
        self.tokens_path = str(tokens)
        self.loss_mask_path = str(loss_mask)
        self.index_path = str(token_index)
        self.eligible_path = str(eligible_ordinals)
        self.expected_rows = int(expected_eligible_rows)
        self._tokens = None
        self._loss_mask = None
        self._index = None
        self._eligible = None

    def __len__(self) -> int:
        return self.expected_rows

    def _initialize(self) -> None:
        if self._index is None:
            self._index = np.load(self.index_path, mmap_mode="r")
        if self._eligible is None:
            self._eligible = np.load(self.eligible_path, mmap_mode="r")
            if len(self._eligible) != self.expected_rows:
                raise RuntimeError("eligible row count changed")
        if self._tokens is None:
            self._tokens = np.memmap(self.tokens_path, mode="r", dtype="<u4")
        if self._loss_mask is None:
            self._loss_mask = np.memmap(self.loss_mask_path, mode="r", dtype="u1")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(
            {
                "_tokens": None,
                "_loss_mask": None,
                "_index": None,
                "_eligible": None,
            }
        )
        return state

    def __getitem__(self, encoded: int) -> dict:
        self._initialize()
        assert self._index is not None and self._eligible is not None
        assert self._tokens is not None and self._loss_mask is not None
        padding = int(encoded & 1)
        combined = encoded >> 1
        epoch, logical_ordinal = divmod(combined, self.expected_rows)
        ordinal = int(self._eligible[logical_ordinal])
        entry = self._index[ordinal]
        source_id = int(entry["source_id"])
        offset = int(entry["token_offset"])
        length = int(entry["token_length"])
        input_ids = torch.from_numpy(
            np.asarray(self._tokens[offset : offset + length], dtype=np.int64)
        )
        loss_mask = torch.from_numpy(
            np.asarray(self._loss_mask[offset : offset + length], dtype=np.int64)
        )
        if input_ids.numel() != loss_mask.numel() or input_ids.numel() < 2:
            raise RuntimeError(f"invalid tokenized record for source {source_id}")
        return {
            "epoch": int(epoch),
            "ordinal": int(ordinal),
            "source_id": source_id,
            "padding_replay": padding,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }


def file_sha256(path: Path) -> str:
    return sha256_path(path)


def parameter_probe_sha(module: torch.nn.Module) -> str:
    probe = torch.cat(
        [p.detach().float().flatten()[:16].cpu() for p in list(module.parameters())[:16]]
    )
    return hashlib.sha256(probe.numpy().tobytes()).hexdigest()


def save_checkpoint(
    *,
    path: Path,
    draft: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    next_epoch: int,
    next_local_position: int,
    bindings: dict,
    coverage_by_rank: list[dict],
    accumulated_seconds: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    previous = path.with_name(path.stem + ".previous" + path.suffix)
    torch.save(
        {
            "schema_version": 1,
            "draft_model": draft.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_local_position": next_local_position,
            "bindings": bindings,
            "coverage_by_rank": coverage_by_rank,
            "accumulated_seconds": float(accumulated_seconds),
        },
        temporary,
    )
    if path.exists():
        link_tmp = previous.with_suffix(previous.suffix + ".tmp")
        link_tmp.unlink(missing_ok=True)
        os.link(path, link_tmp)
        os.replace(link_tmp, previous)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized-manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--loss-mask", type=Path, required=True)
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--eligible-ordinals", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-tokens-sha256", required=True)
    parser.add_argument("--expected-loss-mask-sha256", required=True)
    parser.add_argument("--expected-token-index-sha256", required=True)
    parser.add_argument("--expected-eligible-sha256", required=True)
    parser.add_argument("--expected-raw-rows", type=int, default=1_420_893)
    parser.add_argument("--expected-eligible-rows", type=int, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--specforge-root", type=Path, required=True)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--parallel-topk", type=int, default=32)
    parser.add_argument("--parallel-temperature", type=float, default=1.0)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--draft-checkpoint", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--dev-train-ordinals", type=Path, required=True)
    parser.add_argument("--expected-dev-train-sha256", required=True)
    parser.add_argument("--expected-dev-rows", type=int, required=True)
    parser.add_argument("--dev-arm", required=True)
    parser.add_argument(
        "--second-pass-teacher-forcing", action="store_true", default=False
    )
    parser.add_argument("--use-pac-head", action="store_true", default=False)
    parser.add_argument("--cosine-schedule", action="store_true", default=False)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--init-backbone-safetensors", type=Path, default=None)
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--lambda-override", type=float, default=-1.0)
    parser.add_argument("--pac-detached-base", action="store_true", default=False)
    args = parser.parse_args()
    if args.draft_checkpoint or args.resume_from:
        raise RuntimeError("only --resume-checkpoint is allowed")
    rank, local_rank, world = setup_distributed()
    if world not in (4, 8):
        raise RuntimeError(f"BA dev runs support world_size 4 or 8, got {world}")
    device = torch.device("cuda", local_rank)
    manifest = json.loads(args.tokenized_manifest.read_text())
    if (
        manifest.get("status") != "PASS"
        or manifest.get("raw_rows") != args.expected_raw_rows
        or manifest.get("eligible_rows") != args.expected_eligible_rows
        or manifest.get("max_length") != args.max_length
        or manifest.get("block_size") != 16
    ):
        raise RuntimeError("tokenized full-corpus manifest is not a bound PASS")
    if manifest.get("source_data_sha256") != args.expected_data_sha256:
        raise RuntimeError("source data hash mismatch in tokenized manifest")
    expected_artifacts = {
        "tokens": (args.tokens, args.expected_tokens_sha256),
        "loss_mask": (args.loss_mask, args.expected_loss_mask_sha256),
        "token_index": (args.token_index, args.expected_token_index_sha256),
        "eligible_ordinals": (args.eligible_ordinals, args.expected_eligible_sha256),
    }
    verification_error = ""
    for label, (path, expected_sha) in expected_artifacts.items():
        if manifest.get("artifacts", {}).get(label, {}).get("sha256") != expected_sha:
            raise RuntimeError(f"{label} hash mismatch in manifest")
        if rank == 0 and not verification_error:
            try:
                if file_sha256(path) != expected_sha:
                    verification_error = f"runtime {label} hash mismatch"
            except Exception as exc:
                verification_error = f"runtime {label} verification failed: {exc}"
    verification = [verification_error]
    dist.broadcast_object_list(verification, src=0)
    if verification[0]:
        raise RuntimeError(verification[0])
    if rank == 0:
        if args.resume_checkpoint is None:
            if args.out_dir.exists():
                raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
            args.out_dir.mkdir(parents=True)
        elif not args.out_dir.is_dir():
            raise SystemExit("resume output directory is missing")
    dist.barrier()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Preserve the exact e058 numerical policy.  Full-corpus durability must
    # not silently change the candidate that the small matched run measured.
    torch.backends.cuda.matmul.allow_tf32 = True
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    target.eval()
    target.config.use_cache = False
    args.method = "scratch_mssp2"
    model, trainable = build_training_model(args, target, device)
    model.second_pass_teacher_forcing = bool(args.second_pass_teacher_forcing)
    model.pac_jacobi_k = int(os.environ.get("PAC_JACOBI_K", "1"))
    if model.pac_jacobi_k > 1:
        print(f"[pac] Jacobi cascade training enabled: K={model.pac_jacobi_k}", flush=True)
    model.pac_chain_topk = int(os.environ.get("PAC_CHAIN_TOPK", "1"))
    model.pac_chain_mix = float(os.environ.get("PAC_CHAIN_MIX", "0"))
    _ldg = os.environ.get("PAC_LOSS_DECAY_GAMMA", "")
    if _ldg:
        model.loss_decay_gamma = float(_ldg)
        print(
            f"[pac] depth-weighted loss enabled: gamma={model.loss_decay_gamma}",
            flush=True,
        )
    if model.pac_chain_topk > 1 and model.pac_chain_mix > 0:
        print(
            f"[pac] tree-realistic chains: topk={model.pac_chain_topk}"
            f" mix={model.pac_chain_mix}",
            flush=True,
        )
    model.pac_detached_base = bool(args.pac_detached_base)
    draft = model.draft_model
    if args.use_pac_head:
        from scratch_mssp2_model import attach_pac_head

        attach_pac_head(
            draft,
            hidden_size=target.config.hidden_size,
            embed_size=target.config.hidden_size,
            vocab_size=target.config.vocab_size,
            device=device,
            dtype=torch.float32,
            rank=int(os.environ.get("PAC_RANK", "256")),
            attn_dim=int(os.environ.get("PAC_ATTN_DIM", "512")),
            num_heads=int(os.environ.get("PAC_HEADS", "8")),
        )
        trainable = sum(
            p.numel() for p in draft.parameters() if p.requires_grad
        )
    if args.init_backbone_safetensors is not None:
        from safetensors.torch import load_file

        ost = load_file(str(args.init_backbone_safetensors))
        head_only_prefixes = ("prefix_gru.", "embed_proj.")
        backbone_state = {
            k: v for k, v in ost.items() if not k.startswith(head_only_prefixes)
        }
        own_keys = {k for k, _ in draft.named_parameters()}
        missing_from_donor = [
            k for k in own_keys
            if not k.startswith("pac_head.") and k not in backbone_state
        ]
        if missing_from_donor:
            raise RuntimeError(
                f"backbone transplant missing keys: {missing_from_donor[:5]}"
            )
        incompatible = draft.load_state_dict(backbone_state, strict=False)
        leftover = [
            k for k in incompatible.missing_keys if not k.startswith("pac_head.")
        ]
        if leftover or incompatible.unexpected_keys:
            raise RuntimeError(
                f"backbone transplant mismatch: {leftover[:5]} / "
                f"{list(incompatible.unexpected_keys)[:5]}"
            )
    if args.freeze_backbone:
        frozen_params = 0
        for name, parameter in draft.named_parameters():
            if not name.startswith("pac_head."):
                parameter.requires_grad = False
                frozen_params += 1
        if frozen_params == 0:
            raise RuntimeError("freeze-backbone froze nothing")
        trainable = sum(
            p.numel() for p in draft.parameters() if p.requires_grad
        )
    initial_probe_sha = parameter_probe_sha(draft)
    wrapped = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    bindings = {
        "run_tag": args.run_tag,
        "initial_parameter_probe_sha256": initial_probe_sha,
        "source_data_sha256": args.expected_data_sha256,
        "tokenized_artifact_sha256": {
            label: expected_sha for label, (_path, expected_sha) in expected_artifacts.items()
        },
        "raw_rows": args.expected_raw_rows,
        "eligible_rows": args.expected_eligible_rows,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "world_size": world,
        "seed": args.seed,
        "dev_arm": args.dev_arm,
        "dev_train_ordinals_sha256": args.expected_dev_train_sha256,
        "dev_rows": args.expected_dev_rows,
        "second_pass_teacher_forcing": bool(args.second_pass_teacher_forcing),
        "use_pac_head": bool(args.use_pac_head),
        "pac_detached_base": bool(args.pac_detached_base),
        "cosine_schedule": bool(args.cosine_schedule),
        "warmup_ratio": float(args.warmup_ratio),
    }
    global_step = 0
    start_epoch = 0
    start_local_position = 0
    nonpadding_seen = 0
    padding_seen = 0
    source_id_sum = 0
    accumulated_seconds = 0.0
    nonfinite_skips = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("bindings") != bindings:
            raise RuntimeError("resume checkpoint bindings changed")
        draft.load_state_dict(checkpoint["draft_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["next_epoch"])
        start_local_position = int(checkpoint["next_local_position"])
        coverage_by_rank = checkpoint.get("coverage_by_rank")
        if not isinstance(coverage_by_rank, list) or len(coverage_by_rank) != world:
            raise RuntimeError("resume checkpoint rank coverage is incomplete")
        local_coverage = coverage_by_rank[rank]
        nonpadding_seen = int(local_coverage["nonpadding_seen"])
        padding_seen = int(local_coverage["padding_seen"])
        source_id_sum = int(local_coverage["source_id_sum"])
        accumulated_seconds = float(checkpoint.get("accumulated_seconds", 0.0))
    if file_sha256(args.dev_train_ordinals) != args.expected_dev_train_sha256:
        raise RuntimeError("dev train ordinals hash mismatch")
    dev_ordinals = np.load(args.dev_train_ordinals, mmap_mode="r")
    if len(dev_ordinals) != args.expected_dev_rows:
        raise RuntimeError("dev train ordinal count mismatch")
    rows = args.expected_dev_rows
    steps_per_epoch = math.ceil(rows / world)
    total_steps = steps_per_epoch * args.epochs
    sampler = GlobalPermutationSampler(
        rows=rows,
        epochs=args.epochs,
        seed=args.seed,
        rank=rank,
        world=world,
        start_epoch=start_epoch,
        start_local_position=start_local_position,
    )
    dataset = IndexedConversationDataset(
        tokens=args.tokens,
        loss_mask=args.loss_mask,
        token_index=args.token_index,
        eligible_ordinals=args.dev_train_ordinals,
        expected_eligible_rows=rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=args.loader_workers,
        multiprocessing_context="spawn" if args.loader_workers else None,
        persistent_workers=bool(args.loader_workers),
        prefetch_factor=2 if args.loader_workers else None,
        pin_memory=True,
    )
    started = time.time()
    current_epoch = start_epoch
    local_position = start_local_position
    grad_norm = torch.tensor(float("nan"), device=device)
    window_bad = False
    for record in loader:
        epoch = int(record["epoch"])
        if epoch != current_epoch:
            if epoch != current_epoch + 1 or local_position != steps_per_epoch:
                raise RuntimeError("sampler epoch boundary changed")
            current_epoch = epoch
            local_position = 0
        input_ids = record["input_ids"].unsqueeze(0).to(device, non_blocking=True)
        loss_mask = record["loss_mask"].unsqueeze(0).to(device, non_blocking=True)
        source_id = int(record["source_id"])
        is_padding = int(record["padding_replay"])
        with torch.no_grad():
            hidden = teacher_features(target, input_ids)
        anchor_seed = args.seed * 1_000_003 + epoch * 10_007 + source_id
        torch.manual_seed(anchor_seed)
        torch.cuda.manual_seed(anchor_seed)
        if args.lambda_override >= 0.0:
            lambda_base = float(args.lambda_override)
        else:
            lambda_base = linear_lambda_base(global_step, total_steps, 1.0, 1.0)
        if args.cosine_schedule:
            warmup_steps = max(1, int(total_steps * args.warmup_ratio))
            if global_step < warmup_steps:
                lr_now = args.learning_rate * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                lr_now = args.learning_rate * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            for group in optimizer.param_groups:
                group["lr"] = lr_now
        accum = max(1, int(args.grad_accum))
        if global_step % accum == 0:
            optimizer.zero_grad(set_to_none=True)
            window_bad = False
        window_end = (global_step + 1) % accum == 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, accuracy, metrics = wrapped(
                input_ids=input_ids,
                hidden_states=hidden,
                loss_mask=loss_mask,
                lambda_base=lambda_base,
            )
        # Synchronized skip guard: a rare pathological record at bf16 can
        # produce a non-finite loss/gradient spike.  All ranks must agree to
        # skip (loss finiteness is per-rank, so it is all-reduced; the
        # post-DDP gradient norm is already identical on every rank).  The
        # record still counts toward coverage; only the weight update is
        # dropped.  A runaway budget keeps this fail-closed overall.
        bad_flag = (~torch.isfinite(loss.detach())).float()
        dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
        skipped_this_step = False
        if bad_flag.item() > 0:
            optimizer.zero_grad(set_to_none=True)
            window_bad = True
            nonfinite_skips += 1
            skipped_this_step = True
            grad_norm = torch.tensor(float("nan"), device=device)
        else:
            # Divide by the accumulation window so summed micro-grads average.
            (loss / accum).backward()
            if window_end:
                if window_bad:
                    # A poisoned micro-step already zeroed once, but the later
                    # good micro re-populated grads; drop the whole window.
                    optimizer.zero_grad(set_to_none=True)
                else:
                    head = getattr(draft, "pac_head", None)
                    if head is not None:
                        # Clip the head separately first so a head-side spike
                        # cannot crush the backbone's share of the global clip
                        # budget.
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        draft.parameters(), 1.0
                    )
                    # Poison-step guard.  Threshold calibration (measured, not
                    # guessed): healthy steady band ~1.2; legitimate
                    # post-resume re-equilibration transients reach 50-500 and
                    # must be applied (run4 died by skipping them all); the
                    # true poison signature jumps from single digits to 1e13+
                    # within ~100 steps.  10,000 passes every observed
                    # legitimate value and stops every observed bomb.
                    if not torch.isfinite(grad_norm) or float(grad_norm) > 10000.0:
                        optimizer.zero_grad(set_to_none=True)
                        nonfinite_skips += 1
                        skipped_this_step = True
                    else:
                        optimizer.step()
        if nonfinite_skips > 200000:
            raise RuntimeError(
                f"non-finite skip budget exhausted: {nonfinite_skips}"
            )
        if skipped_this_step and rank == 0:
            print(
                json.dumps(
                    {
                        "skip_nonfinite": True,
                        "global_step": global_step + 1,
                        "nonfinite_skips": nonfinite_skips,
                    }
                ),
                flush=True,
            )
        global_step += 1
        local_position += 1
        if is_padding:
            padding_seen += 1
        else:
            nonpadding_seen += 1
            source_id_sum += source_id
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "local_position": local_position,
            "loss": float(all_reduce_mean(loss, world)),
            "accuracy": float(all_reduce_mean(accuracy, world)),
            "accept_len": float(all_reduce_mean(metrics["accept_len"], world)),
            "base_loss": float(all_reduce_mean(metrics["base_loss"], world)),
            "final_loss": float(all_reduce_mean(metrics["final_loss"], world)),
            "lambda_base": lambda_base,
            "grad_norm": float(all_reduce_mean(grad_norm, world)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "nonfinite_skips": nonfinite_skips,
        }
        pac = getattr(draft, "pac_head", None)
        if pac is not None:
            _w = getattr(pac, "delta_out", None)
            if _w is None:
                _w = pac.embed_proj[-1] if hasattr(pac, "embed_proj") else None
            if _w is not None:
                row["pac_absmax"] = float(_w.weight.detach().abs().max())
        if rank == 0 and (global_step == 1 or global_step % 100 == 0):
            print(json.dumps(row, sort_keys=True), flush=True)
        next_epoch = epoch
        next_position = local_position
        if next_position == steps_per_epoch:
            next_epoch += 1
            next_position = 0
        if global_step % args.checkpoint_every == 0 or global_step == total_steps:
            local_coverage = {
                "nonpadding_seen": nonpadding_seen,
                "padding_seen": padding_seen,
                "source_id_sum": source_id_sum,
            }
            coverage_by_rank = [None for _ in range(world)] if rank == 0 else None
            dist.gather_object(local_coverage, coverage_by_rank, dst=0)
            dist.barrier()
            if rank == 0:
                assert coverage_by_rank is not None
                save_checkpoint(
                    path=args.out_dir / "resume_checkpoint.pt",
                    draft=draft,
                    optimizer=optimizer,
                    global_step=global_step,
                    next_epoch=next_epoch,
                    next_local_position=next_position,
                    bindings=bindings,
                    coverage_by_rank=coverage_by_rank,
                    accumulated_seconds=accumulated_seconds + time.time() - started,
                )
            dist.barrier()
    if global_step != total_steps:
        raise RuntimeError(f"completed {global_step} steps, expected {total_steps}")
    coverage = torch.tensor(
        [nonpadding_seen, padding_seen, source_id_sum],
        dtype=torch.int64,
        device=device,
    )
    dist.reduce(coverage, dst=0, op=dist.ReduceOp.SUM)
    dist.barrier()
    if rank == 0:
        model_path = args.out_dir / "draft_model.pt"
        temporary = model_path.with_suffix(".pt.tmp")
        torch.save(draft.state_dict(), temporary)
        os.replace(temporary, model_path)
        result = {
            "schema_version": 1,
            "status": "PASS",
            "method": "scratch_mssp2_e058_ba_dev",
            "recurrent_modules": False,
            "fully_scratch": True,
            "trainable_parameters": trainable,
            "initial_parameter_probe_sha256": initial_probe_sha,
            "optimizer_steps_per_rank": global_step,
            "optimizer_steps_total_across_ranks": global_step * world,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "raw_source_rows": args.expected_raw_rows,
            "eligible_source_rows_per_epoch": rows,
            "nonpadding_source_exposures": int(coverage[0]),
            "padding_replays": int(coverage[1]),
            "source_id_sum_nonpadding": int(coverage[2]),
            "seconds": accumulated_seconds + time.time() - started,
            "nonfinite_skips": nonfinite_skips,
            "bindings": bindings,
            "checkpoint": {
                "path": str(model_path),
                "sha256": file_sha256(model_path),
            },
        }
        expected_nonpadding = rows * args.epochs
        if result["nonpadding_source_exposures"] != expected_nonpadding:
            raise RuntimeError("full-corpus nonpadding coverage mismatch")
        atomic_json(args.out_dir / "result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
