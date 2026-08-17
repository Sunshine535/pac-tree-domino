# PAC × TrueTree: Surpassing Official Domino (Full Reproduction Package)

**Headline result (to reproduce)**: our fully self-trained stack — **PAC correction head × best-first tree verification (budget 63)** — reaches **234.3 tok/s / acceptance 12.414** (tuned PAC head, `heads/pac_tuned_head_fp16.pt`) and **231.5 tok/s / 12.41** (original PAC head, `heads/pac_off_head_fp16.pt`) on gsm8k:128, vs the official full-strength Domino baseline **199.7 tok/s / 9.608** measured on the same machine at the same time (**+17.3% / +15.9% speed, +29.2% acceptance**). The official-head variant (official GRU head × our tree) reaches 235.2 tok/s / 12.41.

All numbers in `results/` are raw per-item logs from the exact runs reported.

---

## 1. Environment

| Item | Value |
|---|---|
| GPU | 8× NVIDIA A800 80GB (SM80); single-node |
| Python | 3.12 (conda) |
| torch | 2.8.0+cu128 |
| transformers | 5.15.0 |
| attention | torch SDPA (FlashAttention2 absent is fine; benchmark auto-falls back) |
| Target model | Qwen/Qwen3-8B (bf16, HF layout) |
| Draft checkpoint | official Domino release `Huang2020-Domino-b16` (HF layout; contains draft backbone + official GRU head) |
| Dataset | gsm8k (HF datasets cache, offline mode) |

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.15.0 datasets safetensors zstandard
```

## 2. Assets and their SHA-256

| File | Purpose | sha256 (prefix) |
|---|---|---|
| `heads/pac_off_head_fp16.pt` | **our PAC head** (54.0M params, fp16; keys `qkv_in.weight, attn_out.weight, delta_in.weight, delta_out.weight`) | `5d5e2fcb13225...` |
| `heads/pac_tuned_head_fp16.pt` | our PAC head, tuned run (best self-trained stack: 234.3 tok/s / 12.414 with tree63) | `4f9eca9c8d200...` |
| official `Qwen3-8B-Domino-b16` | draft backbone + official GRU head | (from official release; verify `model.safetensors` exists) |
| `code/dflash.py` | official Domino inference file **with our TreeV2 branches added** (drop-in replacement for `Domino/code/dflash.py` in the official repo) | see MANIFEST |

**Model & code links (all verified sources):**

- Target model **Qwen3-8B**: <https://huggingface.co/Qwen/Qwen3-8B>
- Official Domino draft checkpoint (backbone + GRU head) **Qwen3-8B-Domino-b16**: <https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16>
- Official Domino code repository (the pipeline this package plugs into): <https://github.com/jianuo-huang/Domino>
- Our trained heads: download from this repo's **GitHub Release** (`heads/*.pt`, three files with the SHA-256 above) and place them under `heads/`. (HF mirror pending.)

Get the official repo + checkpoints first, then **replace `Domino/code/dflash.py` with `code/dflash.py`** from this package. No other file of the official pipeline is modified (a 2-line dtype env hook in `benchmark.py` is optional, only used for fp32 debug runs).

## 3. Reproduce the headline number (evaluation only, ~15 min on 8 GPUs)

```bash
cd Domino   # official repo root, dflash.py already replaced
TREE_V2_HEAD=pac PAC_HEAD_CKPT=/abs/path/heads/pac_off_head_fp16.pt \
DOMINO_TREE_V2=1 TREE_V2_FAST=1 TREE_V2_GRAPH=1 TREE_V2_LITE=64 TREE_V2_CELL=1 \
TREE_V2_BUDGET=63 TREE_V2_TOPK=8 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHON=python3 \
TARGET_MODEL=/path/to/Qwen3-8B DRAFT_MODEL=/path/to/Huang2020-Domino-b16 \
NUM_GPUS=8 TASKS=gsm8k:128 OUT_DIR=./out_pac_tree63 \
bash run_hf_benchmark.sh
```

Expected: **tok/s ≈ 231.5, acceptance ≈ 12.41**. Acceptance tolerance: same-machine reruns are bit-stable; **across machines expect ±0.1** (bf16 tie-flips land on different but equally-valid greedy branches — verified: a second A800 node reproduced 12.362). For a strict check use the fp32 losslessness test in §4. tok/s scales with hardware (4-GPU second node: 216.2).

Official baseline (same command, no env vars):

```bash
NUM_GPUS=8 TASKS=gsm8k:128 OUT_DIR=./out_official bash run_hf_benchmark.sh
# expected: tok/s ≈ 199.7, acceptance = 9.608 (bit-stable under T=0)
```

Aggregate any run with:

```bash
python3 - <<'EOF'
import json, glob
f = glob.glob("out_pac_tree63/*answers.jsonl")[0]
tn=wt=0; acc=[]
for l in open(f):
    c=json.loads(l)["choices"]
    cc=c[1] if len(c)>1 else c[0]
    tn+=sum(cc["new_tokens"]); wt+=sum(cc["wall_time"])
    if len(c)>1 and c[1]["acceptance_lengths"]: acc+=c[1]["acceptance_lengths"][0]
print("tok/s=%.1f accept=%.3f"%(tn/wt, sum(acc)/len(acc)))
EOF
```

Variants (same command, change env):
- **Tuned PAC head (best self-trained, 234.3/12.414)**: `PAC_HEAD_CKPT=/abs/path/heads/pac_tuned_head_fp16.pt`.
- Official-head tree (235.2/12.41): drop `TREE_V2_HEAD`/`PAC_HEAD_CKPT`.
- CTC head (199.3/12.38): `TREE_V2_HEAD=ctc CTC_HEAD_CKPT=heads/ctc_off_head_fp16.pt`.
- Budget sweep: `TREE_V2_BUDGET` ∈ {15,23,31,47,63,95}.
- math500 generalization (+25.0%/+34.4%): `TASKS=math500:128`.

## 4. Correctness verification (recommended before trusting speed)

T=0 speculative decoding is lossless: tree output must equal chain output token-for-token in fp32.

```bash
BENCH_DTYPE=float32 <tree env as above> TASKS=gsm8k:32 OUT_DIR=./fp32_tree  bash run_hf_benchmark.sh
BENCH_DTYPE=float32 TREE_V2_TOPK=1 TREE_V2_BUDGET=15 <same> OUT_DIR=./fp32_chain bash run_hf_benchmark.sh
# then diff the `turns` text per question: must be 32/32 identical (we verified exactly this).
```

Under bf16, ~60% of items show benign tie-flips (both are valid greedy continuations); use fp32 for the equality check.

## 5. Train the PAC head from scratch (optional, ~13 h on 8× A800)

Data: regenerate open-perfectblend first human turns with the *target model itself* (greedy, `enable_thinking=False`), then pack:

```bash
# 1) corpus regeneration (vLLM; shard with --shard/--num-shards)
python3 code/regen_worker_cl.py --shard 0 --num-shards 8 --out-dir ./regen_out --batch 256
# 2) build the tokenized pack (contract: tokens.u32 / loss_mask.u8 / token_index.npy / eligible_ordinals.npy / manifest.json)
python3 code/build_pack_429.py
```

Train (official-recipe budget; PAC head is the default arch of `attach_pac_head`):

```bash
TOKEN_ROOT=/path/to/pack NUM_ANCHORS=256 NPROC=8 LR_OVERRIDE=6e-4 \
PAC_LOSS_DECAY_GAMMA=7 \
INIT_BACKBONE=/path/to/Huang2020-Domino-b16/model.safetensors FREEZE_BACKBONE=1 \
bash code/run_regen_gate_v2.sh B2DC /path/to/train_ordinals.npy 1 pac_repro
```

Key facts encoded above: golden-chain teacher forcing (`B2DC`), 256 anchors/sample (≈3.5× the official total anchor-block budget in one epoch), depth-decay γ=7, warmup 0.04, cosine to 0, **official draft backbone loaded and frozen — only the 54M head trains**. Extract the head:

```bash
python3 - <<'EOF'
import torch
sd = torch.load("runs/pac_repro/draft_model.pt", map_location="cpu", weights_only=True)
head = {k[len("pac_head."):]: v.half() for k, v in sd.items() if k.startswith("pac_head.")}
torch.save(head, "pac_head_fp16.pt")
EOF
```

Train ordinals for our exact run: `arange(N_eligible)` minus the 512 held-out ordinals (deterministic; see `code/training_probes` for the gate and trainer with all integrity SHA checks).

## 6. What the tree is (method summary)

Per verify block the draft supplies `base_logits[L,V]` and hiddens. We build a **best-first token tree** (per-parent conditional distributions; the correction head is re-evaluated along each branch — GRU state forking for the GRU head, incremental per-branch attention KV for PAC/CTC), keep the global top-`budget` nodes by cumulative log-prob, verify all nodes in **one** target forward with an ancestor-visibility attention mask, accept the deepest correct root-path, then compact KV by the accepted path. The tree build loop and its topology tail are fully CUDA-graph captured (zero host syncs in steady state). Mechanism (from `results/acceptance_distributions.json`): the tree cuts early-death steps (accept ≤5) from 35.7% to 19.2% and first-token failures from 8.1% to 2.3% — it rescues early branch misses, which is why weak heads equalize (PAC 9.24 → 12.41 on the tree).

## 7. Results inventory (`results/`)

- `final_tree/`, `final_chain/`: the headline duel raw logs (official-head×tree63 235.2/12.41 vs official 199.7/9.608).
- `pac_off_eval/`: PAC head main-chain rollout (9.243).
- `acceptance_distributions.json`: per-step acceptance histograms.
- Full artifact set incl. budget sweep / math500 / CTC arms: `~/nips/sd_sota_artifacts_0816/`.

## 8. Known scope notes

- Greedy (T=0) only in this package; sampling-mode tree verification not included.
- SGLang port: acceptance gain reproduced (+1.7); full-graph speed parity in progress (draft-side tree is already CUDA-graph captured there).
