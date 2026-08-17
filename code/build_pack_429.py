import json, hashlib, sys
import numpy as np

SNAP = "/workdir/pac_e/regen_snapshot_0815.jsonl"
OUT = "/workdir/pac_e/pack_ctc_0815"
import os; os.makedirs(OUT, exist_ok=True)

new_tokens, new_mask, rows = [], [], []
off = 0
kept = skip_long = skip_anchor = 0
for line in open(SNAP):
    d = json.loads(line)
    o = int(d["ordinal"])
    p = np.asarray(d["prompt_token_ids"], dtype=np.uint32)
    r = np.asarray(d["response_token_ids"], dtype=np.uint32)
    if len(p) >= 3072 - 8:
        skip_long += 1; continue
    full = np.concatenate([p, r])[:3072]
    mm = np.zeros(len(full), dtype=np.uint8); mm[len(p):] = 1
    win = len(full) - 16 + 1
    anchor_count = int(mm[:win].sum()) if win > 0 else 0
    if anchor_count < 2:
        skip_anchor += 1; continue
    new_tokens.append(full); new_mask.append(mm)
    rows.append((off, len(full), o)); off += len(full)
    kept += 1

tk = np.concatenate(new_tokens); mk = np.concatenate(new_mask)
ix = np.array(rows, dtype=[("token_offset", "<u8"), ("token_length", "<u4"), ("source_id", "<u8")])
tk.tofile(OUT + "/tokens.u32"); mk.tofile(OUT + "/loss_mask.u8"); np.save(OUT + "/token_index.npy", ix)
eligible = np.arange(len(ix), dtype=np.int64)
np.save(OUT + "/eligible_ordinals.npy", eligible)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()

manifest = {
  "schema_version": 1, "status": "PASS",
  "raw_rows": int(len(ix)), "eligible_rows": int(len(eligible)),
  "max_length": 3072, "block_size": 16,
  "source_data_sha256": sha(SNAP),
  "provenance": {
    "generator": "Qwen3-8B greedy empty-think, vLLM (429/432/423 fleet)",
    "source": "regen_snapshot_0815 unique-merged (884k)",
    "builder": "build_pack_429.py"},
  "artifacts": {
    "tokens": {"path": "tokens.u32", "sha256": sha(OUT + "/tokens.u32")},
    "loss_mask": {"path": "loss_mask.u8", "sha256": sha(OUT + "/loss_mask.u8")},
    "token_index": {"path": "token_index.npy", "sha256": sha(OUT + "/token_index.npy")},
    "eligible_ordinals": {"path": "eligible_ordinals.npy", "sha256": sha(OUT + "/eligible_ordinals.npy"), "rows": int(len(eligible))}}}
json.dump(manifest, open(OUT + "/manifest.json", "w"), indent=1)
src_map = {int(r["source_id"]): i for i, r in enumerate(ix)}
holdout_src = np.load("/workdir/pac_e/holdout_source_ids.npy")
hh = sorted(src_map[int(o)] for o in holdout_src if int(o) in src_map)
np.save(OUT + "/regen_holdout_ordinals.npy", np.array(hh, dtype=np.int64))
train = np.array(sorted(set(range(len(ix))) - set(hh)), dtype=np.int64)
np.save(OUT + "/regen_train_ordinals.npy", train)
print("PACKED rows", len(ix), "skip_long", skip_long, "skip_anchor", skip_anchor,
      "holdout", len(hh), "train", len(train), flush=True)
print("tokens_sha", manifest["artifacts"]["tokens"]["sha256"][:16], flush=True)
