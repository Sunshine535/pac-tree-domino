"""CodeLab 重生成 worker:open-perfectblend 首輪人類提問 → 目標模型貪心重答。
口徑對齊原管線:非思考模板 + 空 think 種子,greedy,max_new=2048。
分片 ordinal % num_shards;斷點續作 = 掃描已寫 jsonl 的 ordinal。"""
import argparse, json, os
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--shard", type=int, required=True)
p.add_argument("--num-shards", type=int, default=8)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--out-dir", default="/workdir/pac_e/regen_cl")
p.add_argument("--batch", type=int, default=256)
p.add_argument("--reverse", action="store_true")
p.add_argument("--start-frac", type=float, default=0.0)
p.add_argument("--end-frac", type=float, default=1.0)
p.add_argument("--done-file", default="")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
out_path = f"{args.out_dir}/shard_{args.shard}.jsonl"
done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["ordinal"])
            except Exception:
                pass
if args.done_file:
    import json as _j
    done |= set(_j.load(open(args.done_file)))
print(f"[shard {args.shard}] resume: {len(done)} rows already done", flush=True)

tok = AutoTokenizer.from_pretrained("/workdir/g16/model/Qwen3-8B", local_files_only=True)
ds = load_dataset("/workdir/pac_e/huggingface.co/datasets/mlabonne/open-perfectblend", split="train")

todo = []
for i in range(args.shard, len(ds), args.num_shards):
    if i in done:
        continue
    todo.append(i)
    if args.limit and len(todo) >= args.limit:
        break
if args.reverse:
    todo.reverse()
todo = todo[int(len(todo)*args.start_frac):int(len(todo)*args.end_frac)]
print(f"[shard {args.shard}] todo: {len(todo)}", flush=True)

llm = LLM(model="/workdir/g16/model/Qwen3-8B", dtype="bfloat16",
          gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=False)
sp = SamplingParams(temperature=0.0, max_tokens=2048)

EMPTY_THINK = "<think>\n\n</think>\n\n"

def build_prompt(row):
    conv = row["conversations"]
    human = next((t["value"] for t in conv if t.get("from") == "human"), None)
    if not human:
        return None
    text = tok.apply_chat_template([{"role": "user", "content": human}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    if not text.rstrip().endswith("</think>"):
        text = text + EMPTY_THINK if EMPTY_THINK not in text else text
    return text

fout = open(out_path, "a", buffering=1)
B = args.batch
for s in range(0, len(todo), B):
    chunk = todo[s: s + B]
    prompts, keep = [], []
    for i in chunk:
        pr = build_prompt(ds[i])
        if pr is None or len(pr) > 12000:
            continue
        prompts.append(pr)
        keep.append(i)
    if not prompts:
        continue
    outs = llm.generate(prompts, sp)
    for i, o in zip(keep, outs):
        fout.write(json.dumps({
            "ordinal": i,
            "prompt": prompts[keep.index(i)][:0] or None,
            "prompt_token_ids": list(o.prompt_token_ids),
            "response": o.outputs[0].text,
            "response_token_ids": list(o.outputs[0].token_ids),
        }, ensure_ascii=False) + "\n")
    print(f"[shard {args.shard}] {min(s+B, len(todo))}/{len(todo)}", flush=True)
print(f"[shard {args.shard}] DONE", flush=True)
