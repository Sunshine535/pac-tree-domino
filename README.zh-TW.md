# PAC × 真樹驗證:超越官方滿血 Domino(完整復刻包)

**頭條結果(本包可直接復刻)**:我們完全自訓的組合 —— **PAC 修正頭 × best-first 樹驗證(預算 63)** —— 在 gsm8k:128 上達到 **234.3 tok/s / 接受長度 12.414**(調參版 PAC 頭 `heads/pac_tuned_head_fp16.pt`),原版 PAC 頭為 **231.5 tok/s / 12.41**;同機同時實測的官方滿血 Domino 基線為 **199.7 tok/s / 9.608**(**速度 +17.3% / +15.9%,接受長度 +29.2%**)。官方頭變體(官方 GRU 頭 × 我們的樹)為 235.2 tok/s / 12.41。

`results/` 內全部數字均為所報告運行的原始逐題日誌。

---

## 1. 環境

| 項目 | 值 |
|---|---|
| GPU | 8× NVIDIA A800 80GB(SM80);單機 |
| Python | 3.12(conda) |
| torch | 2.8.0+cu128 |
| transformers | 5.15.0 |
| attention | torch SDPA(無 FlashAttention2 亦可;benchmark 自動回退) |
| 目標模型 | Qwen/Qwen3-8B(bf16,HF 版式) |
| 草稿檢查點 | 官方 Domino 釋出 `Qwen3-8B-Domino-b16`(HF 版式;含草稿骨幹 + 官方 GRU 頭) |
| 資料集 | gsm8k(HF datasets 快取,離線模式) |

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.15.0 datasets safetensors zstandard
```

## 2. 資產與 SHA-256

| 檔案 | 用途 | sha256(前綴) |
|---|---|---|
| `heads/pac_off_head_fp16.pt` | **我們的 PAC 頭**(54.0M 參數,fp16;鍵 `qkv_in.weight, attn_out.weight, delta_in.weight, delta_out.weight`) | `5d5e2fcb13225...` |
| `heads/pac_tuned_head_fp16.pt` | 我們的 PAC 頭(調參爐;最強自訓組合 234.3 tok/s / 12.414) | `4f9eca9c8d200...` |
| 官方 `Qwen3-8B-Domino-b16` | 草稿骨幹 + 官方 GRU 頭 | (取自官方釋出;確認 `model.safetensors` 存在) |
| `code/dflash.py` | 官方 Domino 推理檔 **加入我們的 TreeV2 分支**(直接替換官方 repo 的 `Domino/code/dflash.py`) | 見 MANIFEST |

**模型與程式碼連結(均已實證):**

- 目標模型 **Qwen3-8B**:<https://huggingface.co/Qwen/Qwen3-8B>
- 官方 Domino 草稿檢查點 **Qwen3-8B-Domino-b16**:<https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16>
- 官方 Domino 程式碼倉庫(本包即插入此管線):<https://github.com/jianuo-huang/Domino>
- 我們訓練的頭:從本倉庫的 **GitHub Release** 下載(`heads/*.pt` 三個檔案,SHA-256 見上表)放入 `heads/`。(HF 鏡像待補。)

先取得官方 repo 與檢查點,然後**用本包 `code/dflash.py` 替換 `Domino/code/dflash.py`**。官方管線其餘檔案一概不改(`benchmark.py` 的 2 行 dtype 環境鉤子為可選,僅 fp32 除錯用)。

## 3. 復刻頭條數字(僅評測,8 卡約 15 分鐘)

```bash
cd Domino   # 官方 repo 根目錄,dflash.py 已替換
TREE_V2_HEAD=pac PAC_HEAD_CKPT=/絕對路徑/heads/pac_tuned_head_fp16.pt \
DOMINO_TREE_V2=1 TREE_V2_FAST=1 TREE_V2_GRAPH=1 TREE_V2_LITE=64 TREE_V2_CELL=1 \
TREE_V2_BUDGET=63 TREE_V2_TOPK=8 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHON=python3 \
TARGET_MODEL=/路徑/Qwen3-8B DRAFT_MODEL=/路徑/Qwen3-8B-Domino-b16 \
NUM_GPUS=8 TASKS=gsm8k:128 OUT_DIR=./out_pac_tree63 \
bash run_hf_benchmark.sh
```

預期:**tok/s ≈ 234.3、接受長度 ≈ 12.414**(調參頭);換 `PAC_HEAD_CKPT=heads/pac_off_head_fp16.pt` 則 ≈ 231.5 / 12.41。接受長度容差:同機重跑逐位穩定;**跨機期望 ±0.1**(bf16 平手翻轉落在不同但同樣合法的貪心分支——已實證:另一台 A800 節點復刻得 12.362)。嚴格檢查請用 §4 的 fp32 無損校驗。tok/s 隨硬體伸縮(第二節點 4 卡:216.2)。

官方基線(同命令,去掉所有環境變數):

```bash
NUM_GPUS=8 TASKS=gsm8k:128 OUT_DIR=./out_official bash run_hf_benchmark.sh
# 預期:tok/s ≈ 199.7,接受長度 = 9.608(T=0 下逐位穩定)
```

任意一次運行的聚合:

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

變體(同命令改環境變數):
- **調參 PAC 頭(最強自訓,234.3/12.414)**:`PAC_HEAD_CKPT=/絕對路徑/heads/pac_tuned_head_fp16.pt`。
- 官方頭 × 樹(235.2/12.41):去掉 `TREE_V2_HEAD`/`PAC_HEAD_CKPT`。
- CTC 頭(199.3/12.38):`TREE_V2_HEAD=ctc CTC_HEAD_CKPT=heads/ctc_off_head_fp16.pt`。
- 預算掃描:`TREE_V2_BUDGET` ∈ {15,23,31,47,63,95}。
- math500 泛化(+25.0%/+34.4%):`TASKS=math500:128`。

## 4. 正確性校驗(建議在採信速度前執行)

T=0 投機解碼是無損的:fp32 下樹輸出必須與鏈輸出逐 token 相等。

```bash
BENCH_DTYPE=float32 <同上樹環境> TASKS=gsm8k:32 OUT_DIR=./fp32_tree  bash run_hf_benchmark.sh
BENCH_DTYPE=float32 TREE_V2_TOPK=1 TREE_V2_BUDGET=15 <同上> OUT_DIR=./fp32_chain bash run_hf_benchmark.sh
# 逐題 diff `turns` 文字:必須 32/32 全同(我們正是如此驗證)。
```

bf16 下約 60% 題目出現良性平手翻轉(兩者皆為合法貪心延續);等式檢查請用 fp32。

## 5. 從零訓練 PAC 頭(可選,8× A800 約 13 小時)

資料:用*目標模型本身*重生成 open-perfectblend 首輪人類提問的回答(greedy、`enable_thinking=False`),再打包:

```bash
# 1) 語料重生成(vLLM;--shard/--num-shards 分片)
python3 code/regen_worker_cl.py --shard 0 --num-shards 8 --out-dir ./regen_out --batch 256
# 2) 構建 tokenized 包(契約:tokens.u32 / loss_mask.u8 / token_index.npy / eligible_ordinals.npy / manifest.json)
python3 code/build_pack_429.py
```

訓練(官方配方預算;PAC 頭為 `attach_pac_head` 預設架構):

```bash
TOKEN_ROOT=/路徑/pack NUM_ANCHORS=256 NPROC=8 LR_OVERRIDE=6e-4 \
PAC_LOSS_DECAY_GAMMA=7 \
INIT_BACKBONE=/路徑/Qwen3-8B-Domino-b16/model.safetensors FREEZE_BACKBONE=1 \
bash code/run_regen_gate_v2.sh B2DC /路徑/train_ordinals.npy 1 pac_repro
```

上述命令編碼的關鍵事實:金鏈教師強制(`B2DC`)、每樣本 256 錨(單輪 ≈ 官方總錨塊預算 3.5 倍)、深度衰減 γ=7、warmup 0.04、cosine 到 0、**載入並凍結官方草稿骨幹——只訓 54M 頭**。頭的抽取:

```bash
python3 - <<'EOF'
import torch
sd = torch.load("runs/pac_repro/draft_model.pt", map_location="cpu", weights_only=True)
head = {k[len("pac_head."):]: v.half() for k, v in sd.items() if k.startswith("pac_head.")}
torch.save(head, "pac_head_fp16.pt")
EOF
```

我們實際運行的訓練 ordinals:`arange(N_eligible)` 減去 512 條保留集(確定性;完整性 SHA 檢查見 `code/training_probes` 的門與訓練器)。

## 6. 樹是什麼(方法摘要)

每個 verify 塊,草稿側提供 `base_logits[L,V]` 與 hiddens。我們構建 **best-first token 樹**(逐父條件分佈;修正頭沿每條分支重新求值——GRU 頭用狀態分叉,PAC/CTC 頭用逐分支增量注意力 KV),按累積 log-prob 保留全域 top-`budget` 節點,以祖先可見性注意力遮罩在**單次**目標前向中驗證全部節點,接受最深正確根路徑,再按接受路徑壓實 KV。構樹主迴圈與拓撲尾部全程 CUDA Graph 捕獲(穩態零 host 同步)。機制(出自 `results/acceptance_distributions.json`):樹把早夭步(接受 ≤5)從 35.7% 降到 19.2%、首 token 失敗從 8.1% 降到 2.3%——它救回的是早期分支失誤,這正是弱頭在樹上趨同的原因(PAC 9.24 → 樹上 12.41)。

## 7. 結果清單(`results/`)

- `final_tree/`、`final_chain/`:頭條對決原始日誌(官方頭×樹63 235.2/12.41 vs 官方 199.7/9.608)。
- `pac_off_eval/`:PAC 頭主鏈 rollout(9.243)。
- `acceptance_distributions.json`:逐步接受長度直方圖。
- 完整工件集(含預算掃描 / math500 / CTC 各臂):`~/nips/sd_sota_artifacts_0816/`。

## 8. 範圍說明

- 本包僅含貪心(T=0);取樣模式的樹驗證不在其中。
- SGLang 移植:**速度與接受長度已全面超越官方滿血 sglang Domino**——gsm8k 531.2 tok/s / 接受 10.59 vs 官方 526.83 / 9.564;math500 538.3 / 10.23 vs 532.22 / 9.451(128 題、conc=1、同機)。技術棧:34-node level-fused 構樹 + 單 CTA 接受 kernel + 零同步 verify plan,全程 CUDA Graph 原生。
