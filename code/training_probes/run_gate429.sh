#!/usr/bin/env bash
set -euo pipefail
ARM=${1:?arm}; ORDINALS=${2:?ordinals npy}; EPOCHS=${3:?epochs}; RUN_TAG=${4:?tag}
P=/workdir/pac_e
BADEV=$P/badev429
R4=$P/097ebe8r4_e058_v4_ulpcert_truefresh_tps_specforge2a9f7d0
SPEC=$R4/specforge_src
PY=python3
TOKEN_ROOT=${TOKEN_ROOT:?set TOKEN_ROOT}
TOKEN_MANIFEST=$TOKEN_ROOT/manifest.json
OUT=$BADEV/runs/$RUN_TAG
[[ -n "${RESUME_CKPT:-}" || ! -e "$OUT" ]] || { echo "refusing to reuse $OUT" >&2; exit 2; }
mkdir -p "$BADEV/runs" "$BADEV/logs"
jq_get() { "$PY" -c "import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split(\".\"): d=d[k]
print(d)" "$1" "$2"; }
DATA_SHA=$(jq_get "$TOKEN_MANIFEST" "source_data_sha256")
TOKENS_SHA=$(jq_get "$TOKEN_MANIFEST" "artifacts.tokens.sha256")
LOSS_MASK_SHA=$(jq_get "$TOKEN_MANIFEST" "artifacts.loss_mask.sha256")
TOKEN_INDEX_SHA=$(jq_get "$TOKEN_MANIFEST" "artifacts.token_index.sha256")
ELIGIBLE_SHA=$(jq_get "$TOKEN_MANIFEST" "artifacts.eligible_ordinals.sha256")
ELIGIBLE_ROWS=$(jq_get "$TOKEN_MANIFEST" "eligible_rows")
RAW_ROWS=$(jq_get "$TOKEN_MANIFEST" "raw_rows")
DEV_SHA=$(sha256sum "$ORDINALS" | awk "{print \$1}")
DEV_ROWS=$("$PY" -c "import numpy,sys; print(len(numpy.load(sys.argv[1], mmap_mode=\"r\")))" "$ORDINALS")
TF_FLAG=""
[[ "$ARM" == "B2DC" ]] && TF_FLAG="--second-pass-teacher-forcing --use-pac-head --pac-detached-base --cosine-schedule"
[[ "$ARM" == "B2NC" ]] && TF_FLAG="--use-pac-head --pac-detached-base --cosine-schedule"
export PYTHONPATH=$BADEV/probes:$R4/probes:$SPEC
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
"$PY" -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=localhost:${RDZV_PORT:-29400} --nproc_per_node=${NPROC:-8} "$BADEV/probes/train_e058_ba_dev_v2.py" \
  --tokenized-manifest "$TOKEN_MANIFEST" --tokens "$TOKEN_ROOT/tokens.u32" \
  --loss-mask "$TOKEN_ROOT/loss_mask.u8" --token-index "$TOKEN_ROOT/token_index.npy" \
  --eligible-ordinals "$TOKEN_ROOT/eligible_ordinals.npy" \
  --expected-data-sha256 "$DATA_SHA" \
  --expected-tokens-sha256 "$TOKENS_SHA" \
  --expected-loss-mask-sha256 "$LOSS_MASK_SHA" \
  --expected-token-index-sha256 "$TOKEN_INDEX_SHA" \
  --expected-eligible-sha256 "$ELIGIBLE_SHA" \
  --expected-raw-rows "$RAW_ROWS" --expected-eligible-rows "$ELIGIBLE_ROWS" \
  --target-model /workdir/models/huggingface.co/Qwen/Qwen3-8B \
  --specforge-root "$SPEC" --draft-config "$SPEC/configs/qwen3-8b-domino.json" \
  --out-dir "$OUT" --run-tag "$RUN_TAG" \
  --epochs "$EPOCHS" --max-length 3072 --num-anchors ${NUM_ANCHORS:-8} --grad-accum "${GRAD_ACCUM:-1}" \
  --learning-rate "${LR_OVERRIDE:-3e-4}" --seed 20260718 ${WARMUP_OVERRIDE:+--warmup-ratio "$WARMUP_OVERRIDE"} \
  --dev-train-ordinals "$ORDINALS" \
  --expected-dev-train-sha256 "$DEV_SHA" \
  --expected-dev-rows "$DEV_ROWS" \
  --dev-arm "$ARM" $TF_FLAG ${INIT_BACKBONE:+--init-backbone-safetensors "$INIT_BACKBONE"} ${FREEZE_BACKBONE:+--freeze-backbone} ${LAMBDA_OVERRIDE:+--lambda-override "$LAMBDA_OVERRIDE"} ${RESUME_CKPT:+--resume-checkpoint "$RESUME_CKPT"}
