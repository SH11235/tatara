#!/usr/bin/env bash
# nnue-train の throughput (pos/s) を再現性高く計測するスクリプト。
#
# 基準計測 (5 sb x 200 batches x bs 65536、sb1 は cold
# cache outlier として除外、sb2-5 mean で評価) を RUNS 回まわし、run 間の
# mean / 標準偏差 / 変動係数 (CV%) を出す。小さな perf 改善 (~1-2%) を run 間
# ばらつきと区別して判定するための infra。
#
# GPU clock を固定すると run 間 σ が縮む。BENCH_LOCK_CLOCK=1 で
# `nvidia-smi -lgc` によるロックを試みる (root 権限が要る、権限が無ければ警告
# のみ出して続行)。
#
# 入力データ path (DATA / PROG) は machine 依存なのでリポジトリに焼き込まない。
# 毎回固定したい環境では gitignore 済の local 設定ファイル scripts/bench-pos.env
# を置く (テンプレート: scripts/bench-pos.env.example)。
#   DATA … PSV データファイル
#   PROG … progress 係数ファイル (ARCH=layerstack 時のみ必須)
#
# `ARCH` で計測対象アーキ切替 (default: layerstack):
#   ARCH=layerstack: 既存 9-bucket LayerStack (`progress-coeff` + `bucket-mode` を渡す)
#   ARCH=simple:     bucket-less Simple 4 層 (`SIMPLE_ARCH` / `SIMPLE_ACTIVATION` で
#                    arch preset / 活性化を上書き可)
#
# 使い方:
#   cp scripts/bench-pos.env.example scripts/bench-pos.env  # 初回、自環境の path を書く
#   scripts/bench-pos.sh                       # LayerStack FP32 default、RUNS 回
#   scripts/bench-pos.sh --ft-fp16             # FP16 FT weight モード
#   scripts/bench-pos.sh --ft-fp16 --ft-fp16-out
#   ARCH=simple scripts/bench-pos.sh           # Simple (default arch 512x2-8-64 crelu)
#   ARCH=simple SIMPLE_ARCH=256x2-32-32 scripts/bench-pos.sh
#   RUNS=5 scripts/bench-pos.sh                # run 回数を変える
#   THREADS=32 scripts/bench-pos.sh            # dataloader prefetch worker 数を変える
#   BENCH_LOCK_CLOCK=1 scripts/bench-pos.sh    # GPU clock 固定 (要 root)
#
# 設定ファイルを使わず 1 回だけ path を渡す場合は env var を前置きする:
#   DATA=/path/to/PSV PROG=/path/to/progress.bin scripts/bench-pos.sh
#
# nnue-train への追加 CLI フラグはそのまま "$@" で渡す (subcommand の後ろに付く)。
set -euo pipefail

cd "$(dirname "$0")/.."

# machine 依存の設定 (DATA / PROG 等) を gitignore 済の local ファイルから読む。
# BENCH_ENV_FILE で path 変更可。テンプレート scripts/bench-pos.env.example は
# `:=` 形式なので、明示的に前置きした env var がファイルの値より優先される。
bench_env_file="${BENCH_ENV_FILE:-scripts/bench-pos.env}"
if [[ -f "$bench_env_file" ]]; then
  # shellcheck source=/dev/null
  source "$bench_env_file"
fi

: "${CUDA_OXIDE_TARGET:=sm_86}"
: "${LLVM_LINK_BIN:=/usr/bin/llvm-link-22}"
: "${OPT_BIN:=/usr/bin/opt-22}"
: "${LLC_BIN:=/usr/bin/llc-22}"
export CUDA_OXIDE_TARGET LLVM_LINK_BIN OPT_BIN LLC_BIN

: "${DATA:?PSV データファイルの path を env var で指定する (例: DATA=/path/to/PSV PROG=/path/to/progress.bin scripts/bench-pos.sh)}"
: "${ARCH:=layerstack}"
: "${RUNS:=2}"
: "${SUPERBATCHES:=5}"
: "${BATCHES:=200}"
: "${BATCH_SIZE:=65536}"
: "${THREADS:=16}"
: "${FEATURE_SET:=halfka-hm-merged}"
: "${SIMPLE_ARCH:=512x2-8-64}"
: "${SIMPLE_ACTIVATION:=crelu}"
: "${BIN:=target/release/nnue-train}"
: "${BENCH_LOCK_CLOCK:=0}"

case "$ARCH" in
  layerstack)
    : "${PROG:?ARCH=layerstack で progress 係数ファイル PROG が必須 (例: PROG=/path/to/progress.bin)}"
    arch_args=(layerstack --progress-coeff "$PROG" --bucket-mode progress8kpabs)
    ;;
  simple)
    # benchmark は既存の WRM 既定値 (`--wrm-nnue2score 600`) を使う。Simple の
    # export scale も同じ単位に揃え、起動時の loss / 量子化整合 guard を満たす。
    arch_args=(simple --arch "$SIMPLE_ARCH" --activation "$SIMPLE_ACTIVATION" --scale 600)
    ;;
  *)
    echo "error: ARCH='$ARCH' は未対応 (layerstack または simple)" >&2
    exit 1
    ;;
esac

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN が無い。先に 'cargo build -p nnue-trainer --release' を実行" >&2
  exit 1
fi
# 入力ファイル存在確認: DATA は全 ARCH 共通、PROG は ARCH=layerstack 時のみ。
# ARCH=simple では PROG 未設定でも error にならないよう ARCH ごとに分岐する
# (set -u 下で `"$PROG"` の無条件展開を避けるためでもある)。
input_files=("$DATA")
if [[ "$ARCH" == "layerstack" ]]; then
  input_files+=("$PROG")
fi
for f in "${input_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "error: 入力ファイルが無い: $f" >&2
    exit 1
  fi
done

locked_clock=0
unlock_clock() {
  if [[ "$locked_clock" == 1 ]]; then
    nvidia-smi -rgc >/dev/null 2>&1 || true
    echo "[bench] GPU clock ロック解除"
  fi
}
trap unlock_clock EXIT

if [[ "$BENCH_LOCK_CLOCK" == 1 ]]; then
  # サポート上限クロックでロックする (run 間でブースト挙動が揺れるのを抑える)。
  max_gr=$(nvidia-smi --query-supported-clocks=gr --format=csv,noheader 2>/dev/null \
           | head -1 | awk '{print $1}')
  if [[ -n "${max_gr:-}" ]] && nvidia-smi -lgc "$max_gr" >/dev/null 2>&1; then
    locked_clock=1
    echo "[bench] GPU graphics clock を ${max_gr} MHz にロック"
  else
    echo "[bench] 警告: GPU clock ロック失敗 (root 権限が無い可能性)、ロック無しで続行" >&2
  fi
fi

echo "[bench] ARCH=$ARCH | $RUNS run x ${SUPERBATCHES}sb x ${BATCHES}batch x bs${BATCH_SIZE} threads=$THREADS  extra-args: $*"
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader \
  | sed 's/^/[bench] GPU (start): /'

# 各 run の sb2-5 mean を貯める。
run_means=()
for ((r = 1; r <= RUNS; r++)); do
  log=$(mktemp)
  "$BIN" \
    --data "$DATA" --feature-set "$FEATURE_SET" \
    --output /tmp/bench-pos --net-id bench-pos \
    --superbatches "$SUPERBATCHES" --batches-per-superbatch "$BATCHES" \
    --batch-size "$BATCH_SIZE" \
    --lr 8.75e-4 --win-rate-model --score-drop-abs 32000 \
    --save-rate "$SUPERBATCHES" --threads "$THREADS" \
    "${arch_args[@]}" \
    "$@" >"$log" 2>&1 || { cat "$log" >&2; rm -f "$log"; exit 1; }

  # "[train] superbatch N/5 | loss .. | NNNNNN pos/s | .." 行から sb2 以降を集計。
  mean=$(awk '
    /^\[train\] superbatch / {
      split($3, sb, "/"); n = sb[1]
      for (i = 1; i <= NF; i++) if ($i == "pos/s") ps = $(i - 1)
      if (n >= 2) { sum += ps; cnt++ }
    }
    END { if (cnt > 0) printf "%.0f", sum / cnt }
  ' "$log")
  rm -f "$log"
  if [[ -z "$mean" ]]; then
    echo "error: run $r で pos/s をパースできなかった" >&2
    exit 1
  fi
  run_means+=("$mean")
  printf '[bench] run %d/%d: sb2-%d mean = %s pos/s\n' \
    "$r" "$RUNS" "$SUPERBATCHES" "$mean"
done

printf '%s\n' "${run_means[@]}" | awk '
  { v[NR] = $1; sum += $1 }
  END {
    n = NR; mean = sum / n
    for (i = 1; i <= n; i++) { d = v[i] - mean; ss += d * d }
    sd = (n > 1) ? sqrt(ss / (n - 1)) : 0
    cv = (mean > 0) ? 100 * sd / mean : 0
    printf "[bench] -----------------------------------------\n"
    printf "[bench] runs=%d  mean=%.0f pos/s  sd=%.0f  CV=%.2f%%\n", n, mean, sd, cv
  }
'
