[English](held-out-validation.md) | **日本語**

# held-out validation

過学習や数値発散 (NaN) を SPRT 自己対局を待たずに早期検知するには、held-out
validation を有効化する。validation 用に「勾配更新に一切使わない局面」を別途
保持し、毎 superbatch 末に forward-only パスで集計する。training log は
`test_loss` / `test_acc` を出力 (console 向けの短い field 名)、`experiment.json`
は同じ値を `test_loss` / `test_accuracy` として記録する。opt-in 機能で、学習
全体の流れは [docs/training-quickstart.ja.md](training-quickstart.ja.md) を参照。

## 関連する 3 つの flag

| flag | 役割 | 種類 |
|---|---|---|
| `--test-tail-positions <N>` | held-out の **source**: `--data` の末尾 N 局面 | source A |
| `--test-data <PATH>` | held-out の **source**: 別 PSV ファイル | source B |
| `--test-positions <K>` | 選んだ source から毎 superbatch **評価する局面数** | evaluation size、両 source 共用 |

`--test-tail-positions` と `--test-data` は held-out source の選択肢で、
`clap conflicts_with` 双方向で排他 (どちらか 1 つ、または両方未指定 = held-out
無効)。`--test-positions` は source 選択とは別軸のパラメータで、選ばれた
source の先頭から K 局面 (満タン batch 切上げ) を毎 superbatch 末に評価する。

## どちらの source を選ぶか

- **`--test-tail-positions <N>` (推奨)**: `--data` 自身の末尾 N 局面を切り
  分ける。training は `[0, file_end - N * 40)`、validation は
  `[file_end - N * 40, file_end)` を読み、両者は byte range レベルで disjoint
  なので contamination は構造的に発生しない。教師ファイル 1 本で training と
  validation 両方をまかなえるので、別 file を用意して同期管理する手間が要らな
  い。唯一のコストは training pool が N 局面減ること。教師全件 ≫ N の典型ケース
  (例: 1e9 局面の教師に対し N = 1e6) では 0.1% 未満で実害なし
- **`--test-data <path>`**: validation 専用の別 PSV ファイル。holdout 集合が
  `--data` と独立して用意済 (異なる generator / 異なる時期の局面群) で、その
  独立性を保ちたい積極的な理由があるときに使う。ergonomic 上の理由だけで `--data`
  を 2 本に分割する利点は無い

## 使用例

末尾 100 万局面を holdout に切り分け、うち先頭 1 万局面を毎 superbatch で評価:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --test-tail-positions 1000000 \
  --test-positions 10000 \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> --threads <N> \
  layerstack --progress-coeff <path/to/progress.bin>
```

## 指標の読み方

`test_loss` は `train_loss` と同じ loss kernel (sigmoid-MSE または WRM) +
同じ WDL lambda blend で計算するため、両者は単位・スケールが揃い同 superbatch 内
で直接比較できる。この blend の設定方法（一定の `--wdl` か、線形の
`--start-wdl` / `--end-wdl` taper か）は
[学習スケジュール](training-schedule.ja.md) を参照。`test_loss − train_loss` の差が広がっていけば過学習の兆候、
`test_loss` が `train_loss` より早く異常値に飛ぶようなら NaN 発散の早期検知に
なる。

`test_acc` / `test_accuracy` はモデル出力の符号と実対局結果の一致率 (引き分け
は分母から除外)。scale 不変なので、loss スケールが異なる run / 設定の間でも直接
比較できる。
