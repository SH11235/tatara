[English](progress-bin.md) | **日本語**

# 局面進行度 bucket: `progress.bin` の用意

`layerstack` アーキは各局面を N 個の局面進行度 bucket のいずれかに振り分け、
その割当を `progress.bin`——局面が 1 局の中でどこまで進んだかを推定する KP-abs
係数——で決める。本ページは `progress.bin` の学習方法と、生成された bucket 分布の
確認方法を扱う。`simple` アーキは bucket を持たず不要。学習全体の流れは
[docs/training-quickstart.ja.md](training-quickstart.ja.md) を参照。

進行度を学習して出力 bucket に割り当てる発想は
[nodchip 氏の記事](https://nodchip.hatenablog.com/entry/2026/02/04/000000) に基づく。

## progress.bin を生成

`progress-kpabs-train` で進行度係数を学習する。

> **データはシャッフルしないこと。** `progress-kpabs-train` の `--data` には
> **連続した対局**の PSV(局面が対局順に並び、対局が次々と続くもの)を渡す。
> 進行度係数は「1 局の中で局面がどこまで進んだか」を学習するもので、
> `progress-kpabs-train` はデータを 1 局単位で読み(`game_ply` で対局境界を検出)、
> 各局面にその局内での相対位置をラベル付けする。シャッフル済み PSV だと対局境界が
> 壊れてラベルが無意味になり、正しい係数が学習できない。一般に本体の NNUE 学習
> (`nnue-train`) はシャッフル済み PSV が望ましいが、進行度学習は逆で、シャッフル
> すると正しく学習できない。同じファイルを両方に使い回さないこと。

`--epochs` で総 epoch 数を指定する。epoch ごとに `<run-name>.e<N>.bin` の
checkpoint が出力され、最終 epoch は `--output` の path にも書き出される。

```bash
target/release/progress-kpabs-train \
  --data <path/to/consecutive-psv.bin> \
  --output output/progress/<run-name>.bin \
  --games-per-step 1024 --epochs 5
```

どの epoch の出力 (`<run-name>.e<N>.bin`) を使うかは試行錯誤になる
(progress.bin は bucket 割当を決める係数で、NNUE 学習の収束とは独立なため
何 epoch 必要かはデータ依存)。

どの epoch を使うか決める助けに `--val-fraction <f>`(例 `0.05`)を渡せる。
おおよそ指定割合の対局を入力順の N 局ごとに検証用へ取り分け(データは連続
対局順を保つ必要があるためシャッフルはしない)、各 epoch 末に held-out な
`val_loss` を出力する。有効にすると epoch ごとにデータ走査が 1 回増える。

`val_loss` は健全性チェックと epoch 選びの目安であって、品質の精密な指標では
ない。進行度モデルは単純(特徴ごとの重みを総和して sigmoid に通すだけ)なので
過学習しにくく、`train_loss` と `val_loss` の差は小さいのが正常で、明確に広がる
差が注意すべきサイン。また真の目的は良い bucket 分割で、素の MSE はその近似に
すぎないので、`val_loss` の厳密な最小値を追うより頭打ちになった epoch を選び、
最終的な `progress.bin` の良し悪しはそれで学習した LayerStack NNUE の棋力で
判断する。

## bucket 分布の確認

`progress-bucket-survey` は `progress.bin` が局面を進行度 bucket にどう割り当てる
かを集計する。分布がおおむね均等なら健全で、特定の bucket に偏っていると
LayerStack の出力 bucket ごとの学習データ量が大きく不均衡になる。

```bash
cargo build --release -p progress-bucket-survey
target/release/progress-bucket-survey \
  --data <path/to/consecutive-psv.bin> \
  --progress output/progress/<run-name>.e5.bin \
  --samples 200000
```

bucket ごとの件数・割合と top bucket の占有率を表示する。1 回の実行で読み込める
`progress.bin` は 1 つなので、epoch を比較するときは `<run-name>.e<N>.bin` ごとに
1 回ずつ実行して出力を比べる。

満足のいく `progress.bin` が得られたら、`layerstack` net の学習時に `nnue-train`
へ `--progress-coeff` で渡す([docs/training-quickstart.ja.md](training-quickstart.ja.md) 参照)。
