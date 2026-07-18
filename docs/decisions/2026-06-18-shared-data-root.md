# 共有データ root (`$SHOGI_DATA`) への集約

- Status: Accepted
- Date: 2026-06-18

## Context

教師データ・学習済 NNUE・progress 係数・学習 checkpoint は複数 repo（rshogi 推論
エンジン / tatara NNUE trainer / bullet-shogi reference trainer）を跨いで使われる。
これらを各 repo の working tree 内に置くと、次の問題がある:

- データの生死が repo の生死に結合する（ある係数のためだけに別 repo を clone する
  必要が生じる、repo を消すとデータも消える）。
- consumer が sibling-relative path（`../<other-repo>/data/...`）や machine 絶対パスで
  他 repo のデータを参照し、repo を移動 / rename すると silent に壊れる。
- tracked file（skill / doc / コメント）に `/home/<user>/...` や `/mnt/...` の絶対パスが
  残り、別マシン / 別ユーザで解決できず、公開時に local 構成を漏らす。

一方 consumer 側は元々 path 可変である（engine の USI `EvalFile` / `LS_PROGRESS_COEFF`、
trainer の `--data` / `--output` / `--progress-coeff`、bench の env）。問題は「既定の
置き場所」と「参照の書き方」であって、コードの能力不足ではない。

## Decision

machine 単位の共有データ root を環境変数 `$SHOGI_DATA` で表し、全データをその配下に
集約する。repo の working tree にはデータを置かない。

layout（`$SHOGI_DATA/` 直下）と主な consumer:

- `teachers/` — 学習教師データ（PSV / packed bin）。trainer の `--data`。巨大・machine-local。
- `nnue/` — 配備済み NNUE モデル。engine が USI `EvalFile` で読む。featureset / arch 別。
- `progress/` — progress 係数。engine の `LS_PROGRESS_COEFF`、trainer の `--progress-coeff`。
- `runs/{bullet,tatara}/` — 各 trainer の学習途中 checkpoint（trainer の `--output`）。
  machine-local scratch。

cross-machine で scp 配布するのは小さい `nnue/` + `progress/` のみ。`teachers/`（巨大）・
`runs/`（scratch）は各 host ローカル。`nnue/` 内の symlink は `$SHOGI_DATA` 内の相対
パスにして root 移動に耐える（別マシンへ実体配布するときは symlink を resolve する）。

tracked file の参照規則:

- データ / モデル / 係数 / checkpoint → `$SHOGI_DATA/...`
- 自 repo 内ファイル → repo-relative
- 外部 repo → `/path/to/...` placeholder
- 個人ツール → `~/...`
- machine 固有の具体パスはgitignoredのlocal config（例: `bench-pos.local.toml`の
  `data` / `progress_coeff`）だけに書く。

`$SHOGI_DATA` は各 host の shell rc で定義する（デフォルト例: `$HOME/shogi-data`。大容量
ストレージを別マウントに持つ host はその配下）。具体パスは各 host の shell rc /
gitignored config にのみ書き、本 doc には placeholder のみ残す。

helper script（ckpt 検査ツール等）はデータではないので共有 root ではなく repo 側に置く。

## Consequences

- repo を clone / 削除してもデータは独立。係数 1 つのために無関係な repo を clone する
  必要がなくなる。
- 既定の置き場所が変わるだけで consumer のコード変更は不要（起動時に path を渡す）。
- tracked file から machine 絶対パスが消え、別マシン / OSS reader でも意味が通る。
  再混入は path 検査（grep guard）+ CI で検出する。
- rshogi 内部の bench / position set（`data/{startpos,floodgate,...}`）や各 repo の
  `experiments/` workspace は cross-repo 共有が不要なので repo 内に残す。
