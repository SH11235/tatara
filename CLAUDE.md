# rshogi-nnue リポ運用規約 (Claude Code 向け)

本ファイルは Claude Code セッションが本リポで作業するときに必ず従う運用規約。
人間 collaborator も読む想定だが、現状 user は SH11235 単独。

## CI 規約 (PR push 前必須)

PR 作成 / `git push` の前に `bash scripts/local-ci.sh` を必ず走らせ、exit 0
(`PASS` 表示) を確認する。3 step (fmt / clippy / test) が **全 crate (GPU 依存
含む) で pass** しない限り push 禁止。

```bash
bash scripts/local-ci.sh
```

`.github/workflows/checks.yaml` は GitHub-hosted runner に CUDA / LLVM が無いため
GPU 依存 crate (`gpu-runtime` / `progress-kpabs-train` / `nnue-trainer`) を
clippy / test の workspace から exclude しているが、**本機 (CUDA + LLVM 22
install 済) では exclude なし全 crate check を必須**とする。CI が green でも
local check を skip することは規約違反 (CI が見えない領域に未検出 lint /
test fail が溜まる)。

`scripts/local-ci.sh` の test step は `--release` で実行する。`nnue-trainer` の
GPU 数値同等性テスト (`gpu_cpu_equivalence_tests::*`) は debug build の f32 fma
off で tolerance を満たさず fail するが、release では本番経路と同じ codegen に
なって pass する (warm cache で fmt + clippy + test 計 ~20s)。

## rust-version (MSRV) 規約

`Cargo.toml` の `workspace.package.rust-version` は **保守的に下げない**、現在の
`rust-toolchain.toml` pin (nightly のメジャー version) と揃える。本リポは個人
プロジェクトで外部 consumer ゼロ、crates.io 公開も無い。低い MSRV は clippy
(`clippy::incompatible_msrv`) で `div_ceil` / `is_multiple_of` 等の便利 method
利用を誤ってエラー扱いさせる害がある。toolchain を上げたら rust-version も
同期更新する。

## commit / PR 規約

- commit message は日本語可、`{scope}: {summary}` 形式 (例: `nnue_train: ft_w_grad
  redundant memset 削除 (perf neutral, 論理整理)`)
- perf 改善 commit は計測値 (pos/s mean × 2 run / loss 軌跡 / PTX 変化) を message
  に含める
- negative result も commit を残す (revert commit + Issue 追記の 2 操作)
- main への直接 push 禁止、必ず PR 経由
- PR merge は CI green 必須、`--squash --delete-branch` で main に merge
- `git push --force` は main / merge 済 branch に絶対しない、`--no-verify`
  禁止 (CI を skip して push しない)

## コードコメント規約

コード内コメント (`.rs` / `Cargo.toml` / `.yaml` / `.sh` 等) は **初見の Rust
開発者がそのファイル単独で読んで意味が通る** ものに限る。以下は禁止:

- **作業ログ語彙**: 「削除済」「追加した」「今回」「以前は」「N → M に変更」
- **PM シーケンスラベル**: 「Stage N」「Stage N-M」「Phase N」「Step N」
  「Round N」「Iteration N」「Sprint N」「M1 / M2 / マイルストーン N」が
  プロジェクト/作業の順序を指すとき。
  ただし **algorithm の pass / step を指す場合は許容** (例: `Phase 1 of
  inverse-index sparse_ft_backward: per-feature 出現回数を histogram`、
  `// Step 1: llvm-link <ll> libdevice → linked.bc` 等は OK)
- **Issue / PR 番号参照**: 「Issue #N の」「PR #N で」「#NN review で」
- **Migration history**: 「ここから昇格」「以前は…にあった」「旧 path は」

これらの情報は git log / PR description / `docs/` 配下に置く。コード内に書いて
よいのは:
- 非自明な不変条件 (例:「caller が `n_pos * MAX_ACTIVE` を保証する」)
- 言語仕様で表せない constraint の理由 (例:「cuda-oxide が `f32::clamp` を
  lower できないため if-else 展開」)
- コードを読んでも分からない外部参照 (例:「YaneuraOu progress.bin 形式に追従」、
  論文 / upstream ライブラリの algorithm 出典)

## ドキュメント規約

`docs/` / `README.md` / `ATTRIBUTION.md` 等の `.md` も上記コードコメント規約と
同じ「初見 OSS reader 視点」を採る。加えて以下:

- **ATTRIBUTION.md は license attribution のみ**。vendor 作業履歴 / CHANGELOG /
  PR ごとの追加内容は書かない (CHANGELOG が必要になれば別 file に分ける)。
- **doc 冒頭の dated header 禁止**: `作成: YYYY-MM-DD`、`改訂: YYYY-MM-DD` 等。
  履歴は git log で見る。ADR のように Status / Date field が doc の意味の一部と
  なる場合は OK。
- **設計判断 doc (ADR) は `docs/decisions/YYYY-MM-DD-<slug>.md`**。連番
  (`0NNN-`) は並行 PR で衝突するので使わない。slug は内容トピック。
- **ADR は現アーキの WHY を残す**。執行済 workflow / 完了済ロードマップを ADR
  に残さない (古くなったら削除して良い、ADR は immutable とは限らない)。
- **directory tree / 構成図は現状を反映**。「将来こうする」予定や削除済 directory
  を残さない。
- **dated 検証ブロック禁止**: 「2026-05-11 に X 環境で確認」型の log は
  reference doc に混ぜない (計測経緯は git log / PR description が担当)。
- **略語は README の glossary 章で一回だけ定義**。コード内では glossary に登録
  済の略語を素のまま使ってよい。新規略語を増やしたら glossary も更新する。

## レビュー観点 checklist

コメント / docs / ファイル命名のレビュー (人 / AI) は
[docs/review-checklist.md](docs/review-checklist.md) を参照。本 CLAUDE.md は
「書く前に止める」prevention、checklist は「書かれたものを検出する」detection
の役割分担。

## 作業前 checklist

- 設計判断は ADR (`docs/decisions/`) に記録する
- cuda-oxide / nightly toolchain の構成は壊さない、host 側 unsafe は妥当性を
  コメントで明記する
