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
GPU 依存 crate (`gpu-runtime` / `progress-kpabs-train` / `nnue-trainer` /
`experiments/*`) を clippy / test の workspace から exclude しているが、**本機
(CUDA + LLVM 22 install 済) では exclude なし全 crate check を必須**とする。
CI が green でも local check を skip することは規約違反 (CI が見えない領域に
未検出 lint / test fail が溜まる)。

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

- 作業ログ (「削除済」「追加した」「今回」「N → M に変更」)
- local context 依存の用語 (「Stage X」「PR #N で」「Issue #N の」)
- migration history (「ここから昇格」「以前は…にあった」)

これらの情報は git log / PR description / `docs/` 配下に置く。コード内に書いて
よいのは:
- 非自明な不変条件 (例:「caller が `n_pos * MAX_ACTIVE` を保証する」)
- 言語仕様で表せない constraint の理由 (例:「cuda-oxide が `f32::clamp` を
  lower できないため if-else 展開」)
- コードを読んでも分からない外部参照 (例:「YaneuraOu progress.bin 形式に追従」)

## 作業前 checklist

- 計測ログ + 仮説検証経緯は `docs/experiments/`、設計判断は `docs/01-decisions/`
- cuda-oxide / nightly toolchain の構成は壊さない、host 側 unsafe は妥当性を
  コメントで明記する
