# 新規リポジトリ設計メモ: 将棋 NNUE 専用 ML プロジェクト (cuda-oxide ベース)

> **Status: 設計メモ (歴史的)** — リポ立ち上げ時の初期 plan を記述する。
> 現状の構成は `README.md` と `crates/` / `bins/` 実コードを参照。

作成: 2026-05-09 / 改訂: 2026-05-09

## ビジョン

bullet-shogi (jw1912/bullet のフォーク) とは別系統で、**自分で育てる将棋 NNUE 学習プロジェクト**を立ち上げる。GPU カーネルは **cuda-oxide (NVIDIA Labs の Rust → PTX rustc backend)** で書き、host から device まで Rust 一本で完結させる。

**前提条件**:
- ROCm サポートは不要 (NVIDIA only で割り切る)
- bullet-shogi 上流追従の責務から解放される
- alpha 段階の cuda-oxide のリスクは個人の learning value で相殺できる

## 段階的ロードマップ

| Stage | スコープ | 目的 | 成果物 |
|---|---|---|---|
| **Stage 1** | experiments/001 cuda-oxide で KP-abs progress trainer を実装 | cuda-oxide / Rust GPU の習熟、最小スコープでの稼働確認 | progress.bin (rshogi 互換)、性能ベンチ、技術メモ |
| **Stage 2** | `crates/gpu-kernels` に **hand-fused 学習カーネル**を整備 (RAdam, Ranger, SCReLU, sparse FT 等) | NNUE training への足場作り | カーネルライブラリ + 単体テスト |
| **Stage 3** | `crates/nnue-train` で NNUE training pipeline を構築 (HalfKA_hm 1536-16-32 等) | bullet-shogi 相当の training を Rust 単一言語で再現 | shogi_nnue_train binary、自己対局検証 |
| **Stage 4** | 改良路線 (PSQT、Threat、新アーキテクチャ等)、cuda-oxide が成熟したら検討 | research playground | 各種実験記録 |

各 stage は前 stage の完了を待たず、**experiments/00N で個別検証**しながら main の crates/ に昇格させる流れ。

## 性能設計の核 — runtime fusion 喪失をどう回避するか

bullet-gpu は runtime に PointwiseIR を組み立てて NVRTC で fused kernel を作る。これが element-wise シーケンス (optimizer step、activation gradient、loss + WDL blending 等) で **memory traffic を 1/N に削る**重要な機構。

cuda-oxide は build-time コンパイラなので runtime fusion は不可。だが **本リポは「shogi NNUE 専用」で architecture が固定**なため、必要な fused kernel パターンは **3〜5 種類で打ち止め**:

| Pattern | Op 数 | 用途 |
|---|---|---|
| `fused_radam_step` | 5 | RAdam の m, v 更新 + bias correction + weight 更新 |
| `fused_ranger_step` | RAdam + lookahead | Ranger optimizer (ranger は RAdam + slow params の lerp) |
| `fused_loss_wdl` | 3-5 | sigmoid + WDL blend + scale |
| `fused_screlu_grad` | 2-3 | activation gradient (forward 経路と組合せ) |
| `fused_adamw_step` | 5 | AdamW (decay 込み) |

これらを **build-time に cuda-oxide で 1 度書いておけば**、bullet の runtime-fused 版と同等の memory traffic で動く。ハンドコード労力は合計 100〜300 行程度。

→ **性能ギャップ ~±0% を狙える**。「runtime fusion 喪失で -20〜-40%」の懸念は naive port 限定の話で、本リポでは設計でカバーする。

詳細根拠は `docs/01-decisions/0004-fused-kernel-strategy.md` (新設) に記録。

---

## リポジトリ名の候補

| 案 | 雰囲気 | 既存ネーミングとの整合 |
|---|---|---|
| `rshogi-nnue` | rshogi (将棋エンジン本体) と同じ命名規則 | `rshogi` シリーズの一員になる |
| `shogi-nnue-rs` | crates.io 由来の `-rs` サフィックス、外向きライブラリ風 | crates.io 公開を将来見据えるなら有利 |
| `nnue-shogi-rs` | NNUE 主体、将棋は応用先 | 後続で囲碁等を載せる場合に発展しやすい |
| `bullet-shogi-lab` | bullet-shogi の派生・実験場であることが明示 | 関係性は明示できるが独立性は下がる |
| `shogi-train-rs` | 将棋学習全般 | NNUE 限定でないので将来 PV-MCTS 等も可能 |

**推奨: `rshogi-nnue`** — `rshogi` / `bullet-shogi` と並べたときに位置関係が分かりやすい。将来 crates.io 公開時に名前空間として綺麗。

---

## ディレクトリ構成 (Stage 1〜3 を見据えた最終形)

```
rshogi-nnue/
├── README.md                     # プロジェクト概要、stage / experiments 一覧
├── LICENSE                       # MIT (bullet-shogi 由来コードと互換、cuda-oxide は Apache-2.0)
├── ATTRIBUTION.md                # bullet-shogi / bullet / cuda-oxide からの取り込み元クレジット
├── Cargo.toml                    # workspace
├── rust-toolchain.toml           # cuda-oxide pinning に追従 (nightly-2026-04-03 等)
├── .gitignore                    # data/, target/, *.bin 等
├── .github/
│   └── workflows/
│       ├── checks.yaml           # cargo clippy / fmt / test (GPU 不要なものだけ)
│       └── gpu-checks.yaml       # GPU runner 確保できる場合の build/test (optional)
│
├── crates/
│   ├── shogi-format/             # PSV / Pack の reader、ShogiBoard / Hand / BonaPiece 等
│   │                             # (bullet-shogi の crates/bullet_lib/src/shogi/ から vendor)
│   ├── shogi-features/           # 特徴量定義
│   │   ├── halfka_hm.rs          # NNUE 入力特徴量 (Stage 3 で必要)
│   │   ├── progress_kpabs.rs     # KP-abs progress (Stage 1 で必要)
│   │   └── ...
│   ├── gpu-runtime/              # host 側 CUDA wrapper
│   │                             # cuda-oxide の cuda-core / cuda-host を再利用 or 薄ラッパ
│   ├── gpu-kernels/              # device 側 cuda-oxide kernels (build-time PTX)
│   │   ├── lib.rs                # 公開 kernel の登録
│   │   ├── pointwise/            # element-wise fused kernels
│   │   │   ├── radam_step.rs
│   │   │   ├── ranger_step.rs
│   │   │   ├── loss_wdl.rs
│   │   │   └── screlu_grad.rs
│   │   ├── sparse/               # sparse FT 系
│   │   │   ├── sparse_ft_forward.rs
│   │   │   └── sparse_ft_backward.rs
│   │   └── progress/             # KP-abs progress 用 (Stage 1)
│   │       ├── forward.rs
│   │       ├── grad.rs
│   │       ├── adam_step.rs
│   │       └── eval.rs
│   ├── nnue-train/               # training pipeline (Stage 3 で本格化)
│   │   ├── schedule.rs           # 学習スケジュール (lr, wdl)
│   │   ├── optimizer.rs          # RAdam / Ranger の host 側 state 管理 (kernel は gpu-kernels)
│   │   ├── dataloader.rs         # PSV / Pack の batch 化
│   │   └── trainer.rs            # main loop
│   └── nnue-format/              # NNUE binary IO (rshogi 互換)
│       ├── halfka_psqt.rs        # PSQT 込みの save/load
│       └── header.rs             # net_id, FV_SCALE, QA/QB 等のメタデータ
│
├── bins/
│   ├── progress_kpabs_train/     # Stage 1 の本番 binary (experiments/001 から昇格)
│   ├── nnue_train/               # Stage 3 の本番 binary
│   └── nnue_eval/                # 評価ツール (vs Material, vs other engines)
│
├── experiments/                  # 番号付き実験記録
│   ├── 001-cuda-oxide-kpabs/     # Stage 1 の最初の実験 (gpu-kernels に昇格させる前)
│   │   ├── README.md             # 動機・設計・結果
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── ...                       # 002, 003, ... と continue
│
├── data/                         # .gitignore (PSV/PACK は git 外)
│   └── .gitkeep
│
└── docs/
    ├── 00-overview.md            # プロジェクト全体の意図 + stage roadmap
    ├── 01-decisions/             # ADR (Architecture Decision Records)
    │   ├── 0001-licensing.md
    │   ├── 0002-vendor-vs-dep.md
    │   ├── 0003-cuda-oxide-adoption.md
    │   ├── 0004-fused-kernel-strategy.md
    │   ├── 0005-staged-migration-plan.md
    │   └── 0006-rocm-out-of-scope.md
    ├── format/
    │   └── packed-sfen.md        # PSV 仕様 (bullet-shogi-packed-sfen-spec.md からの抜粋)
    └── kernels/
        └── fused-pattern-catalog.md # どの fused kernel が何を担うか
```

### Stage 1 では crates/ の何を実装するか

最低限:
- `crates/shogi-format` (vendor)
- `crates/shogi-features/progress_kpabs.rs` (KP-abs bucket 算出)
- `crates/gpu-runtime` (cuda-oxide の cuda-core を再利用 or 薄ラッパ)
- `crates/gpu-kernels/progress/` (4 kernel: forward, grad, adam_step, eval)
- `experiments/001-cuda-oxide-kpabs/` (host 側ロジック、bullet-shogi の host コードを参考に)

→ Stage 1 完了で **既存 `shogi_progress_kpabs_train_cuda` と等価の binary が `bins/progress_kpabs_train/` に存在**。

### Stage 2〜3 で増える部分

- `crates/gpu-kernels/pointwise/` に RAdam / Ranger / loss / activation の fused kernel 追加
- `crates/gpu-kernels/sparse/` に HalfKA_hm 用 sparse FT forward/backward
- `crates/nnue-train/` に training loop
- `crates/nnue-format/` に NNUE binary IO (rshogi compatible)
- `bins/nnue_train/` に本番 trainer

---

## 依存関係の取り扱い

### bullet-shogi 由来コード (MIT)

**Vendor 戦略** (最小限切り出し) を推奨:
- 必要部分だけ手動 copy、最初の commit に元 commit hash 記載
- `ATTRIBUTION.md` に取り込み元・元ライセンス・取り込み済 file 一覧
- 利点: 自由に編集、外部依存ゼロ、CI 高速
- 欠点: bullet-shogi 側の更新を手動 sync (frequency: 半年〜年単位で十分)

#### Stage 1 で vendor する file

| 元 path | 新 path | 用途 |
|---|---|---|
| `crates/bullet_lib/src/shogi/packed_sfen.rs` | `crates/shogi-format/src/packed_sfen.rs` | PSV reader |
| `crates/bullet_lib/src/shogi/types.rs` | `crates/shogi-format/src/types.rs` | ShogiBoard, Hand 等 |
| `crates/bullet_lib/src/shogi/bona_piece.rs` | `crates/shogi-format/src/bona_piece.rs` | (Stage 1 では未使用、Stage 3 で必要) |
| `crates/bullet_lib/src/game/outputs.rs` 内の `ShogiProgressKPAbs` 周辺 | `crates/shogi-features/src/progress_kpabs.rs` | progress 8/9 bucket、kp-abs 特徴 |
| `examples/shogi_progress_kpabs_train_cuda.rs` の I/O 部分 | `experiments/001-cuda-oxide-kpabs/src/main.rs` の host 側 | PSV 読込、batch 構築、Adam 状態管理 |

GPU kernel 部分 (`KERNELS_SRC`) は **vendor せず cuda-oxide で書き直す** のが本実験の核心。

#### Stage 3 で追加 vendor

| 元 path | 新 path |
|---|---|
| `crates/bullet_lib/src/game/inputs/shogi_halfka.rs` 等 | `crates/shogi-features/src/halfka_hm.rs` |
| `crates/bullet_lib/src/value/save.rs` の NNUE binary IO | `crates/nnue-format/src/halfka_psqt.rs` |
| `crates/bullet_lib/src/trainer/schedule/` | `crates/nnue-train/src/schedule.rs` |

### cuda-oxide 由来コード (Apache-2.0)

**Git dependency** が現実的:
- `[dependencies] cuda-core = { git = "https://github.com/NVlabs/cuda-oxide.git", rev = "..." }`
- alpha 期間は API 不安定なので **rev pin 必須**
- 自分の検証が済んだ rev に固定し、定期的に上げる
- `crates/rustc-codegen-cuda` は cuda-oxide の build process に乗る (cargo subcommand `cargo oxide` 経由)

### Pliron (cuda-oxide の transitive 依存、git dep)

cuda-oxide が pin している rev を継承。直接触る予定無し。

---

## 最初の実験: experiments/001-cuda-oxide-kpabs

### 動機

NVIDIA `cuda-oxide` (2026-05 公開) を本リポジトリの最小スコープで試す。

- DSL なしで GPU カーネルを Rust ソースから直接生成できる将来性の評価
- 既存 `shogi_progress_kpabs_train_cuda` の `KERNELS_SRC` (~150 行 CUDA C++) を Rust 化して型安全性とリファクタ容易性を測る
- 出力 progress.bin が既存版と一致するか (numerical equivalence) を確認し、cuda-oxide が production ML 用途に成熟しているかの判定材料を得る
- **Stage 2〜3 への足場作り**: ここで gpu-runtime / gpu-kernels の基本パターンを確立すれば、後続の NNUE training 移植が機械的に進む

### スコープ

- 対象 kernel (4 つ): `k_forward`, `k_grad_loss_hist`, `k_adam_step`, `k_eval_loss_hist`
- I/O・CLI 引数・出力ファイル形式は既存版互換
- host 側は cuda-oxide の `cuda-core` / `cuda-host` を再利用 (`gpu-runtime` crate として薄くラップ)

### 対象外 (Stage 1 では)

- bullet-gpu / bullet-shogi 本体への変更
- ROCm 対応 (永久に対象外)
- 本流 NNUE training への適用 (Stage 3 で別実装)
- Tensor cores / TMA / clusters (Hopper/Blackwell 機能、本実験は sm_86 想定)

### 受け入れ条件

- [ ] cuda-oxide で 4 kernel が PTX 化される
- [ ] 新 binary が build & smoke run 成功 (CUDA 12 + sm_86 環境)
- [ ] 既存 `shogi_progress_kpabs_train_cuda` と loss 推移がほぼ一致 (numerical drift < 1e-3 程度)
- [ ] 出力 progress.bin が既存版とバイナリ一致 (or float 誤差範囲)
- [ ] kernel 単体 perf (samples/sec) を README に記載
- [ ] cuda-oxide 不具合に当たった場合は何が動かなかったかを文書化

### リスク

- **sm_86 (Ampere) で動くか未確認**: cuda-oxide は B200 (sm_100) で SoL ベンチマーク、Hopper/Blackwell 機能の例題が中心。基本 atomics / barrier / sharedmem は動く想定だが第一関門
- **Required atomics**: `DeviceAtomicF64::fetch_add` (loss accumulator)、`DeviceAtomicU64::fetch_add` (histogram)、`DeviceAtomicF32::fetch_add` (gradient scatter) — cuda-oxide v0.1.0 の atomics example で全部 test 済みなので OK 想定
- **`llc-22` が atomic syncscope 完全対応に必要**: build setup に LLVM 22 が必要。dev container か手動構築
- **toolchain pin (`nightly-2026-04-03`)**: 他の Rust プロジェクトと同居するなら rustup override で対処
- **Pliron は別 repo**: cuda-oxide の git dep を pin することで暗黙に固定されるが、cuda-oxide upgrade で巻き込まれる可能性

不可と判明した時点で素直に断念して experiment を partial-success / abandon でクローズ。

---

## ADR (Architecture Decision Records、最初の 6 個)

### 0001-licensing
**Decision**: MIT (bullet-shogi 由来 vendor コードと互換、cuda-oxide の Apache-2.0 とも互換、商用利用も含めて寛容)

### 0002-vendor-vs-dep
**Decision**: bullet-shogi 由来は **vendor**、cuda-oxide 由来は **git dep + rev pin**。
- vendor: 編集自由 + メンテ簡素、frequency が低い (将棋データ format は安定) のでコスト小
- cuda-oxide: vendor すると本流追従不可なので git dep。rev pin で alpha 期の API breakage を局所化

### 0003-cuda-oxide-adoption
**Decision**: GPU kernel は cuda-oxide で書く。NVCC / NVRTC は使わない。

理由:
- ROCm 不要なので NVIDIA 専用での割り切りができる
- Rust 一言語で完結する設計が自分の learning value に合致
- alpha リスクは新規リポなので局所化できる
- bullet-shogi 上流追従の責務から解放される

### 0004-fused-kernel-strategy
**Decision**: bullet-gpu の runtime-fused pointwise を build-time hand-fused kernel で代替する。

理由:
- cuda-oxide は build-time コンパイラ、runtime fusion 不可
- 本リポは shogi NNUE 専用で fusion パターンが固定 (RAdam, Ranger, loss+WDL, activ_grad 等の 5 種程度)
- 各パターンを 1 度ハンドコードすれば bullet runtime-fused 相当の memory traffic を達成可能
- naive port (1 op 1 kernel) なら -20〜-40% の slowdown が出るが、hand-fuse すれば ~±0%

実装: `crates/gpu-kernels/pointwise/` 配下に Pattern 1 個 = 1 ファイル で配置。各 fused kernel に benchmark を併設して bullet 版との速度比較を継続検証。

### 0005-staged-migration-plan
**Decision**: 4 stage で段階的に進める (本ファイル冒頭の roadmap 表参照)。各 stage は前 stage の完了を絶対前提とせず、experiments/00N で検証してから main の crates/ に昇格。

### 0006-rocm-out-of-scope
**Decision**: ROCm / AMD GPU サポートは永久に対象外。

理由:
- cuda-oxide が NVIDIA only
- AMD 対応するなら HIP backend が必要だが cuda-oxide には無い
- 本リポは個人 ML playground として割り切り、対応 platform を絞ることで開発速度を確保

---

## 次セッションでの着手手順

```bash
# 1. リポジトリ作成
cd ~/development
gh repo create <username>/rshogi-nnue --private --description "Personal Rust shogi NNUE training lab using cuda-oxide"
git clone git@github.com:<username>/rshogi-nnue.git
cd rshogi-nnue

# 2. 初期 infra
mkdir -p crates/{shogi-format,shogi-features,gpu-runtime,gpu-kernels}/src
mkdir -p experiments/001-cuda-oxide-kpabs/src
mkdir -p docs/01-decisions docs/format docs/kernels
mkdir -p .github/workflows
mkdir -p data
echo -e "/target\n/data\n*.bin" > .gitignore
cat > rust-toolchain.toml <<'EOF'
[toolchain]
channel = "nightly-2026-04-03"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy"]
EOF

# 3. 最初の commit (skeleton + LICENSE + README + roadmap)
git add . && git commit -m "chore: initial repo skeleton with stage roadmap"

# 4. cuda-oxide 動作確認
#    別 dir に cuda-oxide を clone してビルドが通るか確認 (host の CUDA 12 + sm_86)
git clone https://github.com/NVlabs/cuda-oxide.git ~/development/cuda-oxide
cd ~/development/cuda-oxide
cargo oxide run vecadd  # まずは最小例題で動作確認
cargo oxide run atomics  # 必要 atomic 機能 (F64, U64) の確認

# 5. 本リポに戻り、cuda-oxide を git dep として追加
cd ~/development/rshogi-nnue
# experiments/001-cuda-oxide-kpabs/Cargo.toml に
#   [dependencies]
#   cuda-core = { git = "https://github.com/NVlabs/cuda-oxide.git", rev = "..." }
#   cuda-host = { git = "...", rev = "..." }
#   cuda-device = { git = "...", rev = "..." }

# 6. bullet-shogi から PSV reader を vendor
cp /mnt/nvme1/development/bullet-shogi/crates/bullet_lib/src/shogi/packed_sfen.rs \
   crates/shogi-format/src/
# (依存型 ShogiBoard, Hand 等も同様に。最小化を意識)
cp /mnt/nvme1/development/bullet-shogi/crates/bullet_lib/src/shogi/types.rs \
   crates/shogi-format/src/
git add . && git commit -m "feat(shogi-format): vendor PackedSfenValue reader from bullet-shogi"

# 7. experiments/001 で cuda-oxide kernel を 1 つずつ移植
#    - まず host 側 (PSV 読込、batch 構築) を mockup で動かす (kernel は dummy で OK)
#    - 次に k_forward を Rust 化、bit-equivalent を確認
#    - 同様に k_grad_loss_hist, k_adam_step, k_eval_loss_hist
#    - 出力 progress.bin が bullet-shogi 版と一致することを確認
#    - 性能ベンチを記録

# 8. Stage 1 完走したら experiments/001 の中身を bins/progress_kpabs_train + crates/gpu-kernels/progress/ に昇格
```

---

## 参考リンク

- **bullet-shogi** (祖、vendor 元): https://github.com/SH11235/bullet-shogi
- **bullet** (上流): https://github.com/jw1912/bullet
- **cuda-oxide** (中核技術): https://github.com/NVlabs/cuda-oxide
- **cuda-oxide docs**: https://nvlabs.github.io/cuda-oxide/
- **Pliron** (cuda-oxide の依存): https://github.com/vaivaswatha/pliron

## bullet-shogi 側の参照ポイント (移植元の commit / file)

`shogi_progress_kpabs_train_cuda` の現状 (cudarc → bullet-gpu raw FFI 化済) を Stage 1 移植の出発点にする:

- 移植元 path: `bullet-shogi/examples/shogi_progress_kpabs_train_cuda.rs` (~1100 行)
- 移植元 commit: `f275eb9 refactor(examples): cuda example を bullet-gpu raw FFI に書き換えて cudarc 依存除去` (2026-05-08)

KERNELS_SRC 部分 (CUDA C++ ~150 行) が cuda-oxide 化の主対象。それ以外 (PSV 読込、batch 構築、Adam 状態管理、I/O) は host 側ロジックとして移植する。

Stage 3 (NNUE training) の移植元としては:
- `bullet-shogi/examples/shogi_layerstack.rs` (LayerStack 1536x16x32 + PSQT、v101 構成)
- `bullet-shogi/examples/shogi_simple.rs` (HalfKA_hm の最小 trainer)
- `bullet-shogi/crates/bullet_lib/src/game/inputs/shogi_halfka.rs` (特徴量定義)
- `bullet-shogi/crates/bullet_lib/src/trainer/schedule/` (lr / wdl scheduler)

これらを参考に、cuda-oxide で kernel を hand-coding し、host 側は Rust で from-scratch に書き起こす。
