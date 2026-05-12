# Attribution

このリポジトリは以下のオープンソースプロジェクトから派生・参照しています。

## bullet-shogi (MIT)

- Source: https://github.com/SH11235/bullet-shogi
- Upstream: https://github.com/jw1912/bullet
- Use: PSV reader、ShogiBoard / Hand 等の format 周りを vendor (Stage 1〜)
- License: MIT

### 取り込み済 file (時系列で追記)

#### Stage 3 #88 — GpuTrainer 真の checkpoint-resume (`--resume` + raw f32 optimizer state) (2026-05-13、bullet 新規 vendor 無し)

本格学習ラン (v102 = 400 superbatch × 数日) 用の resume 対応。bullet 由来コードの変更/追加なし、本リポ実装の機能追加:

- **raw checkpoint format `b"RNRC"`** (`bins/nnue_train/src/main.rs`、新規 self-contained binary): magic + version u32 + superbatch u64 + step_count u64 + num_groups u64 + 10 weight group ごとに `len u64 + w[f32×len] + m[f32×len] + v[f32×len] + slow[f32×len]` (全 LE、`grad` は resume に不要なので含めない)。group 順は `V102Weights` と同順 (ft_w, ft_b, l1_w, l1_b, l1f_w, l1f_b, l2_w, l2_b, l3_w, l3_b)。`crates/nnue-train::optimizer` の `b"RNGR"` (RangerHostState 単体 format) とは別物 (RNGR は optimizer state だけで weight を持たない) — weight + Ranger state + step + superbatch を 1 file にまとめた resume 専用 format。
- **`GpuTrainer::save_raw_checkpoint(path, superbatch)` / `load_raw_checkpoint(path) -> usize`**: save は device → host download (`DeviceBuffer::to_host_vec`) → `<path>.tmp` へ `BufWriter` 書き → `std::fs::rename` で atomic 置換 (書き込み途中 crash しても `<path>` は前回の完全な checkpoint のまま)。load は magic/version/group数/各 len を v102 arch と照合 (不一致は `InvalidData` で reject、`u64 → usize` は `try_into` で overflow も reject — `RangerHostState::load_from_reader` と同方針、Codex #62 convention)、host → device upload、`step_count` 復元。`grad` buffer は触らない (step ごとに memset される)。量子化 `.bin` (`save_quantised`、推論用 final artifact) は従来どおり別途出力。
- **`TrainerBackend::save_resume_checkpoint(path, superbatch)`** (`crates/nnue-train::trainer`、新 trait method): `crates/nnue-train` は GPU 非依存なので path / superbatch だけ受け取り device IO は backend (`bins/nnue_train::GpuTrainer`) 任せ。`trainer::run` は `save_rate` (および末尾) で `save_checkpoint` (`.bin`) に加えて `save_resume_checkpoint` (`{net_id}-{sb}.ckpt`) を呼ぶ。`TrainingConfig` に `keep_raw_checkpoints: Option<usize>` 追加 (default `None` = 全保持)。`prune_old_raw_checkpoints` (`run` 内、CPU-only): 新 `.ckpt` 書き込み後、`{net_id}-<digits>.ckpt` のうち superbatch 降順に直近 N 個だけ残し古いものを削除 (削除失敗は警告のみで続行、`.bin` には触らない)。
- **CLI** (`bins/nnue_train/src/main.rs`): `--resume <ckpt>` (raw checkpoint から weight + Ranger m/v/slow/step を復元、optimizer state 込みで再開)、`--start-superbatch <N>` (未指定時: `--resume` あり → checkpoint の sb+1、なし → 1。`1 <= N <= --superbatches` 範囲外はエラー)、`--keep-checkpoints <N>`。`--resume` と `--init-from` は排他 (`--init-from` は weight だけ注入し optimizer を reset するため真の resume にならない、明示エラー)。lr/wdl scheduler は superbatch index 駆動 (`StepLR` は `start * gamma^((sb-1)/step)`) なので `start_superbatch` を `TrainingConfig` 経由で渡せば lr が自動で正しい値に戻る。dataloader (`BucketedPrefetchedLoader`) は resume 時 file 先頭から再開 (≈ 次 epoch、bullet も file 位置の厳密復元はしない; `--file-shuffle-seed` 配線時に epoch index も保存する TODO は doc に記載)。
- **テスト**: `crates/nnue-train::trainer` — `run_with_start_superbatch_offset_resumes_loop_and_lr_schedule` (`start_superbatch=3, end=5` で step 回数 / checkpoint 命名 / lr schedule offset を確認)、`keep_raw_checkpoints_prunes_oldest` (実 tmp dir で `.ckpt` prune ロジック exercise、`.bin` / 別 net_id / 非数値名は無傷)、既存 `run_drives_...` に `resume_saves` 確認追加、`config_validate_...` に `keep_raw_checkpoints` 検証追加 (`MockBackend` に `save_resume_checkpoint` 実装)。`bins/nnue_train` — `raw_ckpt_format_tests` (GPU 不要、`write_f32_slice`/`read_f32_vec_io` roundtrip + short-read error + format 定数 pin + `invalid_data` helper)。kernel 追加なし → `nnue_train.ll` の `kernel_names` / kernel 数 (27) は不変、`gpu_cpu_equivalence_tests` の typecheck も維持。
- **検証** (ローカル sm_75 / 8GB / WSL2): CPU 全 crate test ok、workspace clippy / fmt クリーン、GPU bin build OK、`cargo test -p nnue-trainer --release` 24/24 pass、no-arg smoke `PASSED` + `verify_nnue_accumulator` ALL PASSED。resume smoke (sample.psv / `--threads 1` / `--batch-size 100`): 2sb 走らせ `.ckpt` 保存 → `--resume ... --start-superbatch 3` で 4sb まで → `[train] resuming from <ckpt> at superbatch 3` ログ、sb 3 loss = 0.035969 が通し 4sb run の sb 3 と一致 (atomic 非決定性の範囲、本ケースでは bit 一致)、全 weight finite、`verify_nnue_accumulator` ALL PASSED。`--start-superbatch` 省略時は checkpoint の sb+1 から自動再開。`--keep-checkpoints 1` で古い `.ckpt` を prune (`.bin` は全保持)。`--resume` + `--init-from` 同時指定 / `--start-superbatch` 範囲外 はエラー。`--init-from` 単独学習も従来どおり動作 (verify-nnue PASS)。

#### Stage 3 #89 — dataloader を bucket-aware prefetch / 並列パース / decode-once (2026-05-13、bullet 新規 vendor 無し)

EPIC #75 / #81 関連 (perf)。bullet 由来コードの変更/追加なし、本リポ実装の構造変更:

- **decode-once**: `crates/shogi-features` に decode 済み `ShogiBoard` を直接受ける入口を追加 — `ShogiHalfKA_hm::map_features_board` / `ShogiProgressKPAbs::{for_each_active_index_board, progress_board, bucket_board}`。従来 1 局面で `PackedSfenValue::decode()` が 2 回 (`map_features` 内 1 回 + `bucket`/`for_each_active_index` 内 1 回) 走っていたのを、dataloader が `psv.decode()` を 1 回呼んでその `ShogiBoard` を HalfKA_hm 特徴抽出と progress8kpabs bucket 計算の両方に渡す形に。既存 `map_features` / `progress` / `bucket` / `push` は board 版へ委譲 (挙動完全互換、不変条件テスト追加)。`crates/nnue-train::dataloader::Batch::push_decoded` も追加。
- **bucket-aware 並列パース prefetch**: `crates/nnue-train::dataloader::BucketedPrefetchedLoader` (新規)。`--threads` 本の worker が共有 `Mutex<PsvEpochReader>` から短い critical section で生 PSV を `batch_size` 件取り (I/O は逐次)、その外で `decode()` + HalfKA_hm sparse 抽出 + progress bucket 計算を並列実行し `(Batch, per-position bucket Vec)` を main に渡す。`ShogiHalfKA_hm` / `ShogiProgressKPAbs` は ZST + process-global `OnceLock` (read-only) なので thread 間共有可。`trainer.rs::EpochStream` の逐次 read + `--score-drop-abs` skip + EOF wrap (= 次 epoch) + `MAX_BARREN_PASSES` ガードは `dataloader::PsvEpochReader` に移設して loader 内に格納 (挙動同等、bucket 計算だけ worker 側に分離)。**worker 数 ≥ 2 では 1 epoch 内の position 順序が非決定的** (各 worker が batch_size 件ずつ排他的に読むため batch 境界が変わる; training では問題ない — bullet 自体 shuffle、適用される lr/wdl は loop の batch_idx で決まりデータ内容に依らない)。`num_workers = 1` で従来の決定論的逐次 read 相当。
- **ring-buffer return path**: `Batch` / per-position bucket `Vec` を起動時に `prefetch_depth + num_workers + 1` 個確保した std `mpsc::sync_channel` pool から借り、main が `train_step` 後 `BucketedPrefetchedLoader::recycle` で返す → worker が `Batch::reset()` / `Vec::clear()` で reuse。毎 batch の `Vec` 新規 alloc (#81 が指摘していた ~21MB) は起きない。`Batch::reset` の docstring (Stage 3-5 が予告していた follow-up) を更新。
- **`crates/nnue-train::trainer::run`** を新 loader に差し替え (`TrainingConfig` に `threads` フィールド追加)。`bins/nnue_train/src/main.rs` は `--threads` を実際に配線 (旧 `let _ = cli.threads;` no-op 撤廃)、`--epoch-file-shuffle` は引き続き no-op (out of scope; warning 文言更新)。`nnue-train` は `gpu-runtime`-free を維持 (prefetch threading は本 crate 内、std `thread`/`mpsc`/`Arc<Mutex>` のみ)。`crates/nnue-train` は CPU-only / kernel 追加なしのため `nnue_train.ll` の `kernel_names` / kernel 数 (27) は不変。
- **テスト変更**: `trainer.rs` の `run_drives_...` を threads=1 / threads=4 の 2 本に分割 (step 回数 / checkpoint / bucket / lr schedule は不変を確認)、`empty_data_file_errors_...` は loader 内 `PsvEpochReader` の barren ガード経由で error が伝搬することを確認。`dataloader.rs` に `BucketedPrefetchedLoader` の 1/4 worker streaming + recycle + 空 file error + score-drop skip テスト追加。`halfka_hm.rs` / `progress_kpabs.rs` に board 版 == 従来版 の不変条件テスト追加。
- **検証** (ローカル sm_75 / 8GB / WSL2): CPU 全 crate test ok、workspace clippy / fmt クリーン、GPU smoke `PASSED`、sample.psv 学習 (`--threads 1` / `--threads 4` 双方) loss 単調減少、`verify_nnue_accumulator` ALL PASSED、`NNUE_TRAIN_STEP_PROFILE` breakdown 維持。`#76` の GPU-bound vs dataloader-bound 再計測は out of scope。

#### Stage 3 #78 — GpuTrainer perf P4: 中間/grad buffer 永続化 (2026-05-13、bullet 新規 vendor 無し)

EPIC #75 のサブ issue。`bins/nnue_train/src/main.rs` のみ (bullet 由来コードの変更/追加なし、純粋に性能改善のリファクタ):

- **`GpuTrainer::step_impl` の per-step `DeviceBuffer::*::zeroed` を撤廃**: forward/backward の中間 activation (~30 個、`ft_stm_out`/`combined`/`l1_*`/`l2_*`/`l3_out`/`net_output`/`d*` 等) と grad buffer (10 個、`ft_w_grad` だけで ~450MB) を毎 step `DeviceBuffer::zeroed()` で再確保 → `cudaMalloc`/`cudaFree` が stream を stall させていた (nsys 計測 #76 で malloc/free ≈ step の 23%)。中間 activation を `GpuTrainer` 上の永続 `GpuWorkspace` (新規 struct、`GpuTrainer::new` で `batch_size` 分を確保、より大きな batch が来たら grow-only で再 alloc) に移し、grad / `loss_acc` は再 alloc せず `cuMemsetD8Async(0)` (`cuda_core::memory::memset_d8_async` 経由の generic host helper `memset_zero::<T>`) で in-place reset。
- **設計判断**: forward の各 activation は読まれる前に kernel が全 cell を上書きするため memset 不要 (workspace が現 batch より大きい末尾は後続 kernel も `b` で bound するので read されない)。例外は `dl1_total` (`slice_scatter_2d` の host 契約「dst を 0 初期化」を守るため毎 step memset)。入力 H2D buffer (`stm_idx_dev` 等) は per-step `DeviceBuffer::from_host` のまま (永続化は Issue #81 / P5 の範囲)。kernel は追加/削除/改名なし (memset は host-side API、`#[kernel]` ではない) → `compile_ll_to_ptx_via_llc` の `kernel_names` / kernel 数コメントは不変。
- **副作用**: `NNUE_TRAIN_STEP_PROFILE` の `teardown` tick (= per-step buffer の `Drop` = `cuMemFree`) が ~0 に落ちる (drop されるのは入力 H2D buffer のみになるため、期待動作)。数値は不変 (GPU smoke + verify-nnue ALL PASSED、sample.psv 学習 loss 単調減少)。

#### Stage 3 review follow-up (2026-05-13、bullet 新規 vendor 無し)

PR #92 (#84) / PR #91 (#76) の Codex review (P2) 指摘 2 件への follow-up。`bins/nnue_train/src/main.rs` のみ:

- **`GpuTrainer::load_v102_weights` の lookahead `slow` 初期化**: PR #92 で `--init-from` 経路も `slow = 0` にしたが、`--init-from` は量子化済 NNUE の continue-training (bullet checkpoint resume = `slow.bin` 付き、ではない) なので、初回 lookahead lerp (`new_w = alpha*fast + (1-alpha)*slow`) で読み込んだ重みが ~alpha 倍に縮む不具合があった。`load_v102_weights` の `slow` を **loaded weights と同値**に戻し (初回 lerp が `new_w = alpha*fast + (1-alpha)*w_loaded` で、fine-tuning は lr 小さく `fast ≈ w_loaded` なので 0 でなく読み込んだ重みの方へ寄せる anchor になる)、`GpuTrainer::new` (from-scratch) は bullet `RangerLookahead::new` どおり `slow = 0` のまま (v102 厳密再現用) という分担に。
- **`GpuTrainer::step` を `step` (薄い wrapper) + `step_impl` (実体) に分割**: PR #91 の `NNUE_TRAIN_STEP_PROFILE` breakdown は最後の `prof_tick!` が `step()` 関数内で打たれるため、per-step `DeviceBuffer` ローカル (入力 / 中間 activation / 一部 grad) の `Drop` (= `cuMemFree`) が `}` でしか走らず breakdown に含まれていなかった (nsys では free が step の ~2 割)。`step_impl` を別関数にして return 時にローカルを drop させ、wrapper 側で `teardown` tick を打つことで free 時間も計上する (Issue #78 で workspace 永続化すれば per-step alloc/free 自体が消える)。

#### Stage 3 #84 — loss_wrm (bullet win-rate-model loss、2026-05-13、bullet-shogi commit `488d81b`)

3-observer code review (2026-05-12) の L1/L2/L6 指摘に対応。v102 recipe
(`--win-rate-model --wrm-in-scaling 340 --wrm-nnue2score 600 --scale 290 --wdl 0.0`)
を厳密再現するための WRM 損失 + lookahead slow=0 初期化:

- **`crates/gpu-kernels/src/pointwise/loss_wrm.rs`** (新規): `loss_wrm_cpu` reference 実装。
  bullet `examples/shogi_layerstack.rs:2177-2188` の `loss_fn_wrm` (`--win-rate-model` +
  `--wrm-in-scaling` 指定時に選ばれる loss closure) + `crates/bullet_lib/src/value/
  loader.rs:300-316` (data-layer の WRM target `0.5*(1 + sigmoid((score-270)/380) -
  sigmoid((-score-270)/380))` + WDL blend `blend*result + (1-blend)*score`) を NNUE 専用に
  hand-fuse。target 側 in_scaling (380) / offset (270) は bullet ハードコード、prediction 側
  in_scaling は `--wrm-in-scaling` (340) — この非対称も bullet どおり。`loss_wdl` (sigmoid-MSE)
  と違い prediction / target 双方に WRM を適用するため net_output が `out ≈ cp / nnue2score`
  (`= cp/600`、O(1)) で収束し、`crates/nnue-format` の量子化 (`QA=127/QB=64/FV_SCALE=28`、
  bullet の `out ≈ cp/600` スケール前提) と整合する。gradient (`dl_dout = err *
  (nnue2score/in_scaling) * (q(1-q) + qm(1-qm)) * per_pos_norm`、`2` と `0.5` が打ち消し合う形)
  は chain rule から導出し finite-difference テストで照合。`gpu_kernels::pointwise::mod` に登録
- **`bins/nnue_train/src/main.rs::loss_wrm`** (新規 `#[kernel]`): 上記 `loss_wrm_cpu` の GPU 版
  (Stage 1-5 で確立した「`#[kernel]` は bin entry に inline」制約)。`f32::exp` は libdevice
  (`__nv_expf`) に lowering。`kernel_names` list (`compile_ll_to_ptx_via_llc`) と module doc の
  kernel 数も 26 → 27 に更新
- **`crates/nnue-train/src/trainer.rs::LossKind`** (新規 enum、CPU-only): `Sigmoid { scale }`
  (旧来 `loss_wdl`、`out ≈ cp`) / `Wrm { nnue2score, in_scaling }` (bullet WRM、`out ≈ cp/600`)。
  `TrainerBackend::train_step` / `TrainingConfig` の `loss_scale: f32` を `loss: LossKind` に変更
  (kernel 選択を bin 側 `GpuTrainer::step` の `match` に委譲、Stage 3-0 規約: crate は GPU 非依存)
- **CLI 配線**: `--win-rate-model` 指定時に `loss_wrm` 経路 (`--wrm-in-scaling` / `--wrm-nnue2score`
  を `LossKind::Wrm` に)、未指定なら従来 `loss_wdl` + `--scale` (`LossKind::Sigmoid`)。
  Stage 3-8 の「`--win-rate-model` は受けるが未配線で warning」を撤廃
- **lookahead `slow` 初期化を 0 に変更** (`GpuTrainer::new` / `load_v102_weights` の `*_slow`
  buffer)。bullet `crates/trainer/src/optimiser/ranger.rs::RangerLookahead::new`
  (`slow_params: Buffer::from_host(device, &TValue::F32(vec![0.0; size]))?`) と同じで、
  初回 lerp (`step % k == 0`) で `weights = alpha*weights + (1-alpha)*0 = alpha*weights` に
  なる挙動も bullet と一致 (Stage 3-6 で「意図的 divergence」として weight 初期化していたのを
  v102 厳密再現のため bullet に揃えた)

#### Stage 3-quality / perf-P2 (2026-05-12)

bullet からの新規 vendor は無し。3-observer code review (2026-05-12) 指摘の品質修正
(#86) + perf P2 (#77、EPIC #75) のみ:

- `bins/nnue_train/src/main.rs::dense_mm_bwd_weight_bucket` を「1 thread = 1 (bucket,
  out, in) weight cell + batch inner loop」形に書き換え (atomic scatter 撤廃)。bullet
  上流で同等の per-bucket weight grad を runtime-fused でどう生成するかは変わらないが、
  本リポの hand-fused 実装としては non-bucket 版 `dense_mm_bwd_weight` と同流儀に統一
- `crates/nnue-train/src/optimizer.rs::RangerParams::DEFAULT` (const) を追加。bullet
  `RangerParams::default()` の値を single source of truth とし、`bins/nnue_train` の
  Ranger const (BETA1/BETA2/EPS/MIN_W/MAX_W/RANGER_ALPHA/RANGER_K/N_SMA_THRESHOLD) は
  これを参照 (旧 main.rs の二重定義を解消、bullet 由来値との対応が 1 箇所で追える)
- `crates/nnue-format/src/v102_layerstack.rs::compute_fc_hash` を `const fn` 化し、
  `FC_HASH` const を `compute_fc_hash(...)` 評価に置換 (旧実装の手 unroll した別 const
  式を削除、bullet `compute_layerstack_fc_hash` との対応が 1 箇所)
- `crates/shogi-format/src/packed_sfen.rs::sfen()` の `unsafe` 生ポインタ cast に
  詳細 SAFETY コメント補強 (`#[repr(C)]` layout / align=1 / `[u8;N]` の bit pattern /
  lifetime)、`unsafe impl Send/Sync` のコメントも `// SAFETY:` 形式に
- doc 修正: `bins/nnue_train` の kernel 数コメント 24 → 26 (`slice_extract_2d` /
  `slice_scatter_2d` を NEW list に追記)、`dense_mm_bwd_weight_bucket` の説明を
  「atomic」から新実装に更新、`load_quantised` の docstring に「continue-training 時
  l1f を畳んだ影響で bullet の学習軌跡とは厳密一致しない」注を追記、`GpuTrainer::step`
  の末尾 `loss_vec[0]` 無防備 index を `.first().copied().ok_or(...)` に

#### Stage 3-8 (2026-05-12, bullet-shogi commit `f275eb9`)

- **`crates/nnue-train/src/trainer.rs`** (新規): superbatch training loop driver。
  - bullet `crates/bullet_lib/src/value.rs::ValueTrainerInner::train_custom`
    相当の loop を `TrainerBackend` trait 越しに駆動 (per superbatch:
    `LrScheduler::lr` / `WdlScheduler::blend` → PSV batch + per-position bucket
    → `TrainerBackend::train_step` → loss 集計; per `save_rate` superbatch:
    `TrainerBackend::save_checkpoint`)。Stage 1 `bins/progress_kpabs_train::
    train_one_epoch` 流儀の直書き loop で、bullet `bullet_core::Trainer` /
    `DataLoader` trait / `LoggingConfig` には依存しない (Stage 1-1 / 3-1 / 3-4
    と同じ bullet trait 削除ポリシー)。GPU step 本体は trait 越しに bin 側
    (`bins/nnue_train::GpuTrainer`) が担い、本 module は CPU-only に保つ
    (Stage 3-0 規約)
  - per-position output bucket は progress8kpabs (`ShogiProgressKPAbs::bucket`、
    `crates/shogi-features/src/progress_kpabs.rs`、Stage 3-1 で bullet
    `crates/bullet_lib/src/game/outputs.rs::ShogiProgressKPAbs` を vendor 済)
  - bullet `examples/shogi_layerstack.rs:1469-1495` の `display_step_log` 相当の
    progress log (`superbatch X/Y | loss | pos/s | lr | wdl | ETA`)
  - bullet `--score-drop-abs` (`5c4871c`: `|score| >= t` の per-position loss
    weight を 0 にする) は本実装では「batch に push しない (skip)」で近似
    (loss/gradient へ寄与しない点は同じ、batch slot 割当・順序は厳密一致しない)
  - bullet `--epoch-file-shuffle` は本 stage では未実装 (CLI フラグは受けるが
    no-op、`PsvFileLoader` を逐次 read + EOF で同 file を開き直して次 epoch)

- **`bins/nnue_train/src/main.rs`** に CLI 統合 (Stage 3-7 で landed した
  `GpuTrainer` を `nnue_train::trainer::run` で駆動):
  - bullet `examples/shogi_layerstack.rs` の CLI 引数群を v102 recipe (memory
    `project_v102_recipe.md`) に合わせて `clap` で受ける (`--data` / `--output` /
    `--net-id` / `--superbatches` / `--batches-per-superbatch` / `--batch-size` /
    `--lr` / `--lr-gamma` / `--lr-step` / `--wdl` / `--scale` / `--save-rate` /
    `--progress-coeff` / `--score-drop-abs` / `--init-from`、および互換のため
    受けるが本 stage 未配線の `--win-rate-model` / `--wrm-in-scaling` /
    `--wrm-nnue2score` / `--optimizer` / `--weight-decay` / `--threads` /
    `--bucket-mode` / `--epoch-file-shuffle` / `--file-shuffle-seed`)
  - `--data` 省略時は Stage 3-7 の GPU smoke test を実行する (後方互換)
  - `GpuTrainer::step` は Stage 3-7 の固定 const `WDL_LAMBDA` / `LOSS_SCALE`
    を引数 (`wdl_lambda` / `loss_scale`) に変更し、CLI の `--wdl` / `--scale`
    から渡せるようにした (smoke test は引き続き const を渡す)
  - `BatchData::from_batch` で dataloader `Batch` + per-position bucket を
    `GpuTrainer::step` 入力に変換。`per_pos_norm = 1/n_pos` で weight gradient を
    batch 平均にし (bullet の `mean` reduction と同義、`loss_wdl` kernel が
    per-position gradient に `per_pos_norm` を掛ける)、learning rate を batch
    size 非依存にする

新規追加 (bullet 由来ではない):

- `crates/nnue-train/src/trainer.rs` の `TrainerBackend` trait / `TrainingConfig`
  struct (+`validate`: superbatch 範囲 / batch 構成 / save_rate / loss_scale /
  score_drop_abs `>= 1` を reject) / `EpochStream` (PSV を EOF で開き直す stream +
  score-drop skip + per-position bucket 計算 + 空 file / 全 drop の無限ループ検出
  `MAX_BARREN_PASSES`) / `format_hms` — bullet 上流に対応する独立 API は無い
  (loop の組み立てのみ bullet `ValueTrainerInner` を参照)
- `crates/nnue-train/src/trainer.rs` の test 4 件 (`run_drives_superbatches_and_
  writes_checkpoints` / `empty_data_file_errors_instead_of_looping_forever` /
  `config_validate_rejects_bad_ranges` / `format_hms_renders_expected_buckets`、
  mock `TrainerBackend` で GPU 非依存に検証) — `crates/nnue-train` の test は
  本 PR 後 計 44 件 (= 既存 40 + trainer 4)
- `bins/nnue_train/src/main.rs` の `Cli` struct / `run_training` (`--bucket-mode` /
  `--optimizer` の対応値チェック + `--lr` / `--lr-gamma` / `--scale` / `--wdl` の
  finite / 範囲チェックで NaN・不正値を kernel に流さない) / `impl TrainerBackend
  for GpuTrainer` / `BatchData::from_batch`

検証成果:

- ローカル sm_75 (RTX 2070 SUPER) で `sample.psv` (100 records) を入力に
  `--superbatches 30 --batches-per-superbatch 8 --batch-size 100 --lr 1.0` で
  完走、loss が `0.114 → 0.092 → 0.032 → 0.013 → … → 0.006` と (初期 transient 後)
  単調減少することを確認 (`--progress-coeff` 指定版でも同様に減少)。出力
  checkpoint `.bin` が `rshogi-oss/target/release/verify_nnue_accumulator
  --ls-progress-coeff progress.bin --moves 10` で **Total: 10, Pass: 10, ALL
  PASSED**。`--data` 省略時の Stage 3-7 GPU smoke test も引き続き PASS

#### Stage 3-7 (2026-05-12, bullet-shogi commit `f275eb9`)

- **`bins/nnue_train/src/main.rs`** に v102 LayerStack 1536-16-32 + progress8kpabs
  9 buckets の **GpuTrainer** と 26 個の `#[kernel]` を inline 配置 (cuda-oxide
  bin-entry reachability 制約のため):
  - bullet `examples/shogi_layerstack.rs:2206-2289` (`build_trainer_with_input!`
    macro 内 forward closure) → 本 file の `GpuTrainer::step` forward path に
    対応する 15 step:
    1. sparse_ft_forward × 2 (stm, nstm)
    2. ft_post_perspective_fwd (FUSED bias add + CReLU + pairwise_mul + scale)
    3-5. dense_mm_fwd_bucket L1 + dense_mm_fwd L1f shared + elementwise_add
    6-7. slice_extract_2d (l1_main, l1_skip)
    8. abs_pow2_scale_fwd
    9. concat_l1sqr_main_fwd
    10. crelu_fwd
    11. dense_mm_fwd_bucket L2
    12. crelu_fwd
    13. dense_mm_fwd_bucket L3 (per-bucket output)
    14. elementwise_add (+l1_skip)
    15. loss_wdl
  - bullet `crates/trainer/src/model/builder.rs:477-560` の op semantics:
    - `pairwise_mul()`: `slice_rows(0, n/2) * slice_rows(n/2, n)` (前半・後半対応 index 同士の積)
    - `abs_pow(2)`: `|x|^2 = x^2` (本 kernel は `x*x*scale` で実装、abs 不要)
    - `crelu()`: clip(0, 1)
  - 7 STATED kernel (`screlu_grad` / `loss_wdl` / `adamw_step` / `radam_step` /
    `ranger_lookahead_lerp` / `sparse_ft_forward` / `sparse_ft_backward`) は
    Stage 2 (#46-#52) で landed したものを本 file に copy-inline (cuda-oxide
    rustc-codegen-cuda backend は bin entry から reachable な kernel のみ
    NVPTX IR 化する制約、Stage 1-5 で確立)。`screlu_grad` と `adamw_step` は
    v102 path では未使用だが compile-reach のため preserve

- **`crates/nnue-format/src/v102_layerstack.rs`** (新規):
  - bullet `examples/shogi_layerstack.rs:1411-1809` (`build_layerstack_save_format`)
    → 本 module の `V102Weights::save_quantised` / `load_quantised`
  - rshogi-oss `crates/rshogi-core/src/nnue/leb128.rs::LEB128_MAGIC` = `b"COMPRESSED_LEB128"`
    (17 bytes) と同 magic で signed LEB128 i16 encoder/decoder を実装、bullet
    の FT bias / weight 圧縮形式と互換
  - bullet `shogi_layerstack.rs:1706-1715` の per-bucket l1 + shared l1f
    **merge save** (bullet が Factorizer 形式を coalesce してから書き出す動作)
    を再現。本実装 V102Weights は load 時 merged 値を `l1_w` に格納し
    `l1f_w = 0` で復元 (forward 計算上等価)
  - 量子化 scale: bullet `shogi_layerstack.rs:1655, 1731, 1757` 由来
    (FT: `QA=127` i16、L1 bias: `QA*QB=8128` i32、L1 w: `QB=64` i8、L2/L3 bias:
    `127*QB=8128` i32、L2/L3 w: `QB=64` i8)
  - pad32 (SIMD alignment): bullet `shogi_layerstack.rs:1704, 1739, 1765`
  - arch_str format: bullet `shogi_layerstack.rs:1469-1495` 由来、v102 では
    PSQT 無し / Threat 無し / HandCountDense 無し

新規追加 (bullet 由来ではない):

- `bins/nnue_train/src/main.rs` の **17 NEW kernel** (`ft_post_perspective_fwd`/
  `_grad` / `dense_mm_fwd` / `_bwd_input` / `_bwd_weight` / `bias_grad` /
  `dense_mm_fwd_bucket` / `_bwd_input_bucket` / `_bwd_weight_bucket` /
  `bias_grad_bucket` / `crelu_fwd` / `crelu_grad` / `abs_pow2_scale_fwd` /
  `_grad` / `concat_l1sqr_main_fwd` / `_grad` / `elementwise_add` /
  `slice_extract_2d` / `slice_scatter_2d`) — bullet は PointwiseIR runtime fusion
  で同等動作を生成、本リポは hand-fused inline kernel として再実装
- `crates/nnue-format/src/v102_layerstack.rs::V102Weights` struct — bullet
  `SavedFormat` trait 機構は使わず direct quantize/dequantize で完結 (Stage 1-1
  / Stage 3-1 / Stage 3-3 と同方針)
- `crates/nnue-format/src/v102_layerstack.rs` の test (`leb128_*_roundtrip`,
  `pad32_correct`, `weights_zeroed_save_load_roundtrip`, `arch_str_format`)
  および file 依存 (`load_v102_100_reference_if_available`,
  `save_v102_100_resaved_if_available`) — file 不在時は skip するため CI でも
  通る (実機ローカル box で v102-100/quantised.bin が `/tmp` にある場合のみ動作確認)

検証成果:

- 本実装出力 quantised.bin が rshogi-oss `verify_nnue_accumulator`
  (refresh vs differential update 一致 test) を **ALL PASSED 10/10**
- bullet v102-100 (sb=100 reference checkpoint、116MB) との byte 比較:
  116,472,404 bytes 中 **42 bytes 差のみ** (network_hash 4 + 9 fc_hash × 4 = 36 +
  rounding boundary 2)、forward 計算は完全互換

参照リンク:
- bullet `examples/shogi_layerstack.rs:1411-1809, 2206-2289`
- bullet `crates/trainer/src/model/builder.rs:477-560`
- rshogi-oss `crates/rshogi-core/src/nnue/{network_layer_stacks.rs:138-311,
  layer_stacks.rs:203-223, leb128.rs}`
- `docs/bullet_v102_save_format_report.md` (Codex 詳細 recon)

#### Stage 3-6 (2026-05-12, bullet-shogi commit `f275eb9`)

- `crates/nnue-train/src/optimizer.rs` (新規 ~400 行): Ranger (RAdam +
  Lookahead) の **host-side state** + パラメータ + checkpoint serialise +
  12 件の test。
  - **bullet 上流参照** (`crates/trainer/src/optimiser/{radam,ranger}.rs`):
    `RAdam::update` (`radam.rs:198-218` の host pre-compute) と
    `RangerLookahead::update` (`ranger.rs:82-100` の step counter + lerp 起動
    条件) の host 側 control path を本リポ独自 struct に再構築。bullet trait
    `OptimiserState<G: Gpu>` / `WrapOptimiser<Inner, Params>` は **取り込まず**
    (Stage 1-1 / 3-1 / 3-4 / 3-5 と同 trait 削除ポリシー)
  - **device buffer は持たない (CPU-only crate 維持)**: bullet `Buffer<G>` は
    本リポでは `Vec<f32>` (host)。device 側は Stage 3-7 で
    `bins/nnue_train/src/main.rs::GpuTrainer` が `DeviceBuffer<f32>` に
    host-to-device コピー (Stage 1-9 `GpuTrainer` pattern と同流儀)
  - **kernel `build_ranger_op` は不要**: bullet 上流の `PointwiseIR` で kernel
    構築する dynamic build は Stage 2-5 で landed した GPU `#[kernel] fn
    ranger_lookahead_lerp` (bin entry inline) で置き換え済。本 module は
    `radam_compute_step_size_denom` (Stage 2-4 で `gpu-kernels` に landed) を
    `pub use` re-export して trainer に直接呼ばせる
  - **checkpoint format は本 PR で確定** (bullet 2 file 構成 `slow.bin` +
    `step_ranger.txt` を 1 binary に集約、Stage 1 progress.bin 慣行):
    - 0..4 magic `b"RNGR"`
    - 4..8 version u32 LE (1)
    - 8..16 step u64 LE
    - 16..24 n_params u64 LE
    - 24.. momentum / velocity / slow_params 各 f32 LE × n_params
  - **bullet `RangerLookahead::new` の slow_params 0-init を訂正**:
    bullet は `slow_params = Buffer::from_host(.., vec![0.0; size])` で 0 init
    だが、初期 weight が非零の場合 lerp 初回で `alpha * w + (1-alpha) * 0 =
    alpha * w` と weight を半減させてしまう。本リポは `new_with_initial_weights
    (&[f32])` で **明示的に初期 weight を slow にコピー** する API も提供し、
    `new_zeroed(n)` (bullet 互換) と両方公開
  - **MSRV 1.85 罠回避**: bullet `step.is_multiple_of(self.k)` (1.87 stable) は
    `step % (k as u64) == 0` 直書き (Stage 2-5 / 3-3 で踏破済の規約踏襲)
  - **Codex review #62 で追加対応 (本 PR 内 amend)**:
    1. `load_from_reader` に `expected_n_params: Option<usize>` 引数追加、checkpoint
       内 `n_params` と照合し不一致なら `InvalidData` reject (Stage 3-7 trainer 安全策)
    2. `u64 → usize` cast を `try_into()` 経由に変更、32-bit target / 破損ファイルでの
       overflow を `InvalidData` で reject
    3. `RangerHostState::reset` が `slow_params` を 0 fill する挙動は bullet 上流
       `RangerLookahead::reset` (slow 不変) と **意図的 divergence**。docstring +
       `reset_zeros_slow_params_diverging_from_bullet` test で明示化、`new_with_initial_weights`
       後に reset 呼ぶと初期化が失われる旨と回避方法 (radam のみ手動 reset / 新規 `new_with_initial_weights`
       で再構築) を doc 記載
    4. `save_to_writer` の sequential `write_all` × n_params (large N で system call 多発)
       は呼び出し側 `BufWriter` wrap 推奨を docstring に明記
  - **kernel launch は本 module には置かない**: GPU `#[kernel]` 本体は cuda-oxide
    bin-entry inline 制約のため本 crate (CPU-only) では持てず、Stage 3-7
    `bins/nnue_train/src/main.rs::GpuTrainer` が `cuda_launch!` で起動する。
    本 module は host state + step counter + `should_lookahead(k)` 判定 + I/O
    の責務分離 (Stage 1-9 と同流儀)
  - test 12 件: `RangerParams::default` bullet 一致 1 + `RAdamHostState`
    advance/compute/reset 3 + `RangerHostState` new/should_lookahead/reset 3
    + checkpoint round-trip 1 + reject (magic / version / dim mismatch) 3 +
    zero-size state edge 1

#### Stage 3-5 (2026-05-12, bullet-shogi commit `f275eb9`)

- `crates/nnue-train/src/dataloader.rs` (新規 ~360 行): PSV file →
  HalfKA_hm sparse batch (`Batch { stm_indices, nstm_indices, score, wdl,
  per_pos_norm, n_positions }`) + `PsvFileLoader` (single-thread stream
  reader) + `PrefetchedLoader` (background thread + mpsc sync_channel
  prefetch wrapper) + 9 件の test。
  - **弱い vendor 参照** (bullet `value/dataloader.rs` / `value/loader.rs`):
    bullet 上流の `ValueDataLoader<I, O, D, W>` (`bullet_compiler::TValue` /
    `OutputBuckets` / `LoadableDataType` / `WdlScheduler` 等 trait 群 depend) は
    本リポでは **取り込まず**、Stage 1 `bins/progress_kpabs_train/src/host/
    batch.rs` 流儀の直接 struct 実装 + minimum prefetch wrapper に簡素化
    (Stage 1-1 / 3-1 / 3-4 と同じ bullet trait 削除ポリシー)
  - **data-layer WDL blend pre-compute を行わない**: bullet `loader.rs:301-315`
    の `blend * result + (1-blend) * sigmoid(rscale * score)` を data layer で
    やる方式は本リポでは **採用せず**、Stage 2-2 `fused_loss_wdl` kernel が
    GPU 側で blend を fuse する設計 (Stage 2-2 ATTRIBUTION 参照) に合わせて、
    本 dataloader は **`score` (raw cp) と `wdl` (game result {0, 0.5, 1}) を
    別 buffer に保持** する
  - **sparse layout は bullet と同型 `-1` padding**: STM/NSTM 各 `[batch_size,
    max_active]` flat (`bi * max_active + j`)、`map_features` で fill しきれ
    なかった slot は `-1` padding (Stage 2-6 `sparse_ft_forward` の silent
    skip semantics と整合)
  - **map_features (symmetric) のみ使用**: bullet `map_features_split` の
    asymmetric STM/NSTM Option emit は ShogiHalfKA_hm では不要 (HalfKA_hm は
    両視点同時 emit)、本リポは `ShogiHalfKA_hm::map_features` で symmetric fill
  - **multi-thread prefetch は minimum wrapper**: `std::thread::spawn` +
    `std::sync::mpsc::sync_channel(prefetch_depth.max(1))` で background loader
    が batch を 1 つ先まで先読み、main thread が `next_batch()` で取得。bullet の
    多段 thread pool / batched prefetch / hand_count dense input 等は本リポ
    scope 外 (Stage 3-8 trainer integration で必要になれば別 issue で拡張)
  - **`Batch.reset()` の alloc 再利用は single-thread loader (`fill_batch`)
    経路のみ**: Codex review #61 指摘で明示化。`PrefetchedLoader` 経路では
    mpsc channel が所有権ごと main thread に batch を送るため background 側で
    `reset` 再利用不可、毎ループ `Batch::with_capacity` を呼ぶ simplification を
    採用。ring buffer 的な return path は Stage 3-7/3-8 で必要時に追加
  - **`Batch.reset()` の alloc 再利用は `PsvFileLoader::fill_batch` 内部のみ**:
    `PrefetchedLoader` の background loop は channel 経由 send=move のため毎
    iteration `Batch::with_capacity` を新規 alloc する設計 (Codex review #61 で
    設計メモと実装不一致を指摘、docstring に「reuse は fill_batch 経由のみ、
    prefetch 内部は send move で alloc 残る」と訂正明記)。alloc が trainer
    ホットパスでボトルネックになった段階で `Clone` 経由 send / `Arc<Batch>` 化 /
    double-buffer 化等を Stage 3-7/3-8 で検討する follow-up とする (本 PR は
    正しさ優先、性能 follow-up)
  - **`PackedSfenValue::result()` (enum 経由) を使い、{0.0, 0.5, 1.0} に正規化**:
    bullet `loader::GameResult::{Loss=0, Draw=1, Win=2}` 慣行に揃え、bullet
    `loader.rs:312` の `f32::from(pos.result() as u8) / 2.0` と同型。
    **注意 (Codex review #61 で修正)**: `PackedSfenValue::game_result()` は
    **raw i8** で PSV wire 形式の `{-1=Loss, 0=Draw, +1=Win}` を返す。本 PR 初版は
    これを `.max(0) / 2.0` で正規化しており、Draw → 0.0、Win → 0.5 に
    **誤マップする数値バグ**だった。sample.psv は偶然 Draw を含まない fixture
    で test がすり抜けたが、実 PSV では Win 局面が Draw として fused_loss_wdl
    kernel に渡り、学習が壊れる。amend で `pos.result() as u8 / 2.0`
    (`packed_sfen.rs:473`、`GameResult` enum) 経由に修正、回帰防止 test
    (`fill_batch_wdl_covers_loss_and_win_with_correct_values`) を追加
  - test 内訳: Batch init/state 2 + `PsvFileLoader` stream/eof 2 + fill_batch
    sparse 検証 (index 範囲 + wdl 範囲 + EOF partial) 3 + push reject /
    reset 1 + PrefetchedLoader 2 = 計 9 件、sample.psv (`shogi-format/tests/
    data/sample.psv`、100 records / 4000 bytes) を共用 fixture とする

#### Stage 3-4 (2026-05-12, bullet-shogi commit `f275eb9`)

- `crates/bullet_lib/src/trainer/schedule/lr.rs` (215 行) +
  `crates/bullet_lib/src/trainer/schedule/wdl.rs` (141 行) を
  `crates/nnue-train/src/schedule.rs` (新規) に集約 vendor。Stage 3-8 trainer
  loop で `superbatch / batch` index ごとに `LrScheduler::lr` /
  `WdlScheduler::blend` を呼び、Stage 2 `adamw_step` / `radam_step` /
  `fused_loss_wdl` kernel の `lr` / `lambda` 引数として渡す host-side state。
  - **vendor scope**: bullet の lr 系 (ConstantLR / DropLR / StepLR /
    LinearDecayLR / CosineDecayLR / ExponentialDecayLR / Warmup<LR> /
    Sequence<F, S>) + wdl 系 (ConstantWDL / LinearWDL / Warmup<WDL> /
    Sequence<F, S> / WdlSchedulerEnum) を **全て移植**
  - **bullet trait の `colourful(&self) -> String` 削除 → `std::fmt::Display`
    impl に置換**: bullet は `bullet_trainer::run::logger::ansi(value, color_code)`
    で terminal 用 ANSI 色付き string を返す。本リポでは外部 dep を持たず、
    `Display` impl で plain string を返す形に統一 (Stage 1-1 / 3-1 と同じ
    bullet 外部 dep 削除ポリシー、CLI 側で色付けが必要なら別途 wrap)
  - **計算 path (`lr` / `blend`) は bullet 上流と byte 一致**: `saturating_sub` /
    `(max - 1).max(1)` 等の defense / `superbatch == 1 && batch < warmup_batches`
    の warmup condition / `Sequence` の `max - midpoint` 正規化、全て移植元と一致
  - 構造体命名は bullet 上流の `Warmup<LR>` / `Sequence<F, S>` が lr.rs と wdl.rs
    で同名衝突するため、本リポでは `WarmupLR<LR>` / `SequenceLR<F, S>` /
    `WarmupWDL<W>` / `SequenceWDL<F, S>` と接尾辞で分離。bullet 上流は別 module
    namespace で衝突回避していたが、本リポは単一 module 集約方針 (`schedule.rs`)
    のため命名で区別
  - **trait bound `+ Display`**: bullet 上流 trait は `Clone + Debug + Send +
    Sync` のみ、本リポは `Display` を追加して `format!("{lr}")` で plain
    string 取得可能に
  - **`LrScheduler` trait に `'static` を新規追加**: bullet 上流 (`lr.rs:7`) の
    `LrScheduler` は `'static` を持たず、`WdlScheduler` (`wdl.rs:7`) のみ
    `+ 'static` を持つ。本 PR は両 trait を揃えて `+ 'static` を要求 (Stage 3-8
    trainer state で `Arc` 共有 / thread spawn する想定、borrow を持つ
    scheduler は許さない設計)。WdlScheduler の `'static` は上流から保持。
    Codex review で「保持」誤記を指摘され訂正
  - test 14 件 (lr 系 8 件 + wdl 系 6 件、hand-calc fixture で bullet 上流値と
    数値一致確認: warmup interp / sequence midpoint 切替 / linear taper /
    cosine progress=0.5 / exponential mid-factor 等)

#### Stage 3-3 (2026-05-12, bullet-shogi commit `f275eb9`)

- `crates/nnue-format/src/halfka_psqt.rs` (新規、約 430 行): HalfKA_hm + PSQT
  NNUE binary (FT + L1 + PSQT) の `save_quantised` / `load` + 量子化 helper +
  12 件の test (本 PR 内 amend で 10 → 12 に追加、Codex / Claude review 指摘の
  edge case 対応)。
  - **bullet 上流移植**: `crates/trainer/src/model/save.rs::QuantTarget` +
    `quantise` 関数 (`model/save.rs:153-211`、約 60 行) を本リポ `QuantTarget`
    enum + `quantise` / `dequantise` method として移植。i8 / i16 / i32 ×
    multiplier の量子化スキーム、`round_or_trunc`、範囲超過時 `InvalidData`
    return は bullet 上流と同型。`SavedFormat` / `ModelWeights` の trait 機構
    (`SavedFormat::transform`, `SavedFormat::quantise<T: Quant>` 等) は本リポ
    では使わず、`HalfKAPsqtNet` 構造体の field を direct quantise する形に簡素化
    (Stage 1-1 / Stage 3-1 と同じ bullet trait 削除ポリシー)
  - **新規 layout 確定** (本 PR 独自): NNUE binary は header (Stage 3-2) +
    ft_weights i16 LE + ft_bias i16 LE + l1_weights i8 LE + l1_bias i32 LE +
    psqt i32 LE の 5 ブロック。量子化 multiplier は `header.qa` / `qb` /
    `qa*qb` を使う。Stage 3-9 (#64) で rshogi 側 loader と互換性検証の上で調整
  - **scope minimum (本 PR)**: Issue #59 body の `FT + L1 + PSQT` を文字通り
    実装。NNUE 1536-16-32 full arch (FT + 3 linear stack [16, 32, 1] + PSQT、
    bullet `examples/shogi_layerstack.rs` の `layer_sizes = [(16, true),
    (32, true), (1, false)]`) は Stage 3-7 / 3-8 (`bins/nnue_train` 統合) で
    本 layout を拡張する想定 (`HalfKAPsqtNet` を `enum NnueLayout { Minimal,
    LayerStack {...} }` 化など)。本 PR docstring に明記
  - `compute_psqt_material_values` 等の HalfKA_hm 特化 PSQT 初期化
    (`shogi_layerstack.rs:1194-`) は本 PR scope 外 (trainer 側 init で扱う、
    Stage 3-7 / 3-8 で組み込む想定)
  - **PSQT multiplier の bullet 互換修正 (Codex review 指摘で本 PR 内 amend)**:
    旧実装は psqt multiplier に `qa` のみを使っていたが、bullet 上流
    `shogi_layerstack.rs:1555-1570` の LayerStack PSQT は `qa * qb` を使う。
    YaneuraOu / nnue-pytorch 互換のため amend で `qa * qb` に修正
  - **PSQT shape の本 PR scope 限定**: bullet LayerStack の PSQT は
    `output_buckets (9) × num_features` の 2D weight + 9 個 bias 構造だが、
    本 PR は **single bucket (`psqt: Vec<f32>` 長さ `num_features`、bias なし)**
    に限定 (Issue #59 minimum scope)。multi-bucket 化は Stage 3-7 / 3-8 で
    NNUE 1536-16-32 full arch 拡張時に対応する想定 (`HalfKAPsqtNet` enum 化など)
  - test 内訳: `validate` 2 件 / `QuantTarget` 4 件 (i16/i8/i32 round-trip +
    範囲超過 reject) / `HalfKAPsqtNet::save_quantised` round-trip 2 件 (default
    + non-default qa/qb) / dim mismatch reject 1 件 / `NUM_FEATURES` 整合 1 件
    + **本 PR amend 追加 2 件** (`dequantise_rejects_unaligned_len` /
    `load_rejects_truncated_input`、Codex / Claude review 指摘) = 計 12 件。
    production 73_305 × 1536 は test には重すぎるため、mini net (num_features=4,
    ft_out_dim=2, l1_out_dim=1) で round-trip 検証
  - byte-exact 一致確認は本 PR の test では行わず、Stage 3-9 (#64) で bullet
    `cargo run --example shogi_layerstack` 出力との 1:1 比較を docstring に
    手順記載

#### Stage 3-2 (2026-05-12, bullet-shogi commit `f275eb9`)

- `crates/nnue-format/src/header.rs` (新規 200 行強): NNUE binary 先頭の
  固定長 22 bytes header (`NnueHeader { net_id, fv_scale, qa, qb }`) の
  serialise / deserialise + 8 件の test。
  - **弱い vendor 参照**: bullet 上流 `crates/bullet_lib/src/value/save.rs::save_quantised`
    (`save.rs:78-111`) では header 概念が `ModelWeights::write_to_byte_buffer` に
    分散しており、`NnueHeader` のような明示 struct は存在しない。本実装は
    rshogi (将棋エンジン) 互換性のため **本リポ独自の minimal header layout**
    を確定する形 (Stage 3-9 #64 自己対局検証で rshogi 側 loader と整合性検証の
    上で調整可能)。bullet 上流からは「header に `net_id` / `fv_scale` / `qa` / `qb`
    を置く方針」のみ参照、binary layout 自体は本 PR で新規確定
  - layout: net_id 16 bytes (UTF-8 + NUL padding) + fv_scale 2 LE + qa 2 LE +
    qb 2 LE = **22 bytes 固定**。LE 採用は Stage 1
    `bins/progress_kpabs_train/src/host/progress_bin.rs` (YaneuraOu 互換 f64 LE)
    の慣行を踏襲
  - 既定値: `fv_scale = 16` (YaneuraOu typical)、`qa = qb = 64` (placeholder、
    Stage 3-3 `halfka_psqt` で actual quantisation 値が確定したら trainer から
    上書きする想定)
  - test 内訳: default 値 / round-trip (default + non-empty id + 15 bytes 最大 id) /
    write reject (16 bytes 超過) / write byte-level layout / read NUL padding /
    read NUL なし (16 bytes 全部 id) の 8 件
  - 後続: Stage 3-3 (#59) で weight 本体の前段として本 header を呼び出し、
    Stage 3-9 (#64) で rshogi 側 loader との互換性を検証

#### Stage 3-1 (2026-05-12, bullet-shogi commit `f275eb9`)

- `crates/bullet_lib/src/game/inputs/shogi_halfka.rs` の **ShogiHalfKA_hm 関連部分のみ**
  → `crates/shogi-features/src/halfka_hm.rs`。Stage 3 (EPIC #17) の NNUE
  1536-16-32 trainer 入力特徴量として、Stage 3-5 dataloader / Stage 3-7
  sparse_ft_forward から呼ばれる。
  - **vendor scope**: bullet 上流 715 行のうち `ShogiHalfKA_hm` 系のみ
    (定数 `FEATURE_HASH_HM_V2` / `NUM_KING_BUCKETS` / `PIECE_INPUTS` /
    `HALFKA_HM_DIMENSIONS` / `MAX_ACTIVE_FEATURES`、struct `ShogiHalfKA_hm`、
    `map_halfka_features` / `king_bucket` / `is_hm_mirror` / `pack_bonapiece` /
    `king_bonapiece` / `halfka_index` の 6 helper、tests は bullet 上流の
    `ShogiHalfKA_hm` 関連 11 件を移植 (うち `test_shorthand` は static method
    化に伴い `test_shorthand_and_description` として内容拡張) + 本 PR 純粋追加
    2 件 (`test_alias_constants_match_bullet`、`test_map_halfka_features_kings_only`)
    で **計 13 件**)。Non-Mirror 版 (`ShogiHalfKA` / `PIECE_INPUTS_NONMIRROR`
    等、約 250 行) は
    本 Stage 3 で使用しないため **取り込まず**、必要になった時に別 issue で
    追加する scope creep 回避 (Stage 2-8 教訓踏襲)
  - **bullet trait 依存の除去**: bullet `impl SparseInputType for ShogiHalfKA_hm`
    を削除し、`num_inputs` / `max_active` / `map_features` / `shorthand` /
    `description` を **inherent method** として書き直し (Stage 1-1
    `PackedSfenValue` / Stage 1-2 `ShogiProgressKPAbs` と同流儀)。`SparseInputType`
    interface は本リポでは使わず (Stage 3-5 dataloader が直接 inherent method を呼ぶ)
  - **API path 修正**: `crate::shogi::*` → `shogi_format::*` 系に書き換え。
    bullet `ShogiBoard::from_packed_sfen(pos)` 呼び出しは本リポの
    `PackedSfenValue::decode()` に統一 (Stage 1-2 `progress_kpabs.rs` と同 idiom)
  - **新規追加 API** (bullet 由来ではない、本リポ独自):
    - `ShogiHalfKA_hm::collect_active_indices(pos) -> Vec<(usize, usize)>`:
      `map_features` callback を Vec 化 (Stage 1-2 同型、dataloader / smoke test
      用便利 API、容量は `MAX_ACTIVE_FEATURES` 事前確保)
    - 公開 const alias `SHOGI_HALFKA_HM_NUM_FEATURES` /
      `SHOGI_HALFKA_HM_NUM_ACTIVE_INDICES`: Issue #57 受け入れ条件 + 本リポ命名規約
      (`SHOGI_*_NUM_*`、Stage 1-2 `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS` 同型) に
      揃えた alias。bullet 上流命名 (`HALFKA_HM_DIMENSIONS` / `MAX_ACTIVE_FEATURES`) も
      互換のため保持し、両方を公開
  - **数値計算 path は upstream と完全一致**: `map_halfka_features` の駒列挙 / 玉
    特徴量 / 手駒 BonaPiece の 3 セクション、`king_bucket` / `pack_bonapiece` /
    `halfka_index` の helper、いずれも upstream と byte 一致。test 13 件
    (上流 11 件移植 + 本 PR 追加 2 件) で数値挙動を担保
  - **`cargo fmt` 適用** + doc コメント日本語化 + rshogi-nnue 文脈に合わせた
    仕様要約を module docstring に追加

#### Stage 2-7 (2026-05-11, bullet-shogi commit `f275eb9`)

- `crates/compiler/src/tensor/operation/linear/sparse.rs::SparseMatmulBwd::evaluate`
  → `experiments/002-fused-kernels/src/main.rs::sparse_ft_backward` (`#[kernel]`)
  + `crates/gpu-kernels/src/sparse/sparse_ft_backward.rs::sparse_ft_backward_cpu`。
  - 言語移植: bullet `evaluate` (`linear/sparse.rs:118-150`、generic `DValue` 経由) →
    Rust `#[kernel] fn sparse_ft_backward` (1 thread = 1 (batch, row) tuple、
    f32 固定、Stage 2-6 forward と同 layout)
  - **layout は forward と同型 column-major weight**: bullet `o.write(rows * idx +
    ri, ...)` → Rust `grad_weight[(idx as usize) * rows + ri]`。bullet test
    (`linear/sparse.rs::evaluate_bwd`、`:256-269`、batch=2/rows=3/cols=3/nnz=4、
    grad_out=[0..5]、indices=[0,1,-1,-1,2,2,1,0]、expected=[3,5,7,3,5,7,6,8,10])
    と同 fixture を本実装の `matches_bullet_upstream_evaluate_bwd_test` で 1:1 再現
  - **silent skip on `idx >= cols`**: bullet 上流 `if idx >= 0 && (idx as usize)
    < cols` (`linear/sparse.rs:140`) と同型 defensive を再現
  - **accumulate semantics**: bullet `evaluate` (`:130-133` 冒頭で `o.write(idx,
    zero)`) は test 用に zero clear するが、本 kernel は production semantics
    (host が `device.memset(0)` で zero clear する責務、Stage 1-6 grad と同
    convention) に揃える。CPU reference も accumulate (zero clear なし) で同型
  - **atomic scatter**: 複数 (bi, ni) thread が同じ `(idx, ri)` cell に書き込む
    ため `DeviceAtomicF32::fetch_add(_, AtomicOrdering::Relaxed)` で atomic
    scatter (Stage 1-6 grad と完全同 pattern、`unsafe { &*(slice.as_ptr().add(...)
    as *const DeviceAtomicF32) }` reinterpret cast)。`.ll` 上で
    `atomicrmw fadd float ... syncscope("device") monotonic` が 1 箇所出る
    ことを確認
  - thread 配置: bullet 上流は PointwiseIR で per-row reduction、batch 軸別 unroll
    想定。本実装は Stage 2-6 forward と同型 **flat 1D `tid = bi * rows + ri`** で
    batch 軸も込み、atomic scatter で衝突を吸収
  - `SparseMatmulBwdMulti` (bullet `:158-235` の複数 backward 集約) は本 PR 非対応、
    Stage 3 trainer で必要になれば別 issue
  - cuda-oxide API: `+` / `*` / i32 比較 + `DeviceAtomicF32::fetch_add` で
    cuda-oxide 制限非該当 (Stage 1-6 grad と同 atomic scatter pattern)

#### Stage 2-6 (2026-05-11, bullet-shogi commit `f275eb9`)

- `crates/compiler/src/tensor/operation/linear/sparse.rs::SparseMatmul::evaluate`
  → `experiments/002-fused-kernels/src/main.rs::sparse_ft_forward` (`#[kernel]`)
  + `crates/gpu-kernels/src/sparse/sparse_ft_forward.rs::sparse_ft_forward_cpu`。
  - 言語移植: bullet `evaluate` (`linear/sparse.rs:61-91`、generic `DValue` 経由) →
    Rust `#[kernel] fn sparse_ft_forward` (1 thread = 1 (batch, row) tuple、
    f32 固定)。bullet 上流は generic な dtype を受けるが、本実装は NNUE training の
    hot path で f32 のみ扱う前提
  - layout は **column-major weight** で完全一致: bullet `d.read(rows * idx + ri)` →
    Rust `weight[(idx as usize) * rows + ri]`。bullet test (`linear/sparse.rs:243-253`、
    batch=2/rows=2/cols=3/nnz=4、expected=[2,4,10,14]) と同 fixture を本実装の
    `matches_bullet_upstream_evaluate_test` で 1:1 に再現
  - **silent skip on `idx >= cols`**: bullet 上流 `if idx >= 0 && (idx as usize) <
    cols` (`linear/sparse.rs:82`) と同型 defensive を `if idx >= 0 && (idx as u32)
    < cols` で再現。`-1` padding と out-of-range の異常入力どちらも no-op
  - thread 配置: bullet 上流は PointwiseIR の `tid()` で 1 thread = 1 row、batch
    軸は別ループで unroll する想定 (上流側の launch grid 仕様)。本実装は
    **flat 1D `tid = bi * rows + ri`** で batch 軸も込みで thread に展開
    (Stage 1-5 forward / Stage 2-1 screlu_grad と同 idiom、1024 元レンジで
    GPU↔CPU 等価性確認済)
  - bullet は `nnz` ループを runtime fusion で吸収 (上流の PointwiseIR は per-row
    で sum reduction を inline)。本実装は `nnz` を build-time 引数で受けて
    kernel 内 while ループで素直に展開 (Stage 1-5 forward の `while j < max_inds`
    と同 pattern、memory bandwidth bound なので unroll 差は小さい想定、Stage 2-8
    で性能要件が出たら optimize 候補)
  - cuda-oxide API: `+` / `*` / i32 比較のみで cuda-oxide 制限非該当 (Stage 1-5
    forward と同等の単純 pointwise op)、`DisjointSlice<f32>::get_mut` Option
    silent skip pattern (Stage 1-5 / 2-1 / 2-3 / 2-4 と同型)、atomics 不要
  - **Stage 2-7 (backward, #43) は本 forward の column-major weight layout を
    引き継ぐ** (bullet 上流も `SparseMatmulBwd::evaluate` で同 `rows * idx + ri`
    indexing、`linear/sparse.rs:142`)。backward は `grad_weight` の同 cell に
    複数 (bi, ni) が衝突するため `DeviceAtomicF32::fetch_add` で scatter する形
    (Stage 1-6 grad の atomic scatter pattern と同型)

#### Stage 2-5 (2026-05-11, bullet-shogi commit `f275eb9`)

- `crates/trainer/src/optimiser/ranger.rs::build_ranger_op` (PointwiseIR
  `ranger.rs:27-46`) + `RangerLookahead::update` (`ranger.rs:82-100`) →
  `experiments/002-fused-kernels/src/main.rs::ranger_lookahead_lerp`
  (`#[kernel]`) + `crates/gpu-kernels/src/pointwise/ranger_step.rs::
  {ranger_lookahead_lerp_cpu, ranger_step_cpu}`。
  - bullet 上流 Ranger は **RAdam + Lookahead** の 2 段構成。RAdam (Stage 2-4
    `radam_step` を再利用) で fast params (`weights`) を更新しつつ、
    `step % k == 0` のときだけ Lookahead lerp で **slow params (`s`) との SMA**
    を取る host orchestration (bullet `RangerLookahead::update`)
  - 言語移植: bullet PointwiseIR (`build_ranger_op`、`ranger.rs:27-46`) の
    `pntwise.binary(alpha, w, Mul) + pntwise.binary(1-alpha, s, Mul) + Add` →
    Rust `#[kernel] fn ranger_lookahead_lerp` の `alpha * w + (1 - alpha) * s` を
    1 thread = 1 weight の単純 pointwise op に hand-fuse。`+` / `*` のみで
    cuda-oxide 制限に当たらない (Stage 1-5 forward の `+ z` と同等)
  - 同期動作: bullet `pntwise.write(w, ..., new_w); pntwise.write(s, ..., new_w)`
    (`ranger.rs:43-44`) と同型に `weights[i] = slow[i] = new_w` で完全同期する。
    test `ranger_lookahead_lerp_kernel_matches_cpu_reference` の post-condition
    で `weights == slow` を assert
  - host orchestration: `ranger_step_cpu` は bullet `RangerLookahead::update`
    と同 sequence (`radam_step` 毎 step + `step % k == 0` のとき lerp)。Stage 3
    trainer の RangerScheduler が GPU 上で同 sequence を組むときの reference
  - **Stage 2-4 `radam_step` の再利用**: Ranger は RAdam の自然な拡張で、
    本実装は RAdam kernel をそのまま再利用 + lookahead lerp kernel を別個に追加
    する 2 kernel pair として構成。bullet `Ranger = WrapOptimiser<RangerLookahead<G,
    RAdam<G>>, RangerParams>` (`ranger.rs:161`) と同設計
  - **slow params の checkpoint**: bullet は `slow.bin` に書き出して resume
    時に復元 + lookahead step counter を `step_ranger.txt` で persist する
    (`ranger.rs:121-141, 147-159`)。本実装はそれら orchestration までは
    含まず Stage 3 trainer integration で扱う想定
  - cuda-oxide API: lerp kernel は `DisjointSlice<f32>::get_mut` Option destructuring
    (Stage 1-5 / 2-1 / 2-3 / 2-4 と同型 silent skip pattern)、atomics 不要

#### Stage 2-4 (2026-05-11, bullet-shogi commit `f275eb9`)

- `crates/trainer/src/optimiser/radam.rs::RAdam::update` (host pre-compute) +
  `OP` template (`radam.rs:35-54` の `radamOp`) → `experiments/002-fused-kernels/
  src/main.rs::radam_step` (`#[kernel]`) +
  `crates/gpu-kernels/src/pointwise/radam_step.rs::radam_step_cpu` +
  `radam_compute_step_size_denom`。
  - bullet 上流 RAdam は **rectified Adam**: bias correction (`bc1 = 1 - beta1^t`)
    + variance 補正係数 `sqrt((1-beta2_t)*p1*p2*p3)` を **`step_size` に統合**
    し、kernel 側は `rate = lr * step_size` 1 個で受ける。学習初期で variance が
    不安定なときは `denom = 0` で `1/sqrt(v)` の正規化を **off**、十分 accumulate
    された後 (`n_sma > n_sma_threshold`、default 5.0) は `denom = 1` で通常
    Adam-like update
  - 言語移植: bullet C++ `__device__ __forceinline__ void radamOp` (`radam.rs:35-54`) +
    上位 `radam` kernel (`size % 4 == 0` で `float4` vectorize) →
    Rust `#[kernel] fn radam_step` (1 thread = 1 weight scalar、Stage 2-3
    `adamw_step` と同型)。`float4` vectorize は本実装では見送り (Stage 2-8 候補)
  - host pre-compute (`radam_compute_step_size_denom`、`radam.rs:198-218` 由来)
    は数式同一: `n_sma_max = 2/(1-beta2) - 1`、`n_sma = n_sma_max - 2*step*beta2_t/(1-beta2_t)`、
    `step_size = sqrt((1-beta2_t)*p1*p2*p3) / bc1` (n_sma > threshold 時) or
    `1 / bc1` (otherwise)、`denom = (n_sma > threshold) as i32`
  - **Stage 2-3 `adamw_step` (bias correction なし) との設計分離**: AdamW base
    (decay + clip + Adam update) を共有しつつ、bias correction + denom switch
    を加えた版が本 `radam_step`。Stage 2-5 `ranger_step` (#41) で本 RAdam +
    lookahead lerp を加える設計連鎖
  - **`adj_ptr` / `rate_ptr` / `step_size_ptr` / `denom_ptr` の host pre-compute
    値渡し化**: bullet 上流は全て 1-element device buffer で渡す (`radam.rs:DECL`
    参照) が、本実装は `f32` / `i32` 値渡しに簡素化 (Stage 2-3 `adamw_step` と
    同 convention)。Stage 3 trainer integration で device-side scheduling が
    必要になったら別 issue で device buffer 化
  - cuda-oxide API: 4 buffer すべて `DisjointSlice<f32>::get_mut` Option 経路
    (Stage 1-7 / 2-3 silent skip pattern)、clamp は `if-else` ladder
    (Stage 1-7 / 2-3 と同 workaround)、`f32::sqrt` は `__nv_sqrtf` (libdevice)
    に lowering、`if denom != 0` の i32→bool 比較は cuda-oxide で問題なく compile
    される (Stage 1-6 grad の bin clamp `b < 0` 比較と同型)
  - i32 を bool cast せず `!= 0` の比較で使う点は Rust の型安全性に揃えたが
    意味は bullet `if (denom)` と同じ

#### Stage 2-3 (2026-05-11, bullet-shogi commit `f275eb9`)

- `crates/trainer/src/optimiser/adam.rs::AdamWParams::build` (上流 AdamW
  optimizer kernel、`adam.rs:34-50` の `OP` template) →
  `experiments/002-fused-kernels/src/main.rs::adamw_step` (`#[kernel]`) +
  `crates/gpu-kernels/src/pointwise/adamw_step.rs::adamw_step_cpu`。
  - 言語移植: bullet C++ `__device__ __forceinline__ void adamOp` + 上位
    `adamw` kernel (`size % 4 == 0` で `float4` vectorize) →
    Rust `#[kernel] fn adamw_step` (1 thread = 1 weight scalar、Stage 1
    `progress::adam_step` と同型)。`float4` vectorize は本実装では見送り
    (Stage 2-8 で性能要件が出たら追加候補)
  - update 式は upstream と同一: `p *= 1 - decay * lr; m = b1*m + (1-b1)*g;
    v = b2*v + (1-b2)*g^2; p -= lr * m / (sqrt(v) + eps); p = clamp(p, min_w, max_w)`
  - **bias correction なし**: bullet 上流 AdamW は意図的に bias correction
    (`bc1 = 1 - beta1^t` 等) を含まない。Stage 1 `progress::adam_step` (= bullet
    の旧 `KERNELS_SRC::k_adam_step` 由来、bias correction あり) とは別 convention
    なので、`adamw_step_cpu` docstring に明示分離。Stage 2-4 (radam_step) で
    bias correction を加えた版を別途用意する設計
  - **grad reset**: bullet 上流は `gradients` を `const float*` で reset しない
    が、本実装は Stage 1 `progress::adam_step` を踏襲して `grad[i] = 0` を
    kernel 内で行う (host loop が次 batch の `atomicAdd` 累積に使う設計)
  - cuda-oxide API: 4 buffer すべて `DisjointSlice<f32>::get_mut` Option 経路
    (Stage 1-7 adam_step / Stage 2-1 screlu_grad と同型 silent skip pattern)、
    clamp は `f32::clamp` lowering 失敗のため `if-else` ladder に展開
    (Stage 1-7 と同 workaround)、`f32::sqrt` は `__nv_sqrtf` (libdevice) に
    lowering される (Stage 1-7 動作確認済)
  - lambda の関係: `decay = 0.0` で plain Adam (上流 `KERNELS_SRC::k_adam_step`
    の bias-correction を取り除いた形)、`min_w = f32::MIN, max_w = f32::MAX` で
    clip 無効化
  - **`adj` / `rate` の host pre-compute 化**: bullet 上流は `adj_ptr` /
    `rate_ptr` を 1-element device buffer で渡し kernel 内で `adj * rate` を
    取る (`optimiser/adam.rs::DECL` 参照)。本実装は **`adj` を省略** + **`rate`
    (lr) を `f32` 値渡し** に簡素化 (Stage 1 `progress::adam_step` 同型、
    Issue #39 本文の `adj_ptr` / `rate_ptr` 仕様より狭い scope)。Stage 3 trainer
    integration で device-side lr scheduling が必要になった時に `adj_ptr` /
    `rate_ptr` 化を別 issue で扱う想定

#### Stage 2-2 (2026-05-11, bullet-shogi commit `f275eb9`)

- `crates/bullet_lib/src/value/loader.rs::301-316` (data-layer WDL blend) +
  `crates/compiler/src/tensor/operation/autograd/dfo.rs::Sigmoid` →
  `experiments/002-fused-kernels/src/main.rs::loss_wdl` (`#[kernel]`) +
  `crates/gpu-kernels/src/pointwise/loss_wdl.rs::loss_wdl_cpu`。
  - 上流は **data layer で blend を pre-compute** (`results_chunk[i] =
    blend * result + (1 - blend) * sigmoid(rscale * score)`、`loader.rs::315`)
    し、kernel 側は pre-blended target に対する `output.sigmoid().squared_error
    (target)` (`crates/trainer/src/model/builder.rs::505`) を runtime PointwiseIR で
    fuse する 2 段構成。本実装は **kernel 内に WDL blend を畳み込む** ことで
    data layer から sigmoid(score) と blend を消し、forward + backward + loss
    accumulation を 1 kernel に圧縮 (op 数 8〜10、ADR-0004 Pattern table の
    `fused_loss_wdl` に相当)
  - 数値同値: bullet と本実装は同じ式 (sigmoid + blend + MSE + chain-rule grad)
    を計算するが、`out * scale` を kernel 内で取る点で `out` の cp scale を
    runtime 制御できる差分あり。bullet 上流は network architecture で `nnue2score`
    を持って `out` が既に scale 済の cp 単位を持つ場合 (`scale = 1.0` 相当)、
    本式の `* scale` 項が消えるだけで一致する
  - chain rule で sigmoid(out * scale) の `out` 微分には `* scale` が乗る
    (`d/du sigmoid(u) = p (1-p)`、`u = out * scale`)。`loss_wdl_cpu` docstring
    に詳細記載
  - cuda-oxide API: loss accumulator は f64 単一 cell の atomic add で
    `unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) }.fetch_add(_, Relaxed)`
    (Stage 1-6 grad / 1-8 eval の atomic 慣行を踏襲)。grad は 1 thread = 1 index
    で排他的、atomic 不要 (DisjointSlice get_mut Option silent skip パターン、
    Stage 1-5 forward と同型)
  - NaN / Inf 挙動: bullet 上流と一致して NaN を伝搬する (SCReLU と異なり
    握り潰さない)。`loss_wdl_cpu` docstring の "NaN / Inf 挙動" セクション参照

#### Stage 2-1 (2026-05-11, bullet-shogi commit `f275eb9`)

- `crates/compiler/src/tensor/operation/autograd/dfo.rs::SCReLU` (forward /
  backward) → fused gradient kernel として
  `experiments/002-fused-kernels/src/main.rs::screlu_grad` (`#[kernel]`) +
  `crates/gpu-kernels/src/pointwise/screlu_grad.rs::screlu_grad_cpu`
  (numerical equivalence test 用 reference)。
  - 上流は forward (`clamp(x, 0, 1)^2`) と backward (`2 * sqrt(y) *
    IsPositive(1 - sqrt(y))`) を別 op として PointwiseIR で組み合わせる
    runtime fusion 形式。本実装は **forward の input `x` を直接受け取る形に
    hand-fuse** して、`a = clamp(x, 0, 1); dydx = if 0<a<1 { 2a } else { 0 };
    dl_dx = dl_dy * dydx` の 1 fused kernel に圧縮 (op 数 2-3、ADR-0004
    Pattern table の `fused_screlu_grad` に相当)
  - 数値同値: bullet `2 * sqrt(y) * IsPositive(1 - sqrt(y))` と本実装
    `2 * a * (a > 0 && a < 1)` は数学的に同一 (a = sqrt(y) なので)。`sqrt`
    を経由しない分 round-off は本実装のほうが小さい (`a*a` と `sqrt(a*a)`
    の 2 回の中間丸めを避けられる、interior 値で `2 * a` を直接出せる)
  - cuda-oxide 制限: GPU kernel 側は `f32::clamp` を使えない (内部で
    `f32::max` を呼び lowering 失敗、Stage 1-7 で確認済) ため `if-else`
    ladder で展開 (`x < 0 ? 0 : x > 1 ? 1 : x`)。CPU reference は host 実行で
    `f32::clamp` 使用
  - 注: kernel 関数を `experiments/002-fused-kernels/src/main.rs` に直接
    配置しているのは Stage 1-5 と同じ理由 (cuda-oxide rustc-codegen-cuda
    backend は bin entry から到達可能な `#[kernel]` のみ NVPTX IR 化する)。
    reference CPU は `crates/gpu-kernels/src/pointwise/screlu_grad.rs` (Stage 1
    の `progress/` と同列の慣行)

#### Stage 1-1 (2026-05-10, bullet-shogi commit `f275eb9`)

- `crates/bullet_lib/src/shogi/types.rs` → `crates/shogi-format/src/types.rs`
  (Color, PieceType, Square, Piece, Hand。完全一致 + `cargo fmt` 適用)
- `crates/bullet_lib/src/shogi/packed_sfen.rs` → `crates/shogi-format/src/packed_sfen.rs`
  (BitStream, PackedSfen, PackedSfenValue, ShogiBoard。完全一致から下記の差分:
  - `unsafe impl crate::value::loader::CanBeDirectlySequentiallyLoaded for PackedSfenValue {}` を削除 (bullet trait 依存を排除)
  - `impl crate::value::loader::LoadableDataType for PackedSfenValue { ... }` を削除し、`fn result(&self) -> crate::GameResult` を **inherent method** として書き直し
  - `cargo fmt` 適用)
- `crates/bullet_lib/src/shogi/bona_piece.rs` → `crates/shogi-format/src/bona_piece.rs`
  (BonaPiece 定数群。完全一致 + `cargo fmt` 適用)

新規追加 (bullet 由来ではない):

- `crates/shogi-format/src/game_result.rs` — bullet `crate::value::loader::GameResult` の最小サブセット (Loss=0, Draw=1, Win=2)。bullet trait に依存しないために自前定義
- `crates/shogi-format/src/lib.rs` — 上記 4 module の宣言と公開型 re-export
- `crates/shogi-format/Cargo.toml` — workspace member として最小設定
- `crates/shogi-format/tests/psv_smoke.rs` + `tests/data/sample.psv` (smoke_progress/smoke.bin の先頭 4000 bytes / 100 records)

#### Stage 1-5 (2026-05-10, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_forward`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn forward`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/forward.rs` の
  `forward_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]` (cuda-oxide `cuda_device`)
  - `int` → `u32` (符号要らない)、`float* preds` → `mut DisjointSlice<f32>`
  - `for (int j; j<max_inds; ++j)` → `while j < max_inds` (cuda-oxide gemm 上流に倣う)
  - `expf(-z)` → `(-z).exp()` (cuda-oxide が `__nv_expf` に lowering する)
  - C++ `preds[pos] = ...` (上の `if (pos >= n_pos) return;` で bounds 保証
    された unconditional write) → Rust `if let Some(p) = preds.get_mut(pos)
    { *p = ... }`。cuda-oxide の `DisjointSlice<T>::get_mut(idx) -> Option`
    は GPU soundness のため Option を返す API、`pos >= n_pos` 早期 return
    と組み合わせると `preds.len() == n_pos` で必ず Some が返り挙動同一。
    `preds.len() < n_pos` の異常入力に対しては C++ は OOB write (UB)、
    Rust は silent skip という **defensive な差分** あり
  - 計算ロジックは上記 5 点の表面的差分以外 **同一**。reference CPU
    (`forward_cpu`) も GPU kernel と同じ式を素直に書き写しただけで、
    `preds.len() == n_pos` を満たす入力に対し同出力 (浮動小数誤差範囲内) を返す
  - 注: kernel 関数を main.rs に直接配置しているのは、cuda-oxide の
    rustc-codegen-cuda backend が **bin entry から到達可能な #[kernel]
    関数のみ NVPTX IR 化** する設計のため (本リポ内検証で lib.rs 内
    kernel は `cargo oxide build` で `.ll` 出力されないことを確認)。
    `.ll` 生成の正しい invocation は **`cd experiments/001-cuda-oxide-kpabs
    && cargo-oxide build`** (cwd を crate dir にする)。workspace root から
    `cargo-oxide build exp-001-cuda-oxide-kpabs` を呼ぶと cargo-oxide 上流
    実装 (`crates/cargo-oxide/src/backend.rs`) の workspace-root 探索が
    standalone path に落ちず IR 出力が silently no-op になる

#### Stage 1-11 (2026-05-11)

- experiments/001-cuda-oxide-kpabs を **archive 化** し、production target を
  以下 2 crate に昇格:
  - **`crates/gpu-kernels/`** — Stage 1-5..1-8 の reference CPU 実装
    (`forward_cpu` / `grad_cpu` / `adam_step_cpu` / `eval_cpu`) を `progress`
    モジュールに移動。GPU `#[kernel]` は cuda-oxide rustc-codegen-cuda backend の
    "bin entry から到達可能な kernel のみ NVPTX IR 化" 制約のため引き続き bin
    側 (= bins/progress_kpabs_train/src/main.rs) に inline 配置する
  - **`bins/progress_kpabs_train/`** — Stage 1-9 で組み込んだ host loop driver
    (`#[kernel]` × 4 + `GpuTrainer` + `train_one_epoch` + CLI) と host helper
    (`Batch` / `GameIterator` / `progress_bin` / `Args`) を移動。reference
    CPU は新 `gpu-kernels` crate を引き込んで参照する DRY 構成
- experiments/001 は **そのまま残し** (Stage 1 進行中の試行錯誤履歴として参照可能)、
  README に昇格先 link と archive 注意書きを追加
- ファイル内容は `experiments/001-cuda-oxide-kpabs/` から **コピー** (新規実装
  なし)。bullet-shogi 上流に対する関係は Stage 1-1..1-10 の各 entry がそのまま
  有効。本 entry は workspace 構造変更のみ
- workspace `Cargo.toml`: 新 member 2 つを `members` に追加、
  `gpu-kernels = { path = ... }` を `[workspace.dependencies]` に追加
- CI (`.github/workflows/checks.yaml`): `progress-kpabs-train` を `--exclude`
  リストに追加 (cuda-bindings build に CUDA toolkit が必要、GitHub runner で
  build 不能、experiments/001 と同じ理由)。`gpu-kernels` は CPU only なので CI
  に通る
- 動作確認: sm_75 RTX 2070 SUPER で `cargo run --release -p
  progress-kpabs-train -- --data <sample.psv> --output /tmp/progress.bin
  --games-per-step 4 --max-games 8` が `mean_loss=0.085017` の値を experiments/001
  と完全一致で出力、`samples/sec ≈ 290k` を記録

#### Stage 1-10 (2026-05-11, bullet-shogi commit `f275eb9`)

- **新規ファイル** (bullet-shogi 由来ではない、本リポ自前):
  - `docs/experiments/001-stage1-10-numerical-equivalence.md` — Stage 1-10 の
    検証手順とローカル実測値 (sm_75 RTX 2070 SUPER で 218k〜233k samples/sec)、
    bullet-shogi 上流とのクロス検証 manual procedure、cuda-oxide rev `6de0509`
    で遭遇した 4 件の不具合 / 制限 (`Ord::clamp` lowering 失敗 / `f32::max`
    intrinsic 未対応 / libNVVM opaque pointer parse 失敗 / cargo-oxide の
    `.ll` 出力先不整合) を一覧化
  - `experiments/001-cuda-oxide-kpabs/src/main.rs::gpu_cpu_equivalence_tests`
    (`#[cfg(test)]` mod) — Stage 1-5..1-8 の reference CPU 実装 (`*_cpu`) と
    GPU kernel の出力を直接比較する 5 test:
    - `forward_kernel_matches_cpu_reference` (16 pos × 8 inds × 64 weights、
      pad 混在、tolerance 1e-5)
    - `grad_kernel_matches_cpu_reference` (同 setup、grad scatter atomic ↔
      CPU sequential add の round-off を 1e-5 で吸収、loss は f64 atomic で
      1e-8、hist は完全一致)
    - `eval_kernel_matches_cpu_reference` (24 pos、loss + hist 比較)
    - `adam_step_kernel_matches_cpu_reference` (32 weights、1 step 後の
      `weights/m/v/grad` 比較)
    - `samples_per_sec_baseline_on_sample_psv` (sample.psv の 4 games × 8 pos
      = 32 pos/batch × 50 steps の throughput を `println!` で記録)
  - kernel symbol が `main.rs` の bin scope に定義されているため、
    `tests/*.rs` (lib link only) からは届かず、`#[cfg(test)] mod` を main.rs
    inline に置く形式を採用。`cargo test --bin exp-001-cuda-oxide-kpabs
    --release -- --test-threads=1` で実行

- **検証で確認した数値同等性** (sm_75 box 実測):
  - 4 GPU kernel の出力は CPU reference と f32 で 1e-5 以内、f64 で 1e-8 以内、
    u64 hist は完全一致
  - bullet-shogi 上流とのクロス検証は manual procedure として doc 化
    (両 CUDA 環境を両立させる前提が大きく自動化はせず、必要時に
    `docs/experiments/001-stage1-10-numerical-equivalence.md` の手順で実施)

#### Stage 1-9 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs` の **host 側ロジック** (kernel
  以外、約 900 行) を以下の単位で `experiments/001-cuda-oxide-kpabs/` に移植。
  bullet-shogi の multi-thread prefetch / pack interleaving / 学習 epoch
  per-checkpoint / val split は Stage 1-9 の受け入れ条件 (1 epoch 完走 +
  progress.bin 出力) に対し過剰なため意図的に削除し、最小実装に絞った。

  - `src/host/games.rs::PackCursor` / `GameIterator` ←→ 上流 `PackCursor` /
    `GameIterator`。PSV ファイルを 1 record ずつ読み、`game_ply` の減少を
    境界として 1 ゲーム単位に切り出す。bullet 上流の `Vec<u8>` バッファ + size
    検証 path も同等
  - `src/host/batch.rs::Batch` ←→ 上流 `Batch`。`push_game` で 1 ゲーム分の
    flat indices / targets / per_pos_norm を埋め、`finalize` で per_pos_norm
    に `1/n_games` を乗じて batch averaging を完成。target は `i / (game_len - 1)`
    の game-relative ラベル (上流と同式)。`MAX_INDS_PER_POS = 80` も同値
  - `src/host/progress_bin.rs::write_progress_bin` / `read_progress_bin` ←→
    上流の同名関数。YaneuraOu 互換の f64 LE × `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`
    形式 (= 1,003,104 bytes)。f32 ↔ f64 cast は wire format 通り
  - `src/host/cli.rs::Args` ←→ 上流 `Args` の核サブセット (`--data` `--output`
    `--init-from` `--games-per-step` `--max-games` `--epochs` `--lr` `--lr-scale`
    `--log-interval-steps` `--device`)。prefetch / val split 関連 flag は削除
  - `src/main.rs::GpuTrainer` ←→ 上流 `GpuTrainer`。`step` で
    forward → grad/loss/hist → adam_step を順次 `cuda_launch!` で起動、
    `eval_forward` は forward → eval kernel。device buffer 確保は cuda-oxide
    `DeviceBuffer<T>` ベースで bullet の `RawBuf` (raw `malloc/memset`) は不要
  - `src/main.rs::train_one_epoch` ←→ 上流 `train_one_epoch`。multi-thread
    prefetch を **single-threaded** に簡素化 (mpsc / JoinHandle なし)、log /
    epoch 集計はそのまま

- **kernel artifact loader** (`load_kernel_module_with_fallback` /
  `compile_ll_to_ptx_via_llc`) は新規 (上流の NVRTC は Rust kernel に使えない)。
  cuda-oxide が出力する opaque pointer NVVM IR (`define void @grad(ptr ...)`)
  は libNVVM が parse できない (実機エラー: `nvvmCompileProgram error 9:
  parse expected type`、`exp_001_cuda_oxide_kpabs.ll:11` 由来) ため、本 PR は
  **`llvm-link-21 + opt-21 (passes='internalize,globaldce,nvvm-reflect') +
  llc-21`** の 3 段 pipeline で `.ll → .ptx` を生成する。kernel symbol を
  `--internalize-public-api-list=grad,forward,adam_step,eval` で保存し、
  libdevice の未使用関数を `globaldce` で除去、`__nvvm_reflect()` を `nvvm-reflect`
  pass で 0/1 に畳み込む。NVCC の `compileToCubin` 相当だが driver 側の JIT
  にも対応した形で生成。`.ptx` には `.extern .func` が残らず ptxas 単体で完結

- 環境前提: WSL2 sm_75 box (RTX 2070 SUPER)、CUDA 12.9、LLVM 21.1.8 (clang-21
  / llvm-link-21 / opt-21 / llc-21)、`/usr/local/cuda-12.9/nvvm/libdevice/
  libdevice.10.bc`。Stage 1-1〜1-8 と同じ。実行確認: `cargo run -p
  exp-001-cuda-oxide-kpabs -- --data <sample.psv> --output <progress.bin>
  --games-per-step 4 --max-games 8` で 1 epoch 完走 + 1003104 bytes
  progress.bin 出力済 (受け入れ条件達成)

#### Stage 1-8 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_eval_loss_hist`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn eval`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/eval.rs` の
  `eval_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]`
  - C++ `const float* preds / targets` / `double* loss_acc` / `unsigned long long* hist` →
    Rust `&[f32]` / `&[f64]` / `&[u64]` (atomicAdd 経由で書く前提)
  - C++ `atomicAdd(loss_acc, (double)err*(double)err)` → Rust
    `DeviceAtomicF64::fetch_add(_, AtomicOrdering::Relaxed)`、IR で
    `atomicrmw fadd ptr ..., double ... syncscope("device") monotonic` 確認
  - C++ `atomicAdd(&hist[b], 1ULL)` → Rust `DeviceAtomicU64::fetch_add(1, Relaxed)`、
    IR で `atomicrmw add ptr ..., i64 1 syncscope("device") monotonic` 確認
  - C++ `(int)(p * 8.0f); if (b<0) b=0; if (b>7) b=7;` → kernel 側は Stage 1-6
    と同じく verbatim if-else (`#[allow(clippy::manual_clamp)]`)、CPU reference は
    `i32::clamp(0, 7)`
  - 計算ロジックは `grad` の **gradient scatter / per_pos_norm を除いたサブセット** で、
    eval 側 `eval_cpu` と grad 側 `grad_cpu` に同じ `(preds, targets, n_pos)` を渡せば
    `loss_acc` / `hist` が完全一致する不変条件をテスト (`tests/eval_smoke.rs::
    eval_output_matches_grad_loss_hist_subset`)

#### Stage 1-7 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_adam_step`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn adam_step`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/adam_step.rs` の
  `adam_step_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]` (cuda-oxide `cuda_device`)
  - C++ `float* weights / m / v / grad` (in-place 4 buffer) → Rust
    `mut DisjointSlice<f32>` × 4。1 thread = 1 weight で aliasing なし、
    grad のような scatter は発生しないので **atomics 不要** (Stage 1-6 と
    異なる)。host 側で `len() == n` を保証し、`get_mut(i)` の Option を
    全 4 で揃える `if let (Some, Some, Some, Some) = (...)` パターンで
    silent skip 防御
  - C++ `fmaxf(bc, 1e-30f)` → Rust 側 GPU kernel では verbatim な if-else
    `if bc > 1e-30 { bc } else { 1e-30 }` を維持。Rust の `f32::max` は
    内部で `std::intrinsics::maximum_number_nsz_f32` を呼び、cuda-oxide が
    現状その intrinsic を未解決 (実機エラー: `Symbol
    std__intrinsics__maximum_number_nsz_f32 not found`、`f32.rs:993` 由来)。
    CPU reference (`adam_step_cpu`) は host 実行のみのため `f32::max` を使う
  - C++ `sqrtf(v_hat)` → Rust `v_hat.sqrt()`。cuda-oxide は IR で
    `call float @__nv_sqrtf(...)` に lowering する (libdevice 経由、
    `.ll` 出力で確認済)
  - C++ `int n` → Rust `u32`
  - 計算式は表面的差異 (Option-returning DisjointSlice / max の if-else 表現)
    を除き同一

#### Stage 1-6 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_grad_loss_hist`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn grad`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/grad.rs` の
  `grad_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]` (cuda-oxide `cuda_device`)
  - C++ `int` → Rust `u32` (n_pos / max_inds)、`int idx` は `i32` のまま (-1 padding 検出)
  - C++ `float* grad` / `double* loss_acc` / `unsigned long long* hist` →
    Rust `&[f32]` / `&[f64]` / `&[u64]` (atomicAdd 経由でのみ書く前提)
  - C++ `atomicAdd(&grad[idx], gscale)` (f32) → Rust の
    `unsafe { &*(grad.as_ptr().add(idx) as *const DeviceAtomicF32) }
     .fetch_add(gscale, AtomicOrdering::Relaxed)` (cuda-oxide `cuda_device::atomic`)。
    生成 IR は `atomicrmw fadd ... syncscope("device") monotonic` で確認済み
    (sm_60+ の `atom.add.f32` に lowering される、本リポは sm_75 で動作)
  - 同パターンで `loss_acc` (f64) と `hist[bin]` (u64) も `DeviceAtomicF64` /
    `DeviceAtomicU64` に reinterpret cast して `fetch_add(_, Relaxed)`。
    Relaxed 採用は collection 用途で順序保証不要 (bullet 上流 C++ `atomicAdd`
    の暗黙 ordering と同等)
  - C++ `int b = (int)(p * 8.0f); if (b<0) b=0; if (b>7) b=7;` →
    Rust 側 GPU kernel では verbatim な if-else を維持
    (`#[allow(clippy::manual_clamp)]`)。Rust の `i32::clamp` は内部で
    `assert!(min <= max)` の panic 経路 (`Debug::fmt`) を持ち、cuda-oxide の
    rustc-codegen-cuda backend が現状その lowering 未対応 (実機で再現確認)。
    CPU reference (`grad_cpu`) は host 実行のみのため `i32::clamp` を使う
  - 計算ロジックは上記の atomic API / clamp 表現の差異以外 **同一**。reference
    CPU (`grad_cpu`) は同じ式を素直に書き写しただけで、複数 thread の並列
    更新による浮動小数加算順序の差は生じるが (関連: associative でない f32
    の加算)、host 単一 thread 実行では deterministic な値を返す
  - 注: kernel 関数を main.rs に直接配置している理由は Stage 1-5 entry と同じ

#### Stage 1-2 (2026-05-10, bullet-shogi commit `f275eb9`)

- `crates/bullet_lib/src/game/outputs.rs` の `ShogiProgressKPAbs` 周辺
  → `crates/shogi-features/src/progress_kpabs.rs`
  (関連定数 `SHOGI_PROGRESS8_NUM_BUCKETS` `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`
   と static `SHOGI_PROGRESS_KP_ABS_WEIGHTS` `SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS`
   も同 file に同梱。**数値計算 path (for_each_active_index / progress / bucket
   / load_from_bin) は upstream と byte 一致**、下記の差分のみ:
  - `impl OutputBuckets<PackedSfenValue> for ShogiProgressKPAbs { ... }` を削除し、
    `bucket()` を **inherent method** として書き直し (bullet `OutputBuckets` trait
    依存を排除)。失われる `OutputBuckets::BUCKETS` const は
    `ShogiProgressKPAbs::BUCKETS` inherent const で代替
  - import path を `crate::shogi::*` から `shogi_format::*` に書き換え
    (bullet 内部の chess 系 import `bulletformat::*` も削除)
  - module-level および各 method の doc-comment を日本語化・rshogi-nnue
    文脈に合わせて加筆 (英文 upstream → 日本語ローカライズ + 仕様要約追記)
  - `cargo fmt` 適用)

新規追加 (bullet 由来ではない):

- `crates/shogi-features/{Cargo.toml, src/lib.rs}` — workspace member として最小設定、
  shogi-format crate への path dep
- `crates/shogi-features/tests/progress_kpabs_smoke.rs` — shogi-format crate の
  `tests/data/sample.psv` を共有して各 record で `for_each_active_index` /
  `collect_active_indices` / `progress` / `bucket` の挙動を検証 (重み未ロード
  状態で `progress()` が `sigmoid(0)=0.5` / `bucket()` が `4` になることも確認)

## cuda-oxide (Apache-2.0)

- Source: https://github.com/NVlabs/cuda-oxide
- Use: GPU kernel を build-time に PTX 化 (host 側 wrapper も含む)
- License: Apache-2.0
- Dependency style: `Cargo.toml` の git dep + rev pin (vendor せず)
- 採用 rev: **`6de0509`** (NVlabs/cuda-oxide main, 2026-05-08)
  Stage 0-1 で動作確認、Stage 1-3 (#7) で `crates/gpu-runtime` から
  `cuda-core` / `cuda-host` を取り込み

## Pliron (Apache-2.0)

- Source: https://github.com/vaivaswatha/pliron
- Use: cuda-oxide が依存 (transitive)
- License: Apache-2.0

## ライセンス互換性メモ

本リポジトリ自体は MIT。MIT は Apache-2.0 由来コードを含むコンパイル
バイナリ配布と互換。ソース配布時は各依存の `LICENSE` を保持する。
