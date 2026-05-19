# 構造化実験ログ — experiment.json の format 設計

- **Status**: Accepted
- **Date**: 2026-05-17

## Context

`nnue-train` (`bins/nnue_train`) の学習 run は現状、進捗を人間可読の stdout 行
としてのみ出力する (`[train] superbatch N/M | loss ... | pos/s ...`)。run 終了後
に loss 軌跡・パラメータ・throughput を構造化された形で参照する手段がなく、
実行者が stdout を `tee` で拾わない限り何も残らない。

別リポ `nnue-lab` は学習 run を 1 件 = 1 ファイルの `experiment.json` として
取り込み、一覧・loss カーブ・オーバーレイ比較・resume 系列の lineage 連結を
提供する Web アプリである。`nnue-lab` の取り込み schema (`ExperimentJsonV1`、
zod) は当初 `bullet-shogi` の `experiment.json` 出力に合わせて作られた。

本リポでも「実験後にログを構造化して見やすくする」ための出力を持ちたい。
出力先は `nnue-lab` への取り込みを想定する。`bullet-shogi` の format を踏襲
する義務はなく、設計し直して良い。

### nnue-lab の取り込み機構

本 format は `nnue-lab` の取り込み機構に適合する範囲で設計する。下記の auth・
zod・passthrough の挙動は本 ADR の設計対象外の固定制約 (`nnue-lab` 実装
`packages/shared/src/schema.ts` / `apps/web/worker/services/experiment-upload.ts`
の確認に基づく)。重複排除・lineage 解決・DB スキーマは本 ADR が `nnue-lab` 側と
co-design する対象で、下記 Lineage / nnue-lab との連携契約 で扱う。

- 取り込み口は Web UI への drag&drop (`POST .../experiments`、multipart)。
  認証は Cloudflare Access (Google OAuth)、非対話の API 投入口は現状ない。
- zod `ExperimentJsonV1` の必須フィールド: `id` / `name` / `date` (ISO 8601
  datetime、`Z` または offset 必須) / `params.{lr,batch_size,superbatches}` /
  `history[]` (要素は `{superbatch, loss}`)。
- `params` / `data` / `results` は `.passthrough()` — **未知キーは保持され
  R2 まで届く**。本リポ固有情報はこの領域に置く。
- top-level および `history[]` の要素は passthrough でない — 未知キーは
  reject されないが正規化時に削除される。top-level に新フィールド
  (`generator` / `lineage`) を保持・索引させるには `nnue-lab` schema 拡張が
  要る (連携契約を参照)。

## Decision

### スコープ

- `nnue-train` が学習 run ごとに 1 件の構造化ログファイルを **ローカルに
  書き出す**。`nnue-lab` への投入は従来どおり Web UI への手動 drag&drop。
  トレーナーから `nnue-lab` への自動 push は本 ADR のスコープ外 (将来課題)。
- 対象は `nnue-train`。本 format v1 は `nnue-train` 専用とし、もう一方の
  `progress_kpabs_train` への展開は将来課題 (展開時に format 互換性と
  `schema_version` を再評価する)。

### 中核となる 3 つの設計判断

1. **`nnue-lab` ExperimentJsonV1 の互換 superset とする。** 必須フィールドは
   常に zod schema どおりの型で出力し、本 format のファイルは未改修の
   `nnue-lab` にもそのまま取り込める。本リポ固有の情報は passthrough 領域
   (`params`/`data`/`results`) に置いて確実に保持させる。

2. **1 run = 1 ファイル。crash 耐性は atomic incremental write で得る。**
   superbatch ごとに temp ファイル + rename で全体を書き直す。中断時は最後の
   incremental write 結果が `status: "running"` の妥当な JSON として残る
   (= そのまま取り込める)。`bullet-shogi` の `recover_experiment` 相当の
   復元ツールは不要になる。

3. **resume 系列は「1 run = 1 ファイル + lineage 参照」で表現する。** 各 run
   は自分が回した superbatch 範囲の `history` だけを持つ。`--resume` した run
   は親 run と resume 元 checkpoint を指す `lineage` オブジェクトを持つ。
   `history` を世代マージしない (`bullet-shogi` 方式は採らない)。

## Format

`schema_version` は常に明示出力する。値は `nnue-lab` ExperimentJsonV1 と整合
する `1` (= nnue-lab schema 契約の version。producer 自身の version は
`generator.version` が別に持つ)。

注釈付きの骨子 (値は例):

```jsonc
{
  "schema_version": 1,
  "generator": { "name": "rshogi-nnue", "version": "0.1.0" },  // ※1

  "id": "rshogi-20260517t041530z-48213", // net_id + UTC 開始時刻 + pid、run 一意 (※2)
  "name": "rshogi",                   // 既定 = net_id (resume run は別既定、※3)
  "date": "2026-05-17T04:15:30Z",     // run 開始時刻 (ISO 8601 UTC)
  "status": "completed",              // "running" | "completed" のみ (※4)
  "last_updated_at": "2026-05-17T05:10:02Z",
  "commit": "7beb263",                // rshogi-nnue の revision (dirty 時は印付き)
  "command": "nnue-train --data ... --superbatches 400 ...",

  "lineage": {                        // ※5  --resume した run のみ
    "parent_id": "rshogi-20260516t221000z-31002",
    "resumed_from_checkpoint": "rshogi-200.ckpt",
    "resumed_from_superbatch": 200
  },

  "params": {                         // passthrough — 未知キーも保持される
    "architecture": "LayerStack-1536-16-32-9bucket",
    "l0": 1536, "l1": 16, "l2": 32, "num_buckets": 9,
    "optimizer": "ranger",
    "bucket_mode": "progress8kpabs",
    "progress_coeff": "progress.bin",       // basename or null
    "lr": 8.75e-4, "lr_gamma": 0.995, "lr_step": 1,
    "batch_size": 65536, "batches_per_superbatch": 6104, "superbatches": 400,
    "start_superbatch": 1,
    "wdl": 0.0, "scale": 290.0, "weight_decay": 0.0,
    "qa": 127, "qb": 64,

    // 本リポ固有の knob (flat 配置、※6)
    "loss_kind": "wrm",
    "wrm_in_scaling": 340.0, "wrm_nnue2score": 600.0,
    "wrm_target_offset": 270.0, "wrm_target_scaling": 380.0,   // ※7
    "score_drop_abs": null,
    "init_from": null,                      // --init-from の basename or null (※8)
    "tf32": false, "ft_fp16": true, "ft_fp16_out": true, "fp16_opt_state": false,
    "threads": 16
  },

  "data": {                           // passthrough — bullet と同じ意味 (※9)
    "name": "teacher.psv",
    "positions": 800000000,           // データセットファイルの局面数
    "total_positions": 1601320960,    // 学習で消費した局面数 (sb×bps×bs)
    "dataset_passes": 2.0             // total_positions / positions (= epoch 数)
  },

  "results": {                        // passthrough
    "training_time_seconds": 3271,
    "fv_scale": 28,
    "best_loss": 0.011820, "best_loss_superbatch": 392,
    "mean_pos_per_sec": 1421000,      // ※10  run 全体の平均 throughput
    "interrupted": false
  },

  "history": [                        // 要素は {superbatch, loss} に限定 (※11)
    { "superbatch": 1, "loss": 0.041203 },
    { "superbatch": 2, "loss": 0.038551 }
  ],

  "checkpoints": ["rshogi-20.bin", "rshogi-20.ckpt", "rshogi-40.bin"]  // ※12
}
```

- **※1 `generator`**: producer (rshogi-nnue / bullet-shogi) の区別用。top-level
  なので未改修の `nnue-lab` では正規化時に削除される (reject はされない)。
- **※2 `id`**: `{net_id}-{UTC開始時刻}-{process id}` で run を一意化する。
  秒精度 UTC 時刻に process id を足すことで、同一 net_id / output で複数
  プロセスが同一秒に開始しても (sweep / retry script 等) `id` と
  experiment.json の path が衝突しない。同一 run の全 snapshot を通じて不変。
  `nnue-lab` は取り込み時に独自 DB 主キー (ulid) を採番し、本 `id` は別列
  (`producer_id`) に保存して `(tenant_id, producer_id)` upsert と lineage の
  親参照のキーにする (連携契約を参照)。
- **※3 `name`**: 既定は from-scratch run なら `net_id`、`--resume` run なら
  `{net_id} (resume @sb{start})`。CLI flag で上書き可能。resume 既定を分ける
  のは、未改修の `nnue-lab` に取り込んだ degrade 時に lineage 自動連結が効かず、
  同名 run が一覧に並んで世代判別できないため (連携契約の degrade 節を参照)。
- **※4 `status`**: `nnue-lab` の enum が `running`/`completed` のみのため、
  本 format v1 もこの 2 値だけ使う。crash した run は最後の incremental write
  時点の `running` のまま残す (`failed` 等は使わない)。
- **※5 `lineage`**: `--resume` した run のみ持つ。top-level のため未改修の
  `nnue-lab` では正規化時に削除される。保持・自動連結には `nnue-lab` schema
  拡張が要る (連携契約を参照)。
- **※6 本リポ固有 knob を `params` に flat 配置する**理由: `nnue-lab` の
  パラメータ差分表 (ParamDiffTable) は `params` の flat キーを比較する。
  `ft_fp16` / `tf32` の on/off 違いを run 間で並べる用途は本リポの perf
  作業で頻出するため、差分表に出る flat 配置が有用。`params.rshogi` 等に
  ネストすると差分表から見えなくなる。
- **※7 WRM パラメータ**: `loss_kind` が `"wrm"` のとき、loss を再現するには
  `wrm_in_scaling` / `wrm_nnue2score` / `wrm_target_offset` /
  `wrm_target_scaling` の 4 値が必要。4 値すべてを出力する。`loss_kind` が
  `"sigmoid"` のときは `scale` が効き、WRM 系キーは無効。
- **※8 `init_from`**: `--init-from` (weight のみ注入、optimizer reset) の
  入力ファイル basename。これは weight 初期化であって学習継続ではないため
  `lineage` には載せず、`params` の情報フィールドに留める。
- **※9 `data` の局面数フィールド**: `bullet-shogi` の `experiment.json` と
  **同じ意味**にして producer 間で一貫させる (`nnue-lab` は `data` を blob
  として保存するだけで意味検証しないため、意味の一貫性は producer 側責任)。
  `positions` = データセットファイルの局面数、`total_positions` = 学習で消費
  した局面数、`dataset_passes` = `total_positions / positions`。
- **※10 `results.mean_pos_per_sec`**: throughput を成果物に残す。perf 改善
  作業で run 間比較に使える。`nnue-lab` passthrough で保持される。
- **※11 `history` 要素**: `nnue-lab` の history 要素 schema は passthrough で
  ないため、per-superbatch の pos/s や lr を要素に足しても正規化で削除される。
  本 format では `history` を `{superbatch, loss}` に限定し、集約値
  (`mean_pos_per_sec` 等) を `results` 側に置く。
- **※12 `checkpoints`**: その run が書き出した checkpoint ファイル名の
  **生成記録** (informational)。`--keep-checkpoints` で後から `.ckpt` が
  pruning されても過去の experiment.json は書き換えないため、この配列は
  既に削除された名前を含み得る。lineage 解決には使わない (下記 Lineage 参照)。

## Lifecycle

- **書き出し先**: 1 run = 1 ファイル。checkpoint と同じ `--output` 配下の
  サブディレクトリ (例 `{output}/experiments/`) に、`id` を元にした一意名で
  置く (checkpoint 群と混ざらない)。
- **書き込み時点**:
  - run 開始時: `status: "running"`、`history` 空、`params`/`lineage` 確定。
  - 各 superbatch 完了時: その superbatch の処理 (量子化 `.bin` / raw `.ckpt`
    の保存を含む) をすべて終えてから、`history` に 1 点追加・`last_updated_at`
    と `results` 集約値・`checkpoints` を反映して全体を書き直す。checkpoint
    保存 superbatch でも、保存後に experiment.json を書くことで `checkpoints[]`
    に載ったファイルは書き込み時点で実在する。
  - run 正常終了時: `status: "completed"`、`results` 最終値。
- **atomicity**: 毎回 temp ファイルへ書いて同一ディレクトリ内で rename する。
  部分書き込みファイルを残さない。superbatch は数秒オーダー、ファイルは
  数百点 history で数 KB 程度のため、毎 superbatch 書き直しのコストは無視できる。
- **crash 時**: 最後の incremental write が `status: "running"` の妥当な JSON
  として残る。これは `nnue-lab` がそのまま取り込める (`running` は許容 enum)。
  最初の書き込み前に落ちた場合はファイルが存在しないだけで、復元対象がない。
  checkpoint 保存と experiment.json 書き込みの間で crash すると `checkpoints[]`
  が直近 checkpoint を 1 個取りこぼし得るが、`checkpoints[]` は informational
  なので実害はない (lineage は別機構、下記)。

## Lineage

`--resume <net_id>-<sb>.ckpt` で再開した run は `lineage` を持つ。
`resumed_from_checkpoint` / `resumed_from_superbatch` は `--resume` 引数から
直接わかる。残りは 2 段の解決からなる: producer 側で `parent_id` (親 run の
experiment.json の `id`) を特定し、consumer (`nnue-lab`) 側で `lineage.parent_id`
を DB の親子関係に落とす。

### producer 側: 親 run id を特定する

1 run = 1 ファイルかつ checkpoint ファイル名 (`{net_id}-{sb}.ckpt`) は run を
跨いで衝突し得る (同じ `--output` + `--net-id` の再利用で上書きされる) ため、
ファイル名だけからも、`.ckpt` の隣に置いた別ファイルからも「その `.ckpt` を
今所有している run」を確実には特定できない。

**解決機構: `.ckpt` 自身に producer run id を持たせる。** raw checkpoint format
(`.ckpt`) を拡張し、その `.ckpt` を書き出した run の `id` (= その run の
experiment.json の `id`) を `.ckpt` 内に記録する。format version を 1 つ上げる
(この format は version field を持ち、layout 変更時の increment を想定済み)。

- `.ckpt` は従来どおり temp + rename で atomic に書かれる。producer run id は
  その `.ckpt` の中身の一部なので、**weight と常に同一世代**であり、別ファイル
  間の整合をとる必要がない。同名 `.ckpt` を別 run が上書きしても、`.ckpt` と
  その producer run id は単一の atomic write で同時に置き換わる。
- `--resume <X.ckpt>` 時、トレーナーは `X` から producer run id を読み、それを
  `lineage.parent_id` とする。ディレクトリ走査も曖昧性も stale 参照もない。
- 本機能導入前に書かれた古い format version の `.ckpt` には producer run id が
  無い。これらから resume した場合は `parent_id` を省略し、warning を出して
  `lineage` を checkpoint 参照のみとする (resume 自体は妨げない)。古い version
  の `.ckpt` も引き続き resume できる (version で分岐し旧 layout も読む)。

### consumer 側 (nnue-lab): 1 run = 1 row で解決する

`nnue-lab` は producer 側 run id (`id`) を `experiments.producer_id` 列に保存し、
**同一 run の再取り込みを `(tenant_id, producer_id)` で upsert する** — 同じ run
の新しい snapshot が来たら別 row を INSERT せず既存 row を UPDATE する
(完全性順序は連携契約を参照)。結果 **1 run = 1 row** が常に成り立ち、その row
の DB 主キー (`id`、ulid) は run の snapshot 取り込みを跨いで不変になる。

この不変性が lineage 解決を自明にする:

- 親方向: 子 run の取り込み時、`lineage.parent_id` を `producer_id` 列で引くと
  該当 run の row は高々 1 個。その `id` を子 row の `parent_experiment_id` に
  書く。
- 子方向 (out-of-order、子が親より先に取り込まれた場合): 取り込んだ run の
  `producer_id` を `lineage.parent_id` に持つ待機 row を引き、その
  `parent_experiment_id` を埋める。これも各 run = 1 row。
- run を表す row が 1 個なので「どの snapshot 行を親 / 子に繋ぐか」という
  canonical 行選択が要らない。新しい snapshot を取り込んでも row 主キーは
  動かないため、一度張った `parent_experiment_id` 参照が stale 化しない。

1 run の `history` はその run が回した superbatch 範囲に限定する。世代を跨いだ
loss 軌跡は `nnue-lab` が lineage チェーンを辿って連結表示する。

**分岐 (branching) の制約**: `nnue-lab` の lineage は simple chain のみ許容する
(1 親 = 1 子)。同一 checkpoint から複数の resume run を派生させると 2 つ目以降が
同一 parent の child となり、自動連結時に degrade される (繋がれない)。本
format v1 は線形の resume チェーンを前提とし、分岐 resume の自動連結は対象外と
する (`nnue-lab` 側を分岐許容モデルにする変更が将来必要)。

## nnue-lab との連携契約

本 ADR は `nnue-lab` 側の取り込み変更を伴う。下記は producer (experiment.json
出力) と consumer (`nnue-lab`) の双方が満たす契約。

### upsert モデル: 1 run = 1 row

- **構造化 producer は `generator` と run 一意で安定な `id` を出力する。**
  `id` は同一 run の全 snapshot で不変、かつ run を跨いで衝突しない (本リポは
  `{net_id}-{UTC開始時刻}-{pid}`、※2)。`generator` の存在が「この producer は
  run 一意 id 契約に従う」ことの宣言になる。
- `nnue-lab` は `generator` を持つファイルを `(tenant_id, producer_id)` で
  upsert する (`producer_id` = JSON top-level `id`)。同一 run の新しい snapshot
  は既存 row を UPDATE し、別 row を作らない。
- **完全性順序**: upsert が既存 row を UPDATE するのは、取り込む snapshot が
  既存 row 以上に完全なときだけ。順序は (1) `status` が `completed` >
  `running`、(2) `history` 点数が多い方、(3) `last_updated_at` が新しい方。
  これより古い snapshot の取り込みは no-op (既存 row を保つ)。incremental
  write の途中版を順不同で投入しても row が後退しない。
- `generator` を持たないファイル (本機能導入前の experiment.json) は upsert
  対象外で、従来どおり content hash で重複排除し INSERT する。lineage 非対象。
  本リポと `bullet-shogi` は今後 `generator` を必ず出力するため、この経路は
  既存の旧ファイル専用の互換シムに留まる。

### nnue-lab 側に要る変更 (本 ADR と同時に実装する)

- `ExperimentJsonV1` (zod) に optional の `generator` / `lineage` を追加する。
- `experiments` に producer 側 run id 列 (`producer_id`) と宣言された親 run id
  列 (`lineage_parent_id`) を追加し索引する。
- `generator` 付きファイルは `(tenant_id, producer_id)` で upsert する
  (application レベルで既存 row を select し INSERT / UPDATE を分岐)。
  `(tenant_id, producer_id)` に partial unique index を張り race の安全網と
  する。`generator` 無しファイルは従来どおり global `content_hash` UNIQUE で
  重複排除し INSERT する (この UNIQUE は据え置く — generator ファイルの INSERT
  は新規 run の新しい content に限られ衝突しないため、table 再作成を伴う
  migration を避ける)。
- 取り込み時、`lineage.parent_id` を `producer_id` 列で引いて
  `parent_experiment_id` を解決する。親方向・子方向 (out-of-order) の両方を
  解決する (Lineage 節)。解決不能 (親未取り込み / 分岐 / 循環 / 深さ超過) は
  degrade し、取り込み自体は常に成功させる。

### 未改修の nnue-lab に取り込んだ場合の degrade

deploy 過渡期などで上記変更前の `nnue-lab` に取り込んでも、ファイル自体は
取り込める — `id` / `name` / `date` / `params.{lr,batch_size,superbatches}` /
`history[]` が zod 必須を満たし、`params` / `data` / `results` の本リポ固有
キーは passthrough で保持される。`generator` / `lineage` は top-level のため
正規化時に削除され、upsert も lineage 自動連結も効かず、resume 系列は手動
PATCH 連結に degrade する。この degrade 期間でも一覧で世代を見分けられるよう、
`name` の resume 既定を `{net_id} (resume @sb{start})` と分けてある (※3)。

### 将来課題

- `status` enum に `interrupted` 等を追加 (本 format v1 は `running` /
  `completed` のみ使い、未対応でも問題ない)。
- lineage の分岐 (branching) 許容 (上記 Lineage の制約)。

## Rejected alternatives

- **bullet-shogi 方式 (resume で `history` を世代マージし 1 JSON が全 stage を
  含む)**: stage 境界が JSON から判別しにくく、loss 点が世代間で重複する。
  「1 run = 1 ファイル + lineage 参照」を採用 (上記 Decision 3)。
- **`recover_experiment` 相当の復元ツール**: atomic incremental write により
  crash した run も妥当なファイルを残すため不要。最初の書き込み前 crash は
  復元対象がなく、ツールがあっても救えない。
- **1 run = 複数 row (取り込みごとに別 row、`producer_id` 非 UNIQUE)**: content
  hash 重複排除のまま `producer_id` を非 UNIQUE 列として持たせ、read time に
  「最も完全な snapshot 行」を canonical として選ぶ案。incremental write された
  snapshot は別 content hash → 別 row になるため、ある run を親 / 子として
  参照するたびにどの row を指すかを read time に解決する必要があり、新しい
  snapshot 取り込みで canonical 行が動いて `parent_experiment_id` 参照が
  stale 化する。lineage 解決の全体に run-grouping と canonical 行選択が伝播し
  決定論を保てない。`(tenant_id, producer_id)` upsert で 1 run = 1 row を保証
  して row 主キーを不変にする案を採用 (上記 Lineage)。
- **`parent_id` を sibling experiment.json の `checkpoints[]` 走査で解決**:
  checkpoint ファイル名が run を跨いで衝突し得るため、同名 `.ckpt` を書いた
  run が複数あると親を一意に選べない。
- **`.ckpt` の隣に sidecar ファイルを置いて producer run id を記録する**:
  `.ckpt` と sidecar は別ファイルで単一の atomic 操作では書けない。同名
  `.ckpt` を別 run が上書きする際、`.ckpt` の rename 後・sidecar の rename
  前に crash すると「新しい `.ckpt` + 古い run の sidecar」が残り、resume 時に
  誤った `parent_id` を黙って採ってしまう (lineage の誤連結)。producer run id
  を `.ckpt` 内に持たせれば weight と同一の atomic write に入り、この不整合が
  原理的に起きない (上記 Lineage)。
- **本リポ固有フィールドを top-level に置く**: `nnue-lab` の top-level は
  passthrough でなく、未知キーは正規化時に削除されて R2 に残らない。
  passthrough 領域 (`params`/`data`/`results`) に置く。
- **本リポ固有 knob を `params.rshogi` 等にネスト**: `nnue-lab` のパラメータ
  差分表が flat キー前提のため、ネストすると `ft_fp16`/`tf32` 等の run 間
  差分が表に出ない。flat 配置を採用 (上記 ※6)。
- **トレーナーから nnue-lab へ自動 push**: 非対話 auth 投入口が `nnue-lab`
  に必要で、GPU トレーナーに HTTP/auth 依存が乗る。ファイル出力のみに絞り
  push は将来課題とする (別ツール化が候補)。

## Consequences

- 学習 run ごとに構造化ログが残り、`nnue-lab` に取り込んで一覧・loss カーブ・
  比較・lineage 連結ができる。
- crash した run も `running` 状態のファイルとして取り込め、復元ツールを
  保守しなくて済む。
- 本リポ固有の perf knob (`ft_fp16` 等) と throughput が成果物に残り、perf
  改善 run の比較が `nnue-lab` 上でできる。
- `(tenant_id, producer_id)` upsert により 1 run = 1 row。incremental write の
  途中版を順不同で複数回投入しても row は累積せず、最も完全な snapshot に
  収束する。resume 系列の自動連結は `producer_id` → 1 row 直引きで解決する。
- 取り込み変更前の `nnue-lab` に投入した場合は upsert / 自動連結が効かず、
  resume 系列は手動 PATCH 連結に degrade する (取り込み自体は阻害されない)。

## Open questions / future work

- トレーナーから `nnue-lab` への自動 push (別の小さなアップローダ CLI 化、
  `nnue-lab` 側に service token 等の非対話 auth 追加が前提)。
- `progress_kpabs_train` への本 format の展開 (format 互換性と
  `schema_version` の再評価を伴う)。
- per-superbatch の throughput / lr 系列を残す要求が出た場合、`nnue-lab` の
  history 要素 schema を passthrough 化する変更とセットで検討する。
