# optimizer の per-param-group 化 (FT / dense / bias で weight_decay と LR を別値)

- **Status**: Accepted
- **Date**: 2026-06-04

## Context

LayerStack trainer の Ranger optimizer は、全 weight group に `--weight-decay` の
単一値と `--lr` 由来の単一 learning rate を一律に適用している。FT weight
(`ft_w = ft_in × ft_out`、既定 feature set halfka-hm-merged で ~1.1 億・入力次元の
大きい feature set では ~2 億 params) と出力側の dense weight / bias (合計
~25 万 params、大半は `l1_w = num_buckets × l1_out × ft_out`) が同じ weight decay・
同じ LR で更新される。FT が dense より 2 桁以上大きく、役割も違う層に同一の正則化・
学習率を強制するのは粗く、NNUE 学習の標準的な調整余地を塞いでいる。

深層学習一般では **bias に weight decay をかけない (`weight_decay = 0`)** のが定石で
ある (bias は出力の平行移動成分で、decay で 0 へ引くと表現力を不必要に削るため)。
また入力側の大規模 weight と出力側の小さな dense weight とで、最適な正則化・学習率が
異なることも珍しくない。tatara には層グループ単位で `weight_decay` / LR を変える粒度が
無く、こうした標準的な調整を試せない。

optimizer launch は table-drive 化されており、`radam_step` / `ranger_lookahead_lerp` の
launch は一様 weight group を `UniformOptimGroup` 配列で回す形に集約されている。`radam_step`
kernel は `lr` と `decay` (weight_decay) を **per-launch の scalar 引数**として既に
受け取るため、per-group の値を流すのに kernel ABI 変更は要らない。per-group 化を入れる
素地が整っている。

本 ADR は、この per-param-group optimizer をどの粒度・どの scope・どの既定挙動で
入れるかの設計判断を記録する。per-group の weight_decay / LR 差別化は標準的な参照
実装に前例が無く、純粋に first-principles の投機的な調整軸である。Elo 効果そのものの
可否は SPRT で別途検証する。

## Decision

### 1. param-group は FT / dense / bias の 3 分類

全学習対象テンソルを、規模と役割で次の 3 group (入力側 weight / hidden dense weight /
全 bias) に静的に割り当てる。

| group | 含むテンソル | 役割 |
|---|---|---|
| **ft** | `ft_w`, `psqt_w` | feature-indexed な入力側 weight (大規模) |
| **dense** | `l1_w`, `l1f_w`, `l2_w`, `l3_w` | hidden dense layer weight |
| **bias** | `ft_b`, `l1_b`, `l1f_b`, `l2_b`, `l3_b` | 全層の bias |

`psqt_w` は shape `(ft_in, num_buckets)` の feature-indexed な shortcut weight で
性質が `ft_w` に近いため **ft group** に置く。per-layer の細粒度 (l1/l1f/l2/l3 を
個別) は flag 数と探索空間が膨らみ SPRT で寄与を切り分けにくいため採らない
(§Alternatives)。

### 2. group ごとに weight_decay と LR 倍率の両方を可変化

各 group に独立した:

- **weight_decay** (絶対値)
- **lr_mult** (scheduled LR への相対倍率)

を持たせる。per-group LR は `lr_for(group) = scheduled_lr × lr_mult(group)` とし、
LR schedule (`--lr-schedule`) が決めた毎 step の LR に group 倍率を掛ける
(schedule の後段に倍率を適用)。`radam_step` / lerp の step_size・denom は step と
beta のみから決まり LR 非依存なので group 間で共有する。lerp pass は `RANGER_ALPHA`
のみを使い lr/weight_decay を取らないため本変更の影響を受けない (radam pass のみ)。

注意 (knob 間の結合): decoupled weight decay は `radam_step` 内で `rate = lr ×
step_size` を介して `w *= 1 - weight_decay × rate` と適用される。per-group LR が
`scheduled_lr × lr_mult` なので、ある group の `lr_mult` はその group の勾配更新だけ
でなく **実効 weight decay の強さも同じ倍率で scale する** (AdamW-decoupled 由来の
意図された結合)。`weight_decay` を group の絶対 lever として扱う際、および §SPRT で
`lr_mult` と `weight_decay` を同時に振る際はこの結合を踏まえる。

### 3. CLI

per-group の上書き flag を追加する (いずれも `Option`、未指定で従来挙動)。

- `--ft-weight-decay` / `--dense-weight-decay` / `--bias-weight-decay`
- `--ft-lr-mult` / `--dense-lr-mult` / `--bias-lr-mult`

未指定の group は **大域 `--weight-decay` の値** と **lr_mult = 1.0** にフォール
バックする。

### 4. 既定挙動は bit-identical

per-group flag を一つも指定しなければ、全 group が大域 `--weight-decay` と
`lr_mult = 1.0` を使い、現状と**完全に同じ launch 引数**になる。これにより既定経路は
従来と bit-identical を保つ。検証は raw `.ckpt` ではなく **量子化 `.bin` の
bit-identical** で行う (backward の atomicAdd で `.ckpt` は run-to-run 非決定だが、
量子化出力は決定論的; launch 経路の table-drive 化でも用いた等価性検証手法)。psqt
有無の両構成で確認する。

### 5. bias の weight_decay=0 は opt-in (既定にしない)

「全 bias を `weight_decay = 0`」(Context の定石) は **既定にはしない** (§4 の
bit-identical 既定と両立しないため)。`--bias-weight-decay 0` の明示指定で有効化する
位置づけとし、ADR としては **SPRT で最初に試す推奨構成**として bias wd=0 を挙げる
(下記 §SPRT)。

### 6. 実装は table-drive 化された launch 経路に載せる。state 構造は変えない

per-group の `(weight_decay, lr_mult)` を単一の静的 config (`OptimGroupConfig`) に
resolve し、`UniformOptimGroup` には所属 param-group の識別子を持たせて radam loop の
launch ごとに config から `(weight_decay, lr)` を引く。FT
(`ft_w`) は table 外の特殊ケース (fp16 / fp16-opt-state の 4 variant) なので、ft group
の値を個別に配線する。per-group の lr/weight_decay は**静的 config であって stateful
ではない**ため、`RangerHostState` / optimizer state の構造変更は不要 (per-group 値を
stateful に持つ設計なら state のベクトル化が要るが、本設計では発生しない)。

### 7. checkpoint / resume

lr_mult / weight_decay は実行ごとに CLI から供給される hyperparameter であり、
optimizer の m/v state (既に group ごとの device buffer) とは別物。`.bin` / raw
`.ckpt` の format は変更せず、resume 時は CLI で同じ per-group 値を再供給する運用と
する (LR schedule horizon のような「再供給が要る hyperparameter」と同じ扱い)。
experiment.json には有効な per-group 値を記録する。

### 8. per-group の lr_mult / weight_decay は radam step に限定し、norm loss apply は据え置く

同じ `step_impl` 内に、`--norm-loss` 有効時の per-weight-group L2-norm 正則化
(norm loss apply pass) がある。これは radam step と同じテンソル群を更新し、補正は
大域 scheduled `lr` と `--norm-loss-factor` で scale される。本 ADR の per-group
`lr_mult` / `weight_decay` は **radam optimizer step にのみ適用**し、norm loss apply は
従来どおり大域 `lr` を使い、per-group lr_mult の影響を受けない。

理由: norm loss は独立した正則化機能で、強度は専用の `--norm-loss-factor` が持つ。
per-group lr_mult を norm loss の pull にまで波及させると「optimizer の学習率倍率」と
「norm 正則化の強度」という直交した 2 軸が混線し、両機能の解釈・SPRT 切り分けが
難しくなる。radam step に閉じる方が契約が単純で、norm loss の既存 ADR が掲げる
「weight decay とは独立に併用可」の方針とも整合する。group ごとに norm 正則化強度を
変えたくなった場合は、norm loss 側 config の拡張として別途扱う (本 ADR の scope 外)。

## Alternatives considered

- **per-layer の細粒度 (ft/l1/l1f/l2/l3 + bias を個別)**: 柔軟だが flag が 10+ に
  増え、SPRT で各 knob の寄与を切り分けるのが現実的でない。3 分類で必要な調整力は
  得られる。却下。
- **FT vs rest の 2 分類**: 最小実装だが bias を独立させられず、定石の bias wd=0 を
  表現できない。却下。
- **weight_decay のみ (LR-mult なし)**: scope は小さいが、`radam_step` が既に `lr` を
  per-launch で受けるため LR-mult の追加コストは小さく、分割すると 2 度手間になる。
  両方を一度に入れる。却下。
- **bias wd=0 を既定にする**: 定石に寄せられるが、既定経路の bit-identical
  (§4 の既定方針) を破る。opt-in に留める。却下。
- **optimizer state の group ベクトル化**: per-group 値が stateful なら必要だが、
  lr/weight_decay は静的 config なので不要。採らない。

## Consequences

- FT と出力層・bias を別の正則化 / 学習率で調整でき、bias wd=0 や FT/dense で別 LR
  倍率といった構成を tatara で試せるようになる。
- 既定経路は bit-identical を維持するため、既存 recipe・既存 net への影響はない。
- 実装は table-drive 化された launch 経路の上で完結し、kernel ABI 変更・optimizer
  state 構造変更を伴わない (host plumbing + CLI + per-group 値の resolve)。
- CLI surface が 6 flag 増える。help / experiment.json / quickstart doc の更新が要る。
- **Elo 効果は未検証**: 実装 + 既定 bit-identical 検証を先行し、価値検証は SPRT に
  委ねる。
- 将来拡張: 本 group 機構は per-group の *optimizer algorithm* 切り替え (例: dense 層
  のみ Muon、FT/bias は Ranger) にも一般化し得る。ただし dense は全 param の ~0.2% で
  upside が限定的、かつ Newton-Schulz 反復の実装コストがあるため、本 ADR の scope 外の
  研究課題として別 backlog で追跡する。

## SPRT 計画 (実装後)

baseline = 既定 (全 group 一律、bit-identical) に対し、段階的に:

1. **bias wd=0** (`--bias-weight-decay 0`) のみ — 最も確立した lever、単独効果を測る。
2. FT と dense で weight_decay / lr_mult を変える構成 — 1 で得た知見の上で振る。

neutral / regression なら当該構成は不採用 (flag は残すが既定は据え置き)、有意な
+Elo が出た構成を recipe に反映する。
