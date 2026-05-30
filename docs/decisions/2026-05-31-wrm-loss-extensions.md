# WRM loss の nnue-pytorch 一般化パラメータ (pow_exp / qp_asymmetry / weight boost)

- Status: Accepted
- Date: 2026-05-31

## Context

WRM (win-rate-model) loss kernel `loss_wrm` は誤差を `err²` 固定で扱っていた。
nnue-pytorch の `calculate_sf_loss` (`model/lightning_module.py`) は誤差項を
`pow(|pt - qf|, pow_exp)` に一般化し、過大評価ペナルティ (`qp_asymmetry`) と
決着寄り局面の重み増幅 (weight boost) を default で乗せ、全体 loss を
`(loss * weights).sum() / weights.sum()` で正規化する。tatara でも表現力を揃え、
これらを CLI から可変にしたい。

設計上の争点は 2 つ:

1. **weight boost の正規化**。nnue-pytorch は `Σweights` で割る。tatara の
   `loss_wrm` は単一 scalar `per_pos_norm = 1/n` で正規化する単一-pass kernel で、
   `Σweights` は全 position の reduction を要する。
2. **bit-identical default**。既存の学習 run / SPRT baseline をずらさないため、
   既定パラメータでは従来の `err²` 経路と bit 単位で一致させたい。

## Decision

**Σweights を忠実に再現する。** 既定の拡張パラメータ
(`pow_exp=2 / qp_asymmetry=0 / weight_boost_w1=0`) では従来の二乗誤差に帰着する
**bit-identical な default 経路**を残し、いずれかが非既定のときだけ
**extended 経路**に切り替える。

- extended では新 kernel `wrm_weight_sum` が per-position weight
  `w = 1 + (2^w1 - 1) * ((pf-0.5)^2 * pf*(1-pf))^w2` を Σ し device の f64 cell
  (`weight_sum_acc`) に reduce する。同 stream 上で `loss_wrm` の前に launch し、
  `loss_wrm` の extended 経路はその `Σw` を読んで grad を `w_i / Σw`、loss 寄与を
  `L_i * w_i * n / Σw` で正規化する。weight は score (target side) のみに依存し
  net 出力には依らないので、勾配計算前に 1-pass で確定できる。
- default 経路は kernel 内 `if extended == 0` branch で従来式をそのまま実行する。
  `__nv_powf(|err|, 2)` は `err*err` と bit 一致せず、`1/Σw` も host が f32 で
  計算した `1/n` と最終 bit が一致しないため、項をくくり出さず従来の演算順を保つ。
- loss 寄与に `n / Σw` を掛けるのは、学習ループが batch を跨いで `Σ` を取り
  `Σ position 数` で割る集計に合わせるため (per-batch 寄与を
  `position 数 × batch 平均 loss` に揃える)。default の `Σ err²` と同単位。

CLI flag / experiment.json key / `LossKind::Wrm` field は nnue-pytorch の
`w1`/`w2` 名を維持する (upstream 値・SPRT 結果と cross-reference するため)。

## Consequences

- 既定 run は従来 kernel と数値完全一致 (GPU↔CPU 同等性テスト
  `loss_wrm_default_matches_cpu` で検証)。
- extended 有効時は kernel launch が 1 本増える (`wrm_weight_sum`)。weight cell は
  f64 1 要素で、毎 step memset は無視できるコスト。
- `f32::powf` を kernel 内で使う (cuda-oxide が `__nv_powf` に lowering、
  `loss_wrm_extended_matches_cpu` で実機検証済)。
- 有効化時の棋力影響は不明なため SPRT 必須。LR は `Σw` 正規化で吸収されず
  weight 形状が勾配方向を変えるので、weight boost と LR は独立にチューニングする。
