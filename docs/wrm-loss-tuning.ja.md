[English](wrm-loss-tuning.md) | **日本語**

# WRM loss のチューニング

`--win-rate-model` を有効にすると、教師 score と net 出力の双方を勝率 (win-rate) に
変換し、その二乗誤差を最小化する WRM (win-rate-model) loss を使う。net 出力が
`cp / 600` スケールで収束し量子化フォーマットと整合する点など「なぜ WRM を使うか」は
[学習 Quickstart](training-quickstart.ja.md) の `--win-rate-model` を参照。

本ページは WRM の変換式と、それを調整する CLI 引数を説明する: 勝率変換の geometry
(5 引数) と、loss の一般化 (4 引数、nnue-pytorch の loss に倣う)。いずれも
`--win-rate-model` 指定時のみ効く。既定値はそのまま使える標準設定なので、score
分布に合わせたいときだけ調整すればよい。

## WRM 変換

`sigmoid(x) = 1 / (1 + e^(-x))` とする。1 局面ごとに、prediction (net 出力) と
target (教師 score、単位は centipawn) を別々に勝率へ変換する:

```text
# prediction 側 (net 出力)
scorenet = net_output * nnue2score
q   = sigmoid((scorenet  - in_offset) / in_scaling)
qm  = sigmoid((-scorenet - in_offset) / in_scaling)
qf  = 0.5 * (1 + q - qm)

# target 側 (教師 score)
pt         = (score  - target_offset) / target_scaling
pmt        = (-score - target_offset) / target_scaling
target_wrm = 0.5 * (1 + sigmoid(pt) - sigmoid(pmt))
target     = lambda * wdl + (1 - lambda) * target_wrm   # lambda は --wdl (既定 0)

loss = mean((qf - target)^2)   # 既定。下の引数で一般化する
```

`q` / `qm` は「勝ち」「負け」をそれぞれ片側 sigmoid でモデル化し、その対称差が最終
勝率 `qf` になる。`offset` はこの片側 sigmoid の中心 (score がこの値のとき片側勝率が
0.5)、`scaling` は入力スケール (傾きの逆数で、小さいほど勝率が score に鋭敏) を表す。
prediction 側と target 側で offset / scaling を独立に指定できる。

## 5 つの引数

| flag | 既定 | 適用側 | 役割 |
|---|---:|---|---|
| `--wrm-nnue2score` | 600 | 共通 | net 出力を centipawn スケールに戻す係数 (`scorenet = net_output * この値`)。net 出力は `cp / nnue2score` で収束する |
| `--wrm-in-scaling` | 340 | prediction | prediction 片側勝率 sigmoid の入力スケール (傾きの逆数) |
| `--wrm-in-offset` | 270 | prediction | prediction 片側勝率 sigmoid の中心 offset (`scorenet` がこの値で片側勝率 0.5) |
| `--wrm-target-offset` | 270 | target | target 片側勝率 sigmoid の中心 offset |
| `--wrm-target-scaling` | 380 | target | target 片側勝率 sigmoid の入力スケール |

`--wdl` (上式の `lambda`) は target を WRM 勝率と WDL ラベル ({0, 0.5, 1}) で混ぜる
係数。既定 0 では target = `target_wrm` のみ、1 で純 WDL になる。

## loss の一般化 (pow_exp / asymmetry / weight boost)

既定の WRM loss は素の二乗誤差 `mean((qf - target)^2)`。さらに 4 つの引数で誤差項を
一般化できる (nnue-pytorch の loss に倣う)。これらが既定値のときは上の二乗誤差と
bit-identical なので、指定しなければ挙動は変わらない。

1 局面ごとに、`pf = target_wrm` (WDL blend 前の score 由来勝率) として:

```text
err    = qf - target
weight = 1 + (2^w1 - 1) * ((pf - 0.5)^2 * pf * (1 - pf))^w2
asym   = (qf > target) ? 1 + qp_asymmetry : 1     # 過大評価のときだけペナルティ
loss_i = asym * |err|^pow_exp * weight
loss   = sum(loss_i) / sum(weight)                # weight 和で正規化
```

| flag | 既定 | 役割 |
|---|---:|---|
| `--loss-pow-exp` | 2.0 | 誤差項 `\|qf - target\|^pow_exp` の冪。2.0 で二乗誤差、nnue-pytorch は 2.5。1 以上 |
| `--loss-qp-asymmetry` | 0.0 | prediction が target を超える (`qf > target`、過大評価) 局面の loss を `1 + この値` 倍する追加ペナルティ。0 で対称 |
| `--loss-weight-boost-w1` | 0.0 | per-position 重み増幅の強さ。`0` で重み一律 1 (boost 無し)、大きいほど決着寄り (`pf` が 0/1 付近) を強調。0 以上 |
| `--loss-weight-boost-w2` | 0.5 | 重み式の冪 (カーブ形状)。`w1` が 0 のとき無効。0 以上 |

`pow_exp` / `qp_asymmetry` / `w1` のいずれかが非既定のとき、全体 loss は per-position
重みの総和 (`Σ weight`) で正規化される (nnue-pytorch と同じ)。`w1` / `w2` は 0 以上に
制約され、これにより各 weight が 1 以上となって正規化が well-defined になる。引数名
`w1` / `w2` は nnue-pytorch に揃えてあり、その推奨値をそのまま流用できる。

## 既定値と再チューニング

既定値 (offset 270 / target scaling 380 / in scaling 340 / nnue2score 600) は chess の
評価値分布向けに調整された値。将棋の score 分布がこれと異なる場合は、勝率変換が score を
過剰に飽和させる / させなさすぎる可能性があるため再 tune を検討する。prediction 側
(`in_*`) と target 側 (`target_*`) は独立なので、教師の勝率カーブ (target) と net が
学習すべき勝率カーブ (prediction) を別々に合わせられる。

再 tune の良し悪しは loss 値だけでは判断できない。SPRT 自己対局で棋力を比較して検証する。

実際に使われた WRM 値は `experiment.json` に記録される (`loss_kind` が `"wrm"` の
ときのみ): 変換パラメータ `wrm_in_scaling` / `wrm_in_offset` / `wrm_nnue2score` /
`wrm_target_offset` / `wrm_target_scaling` と、一般化 loss パラメータ `wrm_pow_exp` /
`wrm_qp_asymmetry` / `wrm_weight_boost_w1` / `wrm_weight_boost_w2`。

loss の一般化を有効にすると、報告される loss 値だけでなく勾配自体が変わるので、
棋力への影響は再チューニングと同様 SPRT で検証する必要がある。

## 関連

- [学習 Quickstart](training-quickstart.ja.md) — `--win-rate-model` を含む主な option
- [experiment.json スキーマ](decisions/2026-05-17-experiment-json.md) — WRM パラメータの記録形式
