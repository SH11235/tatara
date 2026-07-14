# YaneuraOu 用 LayerStack net 変換

`net_to_yo` は tatara の `HalfKaHmMerged` 1536-16-32、9 bucket LayerStack `.bin`
を YaneuraOu の `SFNN-1536-V2` 評価ファイルへ変換する。

```bash
cargo run --release -p net-to-yo -- \
  --input /path/to/tatara.bin \
  --output /path/to/eval/nn.bin \
  --assume-kingrank9
```

変換対象は PSQT、Threat、EffectBucket を持たない `HalfKaHmMerged` に限定される。
feature set、各層の次元、bucket 数、追加 block の有無が一致しない入力はエラーになる。
量子化 `.bin` は bucket routing mode を記録しないため、変換前に学習時の
`--bucket-mode kingrank9` を確認し、`--assume-kingrank9` で明示する。
既定の `progress8kpabs` で学習した 9 bucket net は、YaneuraOu と bucket の選択規則が
異なるため変換できない。

YaneuraOu の `SFNNwoP1536_V2` loader は FT の bias と weight をそれぞれ signed
LEB128 block として読み、dense 層は bias の i32 LE、続いて canonical row-major の
i8 weight を読む。dense weight は YaneuraOu がロード時に実行用 SIMD layout へ並べ
替えるため、変換ファイルには並べ替え前の順序で格納する。

量子化 scale は両形式とも FT が QA=127、dense weight が QB=64、dense bias が
QA×QB=8128 であり、変換時の scale 変更は行わない。architecture string に
`fv_scale` は含めない。YaneuraOu では読み込み前に `setoption name FV_SCALE value 28`
を指定する。
