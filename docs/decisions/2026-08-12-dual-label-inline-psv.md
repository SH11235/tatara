# Dual-label inline PSV

## Status

Accepted

## Context

教師局面には、探索由来の base score と深層モデル由来の DL score の二種類が必要に
なる場合がある。score と局面を別ファイルに置くと、重複除去や shuffle の各処理が
両ファイルの行対応を維持し続けなければならない。

通常 PSV の offset 34-35 は実指し手を表す。通常形式と dual-label 形式は同じ
40-byte record であり、内容だけから安全に区別できない。

## Decision

dual-label PSV は base score、DL score、選択 gate を一つの固定長 record に格納する。
これにより重複除去や shuffle が record 単位で二つの label を原子的に扱い、行対応の
ずれを構造的に排除する。

読み込みは明示的な CLI mode でのみ有効にし、自動判別しない。通常 PSV の move16 を
DL score と誤読すると妥当な record のまま教師値だけが静かに壊れるためである。
予約 padding bit が非ゼロなら形式違反として停止する。

sidecar score override は独立した入力方式として維持する。既存 run の再現と、PSV
本体を変更せず外部で score を差し替える用途には sidecar が適している。二方式の同時
指定は score の優先順位を曖昧にするため拒否する。
