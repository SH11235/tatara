# Review checklist (コメント / ドキュメント / 命名の冗長性・適切性)

PR / branch のレビュアー (人 / AI 双方) が **コメント・ドキュメント・ファイル
命名の冗長性と適切性** を機械的に検出するための checklist。コード logic /
correctness のレビューは別 (これは `scripts/local-ci.sh` の fmt / clippy /
test で担う)。

ルールの prevention 側 (書く前に止める) は [CLAUDE.md](../CLAUDE.md) の
「コードコメント規約」「ドキュメント規約」を参照。本 checklist は detection
側 (書かれたものを見つけて指摘する) を担う。

## 1. 禁止語彙の検出

### 1.1 PM シーケンスラベル

プロジェクト管理上の順序ラベルは local context 依存で OSS reader に意味が
通らない。

```bash
rg -n --type rust --type-add 'cfg:*.{toml,yaml,yml,sh,md}' --type cfg \
   -e 'Stage [0-9]' -e 'Phase [0-9]' -e 'Step [0-9]+ で' \
   -e 'Round [0-9]' -e 'Iteration [0-9]' -e 'Sprint [0-9]' \
   -e '\bM[0-9]+\b' -e 'マイルストーン' \
   -g '!target/**'
```

ヒットしたら文脈を確認。以下は **false positive** として許容:

- `Phase N of <algorithm>` 型: kernel / アルゴリズムの multi-pass 説明
  (例: `Phase 1 of inverse-index sparse_ft_backward`)
- `// Step 1: llvm-link ... / Step 2: opt ... / Step 3: llc ...` 型: 関数内の
  pipeline 順序を指す行内コメント
- `M[0-9]+`: 数式の変数 (行列 `M1`/`M2`、定数 `M3` 等)
- ファイル名内: `setup.md` の `Stage` 言及など、quote として「Stage」表記
  そのものを引用しているケース (= 規約文書側に出てくる)

PM シーケンス (作業順序のラベル) として使われていれば NG。

### 1.2 Issue / PR / commit 番号参照

```bash
rg -n --type rust --type-add 'cfg:*.{toml,yaml,yml,sh,md}' --type cfg \
   -e 'Issue #[0-9]' -e 'PR #[0-9]' -e '#[0-9]+ で' -e '#[0-9]+ review' \
   -g '!target/**' -g '!.git/**'
```

代わりに git log / GitHub UI から PR を参照する。コード側に PR 番号が必要な
ケースは「workaround for X bug, see GitHub issue X」型の external bug
tracker 参照に限る (本リポでは現状なし)。

### 1.3 作業ログ語彙

```bash
rg -n --type rust --type-add 'cfg:*.{toml,yaml,yml,sh,md}' --type cfg \
   -e '削除済' -e '追加した' -e '今回' -e '以前は' -e '昇格' \
   -e 'から移動' -e '旧 ' -e '新規追加' \
   -g '!target/**'
```

ヒットしたら文脈確認。以下は **false positive**:

- `CLAUDE.md` / `docs/review-checklist.md` 自身: 規約の例として禁止語彙を
  引用している
- 「LLVM 21 → LLVM 22 への昇格」「llc-22 への昇格」型: ツール / version の
  アップグレードを指す通常の語彙 (migration history ではない)

「N → M に変更」型は手動検索 (regex で false positive 多い)。コード差分の
narration は git diff / PR description が担当。

## 2. ドキュメント特有チェック

### 2.1 ATTRIBUTION.md の純度

- license attribution + vendor 範囲のサマリ **のみ**
- 個別 PR の vendor 詳細、CHANGELOG 的 entry、resume 機能の挙動説明等の
  「作業内容」は NG → 別 doc に切り出す
- 期待行数: 100 行以下 (バルク内容は他 file へ)

```bash
wc -l ATTRIBUTION.md   # ≦ 100 を期待
rg -n 'Stage|Issue #|PR #|EPIC' ATTRIBUTION.md   # 0 件を期待
```

### 2.2 dated header

```bash
rg -n -e '^\s*作成:\s*[0-9]{4}-[0-9]{2}' \
     -e '^\s*改訂:\s*[0-9]{4}-[0-9]{2}' \
     -e '^Last updated:' \
     docs/ README.md CLAUDE.md ATTRIBUTION.md
```

ADR の `**Date**: YYYY-MM-DD` field は意味の一部なので OK。それ以外で日付
header は不要 (git log で確認可能)。

### 2.3 ADR (`docs/decisions/`) のルール

- ファイル名: `YYYY-MM-DD-<slug>.md` 形式
- 連番 (`0NNN-...`) はリジェクト (並行 PR 衝突)
- 内容: 現在のアーキの WHY のみ。執行済 workflow / 完了済ロードマップは削除
  候補
- 不要になった ADR は削除して良い (古い ADR は immutable とは限らない、ただし
  削除時は他の ADR / doc からのリンク切れを修正する)

```bash
ls docs/decisions/ | grep -vE '^[0-9]{4}-[0-9]{2}-[0-9]{2}-.+\.md$'
# 出力ゼロを期待 (= 全 ADR が date-based slug)
```

### 2.4 directory tree の現状反映

`README.md` の repo ツリー / 構成テーブル記述が現状と合っているか。
削除済 directory や架空の予定 directory が残っていないか。

```bash
# tree 記述に出てくる directory が実在するか
for d in $(rg -oE '^[│├└\s─]+([a-z_-]+)/' README.md \
            | sed -E 's/.*[│├└\s─]+([a-z_-]+)\/$/\1/' | sort -u); do
  [ -d "$d" ] || echo "MISSING: $d"
done
```

## 3. コード特有チェック

### 3.1 漠然とした上流参照

`bullet 上流` 等の参照は **algorithm の出典 (paper / 関数名 / 数式)** を伴うか
確認する。漠然と「上流参照」「詳細は bullet 上流」のみは NG。

```bash
rg -n --type rust 'bullet 上流' -B1 -A2
# 出力を目視確認。「bullet `ranger_step` のアルゴリズム」「bullet
# `linear/sparse.rs::SparseMatmul::evaluate` と等価」等の具体的な
# 出典を伴っていれば OK
```

### 3.2 「本リポ命名 alias」型コメント

```bash
rg -n --type rust -e '本リポ命名' -e '別名alias' -e '本リポ alias'
```

ヒットが出たら「変数名を本来の名前に rename」または「コメントを削除して
glossary に統合」を検討。コメントで補強せず命名で解決する方が望ましい。

### 3.3 SAFETY コメントの質

```bash
rg -n --type rust 'SAFETY:' -A2
```

各 SAFETY コメントが「**なぜ** その unsafe が soundness を保つか」を具体的に
書いているか確認。「OK」「問題ない」「Stage X と同型」型は NG。

### 3.4 略語の未定義

新しい略語が code / doc に登場したら `README.md` の glossary 章に追加されて
いるか確認。

```bash
# glossary に登録済の略語リスト
GLOSSARY=$(rg -oE '^\| \*\*([A-Z][A-Za-z_]+)\*\*' README.md | sed -E 's/\| \*\*([A-Za-z_]+)\*\*/\1/')
echo "$GLOSSARY"
# 新規 PR で登場した略語と突き合わせ
```

## 4. 命名のルール

### 4.1 連番ファイル名の回避

ADR 以外も「`0NNN-` で始まる連番ファイル」は並行 PR 衝突を招く。新しい連番
ファイル種を作るときは date-based slug or topic-only slug を検討する。

```bash
rg -l '^[0-9]{4}-' --files docs/
# date-based でないファイルが ADR 以外で見つかったら設計再考
```

### 4.2 path 参照の追従

ファイル / ディレクトリを rename / 削除したとき、コメントや doc の path 参照が
追従しているか:

```bash
# 主要な repo 内 path 参照を抽出して実在チェック
rg -oE '\`?(crates|bins|docs|scripts)/[a-zA-Z0-9_/-]+\.(rs|md|toml|sh|yaml)\`?' \
   --type rust --type md --type-add 'cfg:*.{toml,yaml,yml,sh}' --type cfg \
   -g '!target/**' \
   | sort -u | while read -r ref; do
     path=${ref//\`/}
     [ -e "$path" ] || echo "BROKEN REF: $ref"
   done
```

## 5. レビューフロー

1. `bash scripts/local-ci.sh` PASS を前提
2. 本 checklist の 1.x (禁止語彙 grep) を実行 → 0 件
3. 本 checklist の 2.x / 3.x / 4.x を目視 + grep
4. ヒットがあれば PR コメントで指摘、CLAUDE.md の該当規約をリンク
