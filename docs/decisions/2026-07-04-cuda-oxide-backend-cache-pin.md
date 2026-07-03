# cuda-oxide の codegen backend cache も自前で pin 検証する

- **Status**: Accepted
- **Date**: 2026-07-04

## Context

`cargo-oxide`(cuda-oxide の PTX codegen を駆動する cargo subcommand)は、
`librustc_codegen_cuda.so` を次の優先順位で解決する:

1. `CUDA_OXIDE_BACKEND` env var
2. カレントディレクトリが cuda-oxide の checkout 自体である場合のローカル repo
3. `~/.cargo/cuda-oxide/librustc_codegen_cuda.so` の cache(**存在するかどうか
   だけ**を見る)
4. 上記が無ければ `https://github.com/NVlabs/cuda-oxide.git` を
   **rev 指定なしで shallow clone**(`main` HEAD の一発取り)して build・cache

この cache は「その利用者がその機械で初めて `cargo-oxide` の何らかのコマンド
(`build` / `doctor` 等)を実行した瞬間の `main` HEAD」に一度だけ固定され、
以降は本リポジトリの `Cargo.lock` が pin する rev と一切照合されない。
`scripts/setup-cuda-oxide.sh` が `cargo install --rev` で `cargo-oxide` の
CLI 本体を pin rev に入れ直しても、この backend cache には触れないため、
CLI と実際に走る device codegen backend の rev が気づかれないまま乖離しうる。

この乖離は host OS / GPU 世代 / LLVM バージョンの違いでは説明できない
device codegen の不可解な失敗(利用者ごとに再現したりしなかったりする)として
現れる。cuda-oxide の `mir-importer`(MIR → device IR の翻訳部)は活発に変更が
入っている領域のため、pin rev と乖離した backend は host 側 API(pin rev の
`cuda-core` / `cuda-host` / `cuda-device`)との間で codegen の対応関係が
崩れうる。

## Decision

`scripts/setup-cuda-oxide.sh` が、CLI の再 install に加えて
**backend cache 自体も pin rev に揃っているか検証・修復**する:

- backend cache の場所に `.pin-stamp`(pin rev の full SHA + `rustc --version`
  の組)を書き、次回実行時にこれを比較する。full SHA まで見るのは short rev
  の曖昧一致を避けるため、`rustc --version` まで見るのは backend `.so` が
  正確な nightly ABI に紐づくため(toolchain のみが変わる場合もこの stamp が
  検知する)。
- 不一致 (または stamp 不在) のときだけ cache を丸ごと破棄し、pin rev を
  明示指定して取り直す。cuda-oxide 自体の「shallow clone → build」手順を
  再実装するのではなく、`cargo-oxide doctor` が cache 不在時に自動で
  実行する fetch-and-build 経路(cache に `src/Cargo.toml` が既にあれば
  clone 自体は skip される)にそのまま乗せる。
- 一致していれば no-op — 毎回 fetch/rebuild するとその分ビルドが遅くなる。

`scripts/build-kernels.sh` は同じ stamp を**読むだけ**の軽い比較で
fail-fast する。cache の修復は setup 側だけが行い、build 側は cache を
書き換えない。理由: pin bump は `Cargo.lock` の更新だけで起こりうる
(誰も `~/.cargo/cuda-oxide/` に触れなくても、pull した瞬間に CLI・cache が
両方とも古い pin のまま取り残される)ため、build を叩く経路(ローカルの
`bash scripts/build-kernels.sh` も `scripts/local-ci.sh` 経由の CI も)で
乖離を検知できないと、意味不明な codegen エラーとしてしか気づけない。

## Consequences

- backend cache が pin rev と食い違っている(または初回)ときだけ、
  `setup-cuda-oxide.sh` の実行に数十秒〜数分余分にかかる(git fetch +
  backend crate の build)。一致していれば従来通り数秒で終わる。
- 意味不明な device codegen エラーに遭遇したときの復旧手順は
  「`bash scripts/setup-cuda-oxide.sh` を再実行する」の一手に単純化される。
  それでも直らない場合の最終手段として `rm -rf ~/.cargo/cuda-oxide` を
  手動で行ってから再実行する逃げ道は残す(この仕組み自体が cuda-oxide の
  内部実装 — cache パスの位置、`src/Cargo.toml` の有無で clone を skip する
  挙動 — に依存しているため)。
- `CUDA_OXIDE_BACKEND` env var を設定している利用者は、この検証・修復を
  意図的に bypass する(優先度 1 のため)。
- cuda-oxide 本体側のこの cache 設計(rev 未検証)は upstream 側の gap でも
  ある。本リポジトリ側の緩和策とは独立に、upstream への issue 報告は別途
  検討する。
