**English** | [日本語](progress-bin.ja.md)

# Game-progress buckets: preparing `progress.bin`

The `layerstack` architecture routes each position to one of N game-progress
buckets, and the bucket assignment is decided by `progress.bin` — a set of
KP-abs coefficients that estimate how far a position has advanced within its
game. This page covers how to train `progress.bin` and how to check the bucket
split it produces. The `simple` architecture has no buckets and does not need
it; for the overall training flow see
[docs/training-quickstart.md](training-quickstart.md).

The idea of learning game progress and assigning it to output buckets is based
on [a post by nodchip](https://nodchip.hatenablog.com/entry/2026/02/04/000000).

## Generating progress.bin

Train the progress coefficients with `progress-kpabs-train`.

> **Do not shuffle the data.** Pass `progress-kpabs-train`'s `--data` a PSV of
> **consecutive games** (positions in game order, with games following one
> after another). The progress coefficients learn "how far a position has
> advanced within a single game", and `progress-kpabs-train` reads the data one
> game at a time (detecting game boundaries by `game_ply`) and labels each
> position with its relative position within that game. With a shuffled PSV the
> game boundaries break, the labels become meaningless, and correct coefficients
> cannot be learned. As a general rule the main NNUE training (`nnue-train`)
> benefits from a *shuffled* PSV; progress training is the opposite, so do not
> reuse the same file for both.

Specify the total number of epochs with `--epochs`. Each epoch writes a
`<run-name>.e<N>.bin` checkpoint, and the final epoch is also written to the
`--output` path.

```bash
target/release/progress-kpabs-train \
  --data <path/to/consecutive-psv.bin> \
  --output output/progress/<run-name>.bin \
  --games-per-step 1024 --epochs 5
```

Which epoch's output (`<run-name>.e<N>.bin`) to use takes some trial and error
(progress.bin is a coefficient that decides bucket assignment and is independent
of NNUE training convergence, so how many epochs you need is data-dependent).

To help decide which epoch to use, add `--val-fraction <f>` (e.g. `0.05`): it
holds out roughly that fraction of games — every Nth game in input order, since
the data must stay in consecutive-game order — and reports a held-out `val_loss`
at the end of each epoch. This adds one extra pass over the data per epoch.

Treat `val_loss` as a sanity check and an epoch-selection hint, not a precise
quality score. The progress model is simple — per-feature weights summed and
passed through a sigmoid — so it overfits little: a small `train_loss`/`val_loss`
gap is normal, and a clearly widening gap is what to watch for. Because the real
goal is a good bucket split, which a plain MSE only approximates, prefer an
epoch where `val_loss` levels off over chasing its exact minimum, and judge the
final `progress.bin` by the strength of the LayerStack NNUE trained with it.

## Checking the bucket distribution

Use `progress-bucket-survey` to measure the share of PSV records assigned to
each bucket. Give `--progress` and optionally `--num-buckets` (default 8), or
give `--bucket-spec`. To cross-tabulate two definitions, add
`--compare-bucket-spec` or `--compare-progress` with optional
`--compare-num-buckets`.

```bash
cargo build --release -p progress-bucket-survey
target/release/progress-bucket-survey \
  --data teacher.psv --progress progress.bin --num-buckets 8 \
  --compare-bucket-spec 'ownking:2/3/7+oppking:2/3/7' --samples 200000
```

The default samples up to 32 evenly spaced windows per file. A quota of one
position selects the middle of the eligible range. Use `--chunks 1` to read
from the specified offset. Repeat `--data` for multiple teachers and add
`--per-pack` for separate tables. The combined sample
is allocated by file size. Cross-tab cells are percentages of all sampled
positions. Add `--tsv` for counts and fractions, with `all` marking marginal
and grand totals. The `scope` column identifies per-pack rows as `pack`
and combined rows as `aggregate`.

Once you have a `progress.bin` you are happy with, pass it to `nnue-train` via
`--progress-coeff` when training a `layerstack` net (see
[docs/training-quickstart.md](training-quickstart.md)).
