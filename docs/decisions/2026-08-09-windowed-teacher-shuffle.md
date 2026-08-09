# Windowed teacher-data shuffle

## Status

Accepted

## Context

NNUE training reads fixed-size PSV records repeatedly. A direct sequential reader is friendly to
rotating disks, but every dataset pass presents the same order. Fully shuffling a multi-billion
record dataset at runtime would require impractical memory or random I/O.

The input pipeline also needs enough read-ahead to keep decode workers supplied when teacher data
is stored on a slower disk. Batch prefetch alone only reads a small number of batches ahead and
does not decouple longer storage stalls from GPU consumption.

## Decision

Training uses two raw PSV windows. A producer fills one window sequentially while decode workers
consume the other. Each completed window is shuffled with deterministic Fisher-Yates ordering;
the seed includes the configured base seed, physical dataset epoch, and window index. A partial
window at physical EOF is emitted separately, so records from different epochs never share a
shuffle window.

The window size is configured in MiB per window and is aligned down to a whole training batch.
The default `auto` size is one sixteenth of the smaller of total system RAM and the process cgroup
memory limit, clamped to 256 MiB through 4096 MiB per window. This gives larger-memory training
machines a wider shuffle range while bounding the two raw windows to roughly one eighth of the
effective memory limit and at most 8 GiB. Current free memory is deliberately excluded so the same
machine or container configuration resolves consistently across runs. A numeric value fixes the
window size; zero selects the direct reader. Shuffle can be disabled independently to compare
double-buffered I/O with and without reordered training data.

Score sidecars, score filtering, and score clamping are applied before records enter a shuffle
window. Their record indexing therefore remains tied to the original PSV file.

The seed makes each window permutation reproducible. Runs with multiple decode workers can still
deliver completed batches in a different order because worker scheduling is intentionally
parallel.

## Consequences

- Storage access remains sequential and the producer can absorb multi-second throughput stalls.
- Later dataset passes use different local orderings without requiring dataset-sized memory.
- Shuffle is bounded by the configured window; it is not a uniform permutation of the full file.
- Peak raw teacher-data capacity is approximately twice the configured size, in addition to
  decoded batch buffers and model state.
- Runs print and record the effective buffer size, shuffle state, and seed in experiment metadata.
