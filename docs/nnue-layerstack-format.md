# LayerStack NNUE `quantised.bin` save format

Target architecture:

- `HalfKA_hm`, `ft_in = 73305`, `ft_out = 1536`
- `l1_out = 16`, `l1_effective = 15`, `l2_in = 30`, `l2_out = 32`
- `num_buckets = 9`, `PSQT = none`
- `QA = 127`, `QB = 64`, `FV_SCALE = 28`

All offsets below are zero-based byte offsets. Multi-byte scalar fields are little-endian unless stated otherwise. The two Feature Transformer tensor blocks use signed LEB128 compression, so their compressed payload sizes are data-dependent. Define:

- `Cb`: FT bias LEB128 compressed payload byte count, stored in the FT bias block size header.
- `Cw`: FT weight LEB128 compressed payload byte count, stored in the FT weight block size header.
- `S = 257 + Cb + Cw`: first byte of bucket 0 LayerStacks data.

`Cb` and `Cw` are not derivable from dimensions alone because `encode_leb128_tensor_i16` encodes each quantized value with variable length (`examples/shogi_layerstack.rs:1012-1027`, `examples/shogi_layerstack.rs:1030-1042`). Therefore numeric offsets after the FT blocks are `UNKNOWN` without the actual tensor values.

> **Source paths in this document.** `examples/shogi_layerstack.rs` and `crates/bullet_lib/` are writer-side paths in a separate repository (`bullet-shogi`); `crates/rshogi-*` are loader-side paths in the engine repository (`rshogi`). Neither resides in this repository — only `crates/nnue-format/` paths do. The cited line numbers reference those external repositories and may drift over time.

## A. File layout

Producer order is `header`, `ft_hash`, FT biases LEB128, FT weights LEB128, optional PSQT/threat blocks, then LayerStacks data (`examples/shogi_layerstack.rs:1497-1507`, `examples/shogi_layerstack.rs:1508-1543`, `examples/shogi_layerstack.rs:1786-1808`). For this target, PSQT/threat/hand-threat/hand-count are disabled, so no optional block is present (`examples/shogi_layerstack.rs:1442-1468`, `examples/shogi_layerstack.rs:1545-1581`, `examples/shogi_layerstack.rs:1583-1640`).

| Section | Offset | Size (bytes) | Encoding | Description |
|---|---:|---:|---|---|
| `version` | `0` | `4` | `u32 LE` | `0x7AF32F20`; writer appends this as `nnue_version` (`examples/shogi_layerstack.rs:1497-1500`). |
| `network_hash` | `4` | `4` | `u32 LE` | `fc_hash ^ ft_hash`; source computes `ft_hash = FEATURE_HASH_HM_V2 ^ (ft_out * 2)` and `network_hash = fc_hash ^ ft_hash` (`examples/shogi_layerstack.rs:1436-1439`). For this target: `fc_hash = 0x6333714E`, `ft_hash = 0x7F1340B8`, `network_hash = 0x1C2031F6`; `FEATURE_HASH_HM_V2 = 0x7f134cb8` (`crates/bullet_lib/src/game/inputs/shogi_halfka.rs:27`). |
| `arch_str_len` | `8` | `4` | `u32 LE` | Byte length of `arch_str`; source writes `arch_bytes.len()` (`examples/shogi_layerstack.rs:1495-1503`). For this target: `199`. |
| `arch_str` | `12` | `199` | UTF-8 bytes, no NUL | Exact string is listed in section D. Constructed by `arch_desc = format!(...)` (`examples/shogi_layerstack.rs:1469-1495`). |
| `ft_hash` | `211` | `4` | `u32 LE` | `ft_hash` written immediately after header (`examples/shogi_layerstack.rs:1505-1507`, `examples/shogi_layerstack.rs:1786-1787`). For this target: `0x7F1340B8`. |
| FT bias LEB128 magic | `215` | `17` | ASCII | Literal `COMPRESSED_LEB128` (`examples/shogi_layerstack.rs:1037-1040`). |
| FT bias compressed size | `232` | `4` | `u32 LE` | `Cb`, the following compressed byte count (`examples/shogi_layerstack.rs:1037-1041`). |
| FT biases | `236` | `Cb` | signed LEB128 values | `1536` i16 values, each `round(QA * l0b)`; this is the first of the YO-compatible two FT blocks (`examples/shogi_layerstack.rs:1508-1523`). |
| FT weight LEB128 magic | `236 + Cb` | `17` | ASCII | Literal `COMPRESSED_LEB128` (`examples/shogi_layerstack.rs:1037-1040`). |
| FT weight compressed size | `253 + Cb` | `4` | `u32 LE` | `Cw`, the following compressed byte count (`examples/shogi_layerstack.rs:1037-1041`). |
| FT weights | `257 + Cb` | `Cw` | signed LEB128 values | `73305 * 1536 = 112596480` i16 values, feature-major/column-major from `l0w.values[feat * ft_out + out]`, each `round(QA * weight)` (`examples/shogi_layerstack.rs:1525-1543`). |
| Bucket 0 `fc_hash` | `S + 0` | `4` | `u32 LE` | Per-bucket FC hash; source writes it at the start of every bucket (`examples/shogi_layerstack.rs:1680-1683`). |
| Bucket 0 L1 biases | `S + 4` | `64` | `i32[16] LE` | `16` biases, each `round((QA * QB) * (l1b[bucket,out] + l1fb[out]))` (`examples/shogi_layerstack.rs:1684-1691`). |
| Bucket 0 L1 weights | `S + 68` | `24576` | `i8[16][1536]` row-major | `16 * pad32(1536)`, `pad32(1536)=1536`; for FT inputs, `round(QB * (l1w[bucket,out,in] + l1fw[out,in]))`, padding would be zero but none is needed here (`examples/shogi_layerstack.rs:1693-1727`). |
| Bucket 0 L2 biases | `S + 24644` | `128` | `i32[32] LE` | `32` biases, each `round((127 * QB) * l2b[bucket,out])` (`examples/shogi_layerstack.rs:1729-1736`). |
| Bucket 0 L2 weights | `S + 24772` | `1024` | `i8[32][32]` row-major | `32 * pad32(30)`, `pad32(30)=32`; valid inputs `0..30`, last two bytes per row are zero padding (`examples/shogi_layerstack.rs:1738-1753`). |
| Bucket 0 output bias | `S + 25796` | `4` | `i32[1] LE` | `round((127 * QB) * l3b[bucket])` (`examples/shogi_layerstack.rs:1755-1762`). |
| Bucket 0 output weights | `S + 25800` | `32` | `i8[32]` row-major | `pad32(32)=32`, each `round(QB * l3w[bucket,in])` (`examples/shogi_layerstack.rs:1764-1778`). |
| Bucket 1 | `S + 25832` | `25832` | same as bucket 0 | Bucket order is source loop order `for bucket in 0..NUM_BUCKETS` (`examples/shogi_layerstack.rs:1680-1779`). |
| Bucket 2 | `S + 51664` | `25832` | same as bucket 0 | Same per-bucket layout. |
| Bucket 3 | `S + 77496` | `25832` | same as bucket 0 | Same per-bucket layout. |
| Bucket 4 | `S + 103328` | `25832` | same as bucket 0 | Same per-bucket layout. |
| Bucket 5 | `S + 129160` | `25832` | same as bucket 0 | Same per-bucket layout. |
| Bucket 6 | `S + 154992` | `25832` | same as bucket 0 | Same per-bucket layout. |
| Bucket 7 | `S + 180824` | `25832` | same as bucket 0 | Same per-bucket layout. |
| Bucket 8 | `S + 206656` | `25832` | same as bucket 0 | Same per-bucket layout. |
| EOF | `S + 232488` | `0` | EOF | Loader probes for trailing data and errors if any non-EOF byte remains (`crates/rshogi-core/src/nnue/network_layer_stacks.rs:268-298`). |

Total file size is `232745 + Cb + Cw` bytes. There is no final 64-byte `bullet` padding for this custom format: `save_quantised` skips padding when any `SavedFormat` is custom (`crates/bullet_lib/src/value/save.rs:90-105`).

## B. LEB128 encoding spec

`encode_leb128_tensor_i16(values)` produces exactly:

1. Magic: ASCII bytes `COMPRESSED_LEB128`, 17 bytes (`examples/shogi_layerstack.rs:1037-1040`; rshogi defines the same magic at `crates/rshogi-core/src/nnue/leb128.rs:7-8`).
2. Size header: `u32 LE` byte count of the compressed payload only, not including the 17-byte magic or 4-byte size field (`examples/shogi_layerstack.rs:1037-1041`; rshogi reads it at `crates/rshogi-core/src/nnue/leb128.rs:101-104`).
3. Payload: concatenated signed LEB128 encodings, one per i16 value (`examples/shogi_layerstack.rs:1030-1035`).

Per-value encoding is signed LEB128, not zigzag:

- Emit low 7 bits of the signed integer, arithmetic-shift the signed value right by 7, and set continuation bit `0x80` until the termination condition is met (`examples/shogi_layerstack.rs:1012-1027`).
- Terminate when the remaining value is `0` and emitted byte sign bit `0x40` is clear, or remaining value is `-1` and emitted byte sign bit `0x40` is set (`examples/shogi_layerstack.rs:1016-1024`).
- rshogi decodes by accumulating 7-bit groups, using `0x80` as continuation, and sign-extending when the final byte has bit `0x40` set (`crates/rshogi-core/src/nnue/leb128.rs:14-45`, `crates/rshogi-core/src/nnue/leb128.rs:51-87`).

Padding/alignment: `encode_leb128_tensor_i16` adds no padding or alignment bytes (`examples/shogi_layerstack.rs:1037-1042`). rshogi reads exactly `compressed_size` bytes and decodes until payload EOF (`crates/rshogi-core/src/nnue/leb128.rs:101-117`, `crates/rshogi-core/src/nnue/leb128.rs:119-128`).

rshogi accepts the current two-block YO-compatible FT form when the first decoded block has `L1` elements, then reads the second block as FT weights; it also accepts an older one-block `biases + weights` form when the first decoded block has `L1 + weight_size` elements (`crates/rshogi-core/src/nnue/feature_transformer_layer_stacks.rs:326-407`).

## C. Per-layer weight quantization

Constants:

- `NUM_BUCKETS = 9`, `QA = 127`, `QB = 64` in the writer (`examples/shogi_layerstack.rs:90-92`).
- `bias_scale = QA * QB = 8128` for L1 biases (`examples/shogi_layerstack.rs:1655-1656`).

| Layer | Bit width | Scale factor | Memory layout | Bucket order / notes |
|---|---:|---:|---|---|
| FT biases | i16, signed LEB128 block | `QA = 127` | `i16[1536]`, source order from `l0b.values` | Written before FT weights as separate LEB128 block (`examples/shogi_layerstack.rs:1508-1523`). |
| FT weights | i16, signed LEB128 block | `QA = 127` | `i16[73305][1536]`, feature-major: source comment says `l0w.values[feat * ft_out + out]` (`examples/shogi_layerstack.rs:1531-1538`) | Only the HalfKA piece/base part is written: `piece_end = halfka_dim * ft_out` (`examples/shogi_layerstack.rs:1531-1540`). |
| L1 main outputs | L1 bias i32, L1 weights i8 | Bias `QA * QB = 8128`; weights `QB = 64` | Per bucket: biases `i32[16]`, weights row-major `i8[16][pad32(1536)]` | Buckets written `0..8` (`examples/shogi_layerstack.rs:1680-1779`). Each L1 weight row merges bucket and shared factorized weights for FT inputs (`examples/shogi_layerstack.rs:1693-1717`). |
| L1 skip output | Same bytes as L1: i32 bias + i8 weights | Same as L1 | The skip is not a separate section. It is the last L1 output row, `out_idx = 15`, inside the L1 bias/weight arrays. Runtime treats `l1_out[LS_L1_OUT - 1]` as skip (`crates/rshogi-core/src/nnue/layer_stacks.rs:66`, `crates/rshogi-core/src/nnue/layer_stacks.rs:107-127`). |
| L2 | L2 bias i32, L2 weights i8 | Bias `127 * QB = 8128`; weights `QB = 64` | Per bucket: biases `i32[32]`, weights row-major `i8[32][pad32(30)]`; `pad32(30)=32`, padded bytes zero | Source writes valid input bytes for `in_idx < l2_in`, else zero padding (`examples/shogi_layerstack.rs:1729-1753`). |
| L3/output | Output bias i32, output weights i8 | Bias `127 * QB = 8128`; weights `QB = 64` | Per bucket: bias `i32[1]`, weights row-major `i8[pad32(32)]`; `pad32(32)=32` | Source indexes `l3w.values[in_idx * NUM_BUCKETS + bucket]` and writes zero padding only if padded input exceeds `l2_out` (`examples/shogi_layerstack.rs:1755-1778`). |

`pad32(n)` is `ceil(n / 32) * 32` in the writer (`examples/shogi_layerstack.rs:1049-1051`) and the rshogi layer reader uses the same padding rule via `padded_input` (`crates/rshogi-core/src/nnue/layers.rs:11-14`).

PSQT is absent for this target. The writer includes PSQT only when `psqt` is true, in which case it would write `i32[9]` biases then `i32[HALFKA_HM_DIMENSIONS][9]` weights with scale `QA * QB` (`examples/shogi_layerstack.rs:1545-1577`). The local `nnue-format` PSQT reference documents that bullet LayerStack PSQT is bucketed `output_buckets (9) x num_features`, while its older minimum layout was single-bucket and not loader-compatible as-is (`crates/nnue-format/src/halfka_psqt.rs:40-50`). It also mirrors bullet's `qa * qb` PSQT multiplier (`crates/nnue-format/src/halfka_psqt.rs:305-309`, `crates/nnue-format/src/halfka_psqt.rs:332-333`).

## D. `arch_str` exact value

For the target values and `PSQT = none`, no `PSQT=...`, `Threat=...`, `HandThreat=...`, or `HandCountDense=...` fragment is inserted (`examples/shogi_layerstack.rs:1441-1468`). The exact string generated by the `format!` at `examples/shogi_layerstack.rs:1469-1495` is:

```text
Features=HalfKA_hm(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-30](SqrClippedReLU[30](AffineTransform[16<-3072](InputSlice[3072(0:3072)]))))),fv_scale=28
```

Byte length: `199`.

UTF-8 hex:

```text
46656174757265733d48616c664b415f686d28467269656e64295b37333330352d3e3135333678325d2c4e6574776f726b3d416666696e655472616e73666f726d5b313c2d33325d28436c697070656452654c555b33325d28416666696e655472616e73666f726d5b33323c2d33305d28537172436c697070656452654c555b33305d28416666696e655472616e73666f726d5b31363c2d333037325d28496e707574536c6963655b3330373228303a33303732295d29292929292c66765f7363616c653d3238
```

## E. Engine-side load path

Main `rshogi-usi` path:

1. USI exposes `EvalFile` with default `eval/nn.bin` (`crates/rshogi-usi/src/main.rs:245`).
2. `setoption name EvalFile value <path>` stores the path and calls `init_nnue(&value)` immediately; success records `eval_file_explicit = Some(true)`, failure records `Some(false)` (`crates/rshogi-usi/src/main.rs:688-710`).
3. If no `EvalFile` was explicitly set, `isready` auto-loads `eval/nn.bin` via `init_nnue(DEFAULT_EVAL_FILE)` when material eval is disabled and no NNUE is loaded (`crates/rshogi-usi/src/main.rs:280-326`).
4. `init_nnue(path)` calls `NNUENetwork::load(path)`, stores the resulting `Arc<NNUENetwork>`, and marks NNUE initialized (`crates/rshogi-core/src/nnue/network.rs:837-843`).
5. `NNUENetwork::load` opens the file, wraps it in `BufReader`, and calls `NNUENetwork::read` (`crates/rshogi-core/src/nnue/network.rs:295-300`).
6. `NNUENetwork::read` seeks to get file size, reads version/hash/arch length/arch string, detects activation and feature set, and for LayerStacks seeks back to start and dispatches `LayerStacksNetwork::read_with_options(reader, l1, l2, l3, psqt_override)` (`crates/rshogi-core/src/nnue/network.rs:302-385`).
7. For this target, `LayerStacksNetwork::read_with_options` dispatches `(1536, 16, 32)` to `NetworkLayerStacks1536x16x32::read_with_options` when the `layerstacks-1536x16x32` Cargo feature is enabled (`crates/rshogi-core/src/nnue/network_layer_stacks.rs:607-609`, `crates/rshogi-core/src/nnue/network_layer_stacks.rs:737-770`).
8. `NetworkLayerStacks::read_with_options` reads `version`, `network_hash`, `arch_len`, `arch_str`, parses `fv_scale`, rejects `Factorizer`, skips `ft_hash`, reads the FT with `FeatureTransformerLayerStacks::read_leb128`, conditionally reads PSQT/threat, reads LayerStacks, then verifies EOF (`crates/rshogi-core/src/nnue/network_layer_stacks.rs:148-311`).
9. `FeatureTransformerLayerStacks::read_leb128` decodes the two LEB128 FT blocks for this file: first `1536` biases, second `73305 * 1536` weights (`crates/rshogi-core/src/nnue/feature_transformer_layer_stacks.rs:326-407`).
10. `LayerStacks::read` loops over 9 buckets, skips each bucket `fc_hash`, and calls `LayerStackBucket::read`; `LayerStackBucket::read` reads L1, L2, and output `AffineTransform`s in that order (`crates/rshogi-core/src/nnue/layer_stacks.rs:88-94`, `crates/rshogi-core/src/nnue/layer_stacks.rs:203-223`).

Open item:

- `UNKNOWN: Cb and Cw numeric values`, unless a concrete `quantised.bin` instance is supplied or the exact quantized FT tensors are known. The format is byte-precise with `Cb`/`Cw` as stored fields, but fixed numeric offsets after FT cannot be derived from the source code and target dimensions alone.
