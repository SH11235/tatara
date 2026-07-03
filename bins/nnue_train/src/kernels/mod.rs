//! `#[kernel]` device function 群。
//!
//! cuda-oxide の bin-entry reachability 制約により kernel は bin crate 内に置く
//! 必要がある (別 crate に出すと kernel artifact から外れる)。その制約下で kernel を
//! 3 file に分ける: [`common`] (損失 / optimizer 等の共通)、[`layerstack`]
//! (LayerStack 専用)、[`simple`] (Simple アーキ専用)。
//!
//! ## LayerStack アーキテクチャ (FT → L1 → L2 → L3 + progress bucket)
//!
//! PSQT は `--psqt` の opt-in (`psqt_diff_sparse_*`)、hand_count_dense 無し。
//! FT 入力次元 `ft_in` は feature set 依存。FT 出力・L1 / L2 次元・bucket 数は
//! CLI (`--ft-out` / `--l1` / `--l2` / `--num-buckets`) で決まる runtime 値で、
//! 以下の数値は既定値 (1536 / 16 / 32 / 9 bucket) での例:
//!
//! - **L0 (FT)**: sparse_ft_forward — weight (ft_in × ft_out), bias (ft_out, 共有)
//! - **per-perspective post**: bias add → CReLU → pairwise_mul (ft_out→ft_out/2) → ×127/128
//! - **combined**: stm.concat(nstm) → ft_out
//! - **L1 (per-bucket)**: weight (9×16, ft_out) + bias (9×16) → select(bucket) → 16
//! - **L1f (shared)**: weight (ft_out, 16) + bias (16) → 16
//! - **l1_out_t**: L1_select + L1f → 16; slice → l1_main (15) + l1_skip (1)
//! - **l1_sqr**: l1_main^2 * 127/128 → 15
//! - **l2_input**: CReLU(concat(l1_sqr, l1_main)) → 30
//! - **L2 (per-bucket)**: weight (9×32, 30) + bias (9×32) → select(bucket) → CReLU → 32
//! - **L3 (per-bucket)**: weight (9×32) + bias (9) → select(bucket) → 1
//! - **net_output**: l3_out + l1_skip → 1 scalar
//!
//! ## kernel 一覧
//!
//! kernel の一覧は各 file の `#[kernel]` 定義そのもの。各 kernel の役割は定義箇所の
//! doc コメントを参照、アーキ上の繋がりは上記 LayerStack アーキテクチャ節を見る。
//! cuda-oxide は全 `#[kernel]` を `.ll` の `@llvm.used` に列挙するので、`opt` の
//! internalize / globaldce を通っても kernel artifact (PTX) から漏れない。
//!
//! ## cuda-oxide 制限への対応
//!
//! - `f32::clamp` / `f32::max` / `f32::min` lowering 失敗 → `if-else` ladder で展開
//! - `i32::clamp` も同様 (Debug::fmt panic 経路を含む)
//! - `f32::sqrt`, `f32::exp` は libdevice (`__nv_sqrtf`, `__nv_expf`) に lowering OK
//! - atomic add パターン: `unsafe { &*(slice.as_ptr().add(idx) as *const DeviceAtomicX) }
//!   .fetch_add(_, AtomicOrdering::Relaxed)`

// f16 格納値を有限域へ clamp し、発火時は per-thread counter を更新する (計数版。
// 計数不要な clamp は対象外)。if-else 形なのは cuda-oxide が f32::clamp を lower
// できないため (上記「cuda-oxide 制限への対応」参照)。
macro_rules! clamp_f16_value {
    ($value:ident, $local_clamps:ident) => {
        if $value > 65504.0_f32 {
            $local_clamps += 1;
            65504.0_f32
        } else if $value < -65504.0_f32 {
            $local_clamps += 1;
            -65504.0_f32
        } else {
            $value
        }
    };
}

// f16 clamp の発火数を格納後に cumulative counter へ加算する。
macro_rules! finish_f16_clamp_count {
    ($clamp_counter:ident, $local_clamps:ident) => {
        if $local_clamps > 0 {
            // counter は cumulative: host は memset reset を出さない契約。
            // SAFETY: caller は clamp_counter.len() == 1 を保証する。DeviceAtomicU64 は
            // #[repr(transparent)] over UnsafeCell<u64> なので u64 と同じ layout /
            // alignment で、同じ cell への書き込みは全て atomic。
            let cell = unsafe { &*($clamp_counter.as_ptr() as *const DeviceAtomicU64) };
            cell.fetch_add($local_clamps, AtomicOrdering::Relaxed);
        }
    };
}

mod common;
mod layerstack;
mod simple;

// `cuda_launch!` 呼出側 (trainer / smoke / tests) が `use crate::*;` で解決する
// `#[kernel]` marker 型 (`__<name>_CudaKernel`) を crate root から見えるようにする。
pub(crate) use common::*;
pub(crate) use layerstack::*;
pub(crate) use simple::*;
