//! `bins/nnue_train` binary entry point — HalfKA_hm 1536-16-32 NNUE trainer。
//!
//! Stage 3 (EPIC #17) の production target。bullet-shogi 相当の HalfKA_hm
//! 1536-16-32 architecture を Rust + cuda-oxide で再現する。設計方針は
//! Stage 1 の `bins/progress_kpabs_train` と同型:
//!
//! - **kernels** (Stage 2 で landed 済の screlu_grad / loss_wdl / adamw_step /
//!   radam_step / ranger_lookahead_lerp / sparse_ft_forward / sparse_ft_backward)
//!   は Stage 3-7 (#63) で本 file に `#[kernel]` inline 配置する
//!   (cuda-oxide bin-entry 制約、Stage 1-5 で確立、Stage 2 EPIC で踏襲)
//! - **GpuTrainer** (device buffer 所有 + `step` / `eval` 駆動) も Stage 3-7
//!   で本 file に置く
//! - **host helper** (schedule / dataloader / optimizer / trainer loop) は
//!   GPU 非依存なので `crates/nnue-train` 側で実装する (CI でも test 通る)
//!
//! scaffold (Stage 3-0, #56) 段階では `fn main()` は placeholder 出力のみ。
//! Stage 3-8 (#65) で CLI + trainer integrate が完了するまで実装は加えない。

fn main() {
    println!(
        "nnue-train: Stage 3-0 scaffold placeholder (kernels / trainer to be added in Stage 3-7 / 3-8)"
    );
}
