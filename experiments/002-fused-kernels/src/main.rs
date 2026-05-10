//! experiments/002-fused-kernels binary entry point。
//!
//! Stage 2 (EPIC #16) の hand-fused kernel suite を build-time PTX 化するための
//! 受け皿。`#[kernel]` 定義は本 file に inline 配置する (cuda-oxide rustc-codegen-cuda
//! backend の "bin entry から到達可能な `#[kernel]` のみ NVPTX IR 化する" 制約、
//! Stage 1-5 で確立、`ATTRIBUTION.md` 参照)。
//!
//! ## 配置
//!
//! - **kernels** (`screlu_grad`, `loss_wdl`, `adamw_step`, `radam_step`,
//!   `ranger_step`, `sparse_ft_forward`, `sparse_ft_backward`) は
//!   Stage 2-1〜2-7 で各 issue が本 file に inline で追加する。本 PR (Stage 2-0
//!   scaffold) では kernel は未追加で empty `main` のみ
//! - **reference CPU** は `gpu-kernels` crate の `pointwise/` / `sparse/`
//!   module に置く (Stage 1 の `progress/` と同列の慣行)
//! - **GPU↔CPU smoke test** は `tests/` に kernel ごとに 1 file
//!
//! ## 使い方 (Stage 2-1 以降)
//!
//! ```bash
//! cd experiments/002-fused-kernels && \
//! CUDA_OXIDE_TARGET=sm_75 \
//!     /mnt/e/cuda-oxide-target/release/cargo-oxide build
//! ```
//!
//! 出力 `.ll` は workspace root に `exp_002_fused_kernels.ll` として落ちる
//! (`bins/progress_kpabs_train` と同じ慣行、`KernelLoader` が両 path を probe)。
//!
//! ## CI
//!
//! 本 crate は `cuda-host` 経由で transitive に `cuda.h` を要求するため
//! GitHub-hosted runner では build できない。`.github/workflows/checks.yaml` の
//! `--exclude` リストに `exp-002-fused-kernels` を追加済 (Stage 1-9 で
//! `exp-001-cuda-oxide-kpabs` を exclude したのと同じ理由)。

fn main() {
    // Stage 2-0 scaffold は kernel 未配置。Stage 2-1 (#37) 以降で
    // `#[kernel]` を inline 追加し、host 側の launch driver も同梱する。
    println!("exp-002-fused-kernels: Stage 2-0 scaffold (no kernel yet)");
}
