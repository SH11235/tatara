//! KP-absolute progress trainer 用 4 reference kernel (CPU)。
//!
//! 4 kernel (forward / grad / adam_step / eval) の host 側 reference 実装を
//! 集める。bullet-shogi `KERNELS_SRC` 上流との対応は各 module の docstring と
//! `ATTRIBUTION.md` を参照。
//!
//! 利用例 (numerical equivalence test):
//!
//! ```rust
//! use gpu_kernels::progress::forward::forward_cpu;
//!
//! let indices: Vec<i32> = vec![];
//! let weights: Vec<f32> = vec![];
//! let n_pos = 0_usize;
//! let max_inds = 0_usize;
//! let preds = forward_cpu(&indices, &weights, n_pos, max_inds);
//! assert!(preds.is_empty());
//! ```

pub mod adam_step;
pub mod eval;
pub mod forward;
pub mod grad;
