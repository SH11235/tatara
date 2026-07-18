//! bin crate 内に置く `#[cfg(test)]` テスト module 群 (`src/tests/`)。GPU テストが
//! crate root の `#[kernel]` symbol へ path 解決する必要があるため、package root の
//! `tests/` integration test crate ではなく crate 内 `mod` として宣言する。各 module
//! の詳細は file 冒頭の doc を参照。

#[cfg(test)]
mod cli_tests;
#[cfg(all(test, feature = "cuda-oxide"))]
mod ft_factorize_tests;
#[cfg(all(test, feature = "cuda-oxide"))]
mod gpu_cpu_equivalence_tests;
#[cfg(test)]
mod native_bench_schema_tests;
#[cfg(all(test, any(feature = "native-cuda", feature = "native-cuda-host")))]
mod native_cuda_tests;
#[cfg(test)]
mod raw_ckpt_format_tests;
