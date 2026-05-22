use gpu_runtime::{CudaContext, CudaModule};

// ===========================================================================
// Host driver helpers (kernel module loader / launch utilities)
// ===========================================================================

#[allow(dead_code)]
pub(crate) const BLOCK_DIM: u32 = 256;

/// 1 D launch の grid 数を計算 (= ceil(n / block)、n=0 は block=1 個 launch)。
#[allow(dead_code)]
pub(crate) fn grid_dim_1d(n: usize, block: u32) -> (u32, u32, u32) {
    let blocks = ((n as u32).max(1)).div_ceil(block);
    (blocks, 1, 1)
}

/// `cargo-oxide build` が出力した kernel `.ll` を見つけ、`.ptx` に変換した上で
/// CudaModule を load。fallback 順は `.ll` → `.cubin` → `.ptx`。
#[allow(dead_code)]
pub(crate) fn load_kernel_module_with_fallback(
    ctx: &std::sync::Arc<CudaContext>,
    name: &str,
) -> Result<std::sync::Arc<CudaModule>, Box<dyn std::error::Error>> {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| manifest_dir.clone());

    let probe = |dir: &std::path::PathBuf| {
        for ext in ["ll", "cubin", "ptx"] {
            let p = dir.join(format!("{name}.{ext}"));
            if p.exists() {
                return Some(p);
            }
        }
        None
    };

    let path = probe(&manifest_dir)
        .or_else(|| probe(&workspace_root))
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!(
                "kernel artifact `{name}.{{cubin,ptx,ll}}` not found in {} or {}.\n\
                 先に cargo-oxide build を実行してください:\n  \
                 cd {} && CUDA_OXIDE_TARGET=sm_75 cargo-oxide build",
                manifest_dir.display(),
                workspace_root.display(),
                manifest_dir.display(),
            )
            .into()
        })?;

    let to_load = if path.extension().and_then(|s| s.to_str()) == Some("ll") {
        compile_ll_to_ptx_via_llc(&path)?
    } else {
        path
    };

    let module = ctx.load_module_from_file(
        to_load
            .to_str()
            .ok_or("kernel artifact path not valid UTF-8")?,
    )?;
    Ok(module)
}

/// `.ll` を libdevice と link → opt → llc で `.ptx` 生成。`kernel_names` で全 27
/// kernel を internalize する。
#[allow(dead_code)]
pub(crate) fn compile_ll_to_ptx_via_llc(
    ll_path: &std::path::PathBuf,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let stem = ll_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("ll path has no stem")?;
    let dir = ll_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let linked_bc = dir.join(format!("{stem}.linked.bc"));
    let opt_bc = dir.join(format!("{stem}.opt.bc"));
    let ptx_path = dir.join(format!("{stem}.ptx"));

    // `.ll`→`.ptx` の中間/出力ファイル (linked.bc / opt.bc / .ptx) は stem 固定
    // パスのため、複数スレッドが同時に compile すると `llc` が書き込み途中の
    // `.bc` を読んで crash する。`cargo test` は 1 binary のテストを同一プロセスの
    // 複数スレッドで走らせるので、プロセス内 Mutex で直列化すれば足りる。最初の
    // スレッドが compile し、後続は lock 取得後に下の mtime cache で skip する。
    static COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _compile_guard = COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // cache: skip rebuild if .ptx is newer than .ll
    if let (Ok(ll_meta), Ok(ptx_meta)) = (std::fs::metadata(ll_path), std::fs::metadata(&ptx_path))
        && let (Ok(ll_mtime), Ok(ptx_mtime)) = (ll_meta.modified(), ptx_meta.modified())
        && ptx_mtime > ll_mtime
    {
        return Ok(ptx_path);
    }

    let arch = std::env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_75".to_string());
    let llvm_link =
        std::env::var("LLVM_LINK_BIN").unwrap_or_else(|_| discover_llvm_tool("llvm-link"));
    let opt_bin = std::env::var("OPT_BIN").unwrap_or_else(|_| discover_llvm_tool("opt"));
    let llc_bin = std::env::var("LLC_BIN").unwrap_or_else(|_| discover_llvm_tool("llc"));
    let libdevice = find_libdevice_bc()?;

    // module が launch する全 kernel 名。`@<name>` として `.ll` に出ているものを
    // 漏れなく internalize-public-api-list に残す (1 個でも漏れると opt の globaldce
    // で消える)。
    let kernel_names = "sparse_ft_forward,sparse_ft_backward,loss_wdl,loss_wrm,screlu_grad,\
                       adamw_step,radam_step,radam_step_fp16_mirror,\
                       radam_step_f16state,radam_step_f16state_mirror,\
                       ranger_lookahead_lerp,ranger_lookahead_lerp_fp16_mirror,\
                       ft_post_perspective_fwd,ft_post_perspective_grad,\
                       dense_mm_fwd,dense_mm_bwd_input,dense_mm_bwd_weight,bias_grad,\
                       dense_mm_fwd_bucket,dense_mm_bwd_input_bucket,\
                       dense_mm_bwd_weight_bucket,bias_grad_bucket,\
                       crelu_fwd,crelu_grad,screlu_fwd,abs_pow2_scale_fwd,abs_pow2_scale_grad,\
                       concat_l1sqr_main_fwd,concat_l1sqr_main_grad,elementwise_add,\
                       bias_add_per_row,\
                       slice_extract_2d,slice_scatter_2d,\
                       count_buckets,exclusive_scan_aligned,scatter_bucket_perm,\
                       permute_rows_f32,inverse_permute_rows_f32,\
                       dense_mm_fwd_bucket_tiled_l1_sorted,\
                       dense_mm_bwd_weight_bucket_tiled_l1_sorted,\
                       bias_grad_bucket_shared_sorted,\
                       ft_post_perspective_grad_fused,\
                       sparse_ft_forward_fp16,sparse_ft_forward_fp16_out,cast_f32_to_f16,\
                       ft_post_perspective_fwd_fp16,ft_post_perspective_grad_fused_fp16,\
                       ft_post_perspective_grad_fp16,\
                       build_feature_counts,exclusive_prefix_sum_small,scatter_positions,\
                       gather_and_sum_per_feature_overwrite,gather_and_sum_per_feature_add,\
                       gather_and_sum_per_feature_overwrite_fp16,\
                       gather_and_sum_per_feature_add_fp16,\
                       simple_bias_act_fwd_fp16_in_crelu,\
                       simple_act_grad_to_fp16_crelu_with_scale,\
                       simple_bias_act_fwd_fp16_in_screlu,\
                       simple_act_grad_to_fp16_screlu_with_scale,\
                       simple_bias_grad_fp16,simple_sparse_ft_backward_fp16,\
                       simple_bias_grad_dual,simple_bias_grad_dual_fp16,\
                       simple_bwd_ft_act_crelu_fused,simple_bwd_ft_act_screlu_fused,\
                       simple_ft_post_fused_crelu,simple_ft_post_fused_screlu";

    // Step 1: llvm-link <ll> libdevice → linked.bc
    run_or_err(
        &llvm_link,
        &[
            ll_path.as_os_str(),
            libdevice.as_os_str(),
            "-o".as_ref(),
            linked_bc.as_os_str(),
        ],
    )?;

    // Step 2: opt --passes='nvvm-reflect,internalize,globaldce'
    let api = format!("--internalize-public-api-list={kernel_names}");
    run_or_err(
        &opt_bin,
        &[
            "--passes=nvvm-reflect,internalize,globaldce".as_ref(),
            api.as_ref(),
            linked_bc.as_os_str(),
            "-o".as_ref(),
            opt_bc.as_os_str(),
        ],
    )?;

    // Step 3: llc -mcpu=<arch> -O2 opt.bc → .ptx
    let mcpu = format!("--mcpu={arch}");
    run_or_err(
        &llc_bin,
        &[
            "--mtriple=nvptx64-nvidia-cuda".as_ref(),
            mcpu.as_ref(),
            "-O2".as_ref(),
            "-o".as_ref(),
            ptx_path.as_os_str(),
            opt_bc.as_os_str(),
        ],
    )?;

    Ok(ptx_path)
}

/// `.ll`→`.ptx` 変換に使う LLVM tool 名を解決する。`<tool>-22` → `<tool>-21`
/// → 無印 `<tool>` の順で `--version` が通る最初の名前を返す (cuda-oxide 本体の
/// `llc-22 → llc-21` 探索順に揃える)。どれも無ければ `<tool>-21` を返し、spawn
/// 失敗時に `run_or_err` が導入方法を案内する。`LLVM_LINK_BIN` / `OPT_BIN` /
/// `LLC_BIN` env が設定されていればそちらが優先される。
pub(crate) fn discover_llvm_tool(tool: &str) -> String {
    for suffix in ["-22", "-21", ""] {
        let name = format!("{tool}{suffix}");
        let ok = std::process::Command::new(&name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            return name;
        }
    }
    format!("{tool}-21")
}

/// `Command::new` + `args` + `status` を 1 行にまとめる helper。
pub(crate) fn run_or_err(
    bin: &str,
    args: &[&std::ffi::OsStr],
) -> Result<(), Box<dyn std::error::Error>> {
    let status = std::process::Command::new(bin)
        .args(args)
        .status()
        .map_err(|e| {
            format!(
                "failed to spawn {bin}: {e}. \
                 .ll→.ptx 変換は LLVM 21+ の llvm-link / opt / llc を使う \
                 (libNVVM が opaque pointer IR を parse できないため llc 経路)。\
                 -22 / -21 を自動探索するが、見つからなければ \
                 LLVM_LINK_BIN / OPT_BIN / LLC_BIN env で明示指定する。"
            )
        })?;
    if !status.success() {
        return Err(format!("{bin} failed with status {status}").into());
    }
    Ok(())
}

/// `libdevice.10.bc` を CUDA Toolkit から探す。
pub(crate) fn find_libdevice_bc() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    if let Ok(p) = std::env::var("CUDA_OXIDE_LIBDEVICE") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let mut tried = Vec::new();
    let roots: Vec<std::path::PathBuf> = std::env::var("CUDA_HOME")
        .ok()
        .into_iter()
        .chain(std::env::var("CUDA_PATH").ok())
        .map(std::path::PathBuf::from)
        .chain([
            std::path::PathBuf::from("/usr/local/cuda"),
            std::path::PathBuf::from("/usr/local/cuda-13.2"),
            std::path::PathBuf::from("/usr/local/cuda-12.9"),
            std::path::PathBuf::from("/opt/cuda"),
        ])
        .collect();
    for root in roots {
        let candidate = root.join("nvvm/libdevice/libdevice.10.bc");
        tried.push(candidate.display().to_string());
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "libdevice.10.bc not found. CUDA_OXIDE_LIBDEVICE か CUDA_HOME を設定するか、\
         CUDA Toolkit を入れてください。Tried:\n  {}",
        tried.join("\n  ")
    )
    .into())
}
