// `nvidia-smi` / `nvcc` の出力から compute capability を読み取る純粋関数。
//
// build script が `include!` するため inner doc comment は使えない。cargo は build
// script の test を実行しないので、test が実際に走る lib 側の module に実体を置く。

/// `nvidia-smi --query-gpu=compute_cap` の出力から最小の compute capability を返す。
///
/// PTX は前方互換なので、混載環境では最小値を選べば全 device で JIT できる。
pub(crate) fn minimum_compute_capability(stdout: &str) -> Option<u32> {
    stdout.lines().filter_map(parse_compute_capability).min()
}

/// `12.0` 形式を `120` へ正規化する。
pub(crate) fn parse_compute_capability(value: &str) -> Option<u32> {
    let (major, minor) = value.trim().split_once('.')?;
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    format!("{major}{minor}").parse().ok()
}

/// `nvcc --list-gpu-arch` の出力から最大の対応 compute capability を返す。
///
/// `compute_90a` のような suffix 付き arch は数値 parse に失敗するため自然に除外される。
pub(crate) fn max_listed_compute_capability(stdout: &str) -> Option<u32> {
    stdout
        .lines()
        .filter_map(|line| line.trim().strip_prefix("compute_")?.parse().ok())
        .max()
}

#[cfg(test)]
mod tests {
    use super::{
        max_listed_compute_capability, minimum_compute_capability, parse_compute_capability,
    };

    #[test]
    fn selects_minimum_capability_and_ignores_invalid_lines() {
        assert_eq!(
            minimum_compute_capability("12.0\nN/A\ncompute_cap\n8.6\n"),
            Some(86)
        );
    }

    #[test]
    fn rejects_malformed_capability_values() {
        for value in ["", "75", ".5", "8.", "8.x", "8.6.1"] {
            assert_eq!(parse_compute_capability(value), None);
        }
    }

    #[test]
    fn takes_highest_numeric_arch_and_skips_suffixed_ones() {
        assert_eq!(
            max_listed_compute_capability("compute_50\ncompute_90a\ncompute_121\n"),
            Some(121)
        );
    }
}
