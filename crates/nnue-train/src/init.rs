//! 重み初期化の汎用記述と決定論的サンプラ。
//!
//! NNUE の学習結果は初期重みのスケールに敏感で、feature transformer の初期 std が
//! 大きすぎると CReLU が飽和して学習初期の勾配が失われる。ここでは初期化方式を
//! 「分布 (`Dist`) × 広がり (`Scale`) × bucket 複製 (`per_bucket_repeat`) × seed」
//! の直交パラメータで表し、特定方式をハードコードせずに済むようにする。各アーキの
//! 既定値は [`LayerStackInit::default_uniform`] / [`SimpleInit::default_uniform`] が
//! 返し、`--init-{ft,l1,l1f,l2,l3}` SPEC override で層ごとに分布・広がりを差し替え
//! られる。
//!
//! 値生成は xorshift ベースの決定論的 RNG で、同一 seed なら常に同一列を返す
//! (smoke / 数値同等性テストの再現性を保つため)。`Dist::Uniform` + `Scale::Abs`
//! は `±0.01` 一様初期化と bit-identical な列を返すよう実装してあり、unit
//! test (`uniform_abs_is_bit_identical_to_reference_xorshift`) が保証する。

/// 重みをサンプリングする確率分布。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dist {
    /// 全要素 0。
    Zeroed,
    /// `[-h, +h]` の一様分布 (`h` は [`Scale`] が決める半値幅)。
    Uniform,
    /// 平均 0・標準偏差 `s` の正規分布 (`s` は [`Scale`] が決める)。
    Normal,
}

/// 分布の広がりの決め方。`Uniform` では半値幅、`Normal` では標準偏差に解決される。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    /// 絶対値で直接指定する。
    Abs(f32),
    /// `sqrt(gain / fan_in)`。`effective` を指定した場合は実 fan_in の代わりに
    /// その値を使う (入力次元が疎で実効 fan_in が小さい層を模す用途)。
    FanIn { gain: f32, effective: Option<usize> },
}

/// 1 つの weight group (weight か bias) の初期化指定。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerInit {
    pub dist: Dist,
    pub scale: Scale,
    /// `true` なら 1 bucket 分だけ生成して全 bucket に同値を複製する。bucket 付き
    /// Linear を全 bucket 同一初期値で始める方式を表す。
    pub per_bucket_repeat: bool,
    pub seed: u64,
}

impl LayerInit {
    pub const fn zeroed() -> Self {
        Self {
            dist: Dist::Zeroed,
            scale: Scale::Abs(0.0),
            per_bucket_repeat: false,
            seed: 0,
        }
    }

    pub const fn uniform_abs(half_width: f32, seed: u64) -> Self {
        Self {
            dist: Dist::Uniform,
            scale: Scale::Abs(half_width),
            per_bucket_repeat: false,
            seed,
        }
    }

    pub const fn uniform_fan_in(gain: f32, per_bucket_repeat: bool, seed: u64) -> Self {
        Self {
            dist: Dist::Uniform,
            scale: Scale::FanIn {
                gain,
                effective: None,
            },
            per_bucket_repeat,
            seed,
        }
    }
}

/// weight group の形状。`n` は総要素数、`num_buckets` は bucket 複製の分割数
/// (bucket 無し group は 1)、`fan_in` は [`Scale::FanIn`] が参照する入力次元。
#[derive(Debug, Clone, Copy)]
pub struct WeightShape {
    pub n: usize,
    pub num_buckets: usize,
    pub fan_in: usize,
}

impl WeightShape {
    /// bucket 無し group (FT / 共有因子層 / bias など) 用のショートカット。
    pub fn flat(n: usize, fan_in: usize) -> Self {
        Self {
            n,
            num_buckets: 1,
            fan_in,
        }
    }

    /// bucket-major layout (`[bucket][...]` 連続) の group 用。
    pub fn bucketed(n: usize, num_buckets: usize, fan_in: usize) -> Self {
        Self {
            n,
            num_buckets,
            fan_in,
        }
    }
}

/// xorshift64。`(s >> 11) / 2^53` で `[0, 1)` の一様乱数を作る。既定初期化が呼ぶ
/// `Uniform` + `Abs(0.01)` 経路は本実装で bit-identical な列を生成する。
struct XorShift {
    s: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }

    /// `[0, 1)` の一様乱数。
    fn next_unit(&mut self) -> f32 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        (self.s >> 11) as f32 / ((1u64 << 53) as f32)
    }

    /// `[-1, 1)` の一様乱数。
    fn next_signed_unit(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }

    /// Box-Muller で標準正規分布の独立な 2 値を返す。
    fn next_gaussian_pair(&mut self) -> (f32, f32) {
        // ln(0) = -inf を避けるため u1 を下限でクランプする。
        let u1 = self.next_unit().max(1.0e-12);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

fn resolve_scale(scale: Scale, fan_in: usize) -> f32 {
    match scale {
        Scale::Abs(v) => v,
        Scale::FanIn { gain, effective } => {
            let f = effective.unwrap_or(fan_in).max(1) as f32;
            (gain / f).sqrt()
        }
    }
}

/// `shape` の要素数ぶんの初期重みを host buffer として生成する。
///
/// `per_bucket_repeat` のときは 1 bucket 分 (`n / num_buckets`) を生成して
/// `num_buckets` 回複製する。`n` が `num_buckets` で割り切れることを呼び出し側が
/// 保証する (bucket-major layout の group はこれを満たす)。
pub fn sample(shape: WeightShape, init: &LayerInit) -> Vec<f32> {
    if matches!(init.dist, Dist::Zeroed) {
        return vec![0.0; shape.n];
    }

    let mag = resolve_scale(init.scale, shape.fan_in);
    let block_len = if init.per_bucket_repeat {
        assert!(
            shape.num_buckets >= 1 && shape.n.is_multiple_of(shape.num_buckets),
            "per_bucket_repeat requires n ({}) divisible by num_buckets ({})",
            shape.n,
            shape.num_buckets
        );
        shape.n / shape.num_buckets
    } else {
        shape.n
    };

    let mut rng = XorShift::new(init.seed);
    let mut block = Vec::with_capacity(block_len);
    match init.dist {
        Dist::Uniform => {
            for _ in 0..block_len {
                block.push(rng.next_signed_unit() * mag);
            }
        }
        Dist::Normal => {
            let mut i = 0;
            while i < block_len {
                let (z0, z1) = rng.next_gaussian_pair();
                block.push(z0 * mag);
                i += 1;
                if i < block_len {
                    block.push(z1 * mag);
                    i += 1;
                }
            }
        }
        Dist::Zeroed => unreachable!("handled above"),
    }

    if init.per_bucket_repeat && shape.num_buckets > 1 {
        let mut out = Vec::with_capacity(shape.n);
        for _ in 0..shape.num_buckets {
            out.extend_from_slice(&block);
        }
        out
    } else {
        block
    }
}

// 既定初期化の group 別固定 seed。`--init-{ft,l1,...}` override を併用しても seed は
// この固定値を保つので、同じ override 指定なら常に同じ初期重み列を再現する。
const DEFAULT_LS_SEEDS: [u64; 5] = [0x100, 0x101, 0x102, 0x103, 0x104];
const DEFAULT_SIMPLE_SEEDS: [u64; 8] = [
    0x5071_e001,
    0x5071_e002,
    0x5071_e003,
    0x5071_e004,
    0x5071_e005,
    0x5071_e006,
    0x5071_e007,
    0x5071_e008,
];

const DEFAULT_HALF_WIDTH: f32 = 0.01;

/// FT weight の既定 fan-in スケーリング gain。半値幅は `sqrt(gain / fan_in)`。
const DEFAULT_FAN_IN_GAIN: f32 = 1.0;

/// LayerStack (FT → L1(+L1f) → L2 → L3、bucket 付き) の全 weight group の初期化指定。
#[derive(Debug, Clone, Copy)]
pub struct LayerStackInit {
    pub ft_w: LayerInit,
    pub ft_b: LayerInit,
    pub l1_w: LayerInit,
    pub l1_b: LayerInit,
    pub l1f_w: LayerInit,
    pub l1f_b: LayerInit,
    pub l2_w: LayerInit,
    pub l2_b: LayerInit,
    pub l3_w: LayerInit,
    pub l3_b: LayerInit,
}

impl LayerStackInit {
    /// 既定初期化: FT weight は半値幅 `sqrt(1 / fan_in)` の一様分布、それ以外の weight は
    /// `[-0.01, 0.01]` 一様 (いずれも固定 seed)、bias は全て 0。
    ///
    /// FT だけ fan-in スケーリングなのは、FT の fan_in が他層と桁違いに大きく (halfka-hm-merged
    /// で 73,305)、固定半値幅では初期活性が CReLU の上限に張り付いて学習初期の勾配が失われる
    /// ため。L1/L2/L3 は fan_in が小さく、固定半値幅でも飽和しない。
    pub fn default_uniform() -> Self {
        Self {
            ft_w: LayerInit::uniform_fan_in(DEFAULT_FAN_IN_GAIN, false, DEFAULT_LS_SEEDS[0]),
            ft_b: LayerInit::zeroed(),
            l1_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_LS_SEEDS[1]),
            l1_b: LayerInit::zeroed(),
            l1f_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_LS_SEEDS[2]),
            l1f_b: LayerInit::zeroed(),
            l2_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_LS_SEEDS[3]),
            l2_b: LayerInit::zeroed(),
            l3_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_LS_SEEDS[4]),
            l3_b: LayerInit::zeroed(),
        }
    }

    /// CLI override (重み側のみ) を該当 group に適用する。`per_bucket_repeat` と
    /// `seed` は既定値を保ち、`dist` / `scale` だけ差し替える。
    pub fn apply_weight_override(&mut self, layer: WeightLayer, ov: LayerInitOverride) {
        let target = match layer {
            WeightLayer::Ft => &mut self.ft_w,
            WeightLayer::L1 => &mut self.l1_w,
            WeightLayer::L1f => &mut self.l1f_w,
            WeightLayer::L2 => &mut self.l2_w,
            WeightLayer::L3 => &mut self.l3_w,
        };
        ov.apply(target);
    }
}

/// Simple (FT → L1 → L2 → L3、bucket / 共有因子層なし) の全 weight group の初期化指定。
#[derive(Debug, Clone, Copy)]
pub struct SimpleInit {
    pub ft_w: LayerInit,
    pub ft_b: LayerInit,
    pub l1_w: LayerInit,
    pub l1_b: LayerInit,
    pub l2_w: LayerInit,
    pub l2_b: LayerInit,
    pub l3_w: LayerInit,
    pub l3_b: LayerInit,
}

impl SimpleInit {
    /// 既定初期化: weight・bias とも `[-0.01, 0.01]` 一様 (固定 seed)。
    pub fn default_uniform() -> Self {
        Self {
            ft_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[0]),
            ft_b: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[1]),
            l1_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[2]),
            l1_b: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[3]),
            l2_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[4]),
            l2_b: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[5]),
            l3_w: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[6]),
            l3_b: LayerInit::uniform_abs(DEFAULT_HALF_WIDTH, DEFAULT_SIMPLE_SEEDS[7]),
        }
    }

    /// CLI override (重み側のみ) を該当 group に適用する。L1f は Simple に存在しない。
    pub fn apply_weight_override(
        &mut self,
        layer: WeightLayer,
        ov: LayerInitOverride,
    ) -> Result<(), String> {
        let target = match layer {
            WeightLayer::Ft => &mut self.ft_w,
            WeightLayer::L1 => &mut self.l1_w,
            WeightLayer::L2 => &mut self.l2_w,
            WeightLayer::L3 => &mut self.l3_w,
            WeightLayer::L1f => {
                return Err("--init-l1f applies only to the layerstack architecture (Simple has no L1f layer)".to_string());
            }
        };
        ov.apply(target);
        Ok(())
    }
}

/// `--init-<layer>` が指す weight group。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayer {
    Ft,
    L1,
    L1f,
    L2,
    L3,
}

/// CLI `--init-<layer>` で重み側の分布・広がりを差し替える指定。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerInitOverride {
    pub dist: Dist,
    pub scale: Scale,
}

impl LayerInitOverride {
    fn apply(self, target: &mut LayerInit) {
        target.dist = self.dist;
        target.scale = self.scale;
    }
}

/// `--init-<layer>` の SPEC 文字列を解析する。
///
/// 文法:
/// - `zero`
/// - `uniform:abs:<value>` / `normal:abs:<value>`
/// - `uniform:fanin` / `uniform:fanin:<gain>` / `uniform:fanin:<gain>:<effective>`
/// - `normal:` も同じ scale 部を取る
///
/// 例: `uniform:fanin` (半値幅 `sqrt(1/fan_in)`)、`normal:fanin:2:32`
/// (He-normal 風、`std=sqrt(2/32)=0.25`)、`uniform:abs:0.01`。
pub fn parse_layer_init_spec(spec: &str) -> Result<LayerInitOverride, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    let bad = |msg: &str| {
        format!(
            "invalid init spec '{spec}': {msg} (expected 'zero', '<uniform|normal>:abs:<value>', \
             or '<uniform|normal>:fanin[:<gain>[:<effective>]]')"
        )
    };
    let dist = match parts[0] {
        "zero" => {
            if parts.len() != 1 {
                return Err(bad("'zero' takes no parameters"));
            }
            return Ok(LayerInitOverride {
                dist: Dist::Zeroed,
                scale: Scale::Abs(0.0),
            });
        }
        "uniform" => Dist::Uniform,
        "normal" => Dist::Normal,
        other => return Err(bad(&format!("unknown distribution '{other}'"))),
    };
    let scale = match parts.get(1).copied() {
        Some("abs") => {
            let raw = parts.get(2).ok_or_else(|| bad("'abs' requires a value"))?;
            if parts.len() != 3 {
                return Err(bad("'abs' takes exactly one value"));
            }
            let v: f32 = raw
                .parse()
                .map_err(|_| bad(&format!("'{raw}' is not a number")))?;
            if !v.is_finite() || v < 0.0 {
                return Err(bad("abs value must be finite and >= 0"));
            }
            Scale::Abs(v)
        }
        Some("fanin") => {
            let gain = match parts.get(2).copied() {
                None => 1.0,
                Some(g) => {
                    let g: f32 = g
                        .parse()
                        .map_err(|_| bad(&format!("'{g}' is not a number")))?;
                    if !g.is_finite() || g <= 0.0 {
                        return Err(bad("fanin gain must be finite and > 0"));
                    }
                    g
                }
            };
            let effective = match parts.get(3).copied() {
                None => None,
                Some(e) => {
                    let e: usize = e
                        .parse()
                        .map_err(|_| bad(&format!("'{e}' is not a positive integer")))?;
                    if e == 0 {
                        return Err(bad("fanin effective input size must be >= 1"));
                    }
                    Some(e)
                }
            };
            if parts.len() > 4 {
                return Err(bad("too many ':' separated fields"));
            }
            Scale::FanIn { gain, effective }
        }
        Some(other) => return Err(bad(&format!("unknown scale kind '{other}'"))),
        None => return Err(bad("missing scale (e.g. ':fanin' or ':abs:0.01')")),
    };
    Ok(LayerInitOverride { dist, scale })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 既定の `±0.01` 一様初期化と bit-identity 比較するための xorshift 参照実装
    /// (`Dist::Uniform` + `Scale::Abs` と同じ算式を直接書き下す)。
    fn reference_xorshift_init(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed.max(1);
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f32 / ((1u64 << 53) as f32);
            v.push((u * 2.0 - 1.0) * scale);
        }
        v
    }

    #[test]
    fn uniform_abs_is_bit_identical_to_reference_xorshift() {
        for &(seed, n) in &[(0x100_u64, 37), (0x5071_e003, 1024), (1, 9)] {
            let got = sample(
                WeightShape::flat(n, 99),
                &LayerInit::uniform_abs(0.01, seed),
            );
            let want = reference_xorshift_init(seed, n, 0.01);
            assert_eq!(got, want, "seed={seed:#x} n={n}");
        }
    }

    #[test]
    fn fanin_uniform_half_width_matches_sqrt_inv_fan_in() {
        let fan_in = 73_305usize;
        let init = LayerInit::uniform_fan_in(1.0, false, 0xABCD);
        let v = sample(WeightShape::flat(200_000, fan_in), &init);
        let bound = (1.0 / fan_in as f32).sqrt();
        let max = v.iter().cloned().fold(f32::MIN, f32::max);
        let min = v.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            max <= bound && max > bound * 0.98,
            "max={max} bound={bound}"
        );
        assert!(
            min >= -bound && min < -bound * 0.98,
            "min={min} bound={bound}"
        );
        // 一様分布の std は半値幅 / sqrt(3) (fan_in=73305 で 0.00213 付近) を許容誤差で確認。
        let var = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len() as f64;
        let std = var.sqrt();
        let expected_std = (bound as f64) / 3.0_f64.sqrt();
        assert!(
            (std - expected_std).abs() / expected_std < 0.05,
            "std={std} expected={expected_std}"
        );
    }

    #[test]
    fn per_bucket_repeat_copies_bucket_zero_to_all_buckets() {
        let num_buckets = 9;
        let per_bucket = 16 * 32; // l1_out * ft_out 相当
        let n = num_buckets * per_bucket;
        let init = LayerInit::uniform_fan_in(1.0, true, 0x1234);
        let v = sample(WeightShape::bucketed(n, num_buckets, 32), &init);
        assert_eq!(v.len(), n);
        let bucket0 = &v[..per_bucket];
        for b in 1..num_buckets {
            assert_eq!(
                &v[b * per_bucket..(b + 1) * per_bucket],
                bucket0,
                "bucket {b} differs"
            );
        }
    }

    #[test]
    fn same_seed_is_deterministic() {
        let init = LayerInit::uniform_fan_in(2.0, false, 0x9999);
        let a = sample(WeightShape::flat(500, 128), &init);
        let b = sample(WeightShape::flat(500, 128), &init);
        assert_eq!(a, b);
    }

    #[test]
    fn zeroed_is_all_zero_regardless_of_scale() {
        let v = sample(WeightShape::flat(64, 10), &LayerInit::zeroed());
        assert!(v.iter().all(|&x| x == 0.0));
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn normal_fan_in_has_expected_std() {
        let fan_in = 32usize;
        let init = LayerInit {
            dist: Dist::Normal,
            scale: Scale::FanIn {
                gain: 2.0,
                effective: None,
            },
            per_bucket_repeat: false,
            seed: 0x5555,
        };
        let v = sample(WeightShape::flat(400_000, fan_in), &init);
        let var = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len() as f64;
        let std = var.sqrt();
        let expected = (2.0_f64 / fan_in as f64).sqrt(); // sqrt(2/32) = 0.25
        assert!(
            (std - expected).abs() / expected < 0.03,
            "std={std} expected={expected}"
        );
    }

    #[test]
    fn default_uniform_layerstack_biases_are_zeroed() {
        let p = LayerStackInit::default_uniform();
        assert_eq!(p.ft_b.dist, Dist::Zeroed);
        assert_eq!(p.l1_b.dist, Dist::Zeroed);
        assert_eq!(p.l3_b.dist, Dist::Zeroed);
    }

    #[test]
    fn default_layerstack_ft_is_fan_in_and_others_are_abs() {
        let p = LayerStackInit::default_uniform();
        assert_eq!(p.ft_w.dist, Dist::Uniform);
        assert_eq!(
            p.ft_w.scale,
            Scale::FanIn {
                gain: 1.0,
                effective: None
            }
        );
        assert_eq!(p.l1_w.scale, Scale::Abs(0.01));
        assert_eq!(p.l2_w.scale, Scale::Abs(0.01));
        assert_eq!(p.l3_w.scale, Scale::Abs(0.01));
        assert_eq!(p.l1f_w.scale, Scale::Abs(0.01));
    }

    #[test]
    fn default_simple_ft_stays_abs() {
        let p = SimpleInit::default_uniform();
        assert_eq!(p.ft_w.scale, Scale::Abs(0.01));
    }

    #[test]
    fn parse_spec_variants() {
        assert_eq!(
            parse_layer_init_spec("zero").unwrap(),
            LayerInitOverride {
                dist: Dist::Zeroed,
                scale: Scale::Abs(0.0)
            }
        );
        assert_eq!(
            parse_layer_init_spec("uniform:fanin").unwrap(),
            LayerInitOverride {
                dist: Dist::Uniform,
                scale: Scale::FanIn {
                    gain: 1.0,
                    effective: None
                }
            }
        );
        assert_eq!(
            parse_layer_init_spec("normal:fanin:2:32").unwrap(),
            LayerInitOverride {
                dist: Dist::Normal,
                scale: Scale::FanIn {
                    gain: 2.0,
                    effective: Some(32)
                }
            }
        );
        assert_eq!(
            parse_layer_init_spec("uniform:abs:0.01").unwrap(),
            LayerInitOverride {
                dist: Dist::Uniform,
                scale: Scale::Abs(0.01)
            }
        );
    }

    #[test]
    fn parse_spec_rejects_garbage() {
        assert!(parse_layer_init_spec("bogus").is_err());
        assert!(parse_layer_init_spec("uniform").is_err());
        assert!(parse_layer_init_spec("uniform:abs").is_err());
        assert!(parse_layer_init_spec("uniform:abs:notanum").is_err());
        assert!(parse_layer_init_spec("normal:fanin:0").is_err());
        assert!(parse_layer_init_spec("zero:1").is_err());
        assert!(parse_layer_init_spec("uniform:fanin:1:2:3").is_err());
    }
}
