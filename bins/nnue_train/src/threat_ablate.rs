//! threat FT 重みの部分集合 0 化 (ablation 寄与診断) と pair-class 別ノルム分解。
//!
//! `--init-from` で量子化 threat net を load した直後の [`LayerStackWeights::ft_w`]
//! (layout `[base real | threat real | (virtual piece-input rows)]`) に対し、threat real row
//! `[base_ft_in, base_ft_in + threat_dims)` のうち述語一致 pair の row を 0 にする。
//! `--eval-only` と組み合わせ「どの pair-class 部分集合が held-out test_loss に
//! どれだけ寄与するか」を、再学習なしで測る診断経路。

use nnue_format::LayerStackWeights;
use shogi_features::ThreatClass;
use shogi_features::threat::for_each_threat_pair_range;

/// 占有依存 (slider) の attacker class = 香・角・飛・馬・竜。歩・桂・銀・GoldLike は
/// 単発利き (step)。threat の利き列挙コストと index 空間幅はこの slider 群に集中する。
fn is_slider(c: ThreatClass) -> bool {
    matches!(
        c,
        ThreatClass::Lance
            | ThreatClass::Bishop
            | ThreatClass::Rook
            | ThreatClass::Horse
            | ThreatClass::Dragon
    )
}

/// 0 化した範囲の集計。
pub struct AblationStats {
    pub zeroed_dims: usize,
    pub zeroed_pairs: usize,
}

/// `weights.ft_w` の threat row を `spec` 一致分だけ 0 化する。`spec`:
/// `all` | `slider-attacker` | `step-attacker` | `bigslider-attacker` |
/// `defense` | `attack` | `same-class` | `random:<seed>:<dims>`。
pub fn apply(
    weights: &mut LayerStackWeights,
    ft_out: usize,
    spec: &str,
) -> Result<AblationStats, String> {
    let profile = weights.feature_set.threat_profile().ok_or(
        "threat ablation needs a threat-enabled net (the loaded .bin has --threat-profile off)",
    )?;
    let base_ft_in = weights.feature_set.base_ft_in();
    let threat_dims = weights.feature_set.threat_dims();

    // random:<seed>:<dims> は pair 構造に依らず threat feature を一様無作為に 0 化し、
    // 「threat 列を N 本消すこと自体の損」の null baseline を与える (slider/step 等の
    // 構造的 ablation の過大評価分を校正する対照)。
    if let Some(rest) = spec.strip_prefix("random:") {
        let mut it = rest.split(':');
        let seed: u64 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or("random spec must be random:<seed>:<dims>")?;
        let want: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or("random spec must be random:<seed>:<dims>")?;
        let n = want.min(threat_dims);

        // splitmix64 で部分 Fisher-Yates (再現可能・外部 RNG 非依存)。
        let mut state = seed;
        let mut next_u64 = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut idx: Vec<usize> = (0..threat_dims).collect();
        for i in 0..n {
            let j = i + (next_u64() as usize) % (threat_dims - i);
            idx.swap(i, j);
        }
        for &fi in &idx[..n] {
            let s = (base_ft_in + fi) * ft_out;
            weights.ft_w[s..s + ft_out].fill(0.0);
        }
        return Ok(AblationStats {
            zeroed_dims: n,
            zeroed_pairs: 0,
        });
    }

    let pred: fn(usize, ThreatClass, usize, ThreatClass) -> bool = match spec {
        "all" => |_as, _ac, _ds, _dc| true,
        "slider-attacker" => |_as, ac, _ds, _dc| is_slider(ac),
        "step-attacker" => |_as, ac, _ds, _dc| !is_slider(ac),
        "bigslider-attacker" => |_as, ac, _ds, _dc| {
            matches!(
                ac,
                ThreatClass::Bishop | ThreatClass::Rook | ThreatClass::Horse | ThreatClass::Dragon
            )
        },
        "defense" => |a_s, _ac, ds, _dc| a_s == ds,
        "attack" => |a_s, _ac, ds, _dc| a_s != ds,
        "same-class" => |_as, ac, _ds, dc| ac == dc,
        other => {
            return Err(format!(
                "unknown threat-ablate spec '{other}' (expected: all | slider-attacker | \
                 step-attacker | bigslider-attacker | defense | attack | same-class | \
                 random:<seed>:<dims>)"
            ));
        }
    };

    let mut zeroed_dims = 0usize;
    let mut zeroed_pairs = 0usize;
    for_each_threat_pair_range(profile, |a_s, ac, ds, dc, fbase, width| {
        if pred(a_s, ac, ds, dc) {
            let s = (base_ft_in + fbase) * ft_out;
            let e = (base_ft_in + fbase + width) * ft_out;
            weights.ft_w[s..e].fill(0.0);
            zeroed_dims += width;
            zeroed_pairs += 1;
        }
    });
    Ok(AblationStats {
        zeroed_dims,
        zeroed_pairs,
    })
}

/// threat FT 重みを pair-class 軸で L2^2 分解して stdout に出す (eval 不要・即時)。
/// 「モデルが threat 容量をどの class / 攻守 / slider にどれだけ張ったか」の
/// 静的指標 (活性化頻度は加味しないので [`apply`] + `--eval-only` の Δloss と併読する)。
pub fn norm_dump(weights: &LayerStackWeights, ft_out: usize) {
    let Some(profile) = weights.feature_set.threat_profile() else {
        println!("[norm-dump] loaded net has no threat features (--threat-profile off)");
        return;
    };
    let base_ft_in = weights.feature_set.base_ft_in();
    let threat_dims = weights.feature_set.threat_dims();

    let row_sumsq = |fi: usize| -> f64 {
        let s = (base_ft_in + fi) * ft_out;
        weights.ft_w[s..s + ft_out]
            .iter()
            .map(|&w| f64::from(w) * f64::from(w))
            .sum()
    };

    let mut total = 0.0f64;
    let mut by_class = [0.0f64; 9];
    let mut by_class_dims = [0usize; 9];
    let (mut attack, mut defense) = (0.0f64, 0.0f64);
    let (mut slider, mut step) = (0.0f64, 0.0f64);
    let (mut same_class, mut cross_class) = (0.0f64, 0.0f64);

    for_each_threat_pair_range(profile, |a_s, ac, ds, dc, fbase, width| {
        let mut ss = 0.0f64;
        for i in 0..width {
            ss += row_sumsq(fbase + i);
        }
        total += ss;
        by_class[ac as usize] += ss;
        by_class_dims[ac as usize] += width;
        if a_s == ds {
            defense += ss;
        } else {
            attack += ss;
        }
        if is_slider(ac) {
            slider += ss;
        } else {
            step += ss;
        }
        if ac == dc {
            same_class += ss;
        } else {
            cross_class += ss;
        }
    });

    const NAMES: [&str; 9] = [
        "Pawn", "Lance", "Knight", "Silver", "GoldLike", "Bishop", "Rook", "Horse", "Dragon",
    ];
    println!("[norm-dump] profile={profile} threat_dims={threat_dims} ft_out={ft_out}");
    println!("[norm-dump] total L2^2={total:.3} (L2={:.4})", total.sqrt());
    println!("[norm-dump] by attacker_class  (class: L2^2 / dims / per-dim):");
    for (c, name) in NAMES.iter().enumerate() {
        let d = by_class_dims[c].max(1);
        println!(
            "  {:<9} {:>14.3} / {:>6} / {:.6}",
            name,
            by_class[c],
            by_class_dims[c],
            by_class[c] / d as f64
        );
    }
    println!("[norm-dump] attack(as!=ds)={attack:.3}  defense(as==ds)={defense:.3}");
    println!("[norm-dump] slider-attacker={slider:.3}  step-attacker={step:.3}");
    println!("[norm-dump] same-class(ac==dc)={same_class:.3}  cross-class={cross_class:.3}");
}
