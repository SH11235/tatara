//! Threat pair の除外 profile (runtime 選択)。
//!
//! Threat 特徴量は 2 駒の利き関係を `(attacker_side, attacker_class,
//! attacked_side, attacked_class)` の 4 軸 pair で表す。profile はこの pair の
//! 部分集合を選んで次元を間引く。`is_excluded` が true の pair は index 空間から
//! 詰めて除外される (該当 edge は active feature として出力されない)。
//!
//! id と除外規則は bullet-shogi 正準ベクタに揃える:
//!
//! | id | CLI 値 | 除外規則 |
//! |----|--------|---------|
//! | 0  | `full`                  | なし |
//! | 1  | `same-class`            | `ac == dc` |
//! | 2  | `same-class-major-pawn` | `ac == dc \|\| (ac >= 5 && dc == 0)` |
//! | 10 | `cross-side`            | `as == ds \|\| ac == dc` |

/// Threat pair 除外 profile。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreatProfile {
    /// 全 pair (除外なし)。
    Full,
    /// 同種 class pair を全除外。
    SameClass,
    /// 同種 + 大駒 (class >= Bishop) → 歩 を除外。
    SameClassMajorPawn,
    /// 同 side (味方→味方 / 敵→敵) と同種 class を除外し、cross-side 異種のみ残す。
    CrossSide,
}

impl ThreatProfile {
    /// CLI 文字列から変換。未知の文字列は None。
    pub fn from_cli(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Self::Full),
            "same-class" => Some(Self::SameClass),
            "same-class-major-pawn" => Some(Self::SameClassMajorPawn),
            "cross-side" => Some(Self::CrossSide),
            _ => None,
        }
    }

    /// 直列化契約で使う profile ID。
    pub fn profile_id(self) -> u32 {
        match self {
            Self::Full => 0,
            Self::SameClass => 1,
            Self::SameClassMajorPawn => 2,
            Self::CrossSide => 10,
        }
    }

    /// pair を除外すべきか判定する。
    ///
    /// `as_` / `ds` は side (0=味方, 1=敵)、`ac` / `dc` は class index (0..=8)。
    /// `SameClassMajorPawn` の `ac >= 5` は `ThreatClass::Bishop` 以降 (大駒)、
    /// `dc == 0` は `ThreatClass::Pawn` を指す。
    #[inline]
    pub fn is_excluded(self, as_: usize, ac: usize, ds: usize, dc: usize) -> bool {
        match self {
            Self::Full => false,
            Self::SameClass => ac == dc,
            Self::SameClassMajorPawn => ac == dc || (ac >= 5 && dc == 0),
            Self::CrossSide => as_ == ds || ac == dc,
        }
    }

    /// 利用可能な profile 名の一覧 (ヘルプ表示用)。
    pub fn available() -> &'static str {
        "full, same-class, same-class-major-pawn, cross-side"
    }
}

impl std::fmt::Display for ThreatProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Full => "full",
            Self::SameClass => "same-class",
            Self::SameClassMajorPawn => "same-class-major-pawn",
            Self::CrossSide => "cross-side",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_cli_roundtrips_with_display() {
        for p in [
            ThreatProfile::Full,
            ThreatProfile::SameClass,
            ThreatProfile::SameClassMajorPawn,
            ThreatProfile::CrossSide,
        ] {
            assert_eq!(ThreatProfile::from_cli(&p.to_string()), Some(p));
        }
        assert_eq!(ThreatProfile::from_cli("nonexistent"), None);
    }

    #[test]
    fn profile_ids_match_canonical() {
        assert_eq!(ThreatProfile::Full.profile_id(), 0);
        assert_eq!(ThreatProfile::SameClass.profile_id(), 1);
        assert_eq!(ThreatProfile::SameClassMajorPawn.profile_id(), 2);
        assert_eq!(ThreatProfile::CrossSide.profile_id(), 10);
    }

    #[test]
    fn full_excludes_nothing() {
        for as_ in 0..2 {
            for ac in 0..9 {
                for ds in 0..2 {
                    for dc in 0..9 {
                        assert!(!ThreatProfile::Full.is_excluded(as_, ac, ds, dc));
                    }
                }
            }
        }
    }

    #[test]
    fn cross_side_keeps_only_cross_side_distinct_class() {
        // 残るのは side が違い (as != ds) かつ class が違う (ac != dc) pair のみ。
        assert!(!ThreatProfile::CrossSide.is_excluded(0, 5, 1, 0));
        assert!(ThreatProfile::CrossSide.is_excluded(0, 5, 0, 0));
        assert!(ThreatProfile::CrossSide.is_excluded(0, 3, 1, 3));
    }
}
