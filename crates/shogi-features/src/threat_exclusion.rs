//! Threat pair の除外 profile (runtime 選択)。
//!
//! Threat 特徴量は 2 駒の利き関係を `(attacker_side, attacker_class,
//! attacked_side, attacked_class)` の 4 軸 pair で表す。profile はこの pair の
//! 部分集合を選んで次元を間引く。`is_excluded` が true の pair は index 空間から
//! 詰めて除外される (該当 edge は active feature として出力されない)。
//!
//! id と除外規則 (id 0-10) は bullet-shogi 正準ベクタに揃える。`step-attacker`
//! (id 3) は donor に無い engine-native profile で、占有依存 slider を attacker から
//! 外して利き列挙コストを削る狙い (tatara ↔ rshogi 間で id/規則を直接一致させる):
//!
//! | id | CLI 値 | 除外規則 |
//! |----|--------|---------|
//! | 0  | `full`                  | なし |
//! | 1  | `same-class`            | `ac == dc` |
//! | 2  | `same-class-major-pawn` | `ac == dc \|\| (ac >= 5 && dc == 0)` |
//! | 3  | `step-attacker`         | `ac == 1 \|\| ac >= 5` (slider attacker 全除外) |
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
    /// 占有依存 slider (香・角・飛・馬・竜) を attacker から除外し、単発利き駒
    /// (歩・桂・銀・GoldLike) のみ attacker に残す。
    StepAttacker,
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
            "step-attacker" => Some(Self::StepAttacker),
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
            Self::StepAttacker => 3,
            Self::CrossSide => 10,
        }
    }

    /// pair を除外すべきか判定する。
    ///
    /// `as_` / `ds` は side (0=味方, 1=敵)、`ac` / `dc` は class index (0..=8)。
    /// `SameClassMajorPawn` の `ac >= 5` は `ThreatClass::Bishop` 以降 (大駒)、
    /// `dc == 0` は `ThreatClass::Pawn` を指す。`StepAttacker` の `ac == 1 || ac >= 5`
    /// は占有依存 slider (Lance=1 + Bishop/Rook/Horse/Dragon=5..8) を attacker から外す。
    /// 本 trainer では該当 pair を index 空間から除く (active feature として emit されない)
    /// だけで、利き ray 列挙自体は他 profile と同様に行う。engine 側は同 profile で slider
    /// attacker を early-prune し ray 列挙を省いて NPS を削れる (本 crate の責務外)。
    #[inline]
    pub fn is_excluded(self, as_: usize, ac: usize, ds: usize, dc: usize) -> bool {
        match self {
            Self::Full => false,
            Self::SameClass => ac == dc,
            Self::SameClassMajorPawn => ac == dc || (ac >= 5 && dc == 0),
            Self::StepAttacker => ac == 1 || ac >= 5,
            Self::CrossSide => as_ == ds || ac == dc,
        }
    }

    /// 利用可能な profile 名の一覧 (ヘルプ表示用)。
    pub fn available() -> &'static str {
        "full, same-class, same-class-major-pawn, step-attacker, cross-side"
    }
}

impl std::fmt::Display for ThreatProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Full => "full",
            Self::SameClass => "same-class",
            Self::SameClassMajorPawn => "same-class-major-pawn",
            Self::StepAttacker => "step-attacker",
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
            ThreatProfile::StepAttacker,
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
        assert_eq!(ThreatProfile::StepAttacker.profile_id(), 3);
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

    #[test]
    fn step_attacker_keeps_only_step_piece_attackers() {
        // 残るのは attacker class が単発利き駒 (Pawn=0/Knight=2/Silver=3/GoldLike=4)。
        for ac in [0usize, 2, 3, 4] {
            assert!(!ThreatProfile::StepAttacker.is_excluded(0, ac, 1, 0));
        }
        // slider attacker (Lance=1/Bishop=5/Rook=6/Horse=7/Dragon=8) は除外。
        for ac in [1usize, 5, 6, 7, 8] {
            assert!(ThreatProfile::StepAttacker.is_excluded(0, ac, 1, 0));
        }
        // attacked class には依存しない。
        assert!(!ThreatProfile::StepAttacker.is_excluded(0, 0, 0, 6));
    }
}
