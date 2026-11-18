use std::{fmt, str::FromStr};

use shogi_format::{Color, PieceType, ShogiBoard};

/// Maximum number of LayerStack output buckets supported by training.
pub const MAX_LAYERSTACK_BUCKETS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    OwnKing { cuts: [u8; 9], len: u8 },
    OppKing { cuts: [u8; 9], len: u8 },
    HandMajor3,
    HandMajor4,
}

impl Axis {
    fn radix(self) -> usize {
        match self {
            Self::OwnKing { len, .. } | Self::OppKing { len, .. } => usize::from(len) + 1,
            Self::HandMajor3 => 3,
            Self::HandMajor4 => 4,
        }
    }

    fn value(self, board: &ShogiBoard) -> usize {
        match self {
            Self::OwnKing { cuts, len } | Self::OppKing { cuts, len } => {
                let color = if matches!(self, Self::OwnKing { .. }) {
                    board.side_to_move
                } else {
                    board.side_to_move.opponent()
                };
                let rank = board.king_square(color).rank();
                let v = if color == Color::Black {
                    9 - rank
                } else {
                    rank + 1
                };
                cuts[..usize::from(len)].iter().filter(|&&c| c <= v).count()
            }
            Self::HandMajor3 | Self::HandMajor4 => {
                let has_major = |c| {
                    let h = board.hand(c);
                    h.count(PieceType::Rook) > 0 || h.count(PieceType::Bishop) > 0
                };
                match (
                    has_major(board.side_to_move),
                    has_major(board.side_to_move.opponent()),
                ) {
                    (false, false) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (true, true) => {
                        if self == Self::HandMajor3 {
                            0
                        } else {
                            3
                        }
                    }
                }
            }
        }
    }
}

/// Position axes for LayerStack output routing, with the left axis most significant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketSpec {
    axes: [Axis; 2],
    len: u8,
}

impl BucketSpec {
    pub fn num_buckets(self) -> usize {
        self.axes[..usize::from(self.len)]
            .iter()
            .map(|a| a.radix())
            .product()
    }

    pub fn bucket_board(self, board: &ShogiBoard) -> u8 {
        self.axes[..usize::from(self.len)]
            .iter()
            .fold(0, |n, a| n * a.radix() + a.value(board)) as u8
    }
}

impl FromStr for BucketSpec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut result = Self {
            axes: [Axis::HandMajor3; 2],
            len: 0,
        };
        for part in s.split('+') {
            if result.len == 2 {
                return Err("bucket spec requires one or two axes".into());
            }
            let axis = match part {
                "handmajor3" => Axis::HandMajor3,
                "handmajor4" => Axis::HandMajor4,
                _ => {
                    let (name, text) = part.split_once(':').ok_or(
                        "expected ownking:<cuts>, oppking:<cuts>, handmajor3 or handmajor4",
                    )?;
                    if !matches!(name, "ownking" | "oppking") {
                        return Err(format!("unknown bucket axis: {name}"));
                    }
                    let mut cuts = [0; 9];
                    let mut len = 0;
                    let mut previous = 0;
                    for cut in text.split('/') {
                        let v: u8 = cut
                            .parse()
                            .map_err(|_| format!("invalid king-rank cut: {cut}"))?;
                        if !(1..=9).contains(&v) || v <= previous || len == 9 {
                            return Err(
                                "king-rank cuts must be strictly increasing integers in [1, 9]"
                                    .into(),
                            );
                        }
                        cuts[len] = v;
                        len += 1;
                        previous = v;
                    }
                    if name == "ownking" {
                        Axis::OwnKing {
                            cuts,
                            len: len as u8,
                        }
                    } else {
                        Axis::OppKing {
                            cuts,
                            len: len as u8,
                        }
                    }
                }
            };
            result.axes[usize::from(result.len)] = axis;
            result.len += 1;
        }
        if result.num_buckets() > MAX_LAYERSTACK_BUCKETS {
            return Err(format!(
                "bucket spec needs {} buckets; maximum is {MAX_LAYERSTACK_BUCKETS}",
                result.num_buckets()
            ));
        }
        Ok(result)
    }
}

impl fmt::Display for BucketSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, axis) in self.axes[..usize::from(self.len)].iter().enumerate() {
            if i != 0 {
                write!(f, "+")?;
            }
            match axis {
                Axis::HandMajor3 => write!(f, "handmajor3")?,
                Axis::HandMajor4 => write!(f, "handmajor4")?,
                Axis::OwnKing { cuts, len } | Axis::OppKing { cuts, len } => {
                    write!(
                        f,
                        "{}:",
                        if matches!(axis, Axis::OwnKing { .. }) {
                            "ownking"
                        } else {
                            "oppking"
                        }
                    )?;
                    for (j, c) in cuts[..usize::from(*len)].iter().enumerate() {
                        if j != 0 {
                            write!(f, "/")?;
                        }
                        write!(f, "{c}")?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::types::Square;

    fn fixture(sfen: &str) -> ShogiBoard {
        let fields: Vec<_> = sfen.split_whitespace().collect();
        let mut b = ShogiBoard {
            side_to_move: if fields[1] == "b" {
                Color::Black
            } else {
                Color::White
            },
            ..Default::default()
        };
        for (rank, row) in fields[0].split('/').enumerate() {
            let mut x = 0;
            for c in row.chars() {
                if let Some(n) = c.to_digit(10) {
                    x += n;
                    continue;
                }
                let sq = Square::new((8 - x) as u8, rank as u8);
                match c {
                    'K' => b.black_king_sq = sq,
                    'k' => b.white_king_sq = sq,
                    _ => {}
                }
                x += 1;
            }
            assert_eq!(x, 9);
        }
        for c in fields[2].chars() {
            match c {
                'R' => b.black_hand.set_rook(1),
                'B' => b.black_hand.set_bishop(1),
                'r' => b.white_hand.set_rook(1),
                'b' => b.white_hand.set_bishop(1),
                _ => {}
            }
        }
        b
    }

    #[test]
    fn parse_spec() {
        for (s, n) in [
            ("handmajor3+ownking:2/3", 9),
            ("ownking:2/3+oppking:2/3", 9),
            ("handmajor3", 3),
            ("handmajor4", 4),
            ("ownking:2/4/6/8", 5),
            ("ownking:2/3/7+oppking:2/3/7", 16),
            ("handmajor4+ownking:2/3/4/5/6/7/8", 32),
        ] {
            let spec: BucketSpec = s.parse().unwrap();
            assert_eq!(spec.num_buckets(), n);
            assert_eq!(spec.to_string(), s);
        }
        for s in [
            "",
            "spec",
            "handmajor3:",
            "ownking",
            "ownking:",
            "ownking:0",
            "ownking:10",
            "ownking:3/2",
            "ownking:2/2",
            "ownking:2/",
            "ownking:-1",
            "handmajor4+ownking:2/3/4/5/6/7/8/9",
            "handmajor3+handmajor3+handmajor3",
            "+handmajor3",
            "handmajor3+",
        ] {
            assert!(s.parse::<BucketSpec>().is_err(), "{s}");
        }
    }

    #[test]
    fn king_axes_use_each_kings_home_rank() {
        for (placement, own, opp) in [
            ("4k4/9/9/9/9/9/9/9/4K4", 0, 0),
            ("9/4k4/9/9/9/9/9/4K4/9", 1, 1),
            ("9/9/4k4/9/9/9/4K4/9/9", 2, 2),
            ("4K4/9/9/9/9/9/9/9/4k4", 2, 2),
            ("9/4k4/9/9/9/9/9/9/4K4", 0, 1),
        ] {
            for side in ["b", "w"] {
                let b = fixture(&format!("{placement} {side} - 1"));
                let (o, e) = if side == "b" { (own, opp) } else { (opp, own) };
                assert_eq!(
                    "ownking:2/3"
                        .parse::<BucketSpec>()
                        .unwrap()
                        .bucket_board(&b),
                    o
                );
                assert_eq!(
                    "oppking:2/3"
                        .parse::<BucketSpec>()
                        .unwrap()
                        .bucket_board(&b),
                    e
                );
                assert_eq!(
                    "ownking:2/3+oppking:2/3"
                        .parse::<BucketSpec>()
                        .unwrap()
                        .bucket_board(&b),
                    o * 3 + e
                );
            }
        }
    }

    #[test]
    fn hand_axes_and_mixed_radix() {
        for (hand, black3, black4, white3, white4) in [
            ("-", 0, 0, 0, 0),
            ("R", 1, 1, 2, 2),
            ("B", 1, 1, 2, 2),
            ("r", 2, 2, 1, 1),
            ("b", 2, 2, 1, 1),
            ("Rb", 0, 3, 0, 3),
            ("Br", 0, 3, 0, 3),
            ("Pp", 0, 0, 0, 0),
        ] {
            for (side, major3, major4, rank) in [("b", black3, black4, 1), ("w", white3, white4, 2)]
            {
                let b = fixture(&format!("9/9/4k4/9/9/9/9/4K4/9 {side} {hand} 1"));
                assert_eq!(
                    "handmajor3".parse::<BucketSpec>().unwrap().bucket_board(&b),
                    major3
                );
                assert_eq!(
                    "handmajor4".parse::<BucketSpec>().unwrap().bucket_board(&b),
                    major4
                );
                assert_eq!(
                    "handmajor3+ownking:2/3"
                        .parse::<BucketSpec>()
                        .unwrap()
                        .bucket_board(&b),
                    major3 * 3 + rank
                );
                assert_eq!(
                    "ownking:2+handmajor4"
                        .parse::<BucketSpec>()
                        .unwrap()
                        .bucket_board(&b),
                    4 + major4
                );
            }
        }
    }
}
