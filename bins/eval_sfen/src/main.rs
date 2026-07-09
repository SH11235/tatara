use std::io::Cursor;
use std::path::{Path, PathBuf};

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{
    DEFAULT_L1_OUT, DEFAULT_L2_OUT, DEFAULT_NUM_BUCKETS, FV_SCALE, LEGACY_NNUE_VERSION_BUCKETS9,
    NNUE_VERSION, QA, QB,
};
use shogi_features::{EffectBucketConfig, FeatureSet, FeatureSetSpec, ShogiProgressKPAbs};
use shogi_format::ShogiBoard;
use shogi_format::types::{Color, Hand, Piece, PieceType, Square};

const L1_SKIP: usize = 1;
const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

#[derive(Parser)]
struct Args {
    /// Quantised EffectBucket LayerStack net (arch token `EffectBucket=`).
    #[arg(long)]
    nnue_file: PathBuf,

    #[arg(long)]
    progress_coeff: PathBuf,

    /// Quantised LayerStack output scale used to convert raw NNUE output to centipawns.
    #[arg(long, default_value_t = FV_SCALE)]
    fv_scale: i32,

    #[arg(long)]
    sfen: Vec<String>,

    #[arg(long)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let progress = ShogiProgressKPAbs::load_from_bin(&args.progress_coeff)
        .map_err(|e| format!("failed to load progress coeff: {e}"))?;
    let (weights, spec) = load_layerstack(&args.nnue_file)?;
    let positions = if args.sfen.is_empty() {
        vec!["startpos".to_string()]
    } else {
        args.sfen
    };

    for pos in positions {
        let board = parse_position(&pos)?;
        let bucket = progress.bucket_board(&board, weights.num_buckets) as usize;
        let raw = forward_one_raw(&weights, spec, &board, bucket)?;
        let cp = raw / args.fv_scale;
        if args.debug {
            eprintln!("{pos}\tbucket={bucket}\traw={raw}\tcp={cp}");
        }
        println!("{cp}");
    }
    Ok(())
}

fn load_layerstack(
    path: &Path,
) -> Result<(LayerStackWeights, FeatureSetSpec), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let meta = read_arch_meta(&bytes)?;
    let spec = FeatureSet::HalfKaHmMerged
        .spec()
        .with_effect_bucket_config(meta.effect_bucket_config);
    let mut cursor = Cursor::new(&bytes);
    let weights = LayerStackWeights::load_quantised(
        &mut cursor,
        spec,
        meta.ft_out,
        meta.l1_out,
        meta.l2_out,
        meta.num_buckets,
    )?;
    Ok((weights, spec))
}

struct ArchMeta {
    effect_bucket_config: EffectBucketConfig,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
}

fn read_arch_meta(bytes: &[u8]) -> Result<ArchMeta, Box<dyn std::error::Error>> {
    if bytes.len() < 16 {
        return Err("NNUE file is too short".into());
    }
    let version = u32::from_le_bytes(bytes[0..4].try_into().expect("slice has 4 bytes"));
    if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
        return Err(format!("unsupported NNUE version: {version:#x}").into());
    }
    let arch_len = u32::from_le_bytes(bytes[8..12].try_into().expect("slice has 4 bytes")) as usize;
    let arch_start = 12;
    let arch_end = arch_start + arch_len;
    if arch_end > bytes.len() {
        return Err("NNUE arch string extends past end of file".into());
    }
    let arch = std::str::from_utf8(&bytes[arch_start..arch_end])?;
    let num_buckets = if version == LEGACY_NNUE_VERSION_BUCKETS9 {
        DEFAULT_NUM_BUCKETS
    } else {
        if arch_end + 4 > bytes.len() {
            return Err("NNUE num_buckets field extends past end of file".into());
        }
        u32::from_le_bytes(
            bytes[arch_end..arch_end + 4]
                .try_into()
                .expect("slice has 4 bytes"),
        ) as usize
    };
    let affine_outs = parse_affine_outs(arch);
    Ok(ArchMeta {
        effect_bucket_config: parse_effect_bucket_config(arch)?,
        ft_out: parse_between(arch, "->", "x2]")?,
        l1_out: affine_outs.last().copied().unwrap_or(DEFAULT_L1_OUT),
        l2_out: affine_outs.get(1).copied().unwrap_or(DEFAULT_L2_OUT),
        num_buckets: if num_buckets == 0 {
            DEFAULT_NUM_BUCKETS
        } else {
            num_buckets
        },
    })
}

fn parse_effect_bucket_config(
    arch: &str,
) -> Result<EffectBucketConfig, Box<dyn std::error::Error>> {
    if arch.contains("EffectBucket=2x2fixed") {
        Ok(EffectBucketConfig::KINGFIXED_2X2)
    } else if arch.contains("EffectBucket=2x2bucketed") {
        Ok(EffectBucketConfig::KINGBUCKETED_2X2)
    } else if arch.contains("EffectBucket=3x3fixed") {
        Ok(EffectBucketConfig::KINGFIXED_3X3)
    } else if arch.contains("EffectBucket=3x3bucketed") {
        Ok(EffectBucketConfig::KINGBUCKETED_3X3)
    } else {
        Err(format!("unsupported or missing effect bucket token in arch string: {arch}").into())
    }
}

fn parse_affine_outs(arch: &str) -> Vec<usize> {
    let needle = "AffineTransform[";
    let mut start = 0usize;
    let mut vals = Vec::new();
    while let Some(pos) = arch[start..].find(needle) {
        let absolute = start + pos + needle.len();
        if let Some(end) = arch[absolute..].find("<-")
            && let Ok(v) = arch[absolute..absolute + end].parse()
        {
            vals.push(v);
        }
        start = absolute;
    }
    vals
}

fn parse_between(
    s: &str,
    start_pat: &str,
    end_pat: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let start = s
        .find(start_pat)
        .ok_or_else(|| format!("missing `{start_pat}` in arch string"))?
        + start_pat.len();
    let end = s[start..]
        .find(end_pat)
        .ok_or_else(|| format!("missing `{end_pat}` in arch string"))?
        + start;
    Ok(s[start..end].parse()?)
}

fn forward_one_raw(
    weights: &LayerStackWeights,
    spec: FeatureSetSpec,
    board: &ShogiBoard,
    bucket: usize,
) -> Result<i32, Box<dyn std::error::Error>> {
    let ft_out = weights.ft_b.len();
    let l1_out = weights.l1_b.len() / weights.num_buckets;
    let l2_out = weights.l2_b.len() / weights.num_buckets;
    let l1_effective = l1_out
        .checked_sub(L1_SKIP)
        .ok_or("l1_out must be larger than L1_SKIP")?;
    let l2_in = l1_effective * 2;

    let mut stm_idx = vec![-1_i32; spec.max_active()];
    let mut nstm_idx = vec![-1_i32; spec.max_active()];
    let active = spec.extract_active_features(board, &mut stm_idx, &mut nstm_idx);
    if active > stm_idx.len() {
        return Err(format!(
            "active feature count {active} exceeds buffer capacity {}",
            stm_idx.len()
        )
        .into());
    }

    let mut stm_ft = vec![0i32; ft_out];
    let mut nstm_ft = vec![0i32; ft_out];
    sparse_ft_forward_raw(
        weights,
        &stm_idx[..active],
        &mut stm_ft,
        ft_out,
        spec.ft_in(),
    );
    sparse_ft_forward_raw(
        weights,
        &nstm_idx[..active],
        &mut nstm_ft,
        ft_out,
        spec.ft_in(),
    );
    for (raw, bias) in stm_ft.iter_mut().zip(&weights.ft_b) {
        *raw += quant_i32(*bias, QA);
    }
    for (raw, bias) in nstm_ft.iter_mut().zip(&weights.ft_b) {
        *raw += quant_i32(*bias, QA);
    }

    let mut transformed = vec![0u8; ft_out];
    pairwise_crelu_to_u8(&stm_ft, &mut transformed[..ft_out / 2]);
    pairwise_crelu_to_u8(&nstm_ft, &mut transformed[ft_out / 2..]);

    let mut l1_total = vec![0i32; l1_out];
    affine_u8_i8(
        &transformed,
        &weights.l1_w[bucket * l1_out * ft_out..(bucket + 1) * l1_out * ft_out],
        &weights.l1_b[bucket * l1_out..(bucket + 1) * l1_out],
        &mut l1_total,
        ft_out,
        l1_out,
    );

    let mut l2_input = vec![0u8; l2_in];
    l1_sqr_clipped_relu(&l1_total[..l1_effective], &mut l2_input);

    let mut l2_dense_out = vec![0i32; l2_out];
    affine_u8_i8(
        &l2_input,
        &weights.l2_w[bucket * l2_out * l2_in..(bucket + 1) * l2_out * l2_in],
        &weights.l2_b[bucket * l2_out..(bucket + 1) * l2_out],
        &mut l2_dense_out,
        l2_in,
        l2_out,
    );
    let l2_act: Vec<u8> = l2_dense_out.iter().map(|&v| crelu_i32_to_u8(v)).collect();

    let mut out = [0i32; 1];
    affine_u8_i8(
        &l2_act,
        &weights.l3_w[bucket * l2_out..(bucket + 1) * l2_out],
        &weights.l3_b[bucket..bucket + 1],
        &mut out,
        l2_out,
        1,
    );

    Ok(out[0] + l1_total[l1_effective])
}

fn sparse_ft_forward_raw(
    weights: &LayerStackWeights,
    indices: &[i32],
    out: &mut [i32],
    ft_out: usize,
    ft_in: usize,
) {
    for &idx in indices {
        if idx < 0 || idx as usize >= ft_in {
            continue;
        }
        let base = idx as usize * ft_out;
        for (row, out_cell) in out.iter_mut().enumerate().take(ft_out) {
            *out_cell += quant_i32(weights.ft_w[base + row], QA);
        }
    }
}

fn pairwise_crelu_to_u8(input: &[i32], output: &mut [u8]) {
    let half = input.len() / 2;
    debug_assert_eq!(output.len(), half);
    for i in 0..half {
        let a = input[i].clamp(0, QA);
        let b = input[half + i].clamp(0, QA);
        output[i] = ((a * b) >> 7).clamp(0, 126) as u8;
    }
}

fn l1_sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    let main_dim = input.len();
    debug_assert_eq!(output.len(), main_dim * 2);
    for (i, &val) in input.iter().enumerate() {
        let v64 = val as i64;
        output[i] = ((v64 * v64) >> 19).clamp(0, 127) as u8;
        output[main_dim + i] = crelu_i32_to_u8(val);
    }
}

fn crelu_i32_to_u8(val: i32) -> u8 {
    (val >> 6).clamp(0, 127) as u8
}

fn affine_u8_i8(
    input: &[u8],
    weights: &[f32],
    bias: &[f32],
    output: &mut [i32],
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(input.len(), in_dim);
    debug_assert_eq!(output.len(), out_dim);
    for out_idx in 0..out_dim {
        let mut sum = quant_i32(bias[out_idx], QA * QB);
        let row = &weights[out_idx * in_dim..(out_idx + 1) * in_dim];
        for (&x, &w) in input.iter().zip(row) {
            sum += x as i32 * quant_i32(w, QB);
        }
        output[out_idx] = sum;
    }
}

fn quant_i32(value: f32, scale: i32) -> i32 {
    (value * scale as f32).round() as i32
}

fn parse_position(input: &str) -> Result<ShogiBoard, String> {
    let mut tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.first() == Some(&"position") {
        tokens.remove(0);
    }
    if tokens.is_empty() || tokens[0] == "startpos" {
        let mut board = parse_sfen(STARTPOS_SFEN)?;
        if let Some(moves_idx) = tokens.iter().position(|&t| t == "moves") {
            apply_moves(&mut board, &tokens[moves_idx + 1..])?;
        }
        return Ok(board);
    }
    if tokens[0] != "sfen" {
        return parse_sfen(input);
    }
    if tokens.len() < 5 {
        return Err("sfen position must contain board, side, hand, and ply".to_string());
    }
    let mut board = parse_sfen(&tokens[1..5].join(" "))?;
    if let Some(moves_idx) = tokens.iter().position(|&t| t == "moves") {
        apply_moves(&mut board, &tokens[moves_idx + 1..])?;
    }
    Ok(board)
}

fn apply_moves(board: &mut ShogiBoard, moves: &[&str]) -> Result<(), String> {
    for mv in moves {
        apply_move(board, mv)?;
    }
    Ok(())
}

fn apply_move(board: &mut ShogiBoard, mv: &str) -> Result<(), String> {
    let side = board.side_to_move;
    if mv.len() < 4 {
        return Err(format!("invalid move: {mv}"));
    }
    let bytes = mv.as_bytes();
    let to = parse_square(&mv[2..4])?;
    let moving = if bytes[1] == b'*' {
        let pt = piece_type_from_sfen(bytes[0] as char)?;
        remove_hand(board, side, pt)?;
        Piece::new(side, pt)
    } else {
        let from = parse_square(&mv[0..2])?;
        let mut piece = board.piece_on(from);
        if piece.is_none() {
            return Err(format!("move source is empty: {mv}"));
        }
        board.board[from.index()] = Piece::NONE;
        if mv.ends_with('+') {
            piece.piece_type = piece.piece_type.promote();
        }
        piece
    };

    let captured = board.piece_on(to);
    if captured.is_some() {
        hand_mut(board, side).add(captured.piece_type.unpromote(), 1);
    }
    board.board[to.index()] = moving;
    if moving.piece_type == PieceType::King {
        match moving.color {
            Color::Black => board.black_king_sq = to,
            Color::White => board.white_king_sq = to,
        }
    }
    board.side_to_move = side.opponent();
    board.ply = board.ply.saturating_add(1);
    Ok(())
}

fn parse_sfen(sfen: &str) -> Result<ShogiBoard, String> {
    let mut parts = sfen.split_whitespace();
    let board_part = parts.next().ok_or("missing board")?;
    let side_part = parts.next().ok_or("missing side-to-move")?;
    let hand_part = parts.next().ok_or("missing hand")?;
    let ply_part = parts.next().ok_or("missing ply")?;

    let mut board = ShogiBoard {
        side_to_move: parse_side(side_part)?,
        ply: ply_part
            .parse::<u16>()
            .map_err(|_| format!("invalid ply: {ply_part}"))?,
        ..Default::default()
    };

    parse_board(board_part, &mut board)?;
    parse_hand(hand_part, &mut board)?;
    Ok(board)
}

fn parse_side(s: &str) -> Result<Color, String> {
    match s {
        "b" => Ok(Color::Black),
        "w" => Ok(Color::White),
        _ => Err(format!("invalid side-to-move: {s}")),
    }
}

fn parse_board(board_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    let ranks: Vec<&str> = board_part.split('/').collect();
    if ranks.len() != 9 {
        return Err(format!("board must have 9 ranks: {board_part}"));
    }
    for (rank, row) in ranks.iter().enumerate() {
        let mut file_from_left = 0usize;
        let mut promoted = false;
        for ch in row.chars() {
            if ch == '+' {
                if promoted {
                    return Err(format!("duplicate promotion marker in rank: {row}"));
                }
                promoted = true;
                continue;
            }
            if let Some(skip) = ch.to_digit(10) {
                if promoted {
                    return Err(format!(
                        "promotion marker before empty squares in rank: {row}"
                    ));
                }
                file_from_left += skip as usize;
                continue;
            }
            if file_from_left >= 9 {
                return Err(format!("too many files in rank: {row}"));
            }
            let color = if ch.is_ascii_uppercase() {
                Color::Black
            } else {
                Color::White
            };
            let mut pt = piece_type_from_sfen(ch)?;
            if promoted {
                pt = pt.promote();
                promoted = false;
            }
            let sq = Square::new((8 - file_from_left) as u8, rank as u8);
            let piece = Piece::new(color, pt);
            board.board[sq.index()] = piece;
            if pt == PieceType::King {
                match color {
                    Color::Black => board.black_king_sq = sq,
                    Color::White => board.white_king_sq = sq,
                }
            }
            file_from_left += 1;
        }
        if promoted {
            return Err(format!("dangling promotion marker in rank: {row}"));
        }
        if file_from_left != 9 {
            return Err(format!("rank does not contain 9 files: {row}"));
        }
    }
    Ok(())
}

fn parse_hand(hand_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    if hand_part == "-" {
        return Ok(());
    }
    let mut count = 0u8;
    for ch in hand_part.chars() {
        if let Some(digit) = ch.to_digit(10) {
            count = count
                .saturating_mul(10)
                .saturating_add(u8::try_from(digit).expect("single digit fits u8"));
            continue;
        }
        let n = if count == 0 { 1 } else { count };
        count = 0;
        let color = if ch.is_ascii_uppercase() {
            Color::Black
        } else {
            Color::White
        };
        let pt = piece_type_from_sfen(ch)?;
        if !pt.can_be_in_hand() {
            return Err(format!("piece cannot be in hand: {ch}"));
        }
        hand_mut(board, color).add(pt, n);
    }
    if count != 0 {
        return Err(format!("dangling hand count in: {hand_part}"));
    }
    Ok(())
}

fn parse_square(s: &str) -> Result<Square, String> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 || !(b'1'..=b'9').contains(&bytes[0]) {
        return Err(format!("invalid square: {s}"));
    }
    let file = bytes[0] - b'1';
    let rank = match bytes[1] {
        b'a'..=b'i' => bytes[1] - b'a',
        _ => return Err(format!("invalid square: {s}")),
    };
    Ok(Square::new(file, rank))
}

fn piece_type_from_sfen(ch: char) -> Result<PieceType, String> {
    match ch.to_ascii_uppercase() {
        'P' => Ok(PieceType::Pawn),
        'L' => Ok(PieceType::Lance),
        'N' => Ok(PieceType::Knight),
        'S' => Ok(PieceType::Silver),
        'G' => Ok(PieceType::Gold),
        'B' => Ok(PieceType::Bishop),
        'R' => Ok(PieceType::Rook),
        'K' => Ok(PieceType::King),
        _ => Err(format!("invalid piece char: {ch}")),
    }
}

fn hand_mut(board: &mut ShogiBoard, color: Color) -> &mut Hand {
    match color {
        Color::Black => &mut board.black_hand,
        Color::White => &mut board.white_hand,
    }
}

fn remove_hand(board: &mut ShogiBoard, color: Color, pt: PieceType) -> Result<(), String> {
    let hand = hand_mut(board, color);
    let count = hand.count(pt);
    if count == 0 {
        return Err(format!("drop piece is not in hand: {pt:?}"));
    }
    hand.set(pt, count - 1);
    Ok(())
}
