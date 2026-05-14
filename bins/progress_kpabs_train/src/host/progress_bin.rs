//! `progress.bin` I/O (f64 little-endian × N_WEIGHTS、`1_003_104` bytes 固定)。
//!
//! bullet-shogi 上流 (`write_progress_bin` / `read_progress_bin`) を移植。
//! Rust kernel 側は重みを `f32` で持つが、`progress.bin` の wire format は
//! `f64` LE であることに注意。書き込み時 f32→f64 cast、読み込み時 f64→f32 cast。

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use shogi_features::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;

/// `weights` (長さ `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`、f32) を
/// `progress.bin` (f64 LE × N) として書き出す。
pub fn write_progress_bin(path: &Path, weights: &[f32]) -> io::Result<()> {
    if weights.len() != SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "weight slice length {} != expected {}",
                weights.len(),
                SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS
            ),
        ));
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &w in weights {
        let bytes = (w as f64).to_le_bytes();
        writer.write_all(&bytes)?;
    }
    writer.flush()
}

/// progress.bin を読み込んで `Vec<f32>` (N_WEIGHTS 要素) として返す。
pub fn read_progress_bin(path: &Path) -> io::Result<Vec<f32>> {
    let expected_bytes = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() as usize != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "progress.bin size {} != expected {} (= {} f64 LE)",
                metadata.len(),
                expected_bytes,
                SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS
            ),
        ));
    }
    let mut reader = BufReader::new(file);
    let mut buf = vec![0_u8; expected_bytes];
    reader.read_exact(&mut buf)?;
    let weights = buf
        .chunks_exact(std::mem::size_of::<f64>())
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size guaranteed")) as f32)
        .collect();
    Ok(weights)
}
