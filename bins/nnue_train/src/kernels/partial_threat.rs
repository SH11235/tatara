use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn partial_ft_fold(weight: &[f32], pairs: &[u32], shape: &[u32], mut comb: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let k = i.get();
    if k >= comb.len() {
        return;
    }
    let r = shape[0] as usize;
    let base_n = shape[1] as usize * r;
    let w = shape[3] as usize;
    let real_n = base_n + shape[2] as usize * w;
    let mut value = weight[k];
    if shape[6] != 0 {
        let partner;
        if k < base_n {
            partner = real_n + (k / r % shape[5] as usize) * r + k % r;
        } else {
            let f = (k - base_n) / w;
            let mut pair = 0_usize;
            let mut end = pairs.len() - 1;
            while pair + 1 < end {
                let mid = (pair + end) / 2;
                if f >= pairs[mid] as usize {
                    pair = mid;
                } else {
                    end = mid;
                }
            }
            partner = real_n + shape[5] as usize * r + pair * w + (k - base_n) % w;
        }
        value += weight[partner];
    }
    if shape[7] != 0 {
        value = (value as f16) as f32;
    }
    if let Some(cell) = comb.get_mut(i) {
        *cell = value;
    }
}

#[kernel]
pub fn partial_ft_fold_i8x2(
    weight: &[f32],
    pairs: &[u32],
    shape: &[u32],
    mut comb: DisjointSlice<f32>,
) {
    let i = thread::index_1d();
    let k = i.get();
    if k >= comb.len() {
        return;
    }
    let r = shape[0] as usize;
    let base_n = shape[1] as usize * r;
    let w = shape[3] as usize;
    let real_n = base_n + shape[2] as usize * w;
    let mut value = weight[k];
    if shape[6] != 0 {
        let partner;
        if k < base_n {
            partner = real_n + (k / r % shape[5] as usize) * r + k % r;
        } else {
            let f = (k - base_n) / w;
            let mut pair = 0_usize;
            let mut end = pairs.len() - 1;
            while pair + 1 < end {
                let mid = (pair + end) / 2;
                if f >= pairs[mid] as usize {
                    pair = mid;
                } else {
                    end = mid;
                }
            }
            partner = real_n + shape[5] as usize * r + pair * w + (k - base_n) % w;
        }
        value += weight[partner];
    }
    if k < base_n {
        value = super::common::piece_ft_i8x2(value);
    }
    if shape[7] != 0 {
        value = (value as f16) as f32;
    }
    if let Some(cell) = comb.get_mut(i) {
        *cell = value;
    }
}

#[kernel]
pub fn partial_ft_forward(
    weight: &[f32],
    indices: &[i32],
    nnz: &[i32],
    mut out: DisjointSlice<f32>,
    shape: &[u32],
    batch_order: u64,
    order: &[i32],
) {
    let batch = batch_order as u32;
    let order_base = (batch_order >> 32) as u32;
    let rows = shape[0];
    let base = shape[1];
    let threats = shape[2];
    let width = shape[3];
    let active = shape[4];
    let i = thread::index_1d();
    let k = i.get();
    let r = rows as usize;
    if k >= batch as usize * r {
        return;
    }
    let position = k / r;
    let b = if order_base == u32::MAX {
        position
    } else {
        order[order_base as usize + position] as usize - order_base as usize
    };
    let col = k % r;
    let w = width as usize;
    let base = base as usize;
    let threats = threats as usize;
    let mut sum = 0.0_f32;
    for n in 0..nnz[b] as usize {
        let f = indices[b * active as usize + n];
        if f >= 0 && (f as usize) < base + threats {
            let f = f as usize;
            let mut offset = f * r + col;
            let mut enabled = true;
            if f >= base {
                let c = col % (r / 2);
                enabled = c < w / 2;
                let packed_col = c + if col >= r / 2 { w / 2 } else { 0 };
                offset = base * r + (f - base) * w + packed_col;
            }
            if enabled {
                sum += weight[offset];
            }
        }
    }
    // SAFETY: order is a permutation, so each thread owns one distinct output cell.
    unsafe {
        *(out.as_mut_ptr().add(b * r + col)) = sum;
    }
}

#[kernel]
pub fn partial_ft_forward_fp16_out(
    weight: &[f32],
    indices: &[i32],
    nnz: &[i32],
    mut out: DisjointSlice<f16>,
    shape: &[u32],
    batch_order: u64,
    order: &[i32],
) {
    let batch = batch_order as u32;
    let order_base = (batch_order >> 32) as u32;
    let rows = shape[0];
    let base = shape[1];
    let threats = shape[2];
    let width = shape[3];
    let active = shape[4];
    let i = thread::index_1d();
    let k = i.get();
    let r = rows as usize;
    if k >= batch as usize * r {
        return;
    }
    let position = k / r;
    let b = if order_base == u32::MAX {
        position
    } else {
        order[order_base as usize + position] as usize - order_base as usize
    };
    let col = k % r;
    let w = width as usize;
    let base = base as usize;
    let threats = threats as usize;
    let mut sum = 0.0_f32;
    for n in 0..nnz[b] as usize {
        let f = indices[b * active as usize + n];
        if f >= 0 && (f as usize) < base + threats {
            let f = f as usize;
            let mut offset = f * r + col;
            let mut enabled = true;
            if f >= base {
                let c = col % (r / 2);
                enabled = c < w / 2;
                let packed_col = c + if col >= r / 2 { w / 2 } else { 0 };
                offset = base * r + (f - base) * w + packed_col;
            }
            if enabled {
                sum += weight[offset];
            }
        }
    }
    // SAFETY: order is a permutation, so each thread owns one distinct output cell.
    unsafe {
        *(out.as_mut_ptr().add(b * r + col)) = sum as f16;
    }
}

#[kernel]
pub fn partial_ft_gather(
    gradient: &[f32],
    positions: &[u32],
    offsets: &[u32],
    mut result: DisjointSlice<f32>,
    shape: &[u32],
    add: u32,
    scale: f32,
) {
    let rows = shape[0];
    let base = shape[1];
    let threats = shape[2];
    let width = shape[3];
    let i = thread::index_1d();
    let k = i.get();
    let r = rows as usize;
    let base_n = base as usize * r;
    let w = width as usize;
    if k >= base_n + threats as usize * w {
        return;
    }
    let f;
    let col;
    if k < base_n {
        f = k / r;
        col = k % r;
    } else {
        f = base as usize + (k - base_n) / w;
        let c = (k - base_n) % w;
        col = if c < w / 2 { c } else { r / 2 + c - w / 2 };
    }
    let mut sum = 0.0_f32;
    for p in offsets[f]..offsets[f + 1] {
        sum += gradient[positions[p as usize] as usize * r + col] * scale;
    }
    if let Some(cell) = result.get_mut(i) {
        if add != 0 {
            *cell += sum;
        } else {
            *cell = sum;
        }
    }
}

#[kernel]
pub fn partial_ft_gather_fp16(
    gradient: &[f16],
    positions: &[u32],
    offsets: &[u32],
    mut result: DisjointSlice<f32>,
    shape: &[u32],
    add: u32,
    scale: f32,
) {
    let rows = shape[0];
    let base = shape[1];
    let threats = shape[2];
    let width = shape[3];
    let i = thread::index_1d();
    let k = i.get();
    let r = rows as usize;
    let base_n = base as usize * r;
    let w = width as usize;
    if k >= base_n + threats as usize * w {
        return;
    }
    let f;
    let col;
    if k < base_n {
        f = k / r;
        col = k % r;
    } else {
        f = base as usize + (k - base_n) / w;
        let c = (k - base_n) % w;
        col = if c < w / 2 { c } else { r / 2 + c - w / 2 };
    }
    let mut sum = 0.0_f32;
    for p in offsets[f]..offsets[f + 1] {
        sum += (gradient[positions[p as usize] as usize * r + col] as f32) * scale;
    }
    if let Some(cell) = result.get_mut(i) {
        if add != 0 {
            *cell += sum;
        } else {
            *cell = sum;
        }
    }
}

#[kernel]
pub fn partial_ft_reduce(mut grad: DisjointSlice<f32>, pairs: &[u32], shape: &[u32]) {
    let r = shape[0] as usize;
    let base = shape[1] as usize;
    let threats = shape[2] as usize;
    let w = shape[3] as usize;
    let piece = shape[5] as usize;
    let k = thread::index_1d().get();
    let base_n = base * r;
    let real_n = base_n + threats * w;
    let piece_n = piece * r;
    let virtual_n = piece_n + (pairs.len() - 1) * w;
    if k >= virtual_n {
        return;
    }
    let ptr = grad.as_mut_ptr();
    // SAFETY: the prefix contains all real gradients and is read-only during this launch.
    let real_grad = unsafe { core::slice::from_raw_parts(ptr, real_n) };
    let mut sum = 0.0_f32;
    if k < piece_n {
        let p = k / r;
        let c = k % r;
        let mut f = p;
        while f < base {
            sum += real_grad[f * r + c];
            f += piece;
        }
    } else {
        let pair = (k - piece_n) / w;
        let c = (k - piece_n) % w;
        for f in pairs[pair]..pairs[pair + 1] {
            sum += real_grad[base_n + f as usize * w + c];
        }
    }
    // SAFETY: the suffix contains virtual_n elements and each thread owns one element.
    unsafe {
        *ptr.add(real_n + k) = sum;
    }
}

// Split each column across 256 row partitions so the FT reduction can use all SMs.
#[kernel]
pub fn partial_ft_norm(weight: &[f32], shape: &[u32], mut sums: DisjointSlice<f64>) {
    let i = thread::index_1d();
    let k = i.get();
    let r = shape[0] as usize;
    if k >= r * 256 {
        return;
    }
    let col = k % r;
    let partition = k / r;
    let base = shape[1] as usize;
    let threats = shape[2] as usize;
    let w = shape[3] as usize;
    let piece = if shape[6] != 0 { shape[5] as usize } else { 0 };
    let pairs = shape[8] as usize;
    let real_n = base * r + threats * w;
    let c = col % (r / 2);
    let pc = c + if col >= r / 2 { w / 2 } else { 0 };
    let mut sum = 0.0_f64;
    let mut row = partition;
    while row < base + piece + threats + pairs {
        let offset;
        if row < base {
            offset = row * r + col;
        } else if row < base + piece {
            offset = real_n + (row - base) * r + col;
        } else if c < w / 2 {
            if row < base + piece + threats {
                offset = base * r + (row - base - piece) * w + pc;
            } else {
                offset = real_n + piece * r + (row - base - piece - threats) * w + pc;
            }
        } else {
            row += 256;
            continue;
        }
        let v = weight[offset] as f64;
        sum += v * v;
        row += 256;
    }
    if let Some(cell) = sums.get_mut(i) {
        *cell = sum;
    }
}

#[kernel]
pub fn partial_ft_norm_finalize(sums: &[f64], mut norms: DisjointSlice<f32>, rows: u32) {
    let i = thread::index_1d();
    let col = i.get();
    if col >= rows as usize {
        return;
    }
    let mut sum = 0.0_f64;
    for partition in 0..256 {
        sum += sums[partition * rows as usize + col];
    }
    if let Some(cell) = norms.get_mut(i) {
        *cell = sum.sqrt() as f32;
    }
}

#[kernel]
pub fn partial_ft_norm_apply(
    mut weight: DisjointSlice<f32>,
    shape: &[u32],
    norms: &[f32],
    factor: f32,
    lr: f32,
) {
    let i = thread::index_1d();
    let k = i.get();
    if k >= weight.len() {
        return;
    }
    let r = shape[0] as usize;
    let base_n = shape[1] as usize * r;
    let w = shape[3] as usize;
    let real_n = base_n + shape[2] as usize * w;
    let col;
    if k < base_n {
        col = k % r;
    } else if k < real_n {
        let c = (k - base_n) % w;
        col = if c < w / 2 { c } else { r / 2 + c - w / 2 };
    } else if k < real_n + shape[5] as usize * r {
        col = (k - real_n) % r;
    } else {
        let c = (k - real_n - shape[5] as usize * r) % w;
        col = if c < w / 2 { c } else { r / 2 + c - w / 2 };
    }
    let correction = 2.0_f32 * factor * (1.0_f32 - 1.0_f32 / (norms[col] + 1.0e-8_f32));
    if let Some(cell) = weight.get_mut(i) {
        *cell *= 1.0_f32 - lr * correction;
    }
}
