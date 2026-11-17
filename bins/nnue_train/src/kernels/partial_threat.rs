use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn partial_ft_forward(
    weight: &[f32],
    indices: &[i32],
    nnz: &[i32],
    pairs: &[u32],
    mut out: DisjointSlice<f32>,
    shape: &[u32],
    batch: u32,
) {
    let rows = shape[0];
    let base = shape[1];
    let threats = shape[2];
    let width = shape[3];
    let active = shape[4];
    let piece = shape[5];
    let factorize = shape[6];
    let half_weight = shape[7];
    let i = thread::index_1d();
    let k = i.get();
    let r = rows as usize;
    if k >= batch as usize * r {
        return;
    }
    let b = k / r;
    let col = k % r;
    let w = width as usize;
    let base = base as usize;
    let threats = threats as usize;
    let real_n = base * r + threats * w;
    let mut sum = 0.0_f32;
    for n in 0..nnz[b] as usize {
        let f = indices[b * active as usize + n];
        if f >= 0 && (f as usize) < base + threats {
            let f = f as usize;
            let mut offset = f * r + col;
            let mut virtual_offset = real_n + (f % piece as usize) * r + col;
            let mut enabled = true;
            if f >= base {
                let c = col % (r / 2);
                enabled = c < w / 2;
                let packed_col = c + if col >= r / 2 { w / 2 } else { 0 };
                offset = base * r + (f - base) * w + packed_col;
                if factorize != 0 && enabled {
                    let mut pair = 0_usize;
                    while pair + 1 < pairs.len() && (f - base) >= pairs[pair + 1] as usize {
                        pair += 1;
                    }
                    virtual_offset = real_n + piece as usize * r + pair * w + packed_col;
                }
            }
            if enabled {
                let mut value = weight[offset];
                if factorize != 0 {
                    value += weight[virtual_offset];
                }
                if half_weight != 0 {
                    value = (value as f16) as f32;
                }
                sum += value;
            }
        }
    }
    if let Some(cell) = out.get_mut(i) {
        *cell = sum;
    }
}

#[kernel]
pub fn partial_ft_forward_fp16_out(
    weight: &[f32],
    indices: &[i32],
    nnz: &[i32],
    pairs: &[u32],
    mut out: DisjointSlice<f16>,
    shape: &[u32],
    batch: u32,
) {
    let rows = shape[0];
    let base = shape[1];
    let threats = shape[2];
    let width = shape[3];
    let active = shape[4];
    let piece = shape[5];
    let factorize = shape[6];
    let half_weight = shape[7];
    let i = thread::index_1d();
    let k = i.get();
    let r = rows as usize;
    if k >= batch as usize * r {
        return;
    }
    let b = k / r;
    let col = k % r;
    let w = width as usize;
    let base = base as usize;
    let threats = threats as usize;
    let real_n = base * r + threats * w;
    let mut sum = 0.0_f32;
    for n in 0..nnz[b] as usize {
        let f = indices[b * active as usize + n];
        if f >= 0 && (f as usize) < base + threats {
            let f = f as usize;
            let mut offset = f * r + col;
            let mut virtual_offset = real_n + (f % piece as usize) * r + col;
            let mut enabled = true;
            if f >= base {
                let c = col % (r / 2);
                enabled = c < w / 2;
                let packed_col = c + if col >= r / 2 { w / 2 } else { 0 };
                offset = base * r + (f - base) * w + packed_col;
                if factorize != 0 && enabled {
                    let mut pair = 0_usize;
                    while pair + 1 < pairs.len() && (f - base) >= pairs[pair + 1] as usize {
                        pair += 1;
                    }
                    virtual_offset = real_n + piece as usize * r + pair * w + packed_col;
                }
            }
            if enabled {
                let mut value = weight[offset];
                if factorize != 0 {
                    value += weight[virtual_offset];
                }
                if half_weight != 0 {
                    value = (value as f16) as f32;
                }
                sum += value;
            }
        }
    }
    if let Some(cell) = out.get_mut(i) {
        *cell = sum as f16;
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
pub fn partial_ft_reduce(grad: &[f32], pairs: &[u32], shape: &[u32]) {
    let rows = shape[0];
    let base = shape[1];
    let threats = shape[2];
    let width = shape[3];
    let piece = shape[5];
    let i = thread::index_1d();
    let k = i.get();
    let r = rows as usize;
    let w = width as usize;
    let piece_n = piece as usize * r;
    let virtual_n = piece_n + (pairs.len() - 1) * w;
    if k >= virtual_n {
        return;
    }
    let mut sum = 0.0_f32;
    if k < piece_n {
        let p = k / r;
        let c = k % r;
        let mut f = p;
        while f < base as usize {
            sum += grad[f * r + c];
            f += piece as usize;
        }
    } else {
        let pair = (k - piece_n) / w;
        let c = (k - piece_n) % w;
        for f in pairs[pair]..pairs[pair + 1] {
            sum += grad[base as usize * r + f as usize * w + c];
        }
    }
    // SAFETY: each thread owns one virtual element; reads only access the real block.
    unsafe {
        *(grad.as_ptr() as *mut f32).add(base as usize * r + threats as usize * w + k) = sum;
    }
}

#[kernel]
pub fn partial_ft_norm(weight: &[f32], shape: &[u32], mut norms: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let col = i.get();
    let r = shape[0] as usize;
    if col >= r {
        return;
    }
    let base = shape[1] as usize;
    let threats = shape[2] as usize;
    let w = shape[3] as usize;
    let real_n = base * r + threats * w;
    let mut sum = 0.0_f64;
    for f in 0..base {
        let v = weight[f * r + col] as f64;
        sum += v * v;
    }
    if shape[6] != 0 {
        for p in 0..shape[5] as usize {
            let v = weight[real_n + p * r + col] as f64;
            sum += v * v;
        }
    }
    let c = col % (r / 2);
    if c < w / 2 {
        let pc = c + if col >= r / 2 { w / 2 } else { 0 };
        for f in 0..threats {
            let v = weight[base * r + f * w + pc] as f64;
            sum += v * v;
        }
        for p in 0..shape[8] as usize {
            let v = weight[real_n + shape[5] as usize * r + p * w + pc] as f64;
            sum += v * v;
        }
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
