use serde::Serialize;

/// SplitMix64 with rejection sampling keeps diagnostic randomness independent of training.
pub(crate) fn sample_indices(n: u32, count: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut selected = std::collections::BTreeSet::new();
    let bound = u64::from(n);
    let threshold = bound.wrapping_neg() % bound;
    while selected.len() < count.min(n as usize) {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        if z >= threshold {
            selected.insert((z % bound) as u32);
        }
    }
    selected.into_iter().collect()
}

#[derive(Default, Serialize, Debug)]
pub(crate) struct StorageStats {
    samples: u64,
    nonfinite: u64,
    finite: u64,
    input_zero: u64,
    input_nonzero: u64,
    zeroed: u64,
    capped: u64,
    changed_candidate: u64,
    round_to_same: u64,
    stored_zero: u64,
    stored_subnormal: u64,
    uncapped_nonzero: u64,
    uncapped_abs_error_sum: f64,
    uncapped_relative_error_sum: f64,
    uncapped_abs_error_max: f64,
    total_abs_error_sum: f64,
}

impl StorageStats {
    pub(crate) fn observe(
        &mut self,
        old: f32,
        candidate: f32,
        stored: f32,
        scale: f32,
        half: bool,
        velocity: bool,
    ) {
        self.samples += 1;
        if !old.is_finite() || !candidate.is_finite() || !stored.is_finite() {
            self.nonfinite += 1;
            return;
        }
        self.finite += 1;
        self.input_zero += u64::from(candidate == 0.0);
        self.input_nonzero += u64::from(candidate != 0.0);
        let capped = half && (candidate > 65504.0 || (!velocity && candidate < -65504.0));
        self.capped += u64::from(capped);
        self.zeroed += u64::from(candidate != 0.0 && stored == 0.0 && !capped);
        self.changed_candidate += u64::from(candidate != old);
        self.round_to_same +=
            u64::from(candidate != old && stored == old && stored != 0.0 && !capped);
        self.stored_zero += u64::from(stored == 0.0);
        self.stored_subnormal +=
            u64::from(half && stored != 0.0 && stored.abs() < 2.0_f32.powi(-14));
        let error = (f64::from(stored) - f64::from(candidate)).abs() / f64::from(scale);
        self.total_abs_error_sum += error;
        if !capped {
            self.uncapped_abs_error_sum += error;
            self.uncapped_abs_error_max = self.uncapped_abs_error_max.max(error);
            if candidate != 0.0 {
                self.uncapped_nonzero += 1;
                self.uncapped_relative_error_sum +=
                    (f64::from(stored) - f64::from(candidate)).abs() / f64::from(candidate).abs();
            }
        }
    }
}

#[derive(Default, Serialize)]
struct DeltaStats {
    samples: u64,
    finite: u64,
    changed: u64,
    abs_delta_sum: f64,
    abs_delta_max: f64,
}

impl DeltaStats {
    fn observe(&mut self, before: f32, after: f32) {
        self.samples += 1;
        if before.is_finite() && after.is_finite() {
            self.finite += 1;
            self.changed += u64::from(before != after);
            let delta = (f64::from(after) - f64::from(before)).abs();
            self.abs_delta_sum += delta;
            self.abs_delta_max = self.abs_delta_max.max(delta);
        }
    }
}

#[derive(Default, Serialize)]
pub(crate) struct GroupStats {
    sampled_elements: u64,
    optimizer_gradient_nonzero: u64,
    optimizer_gradient_nonfinite: u64,
    m: StorageStats,
    v: StorageStats,
    master_optimizer: DeltaStats,
    master_norm_loss: DeltaStats,
    master_end_of_step: DeltaStats,
    forward_fp16: DeltaStats,
}

pub(crate) fn summarize(
    indices: &[u32],
    records: &[f32],
    base_elements: u32,
    half: bool,
    mirror: bool,
) -> [GroupStats; 2] {
    assert_eq!(records.len(), indices.len() * 13);
    let mut groups = [GroupStats::default(), GroupStats::default()];
    for (&index, r) in indices.iter().zip(records.chunks_exact(13)) {
        let group = &mut groups[usize::from(index >= base_elements)];
        group.sampled_elements += 1;
        group.optimizer_gradient_nonzero += u64::from(r[8].is_finite() && r[8] != 0.0);
        group.optimizer_gradient_nonfinite += u64::from(!r[8].is_finite());
        group.m.observe(
            r[0],
            r[1],
            r[2],
            if half { 268435456.0 } else { 1.0 },
            half,
            false,
        );
        group.v.observe(
            r[3],
            r[4],
            r[5],
            if half { 1099511627776.0 } else { 1.0 },
            half,
            true,
        );
        group.master_optimizer.observe(r[6], r[7]);
        group.master_norm_loss.observe(r[12], r[6]);
        group.master_end_of_step.observe(r[12], r[9]);
        if mirror && index < base_elements {
            group.forward_fp16.observe(r[10], r[11]);
        }
    }
    groups
}

#[cfg(feature = "native")]
pub(crate) struct Diagnostic {
    pub(crate) steps: Vec<u64>,
    pub(crate) indices: Vec<u32>,
    pub(crate) device_indices: gpu_runtime::DeviceBuffer<u32>,
    pub(crate) records: gpu_runtime::DeviceBuffer<f32>,
    pub(crate) dummy_mirror: gpu_runtime::DeviceBuffer<f16>,
    pub(crate) seed: u64,
}

#[cfg(feature = "native")]
impl Diagnostic {
    pub(crate) fn new(
        stream: &gpu_runtime::CudaStream,
        n: usize,
        steps: Vec<u64>,
        count: usize,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = u32::try_from(n)?;
        if n == 0 || count == 0 || count > 65536 || steps.is_empty() || steps.contains(&0) {
            return Err(
                "precision diagnostics require nonempty positive steps and 1..=65536 samples"
                    .into(),
            );
        }
        let indices = sample_indices(n, count, seed);
        Ok(Self {
            device_indices: gpu_runtime::DeviceBuffer::from_host(stream, &indices)?,
            records: gpu_runtime::DeviceBuffer::zeroed(stream, indices.len() * 13)?,
            dummy_mirror: gpu_runtime::DeviceBuffer::zeroed(stream, 1)?,
            indices,
            steps,
            seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_storage_events_and_optimizer_bits() -> Result<(), Box<dyn std::error::Error>> {
        use gpu_runtime::{CudaContext, DeviceBuffer, KernelArgs, LaunchConfig};
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let module = ctx.load_module_from_image(cuda_native_runtime::NATIVE_KERNEL_FATBIN)?;
        for selected in [&[0_u32, 1, 2, 3, 4][..], &[0_u32, 2, 4][..]] {
            let indices = DeviceBuffer::from_host(&stream, selected)?;
            let records = DeviceBuffer::<f32>::zeroed(&stream, 65)?;
            for (denom, decay) in [(0_i32, 0.0_f32), (1, 0.0), (1, 0.01)] {
                for half in [false, true] {
                    for mirror in [false, true] {
                        let mut results = Vec::new();
                        for observe in [false, true] {
                            let w = DeviceBuffer::from_host(&stream, &[0.1_f32; 5])?;
                            let g = DeviceBuffer::from_host(
                                &stream,
                                &[0.0_f32, 1.0e-16, 0.0, 1.0, -1.0],
                            )?;
                            let initial = [0.0_f32, 0.0, 2.0_f32.powi(-24), 0.0, 0.0];
                            let m32 = DeviceBuffer::from_host(&stream, &initial)?;
                            let v32 = DeviceBuffer::from_host(&stream, &initial)?;
                            let initial_half: Vec<f16> =
                                initial.iter().map(|&v| v as f16).collect();
                            let m16 = DeviceBuffer::from_host(&stream, &initial_half)?;
                            let v16 = DeviceBuffer::from_host(&stream, &initial_half)?;
                            let h = DeviceBuffer::<f16>::zeroed(&stream, 5)?;
                            let mut a = KernelArgs::new();
                            a.push_slice(&w);
                            if half {
                                a.push_slice(&m16);
                                a.push_slice(&v16);
                            } else {
                                a.push_slice(&m32);
                                a.push_slice(&v32);
                            }
                            a.push_slice(&g);
                            if observe || mirror {
                                a.push_slice(&h);
                            }
                            if observe {
                                a.push_slice(&indices);
                                a.push_slice(&records);
                            }
                            a.push_scalar(0.001_f32);
                            a.push_scalar(1.0_f32);
                            a.push_scalar(denom);
                            for value in [decay, 0.99, 0.999, 1.0e-8, -100.0, 100.0] {
                                a.push_scalar(value);
                            }
                            if observe || half {
                                a.push_scalar(268435456.0_f32);
                                a.push_scalar(1099511627776.0_f32);
                            }
                            a.push_scalar(5_u32);
                            if observe {
                                a.push_scalar(selected.len() as u32);
                                a.push_scalar(i32::from(half));
                                a.push_scalar(i32::from(mirror));
                            }
                            let kernel = if observe {
                                "precision_radam_step"
                            } else {
                                match (half, mirror) {
                                    (false, false) => "radam_step",
                                    (false, true) => "radam_step_fp16_mirror",
                                    (true, false) => "radam_step_f16state",
                                    (true, true) => "radam_step_f16state_mirror",
                                }
                            };
                            // Native zeroed allocations use the default stream; complete them
                            // before launching on the nonblocking compute stream.
                            ctx.synchronize()?;
                            // SAFETY: all five-element buffers and the thirteen-float sample records
                            // match the selected kernel ABI and live through the following readback.
                            unsafe {
                                module.launch(
                                    kernel,
                                    &stream,
                                    LaunchConfig::for_num_elems(5),
                                    &mut a,
                                )
                            }?;
                            let mut bits: Vec<u32> = w
                                .to_host_vec(&stream)?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect();
                            bits.extend(g.to_host_vec(&stream)?.into_iter().map(f32::to_bits));
                            bits.extend(
                                h.to_host_vec(&stream)?
                                    .into_iter()
                                    .map(|v| u32::from(v.to_bits())),
                            );
                            if half {
                                bits.extend(
                                    m16.to_host_vec(&stream)?
                                        .into_iter()
                                        .map(|v| u32::from(v.to_bits())),
                                );
                                bits.extend(
                                    v16.to_host_vec(&stream)?
                                        .into_iter()
                                        .map(|v| u32::from(v.to_bits())),
                                );
                            } else {
                                bits.extend(
                                    m32.to_host_vec(&stream)?.into_iter().map(f32::to_bits),
                                );
                                bits.extend(
                                    v32.to_host_vec(&stream)?.into_iter().map(f32::to_bits),
                                );
                            }
                            results.push(bits);
                            if observe && half {
                                let raw = records.to_host_vec(&stream)?;
                                let groups = summarize(
                                    selected,
                                    &raw[..selected.len() * 13],
                                    5,
                                    true,
                                    false,
                                );
                                let expected = if selected.len() == 5 {
                                    (1, 1, 1, 2)
                                } else {
                                    (1, 0, 1, 1)
                                };
                                let m = &groups[0].m;
                                let v = &groups[0].v;
                                assert_eq!(
                                    (m.input_zero, m.zeroed, m.round_to_same, m.capped),
                                    expected
                                );
                                assert_eq!(
                                    (v.input_zero, v.zeroed, v.round_to_same, v.capped),
                                    expected
                                );
                            }
                        }
                        assert_eq!(results[0], results[1], "half={half} mirror={mirror}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn storage_events_have_distinct_meaning() {
        let mut s = StorageStats::default();
        s.observe(0.0, 0.0, 0.0, 1.0, true, false);
        s.observe(0.0, 1.0e-9, 0.0, 1.0, true, false);
        s.observe(1.0, 1.0001, 1.0, 1.0, true, false);
        s.observe(65504.0, 70000.0, 65504.0, 1.0, true, false);
        s.observe(0.0, f32::NAN, f32::NAN, 1.0, true, false);
        assert_eq!((s.samples, s.finite, s.nonfinite), (5, 4, 1));
        assert_eq!(
            (s.input_zero, s.zeroed, s.capped, s.round_to_same),
            (1, 1, 1, 1)
        );
        assert_eq!(
            (s.input_nonzero, s.changed_candidate, s.stored_zero),
            (3, 3, 2)
        );
        assert!(s.total_abs_error_sum > s.uncapped_abs_error_sum);
    }

    #[test]
    fn row_groups_and_master_boundaries_are_separate() {
        let base = [
            0.0, 2.0, 2.0, 0.0, 3.0, 3.0, 4.0, 5.0, 0.0, 6.0, 7.0, 8.0, 3.0,
        ];
        let virtual_row = [
            0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 3.0, 1.0, 4.0, 99.0, 101.0, 1.0,
        ];
        let records: Vec<f32> = base.into_iter().chain(virtual_row).collect();
        let groups = summarize(&[1, 10], &records, 10, false, true);
        assert_eq!(groups[0].master_norm_loss.abs_delta_sum, 1.0);
        assert_eq!(groups[0].master_optimizer.abs_delta_sum, 1.0);
        assert_eq!(groups[0].master_end_of_step.abs_delta_sum, 3.0);
        assert_eq!(groups[0].forward_fp16.abs_delta_sum, 1.0);
        assert_eq!(groups[1].forward_fp16.samples, 0);
        assert_eq!(groups[1].optimizer_gradient_nonzero, 1);
        assert_eq!(groups[0].m.total_abs_error_sum, 0.0);
        let no_mirror = summarize(&[1, 10], &records, 10, false, false);
        assert_eq!(no_mirror[0].forward_fp16.samples, 0);
    }

    #[test]
    fn fixed_uniform_sample_is_unique_bounded_and_reproducible() {
        let a = sample_indices(10000, 1024, 20260913);
        assert_eq!(a, sample_indices(10000, 1024, 20260913));
        assert_ne!(a, sample_indices(10000, 1024, 1));
        assert_eq!(a.len(), 1024);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert!(a.iter().all(|&i| i < 10000));
        assert_eq!(sample_indices(3, 10, 0), vec![0, 1, 2]);
    }
}
