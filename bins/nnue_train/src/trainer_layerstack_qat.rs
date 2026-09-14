use super::*;

#[cfg(all(test, feature = "native"))]
#[path = "trainer_layerstack_qat_tests.rs"]
mod tests;

#[cfg(feature = "native")]
fn qat_dense_threads(batch: usize, inputs: usize, outputs: usize) -> Result<usize, &'static str> {
    let lanes = if inputs >= 256 { 32 } else { 1 };
    batch
        .checked_mul(outputs)
        .and_then(|n| n.checked_mul(lanes))
        .filter(|&n| u32::try_from(n).is_ok())
        .ok_or("dense QAT launch exceeds the supported u32 thread count")
}

impl GpuTrainer {
    #[cfg(feature = "native")]
    pub(super) fn qat_activations(
        &self,
        layer: u32,
        batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // SAFETY: each kernel writes only the live batch prefix of its dimensioned workspace.
        unsafe {
            match layer {
                0 => {
                    cuda_launch! {
                        kernel: qat_activation, stream: self.stream, module: self.module,
                        config: cfg_1d(batch * self.ws.ft_out),
                        args: [slice_mut(self.ws.combined), (batch * self.ws.ft_out) as u32]
                    }
                }?,
                1 => {
                    cuda_launch! {
                        kernel: qat_l1_activation, stream: self.stream, module: self.module,
                        config: cfg_1d(batch * (self.ws.l1_out - 1)),
                        args: [slice(self.qat_raw[0]), slice_mut(self.ws.l2_input), batch as u32, self.ws.l1_out as u32]
                    }
                }?,
                2 => {
                    cuda_launch! {
                        kernel: qat_relu_activation, stream: self.stream, module: self.module,
                        config: cfg_1d(batch * self.ws.l2_out),
                        args: [slice(self.qat_raw[1]), slice_mut(self.ws.l2_acted), (batch * self.ws.l2_out) as u32]
                    }
                }?,
                3 => {
                    cuda_launch! {
                        kernel: qat_output, stream: self.stream, module: self.module,
                        config: cfg_1d(batch),
                        args: [slice(self.qat_raw[0]), slice(self.qat_raw[2]), slice_mut(self.ws.net_output), batch as u32, self.ws.l1_out as u32]
                    }
                }?,
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    pub(crate) fn configure_qat(
        &mut self,
        mode: Option<crate::qat::QatMode>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mode) = mode {
            self.qat_mode = mode;
        }
        if self.qat_mode == crate::qat::QatMode::Off {
            return Ok(());
        }
        if !cfg!(feature = "native") {
            return Err("dense QAT requires the native backend; cuda-oxide and oxide-parity are unsupported".into());
        }
        if self.stack_shared_delta.is_some() || self.psqt.is_some() {
            return Err("dense QAT does not support PSQT or L2/L3 shared-delta".into());
        }
        Ok(())
    }

    #[cfg(feature = "native")]
    pub(super) fn qat_swap(&mut self) {
        for (weight, scratch) in [
            &mut self.l1_w,
            &mut self.l1_b,
            &mut self.l1_shared_weight,
            &mut self.l1_shared_bias,
            &mut self.l2_w,
            &mut self.l2_b,
            &mut self.l3_w,
            &mut self.l3_b,
        ]
        .into_iter()
        .zip(&mut self.qat_buffers)
        {
            std::mem::swap(weight, scratch);
        }
    }

    #[cfg(feature = "native")]
    pub(super) fn qat_prepare(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let weights = [
            &self.l1_w,
            &self.l1_b,
            &self.l1_shared_weight,
            &self.l1_shared_bias,
            &self.l2_w,
            &self.l2_b,
            &self.l3_w,
            &self.l3_b,
        ];
        if self.qat_buffers.is_empty() {
            self.qat_buffers = weights
                .iter()
                .map(|w| DeviceBuffer::zeroed(&self.stream, w.len()))
                .collect::<Result<_, _>>()?;
            self.qat_raw = [
                self.ws.l1_total.len(),
                self.ws.l2_dense_out.len(),
                self.ws.l3_out.len(),
            ]
            .into_iter()
            .map(|n| DeviceBuffer::zeroed(&self.stream, n))
            .collect::<Result<_, _>>()?;
        }
        for (i, weight) in weights.into_iter().enumerate() {
            if i == 2 || i == 3 {
                memset_zero(&self.stream, &self.qat_buffers[i])?;
                continue;
            }
            let shared = if i == 1 {
                &self.l1_shared_bias
            } else {
                &self.l1_shared_weight
            };
            // SAFETY: same-length scratch and weights; shared indices follow L1's input-major layout.
            unsafe {
                cuda_launch! {
                    kernel: qat_weight, stream: self.stream, module: self.module,
                    config: cfg_1d(weight.len()),
                    args: [slice(weight), slice(shared), slice_mut(self.qat_buffers[i]),
                        self.ws.ft_out as u32, self.ws.l1_out as u32, (i < 2) as u32, (i % 2 == 1) as u32]
                }
            }?;
        }
        self.qat_swap();
        Ok(())
    }

    #[cfg(feature = "native")]
    pub(super) fn qat_dense_forward(
        &self,
        layer: u32,
        batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // After qat_swap, qat_buffers holds the original masters (including nonzero shared bias).
        // Quantize those biases directly to integers: an f32 dequantize/requantize round trip
        // cannot preserve every i32 bias. The live quantized biases serve the floating kernels only.
        let (input, w, bias, output, inputs, outputs) = match layer {
            1 => (
                &self.ws.combined,
                &self.l1_w,
                &self.qat_buffers[1],
                &self.ws.l1_total,
                self.ws.ft_out,
                self.ws.l1_out,
            ),
            2 => (
                &self.ws.l2_input,
                &self.l2_w,
                &self.qat_buffers[5],
                &self.ws.l2_dense_out,
                self.ws.l2_in(),
                self.ws.l2_out,
            ),
            3 => (
                &self.ws.l2_acted,
                &self.l3_w,
                &self.qat_buffers[7],
                &self.ws.l3_out,
                self.ws.l2_out,
                1,
            ),
            _ => unreachable!(),
        };
        let threads = qat_dense_threads(batch, inputs, outputs)?;
        // SAFETY: workspace and weight dimensions match the selected layer; batch buckets were validated.
        unsafe {
            cuda_launch! {
                kernel: qat_dense, stream: self.stream, module: self.module,
                config: cfg_1d(threads),
                args: [slice(input), slice(w), slice(bias), slice(self.qat_buffers[3]), slice(self.ws.bucket_idx_dev), slice_mut(output), slice_mut(self.qat_raw[(layer - 1) as usize]),
                    batch as u32, inputs as u32, outputs as u32, (layer == 1) as u32]
            }
        }?;
        Ok(())
    }
}
