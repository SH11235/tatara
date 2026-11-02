#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Quantize the effective L1 sum once; shared storage is input-major.
extern "C" __global__ void qat_weight(
    const float* w, unsigned long long n,
    const float* shared, unsigned long long,
    float* out, unsigned long long,
    unsigned int inputs, unsigned int outputs, unsigned int merge, unsigned int bias
) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = w[i];
    if (merge) value += shared[bias ? i % outputs : (i % inputs) * outputs + (i / inputs) % outputs];
    const double scale = bias ? 8128.0 : 64.0;
    double q = round(static_cast<double>(value) * scale);
    q = fmin(bias ? 2147483647.0 : 127.0, fmax(bias ? -2147483648.0 : -128.0, q));
    out[i] = static_cast<float>(q / scale);
}

// Integer MAC avoids flooring an activation just below its lattice point after a floating sum.
extern "C" __global__ void qat_dense(
    const float* x, unsigned long long,
    const float* w, unsigned long long,
    const float* bias, unsigned long long,
    const float* shared_bias, unsigned long long,
    const int* bucket, unsigned long long,
    float* out, unsigned long long,
    int* raw, unsigned long long,
    unsigned int batch, unsigned int inputs, unsigned int outputs, unsigned int merge
) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lanes = inputs >= 256 ? 32 : 1;
    const unsigned int lane = thread % lanes;
    const unsigned int i = thread / lanes;
    if (i >= batch * outputs) return;
    const unsigned int row = i / outputs;
    const unsigned int neuron = bucket[row] * outputs + i % outputs;
    float b = bias[neuron];
    if (merge) b += shared_bias[i % outputs];
    long long sum = 0;
    for (unsigned int j = lane; j < inputs; j += lanes)
        // Inputs are on the [0,127]/127 lattice and weights on the i8/64 lattice.
        // Their f32 round trips are far from half-integers, so nearest-even recovers
        // the same integers as half-away rounding without general f64 conversion.
        sum += static_cast<long long>(__float2int_rn(x[row * inputs + j] * 127.0F))
             * __float2int_rn(w[neuron * inputs + j] * 64.0F);
    // Quantized products sum exactly in i64; warp reduction preserves integer results.
    if (lanes == 32) {
        for (unsigned int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane != 0) return;
    sum += static_cast<long long>(fmin(2147483647.0, fmax(-2147483648.0, round(static_cast<double>(b) * 8128.0))));
    raw[i] = static_cast<int>(sum);
    out[i] = static_cast<float>(static_cast<double>(raw[i]) / 8128.0);
}

// `qat_dense` for wide inputs: one warp per (row, group of 8 outputs). Each lane quantizes its
// strided slice of the input row once and accumulates it into all outputs of the group, so the
// row is read once per group rather than once per output. Products are bounded by 127 * 128 and
// a row has at most `inputs` of them, so for `inputs <= 131072` the dot product fits in `int`
// and the result matches the `long long` accumulation of `qat_dense` bit for bit.
extern "C" __global__ void qat_dense_wide(
    const float* x, unsigned long long,
    const float* w, unsigned long long,
    const float* bias, unsigned long long,
    const float* shared_bias, unsigned long long,
    const int* bucket, unsigned long long,
    float* out, unsigned long long,
    int* raw, unsigned long long,
    unsigned int batch, unsigned int inputs, unsigned int outputs, unsigned int merge
) {
    constexpr unsigned int GROUP = 8;
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = thread % 32;
    const unsigned int warp = thread / 32;
    const unsigned int groups = (outputs + GROUP - 1) / GROUP;
    if (warp >= batch * groups) return;
    const unsigned int row = warp / groups;
    const unsigned int first = (warp % groups) * GROUP;
    const unsigned int count = outputs - first < GROUP ? outputs - first : GROUP;
    const unsigned int neuron_base = static_cast<unsigned int>(bucket[row]) * outputs + first;
    int sums[GROUP] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned int j = lane; j < inputs; j += 32) {
        const int q = __float2int_rn(x[static_cast<unsigned long long>(row) * inputs + j] * 127.0F);
#pragma unroll
        for (unsigned int o = 0; o < GROUP; ++o) {
            if (o < count) {
                sums[o] += q * __float2int_rn(
                    w[static_cast<unsigned long long>(neuron_base + o) * inputs + j] * 64.0F
                );
            }
        }
    }
#pragma unroll
    for (unsigned int o = 0; o < GROUP; ++o) {
        for (unsigned int offset = 16; offset; offset >>= 1)
            sums[o] += __shfl_down_sync(0xffffffff, sums[o], offset);
    }
    if (lane != 0) return;
    for (unsigned int o = 0; o < count; ++o) {
        const unsigned int i = row * outputs + first + o;
        float b = bias[neuron_base + o];
        if (merge) b += shared_bias[first + o];
        const long long sum = static_cast<long long>(sums[o])
            + static_cast<long long>(fmin(2147483647.0, fmax(-2147483648.0, round(static_cast<double>(b) * 8128.0))));
        raw[i] = static_cast<int>(sum);
        out[i] = static_cast<float>(static_cast<double>(raw[i]) / 8128.0);
    }
}

// `qat_dense_wide` over bucket-sorted rows, writing to `permutation[sorted_row]`. A 16-row tile
// shares one bucket because bucket segments start on 16-row boundaries, so its quantized weights
// are staged in shared memory once per tile. The partial sums are integers, so combining the 16
// lanes of a row in any order reproduces `qat_dense_wide` exactly.
extern "C" __global__ void qat_dense_wide_sorted(
    const float* x_sorted, unsigned long long,
    const float* w, unsigned long long,
    const float* bias, unsigned long long,
    const float* shared_bias, unsigned long long,
    const int* bucket_sorted, unsigned long long,
    const int* permutation, unsigned long long,
    float* out, unsigned long long,
    int* raw, unsigned long long,
    unsigned int inputs, unsigned int outputs, unsigned int merge
) {
    constexpr unsigned int ROWS = 16;
    constexpr unsigned int GROUP = 16;
    constexpr unsigned int CHUNK = 512;
    __shared__ int weight_tile[GROUP][CHUNK];
    const unsigned int tile_row = blockIdx.x * ROWS;
    const int bucket = bucket_sorted[tile_row];
    if (bucket < 0) {
        return;
    }
    const unsigned int first = blockIdx.y * GROUP;
    const unsigned int count = outputs - first < GROUP ? outputs - first : GROUP;
    const unsigned int neuron_base = static_cast<unsigned int>(bucket) * outputs + first;
    const unsigned int local_row = threadIdx.x / GROUP;
    const unsigned int lane = threadIdx.x % GROUP;
    const unsigned long long row_base =
        static_cast<unsigned long long>(tile_row + local_row) * inputs;
    int sums[GROUP] = {};
    for (unsigned int base = 0; base < inputs; base += CHUNK) {
        const unsigned int width = inputs - base < CHUNK ? inputs - base : CHUNK;
        for (unsigned int k = threadIdx.x; k < GROUP * CHUNK; k += blockDim.x) {
            const unsigned int o = k / CHUNK;
            const unsigned int j = k % CHUNK;
            weight_tile[o][j] = o < count && j < width
                ? __float2int_rn(
                      w[static_cast<unsigned long long>(neuron_base + o) * inputs + base + j] * 64.0F
                  )
                : 0;
        }
        __syncthreads();
        for (unsigned int j = lane; j < width; j += GROUP) {
            const int q = __float2int_rn(x_sorted[row_base + base + j] * 127.0F);
#pragma unroll
            for (unsigned int o = 0; o < GROUP; ++o) {
                sums[o] += q * weight_tile[o][j];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned int o = 0; o < GROUP; ++o) {
        for (unsigned int offset = GROUP / 2; offset; offset >>= 1) {
            sums[o] += __shfl_xor_sync(0xffffffff, sums[o], offset, GROUP);
        }
    }
    const int row = permutation[tile_row + local_row];
    if (row < 0 || lane >= count) {
        return;
    }
    int total = 0;
#pragma unroll
    for (unsigned int o = 0; o < GROUP; ++o) {
        if (o == lane) {
            total = sums[o];
        }
    }
    const unsigned int i = static_cast<unsigned int>(row) * outputs + first + lane;
    float b = bias[neuron_base + lane];
    if (merge) b += shared_bias[first + lane];
    const long long sum = static_cast<long long>(total)
        + static_cast<long long>(fmin(2147483647.0, fmax(-2147483648.0, round(static_cast<double>(b) * 8128.0))));
    raw[i] = static_cast<int>(sum);
    out[i] = static_cast<float>(static_cast<double>(raw[i]) / 8128.0);
}

// `floor(clamp(x * 127, 0, 127)) / 127` evaluated in single precision without rounding the
// product first. While `x * 127` is finite, `product` is its nearest float and `error`, from
// `fmaf`, the exact remainder. Inside the unsaturated range, a `product` that is not an integer
// floors like the exact value, because an integer between the two would have been the nearer
// float, and an integer `product` exceeds the exact value only when `error` is negative. Outside
// that range the correction may be lost to rounding or to an infinite / NaN `error`, but `level`
// stays on the same side of the clamp, so the clamp gives the same endpoint. Clamping the floored
// integer equals flooring the clamped value, and `fmaxf` returns +0 for NaN and for -0.
extern "C" __global__ void qat_activation(
    float* x, unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float value = x[i];
    const float product = __fmul_rn(value, 127.0F);
    const float error = fmaf(value, 127.0F, -product);
    float level = floorf(product);
    if (level == product && error < 0.0F) {
        level -= 1.0F;
    }
    x[i] = fminf(127.0F, fmaxf(0.0F, level)) / 127.0F;
}

extern "C" __global__ void qat_l1_activation(
    const int* raw, unsigned long long,
    float* out, unsigned long long,
    unsigned int batch, unsigned int outputs
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int main = outputs - 1;
    if (i >= batch * main) return;
    const unsigned int row = i / main, col = i % main;
    const long long q = raw[row * outputs + col];
    out[row * main * 2 + col] = static_cast<float>(min(127LL, (q * q) >> 19)) / 127.0F;
    out[row * main * 2 + main + col] = static_cast<float>(fmin(127.0, fmax(0.0, floor(q / 64.0)))) / 127.0F;
}

extern "C" __global__ void qat_relu_activation(
    const int* raw, unsigned long long,
    float* out, unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double q = raw[i];
    out[i] = static_cast<float>(fmin(127.0, fmax(0.0, floor(q / 64.0)))) / 127.0F;
}

extern "C" __global__ void qat_output(
    const int* l1, unsigned long long,
    const int* l3, unsigned long long,
    float* out, unsigned long long,
    unsigned int batch, unsigned int outputs
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    const int raw = static_cast<int>(static_cast<long long>(l3[i]) + l1[i * outputs + outputs - 1]);
    out[i] = static_cast<float>(static_cast<double>(raw) / 8128.0);
}

// LayerStack bucket 数の上限。per-bucket accumulator を register array で持つ kernel が
// この値で配列を確保する。各アクセスは `min(num_buckets, kMaxSupportedNumBuckets)` で
// clamp するので範囲外 index (UB) にはならないが、host 側 `arch::MAX_SUPPORTED_NUM_BUCKETS`
// がこの値を超えると上位 bucket の勾配が黙って落ちる。両定数の一致を
// `native_bucket_capacity_matches_host` test で固定する。
constexpr unsigned int kMaxSupportedNumBuckets = 9;

extern "C" __global__ void native_vec_add(
    const float* lhs,
    const float* rhs,
    float* output,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = lhs[i] + rhs[i];
    }
}

extern "C" __global__ void native_loss_wrm_default(
    const float* output,
    const float* score,
    const float* wdl,
    float per_pos_norm,
    float* output_gradient,
    double* loss_accumulator,
    float lambda,
    float nnue2score,
    float input_scaling,
    float input_offset,
    float target_offset,
    float target_scaling,
    unsigned int ignore_draws,
    unsigned int n
) {
    __shared__ double partial[256];
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    double contribution = 0.0;

    if (i < n) {
        const float s = score[i];
        const float target_positive = 1.0F / (1.0F + expf(-((s - target_offset) / target_scaling)));
        const float target_negative = 1.0F / (1.0F + expf(-((-s - target_offset) / target_scaling)));
        const float target_wrm = 0.5F * (1.0F + target_positive - target_negative);
        const float lam = ignore_draws != 0 && wdl[i] == 0.5F ? 0.0F : lambda;
        const float target = lam * wdl[i] + (1.0F - lam) * target_wrm;

        const float score_net = output[i] * nnue2score;
        const float q = 1.0F / (1.0F + expf(-((score_net - input_offset) / input_scaling)));
        const float qm = 1.0F / (1.0F + expf(-((-score_net - input_offset) / input_scaling)));
        const float prediction = 0.5F * (1.0F + q - qm);
        const float error = prediction - target;

        output_gradient[i] = error
            * (nnue2score / input_scaling)
            * (q * (1.0F - q) + qm * (1.0F - qm))
            * per_pos_norm;
        contribution = static_cast<double>(error) * static_cast<double>(error);
    }

    partial[tid] = contribution;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(loss_accumulator, partial[0]);
    }
}

extern "C" __global__ void native_radam_step(
    float* weights,
    float* momentum,
    float* velocity,
    float* gradient,
    float learning_rate,
    float step_size,
    int use_variance_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float g = gradient[i];
    const float rate = learning_rate * step_size;
    float p = weights[i];
    p *= 1.0F - decay * rate;
    const float m = beta1 * momentum[i] + (1.0F - beta1) * g;
    const float v = beta2 * velocity[i] + (1.0F - beta2) * g * g;
    momentum[i] = m;
    velocity[i] = v;
    float value = m;
    if (use_variance_denom != 0) {
        value /= sqrtf(v) + epsilon;
    }
    p -= rate * value;
    if (p < min_weight) {
        p = min_weight;
    } else if (p > max_weight) {
        p = max_weight;
    }
    weights[i] = p;
    gradient[i] = 0.0F;
}

extern "C" __global__ void native_sparse_ft_forward(
    const float* weight,
    const int* indices,
    const int* nonzero_counts,
    float* output,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    const unsigned int packed_rows = rows / 4;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * packed_rows) {
        return;
    }
    const unsigned int batch_index = tid / packed_rows;
    const unsigned int row = (tid % packed_rows) * 4;
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;
    const unsigned int index_base = batch_index * max_active;
    const int count = nonzero_counts[batch_index];
    for (int active = 0; active < count; ++active) {
        const int column = indices[index_base + static_cast<unsigned int>(active)];
        if (column >= 0 && static_cast<unsigned int>(column) < columns) {
            const unsigned int weight_base = static_cast<unsigned int>(column) * rows + row;
            sum0 += weight[weight_base];
            sum1 += weight[weight_base + 1];
            sum2 += weight[weight_base + 2];
            sum3 += weight[weight_base + 3];
        }
    }
    const unsigned int output_base = batch_index * rows + row;
    output[output_base] = sum0;
    output[output_base + 1] = sum1;
    output[output_base + 2] = sum2;
    output[output_base + 3] = sum3;
}

// Trainer ABI wrappers. cuda-oxide marshals every Rust slice as a device pointer followed by a
// u64 length. Both host backends use this packet layout so one fat binary can be checked against
// the cuda-oxide reference without changing allocation, stream, or cuBLAS semantics.
extern "C" __global__ void sparse_ft_forward(
    const float* weight,
    unsigned long long,
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    const unsigned int packed_rows = rows / 4;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * packed_rows) {
        return;
    }
    const unsigned int batch_index = tid / packed_rows;
    const unsigned int row = (tid % packed_rows) * 4;
    float sums[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const unsigned int index_base = batch_index * max_active;
    const int count = nonzero_counts[batch_index];
    for (int active = 0; active < count; ++active) {
        const int column = indices[index_base + static_cast<unsigned int>(active)];
        if (column >= 0 && static_cast<unsigned int>(column) < columns) {
            const unsigned long long weight_base =
                static_cast<unsigned long long>(column) * rows + row;
#pragma unroll
            for (unsigned int lane = 0; lane < 4; ++lane) {
                sums[lane] += weight[weight_base + lane];
            }
        }
    }
    const unsigned int output_base = batch_index * rows + row;
#pragma unroll
    for (unsigned int lane = 0; lane < 4; ++lane) {
        output[output_base + lane] = sums[lane];
    }
}

// Index of the factorizer virtual weight folded into real FT element (feature, column), or
// `~0ULL` when the feature has no virtual partner. Shared by the fold kernels and the fused FT
// optimizer so both resolve the same virtual row.
__device__ __forceinline__ unsigned long long ft_fold_partner_index(
    unsigned long long feature,
    unsigned int column,
    const unsigned int* threat_pair_starts,
    unsigned long long threat_pair_starts_len,
    unsigned int base_ft_in,
    unsigned int ft_in,
    unsigned int ft_out,
    unsigned int piece_inputs,
    unsigned int effect_bucket_factorize
) {
    const unsigned int nb = effect_bucket_factorize & 0xffffU;
    const unsigned int mode = effect_bucket_factorize >> 16;
    if (feature < base_ft_in) {
        const unsigned long long piece = mode == 0
            ? feature % piece_inputs
            : (feature / nb) % piece_inputs;
        const unsigned long long virtual_row = mode == 2
            ? piece * nb + feature % nb
            : piece;
        return (static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column;
    }
    if (threat_pair_starts_len >= 2) {
        const unsigned long long relative = feature - base_ft_in;
        unsigned long long low = 0;
        unsigned long long high = threat_pair_starts_len - 1;
        while (low + 1 < high) {
            const unsigned long long middle = (low + high) / 2;
            if (threat_pair_starts[middle] <= relative) {
                low = middle;
            } else {
                high = middle;
            }
        }
        const unsigned long long base_virtual_rows = mode == 2
            ? static_cast<unsigned long long>(piece_inputs) * nb
            : piece_inputs;
        return (static_cast<unsigned long long>(ft_in) + base_virtual_rows + low) * ft_out + column;
    }
    return ~0ULL;
}

extern "C" __global__ void ft_fold_virtual(
    const float* weights,
    unsigned long long,
    float* combined,
    unsigned long long,
    const unsigned int* threat_pair_starts,
    unsigned long long threat_pair_starts_len,
    unsigned long long ft_bounds,
    unsigned int ft_out,
    unsigned int piece_inputs,
    unsigned int effect_bucket_factorize
) {
    const unsigned int base_ft_in = static_cast<unsigned int>(ft_bounds);
    const unsigned int ft_in = static_cast<unsigned int>(ft_bounds >> 32);
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long n = static_cast<unsigned long long>(ft_in) * ft_out;
    if (i >= n) {
        return;
    }

    const unsigned long long feature = i / ft_out;
    const unsigned int column = static_cast<unsigned int>(i - feature * ft_out);
    float value = weights[i];
    const unsigned long long partner = ft_fold_partner_index(
        feature, column, threat_pair_starts, threat_pair_starts_len, base_ft_in, ft_in, ft_out,
        piece_inputs, effect_bucket_factorize
    );
    if (partner != ~0ULL) {
        value += weights[partner];
    }
    combined[i] = value;
}

extern "C" __global__ void ft_fold_virtual_f16(
    const float* weights,
    unsigned long long,
    __half* combined,
    unsigned long long,
    const unsigned int* threat_pair_starts,
    unsigned long long threat_pair_starts_len,
    unsigned long long ft_bounds,
    unsigned int ft_out,
    unsigned int piece_inputs,
    unsigned int effect_bucket_factorize
) {
    const unsigned int base_ft_in = static_cast<unsigned int>(ft_bounds);
    const unsigned int ft_in = static_cast<unsigned int>(ft_bounds >> 32);
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long n = static_cast<unsigned long long>(ft_in) * ft_out;
    if (i >= n) {
        return;
    }

    const unsigned long long feature = i / ft_out;
    const unsigned int column = static_cast<unsigned int>(i - feature * ft_out);
    float value = weights[i];
    const unsigned long long partner = ft_fold_partner_index(
        feature, column, threat_pair_starts, threat_pair_starts_len, base_ft_in, ft_in, ft_out,
        piece_inputs, effect_bucket_factorize
    );
    if (partner != ~0ULL) {
        value += weights[partner];
    }
    combined[i] = __float2half_rn(value);
}

// Real-row FT gradient for the virtual-row reduction. Rows with `active_rows[row] == 0` were not
// written by the masked backward gather and hold stale values, so they read as zero.
__device__ __forceinline__ float ft_real_row_grad(
    const float* gradient,
    const unsigned int* active_rows,
    unsigned long long row,
    unsigned int ft_out,
    unsigned int column
) {
    if (active_rows != nullptr && active_rows[row] == 0U) {
        return 0.0F;
    }
    return gradient[row * ft_out + column];
}

__device__ __forceinline__ void ft_reduce_virtual_grad_impl(
    float* gradient,
    const unsigned int* active_rows,
    const unsigned int* threat_pair_starts,
    unsigned long long threat_pair_starts_len,
    unsigned long long ft_bounds,
    unsigned int ft_out,
    unsigned int piece_inputs,
    unsigned int effect_bucket_factorize
) {
    const unsigned int base_ft_in = static_cast<unsigned int>(ft_bounds);
    const unsigned int ft_in = static_cast<unsigned int>(ft_bounds >> 32);
    const unsigned int nb = effect_bucket_factorize & 0xffffU;
    const unsigned int mode = effect_bucket_factorize >> 16;
    const unsigned int base_virtual_rows = mode == 2 ? piece_inputs * nb : piece_inputs;
    const unsigned long long threat_virtual_rows = threat_pair_starts_len == 0
        ? 0
        : threat_pair_starts_len - 1;
    const unsigned long long virtual_rows = base_virtual_rows + threat_virtual_rows;
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= virtual_rows * ft_out) {
        return;
    }

    const unsigned long long virtual_row = i / ft_out;
    const unsigned int column = static_cast<unsigned int>(i - virtual_row * ft_out);
    float sum = 0.0F;
    if (virtual_row >= base_virtual_rows) {
        const unsigned long long pair = virtual_row - base_virtual_rows;
        const unsigned long long start =
            static_cast<unsigned long long>(base_ft_in) + threat_pair_starts[pair];
        const unsigned long long end =
            static_cast<unsigned long long>(base_ft_in) + threat_pair_starts[pair + 1];
        for (unsigned long long feature = start; feature < end; ++feature) {
            sum += ft_real_row_grad(gradient, active_rows, feature, ft_out, column);
        }
        gradient[(static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column] = sum;
        return;
    }

    const unsigned int piece = mode == 2 ? virtual_row / nb : virtual_row;
    const unsigned int bucket = mode == 2 ? virtual_row - piece * nb : 0;
    const unsigned int king_buckets = mode == 0
        ? base_ft_in / piece_inputs
        : base_ft_in / (piece_inputs * nb);
    if (mode == 1) {
        const unsigned long long king_stride = static_cast<unsigned long long>(piece_inputs) * nb;
        for (unsigned int king_bucket = 0; king_bucket < king_buckets; ++king_bucket) {
            const unsigned long long base =
                static_cast<unsigned long long>(king_bucket) * king_stride
                + static_cast<unsigned long long>(piece) * nb;
            for (unsigned int effect_bucket = 0; effect_bucket < nb; ++effect_bucket) {
                sum += ft_real_row_grad(
                    gradient, active_rows, base + effect_bucket, ft_out, column
                );
            }
        }
        gradient[(static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column] = sum;
        return;
    }

    const unsigned long long row_stride = mode == 0
        ? static_cast<unsigned long long>(piece_inputs)
        : static_cast<unsigned long long>(piece_inputs) * nb;
    const unsigned long long base = mode == 0
        ? static_cast<unsigned long long>(piece)
        : static_cast<unsigned long long>(piece) * nb + bucket;
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;
    unsigned int king_bucket = 0;
    const unsigned int unroll_end = king_buckets > 3 ? king_buckets - 3 : 0;
    while (king_bucket < unroll_end) {
        const unsigned long long row = base + static_cast<unsigned long long>(king_bucket) * row_stride;
        sum0 += ft_real_row_grad(gradient, active_rows, row, ft_out, column);
        sum1 += ft_real_row_grad(gradient, active_rows, row + row_stride, ft_out, column);
        sum2 += ft_real_row_grad(gradient, active_rows, row + 2 * row_stride, ft_out, column);
        sum3 += ft_real_row_grad(gradient, active_rows, row + 3 * row_stride, ft_out, column);
        king_bucket += 4;
    }
    while (king_bucket < king_buckets) {
        sum0 += ft_real_row_grad(
            gradient, active_rows, base + static_cast<unsigned long long>(king_bucket) * row_stride,
            ft_out, column
        );
        ++king_bucket;
    }
    sum = (sum0 + sum1) + (sum2 + sum3);
    gradient[(static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column] = sum;
}

extern "C" __global__ void ft_reduce_virtual_grad(
    float* gradient,
    unsigned long long,
    const unsigned int* threat_pair_starts,
    unsigned long long threat_pair_starts_len,
    unsigned long long ft_bounds,
    unsigned int ft_out,
    unsigned int piece_inputs,
    unsigned int effect_bucket_factorize
) {
    ft_reduce_virtual_grad_impl(
        gradient, nullptr, threat_pair_starts, threat_pair_starts_len, ft_bounds, ft_out,
        piece_inputs, effect_bucket_factorize
    );
}

extern "C" __global__ void ft_reduce_virtual_grad_active(
    float* gradient,
    unsigned long long,
    const unsigned int* active_rows,
    unsigned long long,
    const unsigned int* threat_pair_starts,
    unsigned long long threat_pair_starts_len,
    unsigned long long ft_bounds,
    unsigned int ft_out,
    unsigned int piece_inputs,
    unsigned int effect_bucket_factorize
) {
    ft_reduce_virtual_grad_impl(
        gradient, active_rows, threat_pair_starts, threat_pair_starts_len, ft_bounds, ft_out,
        piece_inputs, effect_bucket_factorize
    );
}

extern "C" __global__ void loss_wrm(
    const float* output,
    unsigned long long,
    const float* score,
    unsigned long long,
    const float* wdl,
    unsigned long long,
    float per_pos_norm,
    float* output_gradient,
    unsigned long long,
    double* loss_accumulator,
    unsigned long long,
    float lambda,
    float nnue2score,
    float input_scaling,
    float input_offset,
    float target_offset,
    float target_scaling,
    float pow_exp,
    float qp_asymmetry,
    float weight_boost_w1,
    float weight_boost_w2,
    const double* weight_sum_accumulator,
    unsigned long long,
    unsigned int extended,
    unsigned int ignore_draws,
    unsigned int n
) {
    __shared__ double partial[256];
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    double contribution = 0.0;

    if (i < n) {
        const float s = score[i];
        const float target_positive = 1.0F / (1.0F + expf(-((s - target_offset) / target_scaling)));
        const float target_negative = 1.0F / (1.0F + expf(-((-s - target_offset) / target_scaling)));
        const float target_wrm = 0.5F * (1.0F + target_positive - target_negative);
        const float lam = ignore_draws != 0 && wdl[i] == 0.5F ? 0.0F : lambda;
        const float target = lam * wdl[i] + (1.0F - lam) * target_wrm;
        const float score_net = output[i] * nnue2score;
        const float q = 1.0F / (1.0F + expf(-((score_net - input_offset) / input_scaling)));
        const float qm = 1.0F / (1.0F + expf(-((-score_net - input_offset) / input_scaling)));
        const float prediction = 0.5F * (1.0F + q - qm);
        const float error = prediction - target;
        if (extended == 0) {
            output_gradient[i] = error
                * (nnue2score / input_scaling)
                * (q * (1.0F - q) + qm * (1.0F - qm))
                * per_pos_norm;
            contribution = static_cast<double>(error) * static_cast<double>(error);
        } else {
            const float weight_base = (target_wrm - 0.5F) * (target_wrm - 0.5F)
                * target_wrm * (1.0F - target_wrm);
            const float weight = 1.0F
                + (powf(2.0F, weight_boost_w1) - 1.0F) * powf(weight_base, weight_boost_w2);
            const float asymmetry = prediction > target ? 1.0F + qp_asymmetry : 1.0F;
            const float absolute_error = fabsf(error);
            const float absolute_power = powf(absolute_error, pow_exp - 1.0F);
            const float signed_power = error < 0.0F ? -absolute_power : absolute_power;
            const float inverse_weight_sum = static_cast<float>(1.0 / weight_sum_accumulator[0]);
            const float prediction_gradient = 0.5F * (nnue2score / input_scaling)
                * (q * (1.0F - q) + qm * (1.0F - qm));
            output_gradient[i] = (weight * inverse_weight_sum)
                * (asymmetry * pow_exp * signed_power) * prediction_gradient;
            contribution = static_cast<double>(
                asymmetry * powf(absolute_error, pow_exp) * weight
                * static_cast<float>(n) * inverse_weight_sum
            );
        }
    }

    partial[tid] = contribution;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(loss_accumulator, partial[0]);
    }
}

extern "C" __global__ void loss_wdl(
    const float* output,
    unsigned long long,
    const float* score,
    unsigned long long,
    const float* wdl,
    unsigned long long,
    float per_pos_norm,
    float* output_gradient,
    unsigned long long,
    double* loss_accumulator,
    unsigned long long,
    float lambda,
    float scale,
    unsigned int ignore_draws,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float prediction = 1.0F / (1.0F + expf(-(output[i] * scale)));
    const float score_target = 1.0F / (1.0F + expf(-(score[i] * scale)));
    const float lam = ignore_draws != 0 && wdl[i] == 0.5F ? 0.0F : lambda;
    const float target = lam * wdl[i] + (1.0F - lam) * score_target;
    const float error = prediction - target;
    output_gradient[i] = 2.0F * error * prediction * (1.0F - prediction)
        * scale * per_pos_norm;
    atomicAdd(loss_accumulator, static_cast<double>(error) * static_cast<double>(error));
}

extern "C" __global__ void wrm_weight_sum(
    const float* score,
    unsigned long long,
    double* weight_sum_accumulator,
    unsigned long long,
    float weight_boost_w1,
    float weight_boost_w2,
    float target_offset,
    float target_scaling,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float s = score[i];
    const float target_positive = 1.0F / (1.0F + expf(-((s - target_offset) / target_scaling)));
    const float target_negative = 1.0F / (1.0F + expf(-((-s - target_offset) / target_scaling)));
    const float target_wrm = 0.5F * (1.0F + target_positive - target_negative);
    const float weight_base = (target_wrm - 0.5F) * (target_wrm - 0.5F)
        * target_wrm * (1.0F - target_wrm);
    const float weight = 1.0F
        + (powf(2.0F, weight_boost_w1) - 1.0F) * powf(weight_base, weight_boost_w2);
    atomicAdd(weight_sum_accumulator, static_cast<double>(weight));
}

extern "C" __global__ void norm_loss_reduce(
    const float* weight,
    unsigned long long,
    float* norms,
    unsigned long long,
    unsigned int n_groups,
    unsigned int group_pitch,
    unsigned int element_stride,
    unsigned int group_length
) {
    const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= n_groups) {
        return;
    }
    const unsigned long long base = static_cast<unsigned long long>(group) * group_pitch;
    float sum_squared = 0.0F;
    for (unsigned int position = blockIdx.y; position < group_length; position += gridDim.y) {
        const float value = weight[base + static_cast<unsigned long long>(position) * element_stride];
        sum_squared += value * value;
    }
    if (sum_squared != 0.0F) {
        atomicAdd(norms + group, sum_squared);
    }
}

extern "C" __global__ void bias_grad(
    const float* output_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int output_dimension
) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(batch) * output_dimension;
    if (index < count) {
        atomicAdd(bias_gradient + index % output_dimension, output_gradient[index]);
    }
}

extern "C" __global__ void norm_loss_finalize(
    float* norms,
    unsigned long long,
    unsigned int n_groups
) {
    const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group < n_groups) {
        norms[group] = sqrtf(norms[group]);
    }
}

extern "C" __global__ void norm_loss_finalize_f64(
    const double* sum_squared,
    unsigned long long,
    float* norms,
    unsigned long long,
    unsigned int n_groups
) {
    const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group < n_groups) {
        norms[group] = static_cast<float>(sqrt(sum_squared[group]));
    }
}

extern "C" __global__ void norm_loss_apply(
    float* weight,
    unsigned long long,
    const float* norms,
    unsigned long long,
    float factor,
    float learning_rate,
    float epsilon,
    unsigned int n_groups,
    unsigned int group_pitch,
    unsigned int element_stride,
    unsigned int group_length
) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(n_groups) * group_length;
    if (index >= count) {
        return;
    }
    unsigned int group;
    unsigned int position;
    if (group_pitch == 1) {
        group = static_cast<unsigned int>(index % n_groups);
        position = static_cast<unsigned int>(index / n_groups);
    } else {
        group = static_cast<unsigned int>(index / group_length);
        position = static_cast<unsigned int>(index % group_length);
    }
    const unsigned long long offset = static_cast<unsigned long long>(group) * group_pitch
        + static_cast<unsigned long long>(position) * element_stride;
    const float correction = 2.0F * factor * (1.0F - 1.0F / (norms[group] + epsilon));
    weight[offset] *= 1.0F - learning_rate * correction;
}

__device__ __forceinline__ float native_clamp_unit(float value) {
    return value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
}

__device__ __forceinline__ float native_clamp_half(float value, unsigned long long* clamps) {
    if (value > 65504.0F) {
        atomicAdd(clamps, 1ULL);
        return 65504.0F;
    }
    if (value < -65504.0F) {
        atomicAdd(clamps, 1ULL);
        return -65504.0F;
    }
    return value;
}

extern "C" __global__ void cast_f32_to_f16(
    const float* source,
    unsigned long long,
    __half* destination,
    unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        destination[i] = __float2half_rn(source[i]);
    }
}

template <typename Output>
__device__ __forceinline__ void native_sparse_ft_forward_fp16(
    const __half* weight,
    const int* indices,
    const int* nonzero_counts,
    Output* output,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    const unsigned int rows_quarter = rows / 4;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(batch) * rows_quarter;
    if (index >= count) {
        return;
    }
    const unsigned int batch_index = index / rows_quarter;
    const unsigned int row = (index % rows_quarter) * 4;
    float sums[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const unsigned int index_base = batch_index * max_active;
    const int row_nonzero_count = nonzero_counts[batch_index];
    for (int active = 0; active < row_nonzero_count; ++active) {
        const int column = indices[index_base + static_cast<unsigned int>(active)];
        if (column >= 0 && static_cast<unsigned int>(column) < columns) {
            const unsigned long long weight_base =
                static_cast<unsigned long long>(column) * rows + row;
#pragma unroll
            for (unsigned int lane = 0; lane < 4; ++lane) {
                sums[lane] += __half2float(weight[weight_base + lane]);
            }
        }
    }
    const unsigned long long output_base =
        static_cast<unsigned long long>(batch_index) * rows + row;
#pragma unroll
    for (unsigned int lane = 0; lane < 4; ++lane) {
        output[output_base + lane] = sums[lane];
    }
}

template <>
__device__ __forceinline__ void native_sparse_ft_forward_fp16<__half>(
    const __half* weight,
    const int* indices,
    const int* nonzero_counts,
    __half* output,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    const unsigned int rows_quarter = rows / 4;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(batch) * rows_quarter;
    if (index >= count) {
        return;
    }
    const unsigned int batch_index = index / rows_quarter;
    const unsigned int row = (index % rows_quarter) * 4;
    float sums[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const unsigned int index_base = batch_index * max_active;
    const int row_nonzero_count = nonzero_counts[batch_index];
    for (int active = 0; active < row_nonzero_count; ++active) {
        const int column = indices[index_base + static_cast<unsigned int>(active)];
        if (column >= 0 && static_cast<unsigned int>(column) < columns) {
            const unsigned long long weight_base =
                static_cast<unsigned long long>(column) * rows + row;
#pragma unroll
            for (unsigned int lane = 0; lane < 4; ++lane) {
                sums[lane] += __half2float(weight[weight_base + lane]);
            }
        }
    }
    const unsigned long long output_base =
        static_cast<unsigned long long>(batch_index) * rows + row;
#pragma unroll
    for (unsigned int lane = 0; lane < 4; ++lane) {
        output[output_base + lane] = __float2half_rn(sums[lane]);
    }
}

extern "C" __global__ void sparse_ft_forward_fp16(
    const __half* weight,
    unsigned long long,
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    native_sparse_ft_forward_fp16(
        weight, indices, nonzero_counts, output, batch, rows, columns, max_active
    );
}

extern "C" __global__ void sparse_ft_forward_fp16_out(
    const __half* weight,
    unsigned long long,
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    __half* output,
    unsigned long long,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    native_sparse_ft_forward_fp16(
        weight, indices, nonzero_counts, output, batch, rows, columns, max_active
    );
}

// Sort key that groups positions by king bucket for `sparse_ft_forward*_ordered`. Entry `i < batch`
// is the stm key of position `i`, entry `batch + i` its nstm key offset by `king_buckets`, so one
// counting sort over `2 * king_buckets` bins yields both orders back to back. Base features of one
// king bucket occupy `king_stride` consecutive indices (`piece_inputs`, times the effect bucket
// count when effect buckets expand the index), so the bucket is the smallest base feature divided
// by `king_stride`; a position without one goes to the last bucket of its perspective.
extern "C" __global__ void ft_king_group_keys(
    const int* stm_indices,
    unsigned long long,
    const int* nstm_indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    int* keys,
    unsigned long long,
    unsigned int batch,
    unsigned int max_active,
    unsigned int king_stride,
    unsigned int king_buckets
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2U * batch) {
        return;
    }
    const unsigned int perspective = i >= batch ? 1U : 0U;
    const unsigned int position = i - perspective * batch;
    const int* indices = perspective == 0U ? stm_indices : nstm_indices;
    const unsigned int base_columns = king_stride * king_buckets;
    const unsigned int index_base = position * max_active;
    const int count = nonzero_counts[position];
    unsigned int smallest = base_columns;
    for (int active = 0; active < count; ++active) {
        const int column = indices[index_base + static_cast<unsigned int>(active)];
        if (column >= 0 && static_cast<unsigned int>(column) < smallest) {
            smallest = static_cast<unsigned int>(column);
        }
    }
    const unsigned int group =
        smallest < base_columns ? smallest / king_stride : king_buckets - 1U;
    keys[i] = static_cast<int>(perspective * king_buckets + group);
}

// Loads 4 consecutive FT weights as one vector access. `source` must be aligned to 4 elements.
__device__ __forceinline__ void ft_forward_load4(const float* source, float* values) {
    const float4 packed = *reinterpret_cast<const float4*>(source);
    values[0] = packed.x;
    values[1] = packed.y;
    values[2] = packed.z;
    values[3] = packed.w;
}

// The four `__half` bit patterns arrive as one 8-byte integer load and are converted with
// `__ushort_as_half`, so no integer object is read through a `__half2` pointer.
__device__ __forceinline__ void ft_forward_load4(const __half* source, float* values) {
    const uint2 packed = *reinterpret_cast<const uint2*>(source);
    values[0] = __half2float(__ushort_as_half(static_cast<unsigned short>(packed.x & 0xffffU)));
    values[1] = __half2float(__ushort_as_half(static_cast<unsigned short>(packed.x >> 16)));
    values[2] = __half2float(__ushort_as_half(static_cast<unsigned short>(packed.y & 0xffffU)));
    values[3] = __half2float(__ushort_as_half(static_cast<unsigned short>(packed.y >> 16)));
}

__device__ __forceinline__ void ft_forward_store(float* destination, float value) {
    *destination = value;
}

__device__ __forceinline__ void ft_forward_store(__half* destination, float value) {
    *destination = __float2half_rn(value);
}

// `sparse_ft_forward*` with positions visited in `order[order_base + j] - order_base` order
// instead of `j`. Each position's sum and output row are the same as the unordered kernels; only
// which positions run concurrently changes. The base features of one king bucket form a
// contiguous block of weight rows, so positions running together under this order draw their rows
// from a few such blocks rather than from the whole table, which gives L2 a chance to reuse them.
template <typename Weight, typename Output>
__device__ __forceinline__ void native_sparse_ft_forward_ordered(
    const Weight* weight,
    const int* indices,
    const int* nonzero_counts,
    const int* order,
    unsigned int order_base,
    Output* output,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    const unsigned int rows_quarter = rows / 4;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(batch) * rows_quarter;
    if (index >= count) {
        return;
    }
    const unsigned int batch_index =
        static_cast<unsigned int>(order[order_base + index / rows_quarter]) - order_base;
    const unsigned int row = (index % rows_quarter) * 4;
    float sums[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const unsigned int index_base = batch_index * max_active;
    const int row_nonzero_count = nonzero_counts[batch_index];
    for (int active = 0; active < row_nonzero_count; ++active) {
        const int column = indices[index_base + static_cast<unsigned int>(active)];
        if (column >= 0 && static_cast<unsigned int>(column) < columns) {
            // `rows` and `row` are multiples of 4, so `weight_base` keeps the vector load aligned.
            const unsigned long long weight_base =
                static_cast<unsigned long long>(column) * rows + row;
            float values[4];
            ft_forward_load4(weight + weight_base, values);
#pragma unroll
            for (unsigned int lane = 0; lane < 4; ++lane) {
                sums[lane] += values[lane];
            }
        }
    }
    const unsigned long long output_base =
        static_cast<unsigned long long>(batch_index) * rows + row;
#pragma unroll
    for (unsigned int lane = 0; lane < 4; ++lane) {
        ft_forward_store(output + output_base + lane, sums[lane]);
    }
}

extern "C" __global__ void sparse_ft_forward_ordered(
    const float* weight,
    unsigned long long,
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    const int* order,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int order_base,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    native_sparse_ft_forward_ordered(
        weight, indices, nonzero_counts, order, order_base, output, batch, rows, columns,
        max_active
    );
}

extern "C" __global__ void sparse_ft_forward_fp16_ordered(
    const __half* weight,
    unsigned long long,
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    const int* order,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int order_base,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    native_sparse_ft_forward_ordered(
        weight, indices, nonzero_counts, order, order_base, output, batch, rows, columns,
        max_active
    );
}

extern "C" __global__ void sparse_ft_forward_fp16_out_ordered(
    const __half* weight,
    unsigned long long,
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    const int* order,
    unsigned long long,
    __half* output,
    unsigned long long,
    unsigned int order_base,
    unsigned int batch,
    unsigned int rows,
    unsigned int columns,
    unsigned int max_active
) {
    native_sparse_ft_forward_ordered(
        weight, indices, nonzero_counts, order, order_base, output, batch, rows, columns,
        max_active
    );
}

template <bool Squared>
__device__ __forceinline__ void native_simple_bias_act_fwd_fp16(
    const __half* ft_output,
    const float* bias,
    float* combined,
    unsigned int combined_stride,
    unsigned int column_offset,
    unsigned int batch,
    unsigned int ft_dimension
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(batch) * ft_dimension;
    if (index >= count) {
        return;
    }
    const unsigned int row = index % ft_dimension;
    const unsigned int batch_index = index / ft_dimension;
    const float activated = native_clamp_unit(__half2float(ft_output[index]) + bias[row]);
    combined[static_cast<unsigned long long>(batch_index) * combined_stride
        + column_offset + row] = Squared ? activated * activated : activated;
}

extern "C" __global__ void simple_bias_act_fwd_fp16_in_crelu(
    const __half* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* combined,
    unsigned long long,
    unsigned int combined_stride,
    unsigned int column_offset,
    unsigned int batch,
    unsigned int ft_dimension
) {
    native_simple_bias_act_fwd_fp16<false>(
        ft_output, bias, combined, combined_stride, column_offset, batch, ft_dimension
    );
}

extern "C" __global__ void simple_bias_act_fwd_fp16_in_screlu(
    const __half* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* combined,
    unsigned long long,
    unsigned int combined_stride,
    unsigned int column_offset,
    unsigned int batch,
    unsigned int ft_dimension
) {
    native_simple_bias_act_fwd_fp16<true>(
        ft_output, bias, combined, combined_stride, column_offset, batch, ft_dimension
    );
}

extern "C" __global__ void ft_post_perspective_fwd_fp16(
    const __half* stm_output,
    unsigned long long,
    const __half* nstm_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* combined,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_dimension,
    float scale
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(batch) * ft_dimension;
    if (index >= count) {
        return;
    }
    const unsigned int batch_index = index / ft_dimension;
    const unsigned int row = index % ft_dimension;
    const unsigned int half = ft_dimension / 2;
    const __half* source = row < half ? stm_output : nstm_output;
    const unsigned int pair = row < half ? row : row - half;
    const unsigned long long base = static_cast<unsigned long long>(batch_index) * ft_dimension;
    const float first = native_clamp_unit(__half2float(source[base + pair]) + bias[pair]);
    const float second = native_clamp_unit(
        __half2float(source[base + half + pair]) + bias[half + pair]
    );
    combined[index] = first * second * scale;
}

template <bool Squared>
__device__ __forceinline__ void native_simple_act_grad_to_fp16(
    const __half* ft_output,
    const float* bias,
    const float* combined_gradient,
    unsigned int combined_stride,
    unsigned int column_offset,
    __half* ft_gradient,
    unsigned long long* clamp_counter,
    unsigned int batch,
    unsigned int ft_dimension,
    float gradient_scale
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(batch) * ft_dimension;
    if (index >= count) {
        return;
    }
    const unsigned int row = index % ft_dimension;
    const unsigned int batch_index = index / ft_dimension;
    const float x = __half2float(ft_output[index]) + bias[row];
    const float activated = native_clamp_unit(x);
    const float derivative = Squared
        ? (activated > 0.0F && activated < 1.0F ? 2.0F * activated : 0.0F)
        : (x > 0.0F && x < 1.0F ? 1.0F : 0.0F);
    const float upstream = combined_gradient[
        static_cast<unsigned long long>(batch_index) * combined_stride + column_offset + row
    ];
    const float scaled = upstream * derivative * gradient_scale;
    ft_gradient[index] = __float2half_rn(native_clamp_half(scaled, clamp_counter));
}

extern "C" __global__ void simple_act_grad_to_fp16_crelu_with_scale(
    const __half* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    const float* combined_gradient,
    unsigned long long,
    unsigned int combined_stride,
    unsigned int column_offset,
    __half* ft_gradient,
    unsigned long long,
    unsigned long long* clamp_counter,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_dimension,
    float gradient_scale
) {
    native_simple_act_grad_to_fp16<false>(
        ft_output, bias, combined_gradient, combined_stride, column_offset, ft_gradient,
        clamp_counter, batch, ft_dimension, gradient_scale
    );
}

extern "C" __global__ void simple_act_grad_to_fp16_screlu_with_scale(
    const __half* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    const float* combined_gradient,
    unsigned long long,
    unsigned int combined_stride,
    unsigned int column_offset,
    __half* ft_gradient,
    unsigned long long,
    unsigned long long* clamp_counter,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_dimension,
    float gradient_scale
) {
    native_simple_act_grad_to_fp16<true>(
        ft_output, bias, combined_gradient, combined_stride, column_offset, ft_gradient,
        clamp_counter, batch, ft_dimension, gradient_scale
    );
}

extern "C" __global__ void ft_post_perspective_grad_fp16(
    const float* combined_gradient,
    unsigned long long,
    const __half* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    __half* ft_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned long long* clamp_counter,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_dimension,
    unsigned int combined_offset,
    unsigned int combined_stride,
    float scale,
    float gradient_scale
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = ft_dimension / 2;
    const unsigned long long count = static_cast<unsigned long long>(batch) * half;
    if (index >= count) {
        return;
    }
    const unsigned int batch_index = index / half;
    const unsigned int pair = index % half;
    const float upstream = combined_gradient[
        static_cast<unsigned long long>(batch_index) * combined_stride + combined_offset + pair
    ];
    const unsigned long long base = static_cast<unsigned long long>(batch_index) * ft_dimension;
    const float x_first = __half2float(ft_output[base + pair]) + bias[pair];
    const float x_second = __half2float(ft_output[base + half + pair]) + bias[half + pair];
    const float first = native_clamp_unit(x_first);
    const float second = native_clamp_unit(x_second);
    const float first_gradient = x_first > 0.0F && x_first < 1.0F
        ? upstream * second * scale : 0.0F;
    const float second_gradient = x_second > 0.0F && x_second < 1.0F
        ? upstream * first * scale : 0.0F;
    ft_gradient[base + pair] = __float2half_rn(native_clamp_half(
        first_gradient * gradient_scale, clamp_counter
    ));
    ft_gradient[base + half + pair] = __float2half_rn(native_clamp_half(
        second_gradient * gradient_scale, clamp_counter
    ));
    atomicAdd(bias_gradient + pair, first_gradient);
    atomicAdd(bias_gradient + half + pair, second_gradient);
}

extern "C" __global__ void simple_bias_grad_dual_fp16(
    const __half* stm_gradient,
    unsigned long long,
    const __half* nstm_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_dimension,
    float inverse_gradient_scale,
    unsigned int items
) {
    const unsigned int output = blockIdx.y * blockDim.x + threadIdx.x;
    if (output >= ft_dimension) {
        return;
    }
    const unsigned int start = blockIdx.x * items;
    const unsigned int end = min(start + items, batch);
    float sum = 0.0F;
    for (unsigned int position = start; position < end; ++position) {
        const unsigned long long index =
            static_cast<unsigned long long>(position) * ft_dimension + output;
        sum += __half2float(stm_gradient[index]) * inverse_gradient_scale
            + __half2float(nstm_gradient[index]) * inverse_gradient_scale;
    }
    atomicAdd(bias_gradient + output, sum);
}

// Sums the f16 gradients of `Columns` adjacent columns `row .. row + Columns` of one feature.
// Each column keeps its own four interleaved partial sums, so the result per cell does not depend
// on `Columns`; `Columns == 2` reads both halves with one 4-byte load.
template <bool Add, unsigned int Columns>
__device__ __forceinline__ void gather_feature_gradient_fp16_columns(
    const __half* output_gradient,
    const unsigned int* positions,
    unsigned int start,
    unsigned int end,
    float* weight_gradient,
    unsigned int feature,
    unsigned int row,
    unsigned int ft_output,
    float inverse_gradient_scale
) {
    float sums[4][Columns] = {};
    auto add_position = [&](unsigned int slot, unsigned int index) {
        const __half* source =
            output_gradient + static_cast<unsigned long long>(positions[index]) * ft_output + row;
        if (Columns == 2) {
            const unsigned int packed = *reinterpret_cast<const unsigned int*>(source);
            sums[slot][0] +=
                __half2float(__ushort_as_half(static_cast<unsigned short>(packed & 0xffffU)));
            sums[slot][1] += __half2float(__ushort_as_half(static_cast<unsigned short>(packed >> 16)));
        } else {
            sums[slot][0] += __half2float(source[0]);
        }
    };
    unsigned int i = start;
    const unsigned int unroll_end = end >= start + 3 ? end - 3 : start;
    while (i < unroll_end) {
        add_position(0, i);
        add_position(1, i + 1);
        add_position(2, i + 2);
        add_position(3, i + 3);
        i += 4;
    }
    while (i < end) {
        add_position(0, i);
        ++i;
    }
#pragma unroll
    for (unsigned int column = 0; column < Columns; ++column) {
        const float sum = (sums[0][column] + sums[1][column]) + (sums[2][column] + sums[3][column]);
        const unsigned long long output_index =
            static_cast<unsigned long long>(feature) * ft_output + row + column;
        const float scaled = sum * inverse_gradient_scale;
        if (Add) {
            if (sum != 0.0F) {
                atomicAdd(weight_gradient + output_index, scaled);
            }
        } else {
            weight_gradient[output_index] = scaled;
        }
    }
}

// Each thread covers `ft_output / (gridDim.y * blockDim.x)` adjacent columns: 2 when the grid
// spans the columns with half as many threads (the native launch, 64 x 2 blocks over
// `ft_output / 128` slices), otherwise 1.
template <bool Add>
__device__ __forceinline__ void native_gather_and_sum_per_feature_fp16(
    const __half* output_gradient,
    const unsigned int* positions,
    const unsigned int* offsets,
    float* weight_gradient,
    const unsigned int* active_rows,
    unsigned int n_features,
    unsigned int ft_output,
    float inverse_gradient_scale
) {
    const bool pairs = ft_output / (gridDim.y * blockDim.x) == 2U;
    const unsigned int feature = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int row = (blockIdx.y * blockDim.x + threadIdx.x) * (pairs ? 2U : 1U);
    if (feature >= n_features || row >= ft_output) {
        return;
    }
    if (active_rows != nullptr && active_rows[feature] == 0U) {
        return;
    }
    const unsigned int start = offsets[feature];
    const unsigned int end = offsets[feature + 1];
    if (pairs) {
        gather_feature_gradient_fp16_columns<Add, 2>(
            output_gradient, positions, start, end, weight_gradient, feature, row, ft_output,
            inverse_gradient_scale
        );
    } else {
        gather_feature_gradient_fp16_columns<Add, 1>(
            output_gradient, positions, start, end, weight_gradient, feature, row, ft_output,
            inverse_gradient_scale
        );
    }
}

extern "C" __global__ void gather_and_sum_per_feature_overwrite_fp16(
    const __half* output_gradient,
    unsigned long long,
    const unsigned int* positions,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int n_features,
    unsigned int ft_output,
    float inverse_gradient_scale
) {
    native_gather_and_sum_per_feature_fp16<false>(
        output_gradient, positions, offsets, weight_gradient, nullptr, n_features, ft_output,
        inverse_gradient_scale
    );
}

extern "C" __global__ void gather_and_sum_per_feature_overwrite_fp16_active(
    const __half* output_gradient,
    unsigned long long,
    const unsigned int* positions,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    const unsigned int* active_rows,
    unsigned long long,
    unsigned int n_features,
    unsigned int ft_output,
    float inverse_gradient_scale
) {
    native_gather_and_sum_per_feature_fp16<false>(
        output_gradient, positions, offsets, weight_gradient, active_rows, n_features, ft_output,
        inverse_gradient_scale
    );
}

extern "C" __global__ void gather_and_sum_per_feature_add_fp16(
    const __half* output_gradient,
    unsigned long long,
    const unsigned int* positions,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int n_features,
    unsigned int ft_output,
    float inverse_gradient_scale
) {
    native_gather_and_sum_per_feature_fp16<true>(
        output_gradient, positions, offsets, weight_gradient, nullptr, n_features, ft_output,
        inverse_gradient_scale
    );
}

template <bool HalfState, bool Mirror, bool Observe = false>
__device__ __forceinline__ void native_radam_step_fp16(
    float* weights,
    void* momentum_storage,
    void* velocity_storage,
    float* gradient,
    __half* mirror,
    float learning_rate,
    float step_size,
    int use_variance_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    float momentum_scale,
    float velocity_scale,
    unsigned int n,
    const unsigned int* sample_indices = nullptr,
    unsigned int sample_count = 0,
    float* observations = nullptr
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float* record = nullptr;
    if (Observe) {
        unsigned int lo = 0, hi = sample_count;
        while (lo < hi) {
            const unsigned int mid = lo + (hi - lo) / 2;
            if (sample_indices[mid] < i) lo = mid + 1;
            else hi = mid;
        }
        if (lo < sample_count && sample_indices[lo] == i) {
            record = observations + lo * 13;
            record[0] = HalfState ? __half2float(static_cast<__half*>(momentum_storage)[i])
                                 : static_cast<float*>(momentum_storage)[i];
            record[3] = HalfState ? __half2float(static_cast<__half*>(velocity_storage)[i])
                                 : static_cast<float*>(velocity_storage)[i];
            record[6] = weights[i];
            record[8] = gradient[i];
        }
    }
    const float g = gradient[i];
    const float rate = learning_rate * step_size;
    float p = weights[i] * (1.0F - decay * rate);
    float momentum;
    float velocity;
    if (HalfState) {
        __half* momentum_half = static_cast<__half*>(momentum_storage);
        __half* velocity_half = static_cast<__half*>(velocity_storage);
        const float previous_momentum = __half2float(momentum_half[i]) / momentum_scale;
        const float previous_velocity = __half2float(velocity_half[i]) / velocity_scale;
        momentum = beta1 * previous_momentum + (1.0F - beta1) * g;
        velocity = beta2 * previous_velocity + (1.0F - beta2) * g * g;
        float stored_momentum = momentum * momentum_scale;
        stored_momentum = stored_momentum > 65504.0F
            ? 65504.0F : (stored_momentum < -65504.0F ? -65504.0F : stored_momentum);
        float stored_velocity = velocity * velocity_scale;
        // Comparison-based clamping preserves NaN like the cuda-oxide if-else lowering.
        stored_velocity = stored_velocity > 65504.0F ? 65504.0F : stored_velocity;
        momentum_half[i] = __float2half_rn(stored_momentum);
        velocity_half[i] = __float2half_rn(stored_velocity);
    } else {
        float* momentum_float = static_cast<float*>(momentum_storage);
        float* velocity_float = static_cast<float*>(velocity_storage);
        momentum = beta1 * momentum_float[i] + (1.0F - beta1) * g;
        velocity = beta2 * velocity_float[i] + (1.0F - beta2) * g * g;
        momentum_float[i] = momentum;
        velocity_float[i] = velocity;
    }
    float value = momentum;
    if (use_variance_denom != 0) {
        value /= sqrtf(velocity) + epsilon;
    }
    p -= rate * value;
    p = p < min_weight ? min_weight : (p > max_weight ? max_weight : p);
    weights[i] = p;
    if (Observe && record != nullptr) {
        record[1] = momentum * momentum_scale;
        record[2] = HalfState ? __half2float(static_cast<__half*>(momentum_storage)[i]) : momentum;
        record[4] = velocity * velocity_scale;
        record[5] = HalfState ? __half2float(static_cast<__half*>(velocity_storage)[i]) : velocity;
        record[7] = p;
    }
    if (Mirror) {
        mirror[i] = __float2half_rn(p);
    }
}

extern "C" __global__ void precision_radam_step(
    float* weights, unsigned long long,
    void* momentum, unsigned long long,
    void* velocity, unsigned long long,
    float* gradient, unsigned long long,
    __half* mirror, unsigned long long,
    const unsigned int* indices, unsigned long long,
    float* observations, unsigned long long,
    float learning_rate, float step_size, int denom, float decay, float beta1, float beta2,
    float epsilon, float min_weight, float max_weight, float m_scale, float v_scale,
    unsigned int n, unsigned int samples, int half_state, int write_mirror
) {
    if (half_state) {
        if (write_mirror) {
            native_radam_step_fp16<true, true, true>(weights, momentum, velocity, gradient, mirror,
                learning_rate, step_size, denom, decay, beta1, beta2, epsilon, min_weight, max_weight,
                m_scale, v_scale, n, indices, samples, observations);
        } else {
            native_radam_step_fp16<true, false, true>(weights, momentum, velocity, gradient, mirror,
                learning_rate, step_size, denom, decay, beta1, beta2, epsilon, min_weight, max_weight,
                m_scale, v_scale, n, indices, samples, observations);
        }
    } else {
        if (write_mirror) {
            native_radam_step_fp16<false, true, true>(weights, momentum, velocity, gradient, mirror,
                learning_rate, step_size, denom, decay, beta1, beta2, epsilon, min_weight, max_weight,
                1.0F, 1.0F, n, indices, samples, observations);
        } else {
            native_radam_step_fp16<false, false, true>(weights, momentum, velocity, gradient, mirror,
                learning_rate, step_size, denom, decay, beta1, beta2, epsilon, min_weight, max_weight,
                1.0F, 1.0F, n, indices, samples, observations);
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            // Match the FP32 non-mirror entry point's gradient-clear contract.
            if (i < n) gradient[i] = 0.0F;
        }
    }
}

extern "C" __global__ void precision_snapshot(
    const float* weights, unsigned long long,
    const __half* mirror, unsigned long long,
    const unsigned int* indices, unsigned long long,
    float* observations, unsigned long long,
    unsigned int samples, unsigned int base_elements, int has_mirror, int after
) {
    const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= samples) return;
    const unsigned int i = indices[k];
    observations[k * 13 + (after ? 9 : 12)] = weights[i];
    if (has_mirror && i < base_elements) {
        observations[k * 13 + (after ? 11 : 10)] = __half2float(mirror[i]);
    }
}

extern "C" __global__ void radam_step_fp16_mirror(
    float* weights,
    unsigned long long,
    float* momentum,
    unsigned long long,
    float* velocity,
    unsigned long long,
    float* gradient,
    unsigned long long,
    __half* mirror,
    unsigned long long,
    float learning_rate,
    float step_size,
    int use_variance_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    unsigned int n
) {
    native_radam_step_fp16<false, true>(
        weights, momentum, velocity, gradient, mirror, learning_rate, step_size,
        use_variance_denom, decay, beta1, beta2, epsilon, min_weight, max_weight,
        1.0F, 1.0F, n
    );
}

extern "C" __global__ void radam_step_f16state(
    float* weights,
    unsigned long long,
    __half* momentum,
    unsigned long long,
    __half* velocity,
    unsigned long long,
    float* gradient,
    unsigned long long,
    float learning_rate,
    float step_size,
    int use_variance_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    float momentum_scale,
    float velocity_scale,
    unsigned int n
) {
    native_radam_step_fp16<true, false>(
        weights, momentum, velocity, gradient, nullptr, learning_rate, step_size,
        use_variance_denom, decay, beta1, beta2, epsilon, min_weight, max_weight,
        momentum_scale, velocity_scale, n
    );
}

extern "C" __global__ void radam_step_f16state_mirror(
    float* weights,
    unsigned long long,
    __half* momentum,
    unsigned long long,
    __half* velocity,
    unsigned long long,
    float* gradient,
    unsigned long long,
    __half* mirror,
    unsigned long long,
    float learning_rate,
    float step_size,
    int use_variance_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    float momentum_scale,
    float velocity_scale,
    unsigned int n
) {
    native_radam_step_fp16<true, true>(
        weights, momentum, velocity, gradient, mirror, learning_rate, step_size,
        use_variance_denom, decay, beta1, beta2, epsilon, min_weight, max_weight,
        momentum_scale, velocity_scale, n
    );
}

extern "C" __global__ void ranger_lookahead_lerp_fp16_mirror(
    float* weights,
    unsigned long long,
    float* slow_weights,
    unsigned long long,
    __half* mirror,
    unsigned long long,
    float alpha,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = alpha * weights[i] + (1.0F - alpha) * slow_weights[i];
        weights[i] = value;
        slow_weights[i] = value;
        mirror[i] = __float2half_rn(value);
    }
}

// One pass over FT master weights in `[element_begin, element_end)`: optional norm-loss
// scaling, the RAdam step, optional Ranger lookahead and an optional forward copy. Each element
// repeats the arithmetic of `norm_loss_apply`, `radam_step` / `native_radam_step_fp16`,
// `ranger_lookahead_lerp` and `ft_fold_virtual*` in that order, so the result matches running
// those kernels back to back. `norms` holds the per-column L2 norms of the weights before this
// step. A factorizer comb reads virtual rows from `weights`, so the caller must finish the
// virtual-row range before launching the real-row range with a comb output.
//
// The kernel is a grid-stride loop and the host sizes the grid so that the stride is a multiple
// of `ft_out`: every thread stays on one column. With norm accumulation each thread adds the
// squares of the weights it wrote to `column_sum_squared[column]` (zeroed by the caller), so the
// next step's norms need no separate pass over the weights. The accumulator is `double` because
// each column receives one atomic per thread; `float` atomics in that order lose about an order
// of magnitude more precision than `norm_loss_reduce`.
//
// With the active-row mask, rows below `active_row_count` whose `active_rows` entry is zero take
// a zero gradient without reading `gradient`, and their gradient storage is left untouched: the
// masked backward gather skips them, so they hold stale values. Rows at or above
// `active_row_count` (factorizer virtual rows) always read the gradient.
//
// flags: bit 0 = FP16 moment storage, bit 1 = norm-loss, bit 2 = lookahead,
// bit 3 = active-row mask, bits 4..5 = forward output (0 none, 1 FP16 mirror, 2 FP16 comb,
// 3 FP32 comb), bit 6 = norm accumulation.
extern "C" __global__ void ft_fused_optimizer_step(
    float* weights,
    unsigned long long,
    void* momentum_storage,
    unsigned long long,
    void* velocity_storage,
    unsigned long long,
    float* gradient,
    unsigned long long,
    float* slow_weights,
    unsigned long long,
    const float* norms,
    unsigned long long,
    double* column_sum_squared,
    unsigned long long,
    const unsigned int* active_rows,
    unsigned long long active_row_count,
    void* forward_output,
    unsigned long long,
    const unsigned int* threat_pair_starts,
    unsigned long long threat_pair_starts_len,
    unsigned long long ft_bounds,
    unsigned int ft_out,
    unsigned int piece_inputs,
    unsigned int effect_bucket_factorize,
    unsigned long long element_begin,
    unsigned long long element_end,
    float norm_learning_rate,
    float norm_factor,
    float norm_epsilon,
    float learning_rate,
    float step_size,
    int use_variance_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    float momentum_scale,
    float velocity_scale,
    float lookahead_alpha,
    unsigned int flags
) {
    const bool half_state = (flags & 1U) != 0;
    const unsigned int output_mode = (flags >> 4) & 3U;
    const bool accumulate_norms = (flags & 64U) != 0;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    const unsigned long long first = element_begin
        + static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int column = static_cast<unsigned int>(first % ft_out);
    float thread_sum_squared = 0.0F;
    for (unsigned long long i = first; i < element_end; i += stride) {
        float w = weights[i];
        if ((flags & 2U) != 0) {
            const float correction = 2.0F * norm_factor * (1.0F - 1.0F / (norms[column] + norm_epsilon));
            w *= 1.0F - norm_learning_rate * correction;
        }

        const unsigned long long row = i / ft_out;
        const bool gradient_live = (flags & 8U) == 0 || row >= active_row_count
            || active_rows[row] != 0U;
        const float g = gradient_live ? gradient[i] : 0.0F;
        const float rate = learning_rate * step_size;
        float p = w * (1.0F - decay * rate);
        float momentum;
        float velocity;
        if (half_state) {
            __half* momentum_half = static_cast<__half*>(momentum_storage);
            __half* velocity_half = static_cast<__half*>(velocity_storage);
            const float previous_momentum = __half2float(momentum_half[i]) / momentum_scale;
            const float previous_velocity = __half2float(velocity_half[i]) / velocity_scale;
            momentum = beta1 * previous_momentum + (1.0F - beta1) * g;
            velocity = beta2 * previous_velocity + (1.0F - beta2) * g * g;
            float stored_momentum = momentum * momentum_scale;
            stored_momentum = stored_momentum > 65504.0F
                ? 65504.0F : (stored_momentum < -65504.0F ? -65504.0F : stored_momentum);
            float stored_velocity = velocity * velocity_scale;
            stored_velocity = stored_velocity > 65504.0F ? 65504.0F : stored_velocity;
            momentum_half[i] = __float2half_rn(stored_momentum);
            velocity_half[i] = __float2half_rn(stored_velocity);
        } else {
            float* momentum_float = static_cast<float*>(momentum_storage);
            float* velocity_float = static_cast<float*>(velocity_storage);
            momentum = beta1 * momentum_float[i] + (1.0F - beta1) * g;
            velocity = beta2 * velocity_float[i] + (1.0F - beta2) * g * g;
            momentum_float[i] = momentum;
            velocity_float[i] = velocity;
            // `radam_step` clears the gradient it consumed; the mirror variant
            // `radam_step_fp16_mirror` leaves it in place.
            if (output_mode != 1U && gradient_live) {
                gradient[i] = 0.0F;
            }
        }
        float value = momentum;
        if (use_variance_denom != 0) {
            value /= sqrtf(velocity) + epsilon;
        }
        p -= rate * value;
        p = p < min_weight ? min_weight : (p > max_weight ? max_weight : p);

        if ((flags & 4U) != 0) {
            p = lookahead_alpha * p + (1.0F - lookahead_alpha) * slow_weights[i];
            slow_weights[i] = p;
        }
        weights[i] = p;
        thread_sum_squared += p * p;

        if (output_mode == 1U) {
            static_cast<__half*>(forward_output)[i] = __float2half_rn(p);
        } else if (output_mode >= 2U) {
            const unsigned int base_ft_in = static_cast<unsigned int>(ft_bounds);
            const unsigned int ft_in = static_cast<unsigned int>(ft_bounds >> 32);
            float folded = p;
            const unsigned long long partner = ft_fold_partner_index(
                row, column, threat_pair_starts, threat_pair_starts_len, base_ft_in, ft_in,
                ft_out, piece_inputs, effect_bucket_factorize
            );
            if (partner != ~0ULL) {
                folded += weights[partner];
            }
            if (output_mode == 2U) {
                static_cast<__half*>(forward_output)[i] = __float2half_rn(folded);
            } else {
                static_cast<float*>(forward_output)[i] = folded;
            }
        }
    }
    if (accumulate_norms && thread_sum_squared != 0.0F) {
        atomicAdd(column_sum_squared + column, static_cast<double>(thread_sum_squared));
    }
}

extern "C" __global__ void radam_step(
    float* weights,
    unsigned long long,
    float* momentum,
    unsigned long long,
    float* velocity,
    unsigned long long,
    float* gradient,
    unsigned long long,
    float learning_rate,
    float step_size,
    int use_variance_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float g = gradient[i];
    const float rate = learning_rate * step_size;
    float p = weights[i] * (1.0F - decay * rate);
    const float m = beta1 * momentum[i] + (1.0F - beta1) * g;
    const float v = beta2 * velocity[i] + (1.0F - beta2) * g * g;
    momentum[i] = m;
    velocity[i] = v;
    const float value = use_variance_denom != 0 ? m / (sqrtf(v) + epsilon) : m;
    p -= rate * value;
    p = p < min_weight ? min_weight : (p > max_weight ? max_weight : p);
    weights[i] = p;
    gradient[i] = 0.0F;
}

extern "C" __global__ void crelu_fwd(
    const float* input,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = input[i];
        output[i] = x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : x);
    }
}

extern "C" __global__ void crelu_grad(
    const float* input,
    unsigned long long,
    const float* output_gradient,
    unsigned long long,
    float* input_gradient,
    unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = input[i];
        input_gradient[i] = x > 0.0F && x < 1.0F ? output_gradient[i] : 0.0F;
    }
}

extern "C" __global__ void screlu_fwd(
    const float* input,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = input[i];
        const float clipped = x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : x);
        output[i] = clipped * clipped;
    }
}

extern "C" __global__ void screlu_grad(
    const float* input,
    unsigned long long,
    const float* output_gradient,
    unsigned long long,
    float* input_gradient,
    unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = input[i];
        const float clipped = x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : x);
        const float derivative = clipped > 0.0F && clipped < 1.0F ? 2.0F * clipped : 0.0F;
        input_gradient[i] = output_gradient[i] * derivative;
    }
}

extern "C" __global__ void bias_add_per_row(
    const float* bias,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int columns
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * columns) {
        output[i] += bias[i % columns];
    }
}

extern "C" __global__ void simple_ft_post_fused_crelu(
    float* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* combined,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_columns,
    unsigned int destination_offset
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * ft_columns) {
        return;
    }
    const unsigned int row = i / ft_columns;
    const unsigned int column = i % ft_columns;
    const float value = ft_output[i] + bias[column];
    ft_output[i] = value;
    combined[row * (2 * ft_columns) + destination_offset + column] =
        value <= 0.0F ? 0.0F : (value >= 1.0F ? 1.0F : value);
}

extern "C" __global__ void simple_ft_post_fused_screlu(
    float* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* combined,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_columns,
    unsigned int destination_offset
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * ft_columns) {
        return;
    }
    const unsigned int row = i / ft_columns;
    const unsigned int column = i % ft_columns;
    const float value = ft_output[i] + bias[column];
    ft_output[i] = value;
    const float clipped = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
    combined[row * (2 * ft_columns) + destination_offset + column] = clipped * clipped;
}

extern "C" __global__ void simple_bwd_ft_act_crelu_fused(
    const float* combined_gradient,
    unsigned long long,
    const float* ft_pre_activation,
    unsigned long long,
    float* ft_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_columns,
    unsigned int source_offset
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * ft_columns) {
        return;
    }
    const unsigned int row = i / ft_columns;
    const unsigned int column = i % ft_columns;
    const float x = ft_pre_activation[i];
    ft_gradient[i] = x > 0.0F && x < 1.0F
        ? combined_gradient[row * (2 * ft_columns) + source_offset + column]
        : 0.0F;
}

extern "C" __global__ void simple_bwd_ft_act_screlu_fused(
    const float* combined_gradient,
    unsigned long long,
    const float* ft_pre_activation,
    unsigned long long,
    float* ft_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_columns,
    unsigned int source_offset
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * ft_columns) {
        return;
    }
    const unsigned int row = i / ft_columns;
    const unsigned int column = i % ft_columns;
    const float x = ft_pre_activation[i];
    const float clipped = x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : x);
    const float derivative = clipped > 0.0F && clipped < 1.0F ? 2.0F * clipped : 0.0F;
    ft_gradient[i] =
        combined_gradient[row * (2 * ft_columns) + source_offset + column] * derivative;
}

extern "C" __global__ void ft_post_perspective_fwd(
    const float* stm_ft_output,
    unsigned long long,
    const float* nstm_ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* combined,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_columns,
    float scale
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * ft_columns) {
        return;
    }
    const unsigned int row = i / ft_columns;
    const unsigned int column = i % ft_columns;
    const unsigned int half = ft_columns / 2;
    const unsigned int pair = column < half ? column : column - half;
    const float* ft_output = column < half ? stm_ft_output : nstm_ft_output;
    const unsigned int base = row * ft_columns;
    const float xa = ft_output[base + pair] + bias[pair];
    const float xb = ft_output[base + half + pair] + bias[half + pair];
    const float ya = xa < 0.0F ? 0.0F : (xa > 1.0F ? 1.0F : xa);
    const float yb = xb < 0.0F ? 0.0F : (xb > 1.0F ? 1.0F : xb);
    combined[i] = ya * yb * scale;
}

extern "C" __global__ void ft_post_perspective_grad(
    const float* combined_gradient,
    unsigned long long,
    const float* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* ft_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_columns,
    unsigned int combined_offset,
    unsigned int combined_stride,
    float scale
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = ft_columns / 2;
    if (i >= batch * half) {
        return;
    }
    const unsigned int row = i / half;
    const unsigned int pair = i % half;
    const float output_gradient =
        combined_gradient[row * combined_stride + combined_offset + pair];
    const unsigned int base = row * ft_columns;
    const float xa = ft_output[base + pair] + bias[pair];
    const float xb = ft_output[base + half + pair] + bias[half + pair];
    const float ya = xa < 0.0F ? 0.0F : (xa > 1.0F ? 1.0F : xa);
    const float yb = xb < 0.0F ? 0.0F : (xb > 1.0F ? 1.0F : xb);
    const float gradient_a = xa > 0.0F && xa < 1.0F
        ? output_gradient * yb * scale
        : 0.0F;
    const float gradient_b = xb > 0.0F && xb < 1.0F
        ? output_gradient * ya * scale
        : 0.0F;
    ft_gradient[base + pair] = gradient_a;
    ft_gradient[base + half + pair] = gradient_b;
    atomicAdd(bias_gradient + pair, gradient_a);
    atomicAdd(bias_gradient + half + pair, gradient_b);
}

extern "C" __global__ void simple_bias_grad_dual(
    const float* stm_gradient,
    unsigned long long,
    const float* nstm_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_columns,
    unsigned int items
) {
    const unsigned int output = blockIdx.y * blockDim.x + threadIdx.x;
    if (output >= ft_columns) {
        return;
    }
    const unsigned int start = blockIdx.x * items;
    const unsigned int end = min(start + items, batch);
    float sum = 0.0F;
    for (unsigned int row = start; row < end; ++row) {
        const unsigned int i = row * ft_columns + output;
        sum += stm_gradient[i] + nstm_gradient[i];
    }
    atomicAdd(bias_gradient + output, sum);
}

extern "C" __global__ void dense_bias_grad_tiled(
    const float* output_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int output_columns
) {
    __shared__ float partial[256];
    const unsigned int tid = threadIdx.x;
    const unsigned int output = tid % output_columns;
    const unsigned int reducer = tid / output_columns;
    const unsigned int rows_per_iteration = blockDim.x / output_columns;
    const unsigned int stride = gridDim.x * rows_per_iteration;
    unsigned int row = blockIdx.x * rows_per_iteration + reducer;
    float sum = 0.0F;
    while (row < batch) {
        sum += output_gradient[row * output_columns + output];
        row += stride;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = rows_per_iteration / 2; offset >= 1; offset /= 2) {
        if (reducer < offset) {
            partial[tid] += partial[tid + offset * output_columns];
        }
        __syncthreads();
    }
    if (reducer == 0) {
        atomicAdd(bias_gradient + output, partial[tid]);
    }
}

extern "C" __global__ void build_feature_counts(
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    unsigned int* counts,
    unsigned long long,
    unsigned int batch,
    unsigned int max_active,
    unsigned int columns
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * max_active) {
        return;
    }
    const unsigned int row = i / max_active;
    const unsigned int slot = i % max_active;
    if (static_cast<int>(slot) >= nonzero_counts[row]) {
        return;
    }
    const int feature = indices[i];
    if (feature >= 0 && static_cast<unsigned int>(feature) < columns) {
        atomicAdd(counts + feature, 1U);
    }
}

// Sets `active_rows[feature] = 1` for every feature either perspective uses in the batch. The
// caller zeroes `active_rows` first.
extern "C" __global__ void mark_active_feature_rows(
    const int* stm_indices,
    unsigned long long,
    const int* nstm_indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    unsigned int* active_rows,
    unsigned long long,
    unsigned int batch,
    unsigned int max_active,
    unsigned int columns
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * max_active) {
        return;
    }
    if (static_cast<int>(i % max_active) >= nonzero_counts[i / max_active]) {
        return;
    }
    const int stm_feature = stm_indices[i];
    if (stm_feature >= 0 && static_cast<unsigned int>(stm_feature) < columns) {
        atomicOr(active_rows + stm_feature, 1U);
    }
    const int nstm_feature = nstm_indices[i];
    if (nstm_feature >= 0 && static_cast<unsigned int>(nstm_feature) < columns) {
        atomicOr(active_rows + nstm_feature, 1U);
    }
}

extern "C" __global__ void prefix_sum_block_local(
    const unsigned int* counts,
    unsigned long long,
    unsigned int* offsets,
    unsigned long long,
    unsigned int* block_sums,
    unsigned long long,
    unsigned int n
) {
    __shared__ unsigned int partial[1024];
    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + tid;
    partial[tid] = index < n ? counts[index] : 0U;
    __syncthreads();
    for (unsigned int step = 1; step < blockDim.x; step <<= 1) {
        const unsigned int add = tid >= step ? partial[tid - step] : 0U;
        __syncthreads();
        partial[tid] += add;
        __syncthreads();
    }
    if (index < n) {
        offsets[index] = tid == 0 ? 0U : partial[tid - 1];
    }
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = partial[tid];
    }
}

extern "C" __global__ void exclusive_prefix_sum_small(
    const unsigned int* counts,
    unsigned long long,
    unsigned int* offsets,
    unsigned long long,
    unsigned int n
) {
    __shared__ unsigned int partial[1024];
    const unsigned int tid = threadIdx.x;
    const unsigned int chunk = (n + blockDim.x - 1) / blockDim.x;
    const unsigned int start = tid * chunk;
    const unsigned int end = min(start + chunk, n);
    unsigned int local_sum = 0;
    for (unsigned int i = start; i < end; ++i) {
        local_sum += counts[i];
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (unsigned int step = 1; step < blockDim.x; step <<= 1) {
        const unsigned int add = tid >= step ? partial[tid - step] : 0U;
        __syncthreads();
        partial[tid] += add;
        __syncthreads();
    }
    unsigned int sum = tid == 0 ? 0U : partial[tid - 1];
    __syncthreads();
    for (unsigned int i = start; i < end; ++i) {
        offsets[i] = sum;
        sum += counts[i];
    }
    if (tid == blockDim.x - 1) {
        offsets[n] = sum;
    }
}

extern "C" __global__ void prefix_sum_add_block_offset(
    unsigned int* offsets,
    unsigned long long,
    const unsigned int* block_offsets,
    unsigned long long,
    unsigned int n,
    unsigned int num_blocks
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        offsets[index] += block_offsets[blockIdx.x];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        offsets[n] = block_offsets[num_blocks];
    }
}

extern "C" __global__ void scatter_positions(
    const int* indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    unsigned int* write_counters,
    unsigned long long,
    unsigned int* positions,
    unsigned long long,
    unsigned int batch,
    unsigned int max_active,
    unsigned int columns
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * max_active) {
        return;
    }
    const unsigned int row = i / max_active;
    const unsigned int slot = i % max_active;
    if (static_cast<int>(slot) >= nonzero_counts[row]) {
        return;
    }
    const int feature = indices[i];
    if (feature >= 0 && static_cast<unsigned int>(feature) < columns) {
        const unsigned int rank = atomicAdd(write_counters + feature, 1U);
        positions[offsets[feature] + rank] = row;
    }
}

__device__ void gather_feature_gradient(
    const float* output_gradient,
    const unsigned int* positions,
    const unsigned int* offsets,
    float* weight_gradient,
    const unsigned int* active_rows,
    unsigned int feature_count,
    unsigned int ft_columns,
    bool add
) {
    const unsigned int feature = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int output = blockIdx.y * blockDim.x + threadIdx.x;
    if (feature >= feature_count || output >= ft_columns) {
        return;
    }
    if (active_rows != nullptr && active_rows[feature] == 0U) {
        return;
    }
    const unsigned int start = offsets[feature];
    const unsigned int end = offsets[feature + 1];
    float sums[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    unsigned int i = start;
    const unsigned int unroll_end = end >= start + 3 ? end - 3 : start;
    while (i < unroll_end) {
        sums[0] += output_gradient[positions[i] * ft_columns + output];
        sums[1] += output_gradient[positions[i + 1] * ft_columns + output];
        sums[2] += output_gradient[positions[i + 2] * ft_columns + output];
        sums[3] += output_gradient[positions[i + 3] * ft_columns + output];
        i += 4;
    }
    while (i < end) {
        sums[0] += output_gradient[positions[i] * ft_columns + output];
        ++i;
    }
    const float sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    float* destination = weight_gradient + feature * ft_columns + output;
    if (add) {
        if (sum != 0.0F) {
            atomicAdd(destination, sum);
        }
    } else {
        *destination = sum;
    }
}

extern "C" __global__ void gather_and_sum_per_feature_overwrite(
    const float* output_gradient,
    unsigned long long,
    const unsigned int* positions,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int feature_count,
    unsigned int ft_columns
) {
    gather_feature_gradient(
        output_gradient, positions, offsets, weight_gradient, nullptr, feature_count, ft_columns,
        false
    );
}

// Overwrite gather that leaves rows with `active_rows[feature] == 0` untouched. Only valid when
// every later reader of those rows treats them as zero (see `ft_fused_optimizer_step`).
extern "C" __global__ void gather_and_sum_per_feature_overwrite_active(
    const float* output_gradient,
    unsigned long long,
    const unsigned int* positions,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    const unsigned int* active_rows,
    unsigned long long,
    unsigned int feature_count,
    unsigned int ft_columns
) {
    gather_feature_gradient(
        output_gradient, positions, offsets, weight_gradient, active_rows, feature_count,
        ft_columns, false
    );
}

extern "C" __global__ void gather_and_sum_per_feature_add(
    const float* output_gradient,
    unsigned long long,
    const unsigned int* positions,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int feature_count,
    unsigned int ft_columns
) {
    gather_feature_gradient(
        output_gradient, positions, offsets, weight_gradient, nullptr, feature_count, ft_columns,
        true
    );
}

extern "C" __global__ void ranger_lookahead_lerp(
    float* weights,
    unsigned long long,
    float* slow_weights,
    unsigned long long,
    float alpha,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = alpha * weights[i] + (1.0F - alpha) * slow_weights[i];
        weights[i] = value;
        slow_weights[i] = value;
    }
}

extern "C" __global__ void count_buckets(
    const int* bucket_idx,
    unsigned long long,
    unsigned int* counts,
    unsigned long long,
    unsigned int batch,
    unsigned int num_buckets
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) {
        return;
    }
    const int bucket = bucket_idx[i];
    const unsigned int bin = bucket >= 0 && static_cast<unsigned int>(bucket) < num_buckets
        ? static_cast<unsigned int>(bucket)
        : num_buckets;
    atomicAdd(counts + bin, 1U);
}

extern "C" __global__ void exclusive_scan_aligned(
    const unsigned int* counts,
    unsigned long long,
    unsigned int* offsets,
    unsigned long long,
    unsigned int n,
    unsigned int align
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    unsigned int accumulator = 0;
    for (unsigned int i = 0; i < n; ++i) {
        const unsigned int remainder = accumulator % align;
        if (remainder != 0) {
            accumulator += align - remainder;
        }
        offsets[i] = accumulator;
        accumulator += counts[i];
    }
}

extern "C" __global__ void scatter_bucket_perm(
    const int* bucket_idx,
    unsigned long long,
    const unsigned int* offsets,
    unsigned long long,
    unsigned int* write_counters,
    unsigned long long,
    int* permutation,
    unsigned long long,
    int* sorted_bucket,
    unsigned long long,
    unsigned int batch,
    unsigned int num_buckets
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) {
        return;
    }
    const int bucket = bucket_idx[i];
    const unsigned int bin = bucket >= 0 && static_cast<unsigned int>(bucket) < num_buckets
        ? static_cast<unsigned int>(bucket)
        : num_buckets;
    const unsigned int rank = atomicAdd(write_counters + bin, 1U);
    const unsigned int destination = offsets[bin] + rank;
    permutation[destination] = static_cast<int>(i);
    sorted_bucket[destination] = bucket;
}

extern "C" __global__ void permute_rows_f32(
    const float* input,
    unsigned long long,
    const int* permutation,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int dimension
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / dimension);
    const unsigned int column = static_cast<unsigned int>(i % dimension);
    const int source_row = permutation[row];
    output[i] = source_row >= 0 && static_cast<unsigned int>(source_row) < batch
        ? input[static_cast<unsigned long long>(source_row) * dimension + column]
        : 0.0F;
}

extern "C" __global__ void inverse_permute_rows_f32(
    const float* input,
    unsigned long long,
    const int* permutation,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int dimension
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / dimension);
    const unsigned int column = static_cast<unsigned int>(i % dimension);
    const int destination_row = permutation[row];
    if (destination_row >= 0 && static_cast<unsigned int>(destination_row) < batch) {
        output[static_cast<unsigned long long>(destination_row) * dimension + column] = input[i];
    }
}

extern "C" __global__ void abs_pow2_scale_fwd(
    const float* input,
    unsigned long long,
    float* output,
    unsigned long long,
    float scale,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * input[i] * scale;
    }
}

extern "C" __global__ void abs_pow2_scale_grad(
    const float* input,
    unsigned long long,
    const float* output_gradient,
    unsigned long long,
    float* input_gradient,
    unsigned long long,
    float scale,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        input_gradient[i] = 2.0F * input[i] * scale * output_gradient[i];
    }
}

extern "C" __global__ void concat_l1sqr_main_fwd(
    const float* lhs,
    unsigned long long,
    const float* rhs,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int lhs_dimension,
    unsigned int rhs_dimension
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int output_dimension = lhs_dimension + rhs_dimension;
    const unsigned long long total = static_cast<unsigned long long>(batch) * output_dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / output_dimension);
    const unsigned int column = static_cast<unsigned int>(i % output_dimension);
    output[i] = column < lhs_dimension
        ? lhs[static_cast<unsigned long long>(row) * lhs_dimension + column]
        : rhs[static_cast<unsigned long long>(row) * rhs_dimension + column - lhs_dimension];
}

extern "C" __global__ void concat_l1sqr_main_grad(
    const float* output_gradient,
    unsigned long long,
    float* lhs_gradient,
    unsigned long long,
    float* rhs_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int dimension
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / dimension);
    const unsigned int column = static_cast<unsigned int>(i % dimension);
    const unsigned long long base = static_cast<unsigned long long>(row) * 2U * dimension;
    lhs_gradient[i] = output_gradient[base + column];
    rhs_gradient[i] = output_gradient[base + dimension + column];
}

extern "C" __global__ void bias_add_per_bucket_row(
    const float* bias,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * output_dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / output_dimension);
    const unsigned int column = static_cast<unsigned int>(i % output_dimension);
    const int bucket = bucket_idx[row];
    if (bucket >= 0 && static_cast<unsigned int>(bucket) < num_buckets) {
        output[i] += bias[static_cast<unsigned long long>(bucket) * output_dimension + column];
    }
}

extern "C" __global__ void elementwise_add(
    const float* lhs,
    unsigned long long,
    const float* rhs,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = lhs[i] + rhs[i];
    }
}

extern "C" __global__ void stack_shared_delta_fold(
    const float* bucketed,
    unsigned long long,
    const float* shared,
    unsigned long long,
    float* folded,
    unsigned long long,
    unsigned int num_buckets,
    unsigned int group_len
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total =
        static_cast<unsigned long long>(num_buckets) * group_len;
    if (i < total) {
        folded[i] = bucketed[i] + shared[i % group_len];
    }
}

extern "C" __global__ void stack_shared_delta_reduce_grad(
    const float* bucketed_gradient,
    unsigned long long,
    float* shared_gradient,
    unsigned long long,
    unsigned int num_buckets,
    unsigned int group_len
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= group_len) {
        return;
    }
    float sum = 0.0F;
    for (unsigned int bucket = 0; bucket < num_buckets; ++bucket) {
        sum += bucketed_gradient[static_cast<unsigned long long>(bucket) * group_len + i];
    }
    shared_gradient[i] = sum;
}

extern "C" __global__ void slice_extract_2d(
    const float* source,
    unsigned long long,
    float* destination,
    unsigned long long,
    unsigned int batch,
    unsigned int source_stride,
    unsigned int source_offset,
    unsigned int output_dimension
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * output_dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / output_dimension);
    const unsigned int column = static_cast<unsigned int>(i % output_dimension);
    destination[i] = source[static_cast<unsigned long long>(row) * source_stride + source_offset + column];
}

extern "C" __global__ void slice_scatter_2d(
    const float* source,
    unsigned long long,
    float* destination,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int destination_stride,
    unsigned int destination_offset
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * input_dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / input_dimension);
    const unsigned int column = static_cast<unsigned int>(i % input_dimension);
    destination[static_cast<unsigned long long>(row) * destination_stride + destination_offset + column] =
        source[i];
}

extern "C" __global__ void psqt_diff_sparse_fwd_inplace(
    const float* weights,
    unsigned long long,
    const int* stm_indices,
    unsigned long long,
    const int* nstm_indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* network_output,
    unsigned long long,
    unsigned int batch,
    unsigned int max_active,
    unsigned int num_buckets,
    unsigned int feature_count
) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) {
        return;
    }
    const int bucket = bucket_idx[row];
    if (bucket < 0 || static_cast<unsigned int>(bucket) >= num_buckets) {
        return;
    }
    float stm_sum = 0.0F;
    float nstm_sum = 0.0F;
    const unsigned long long base = static_cast<unsigned long long>(row) * max_active;
    for (int active = 0; active < nonzero_counts[row]; ++active) {
        const int stm = stm_indices[base + static_cast<unsigned int>(active)];
        const int nstm = nstm_indices[base + static_cast<unsigned int>(active)];
        if (stm >= 0 && static_cast<unsigned int>(stm) < feature_count) {
            stm_sum += weights[static_cast<unsigned long long>(stm) * num_buckets + bucket];
        }
        if (nstm >= 0 && static_cast<unsigned int>(nstm) < feature_count) {
            nstm_sum += weights[static_cast<unsigned long long>(nstm) * num_buckets + bucket];
        }
    }
    network_output[row] += 0.5F * (stm_sum - nstm_sum);
}

extern "C" __global__ void psqt_diff_sparse_bwd(
    const float* network_gradient,
    unsigned long long,
    const int* stm_indices,
    unsigned long long,
    const int* nstm_indices,
    unsigned long long,
    const int* nonzero_counts,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int max_active,
    unsigned int num_buckets,
    unsigned int feature_count
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * max_active;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / max_active);
    const unsigned int active = static_cast<unsigned int>(i % max_active);
    if (active >= static_cast<unsigned int>(nonzero_counts[row])) {
        return;
    }
    const int bucket = bucket_idx[row];
    if (bucket < 0 || static_cast<unsigned int>(bucket) >= num_buckets) {
        return;
    }
    const float gradient = 0.5F * network_gradient[row];
    const int stm = stm_indices[i];
    const int nstm = nstm_indices[i];
    if (stm >= 0 && static_cast<unsigned int>(stm) < feature_count) {
        atomicAdd(weight_gradient + static_cast<unsigned long long>(stm) * num_buckets + bucket, gradient);
    }
    if (nstm >= 0 && static_cast<unsigned int>(nstm) < feature_count) {
        atomicAdd(weight_gradient + static_cast<unsigned long long>(nstm) * num_buckets + bucket, -gradient);
    }
}

template <typename Activation>
__device__ __forceinline__ void native_ft_post_perspective_grad_fused(
    const float* combined_gradient_a,
    const float* combined_gradient_b,
    const Activation* ft_output,
    const float* bias,
    Activation* ft_gradient,
    float* bias_gradient,
    unsigned long long* clamp_counter,
    unsigned int batch,
    unsigned int ft_dimension,
    unsigned int combined_offset,
    unsigned int combined_stride,
    float scale,
    float gradient_scale
) {
    const unsigned long long pair =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int half = ft_dimension / 2;
    const unsigned long long total = static_cast<unsigned long long>(batch) * half;
    if (pair >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(pair / half);
    const unsigned int column = static_cast<unsigned int>(pair % half);
    const unsigned long long combined_index =
        static_cast<unsigned long long>(row) * combined_stride + combined_offset + column;
    const float dy = combined_gradient_a[combined_index] + combined_gradient_b[combined_index];
    const unsigned long long base = static_cast<unsigned long long>(row) * ft_dimension;
    float xa;
    float xb;
    if constexpr (sizeof(Activation) == sizeof(float)) {
        xa = ft_output[base + column] + bias[column];
        xb = ft_output[base + half + column] + bias[half + column];
    } else {
        xa = __half2float(ft_output[base + column]) + bias[column];
        xb = __half2float(ft_output[base + half + column]) + bias[half + column];
    }
    const float ya = native_clamp_unit(xa);
    const float yb = native_clamp_unit(xb);
    const float grad_a = xa > 0.0F && xa < 1.0F ? dy * yb * scale : 0.0F;
    const float grad_b = xb > 0.0F && xb < 1.0F ? dy * ya * scale : 0.0F;
    if constexpr (sizeof(Activation) == sizeof(float)) {
        ft_gradient[base + column] = grad_a;
        ft_gradient[base + half + column] = grad_b;
    } else {
        const float scaled_a = grad_a * gradient_scale;
        const float scaled_b = grad_b * gradient_scale;
        const float clamped_a = native_clamp_half(scaled_a, clamp_counter);
        const float clamped_b = native_clamp_half(scaled_b, clamp_counter);
        ft_gradient[base + column] = __float2half_rn(clamped_a);
        ft_gradient[base + half + column] = __float2half_rn(clamped_b);
    }
    atomicAdd(bias_gradient + column, grad_a);
    atomicAdd(bias_gradient + half + column, grad_b);
}

extern "C" __global__ void ft_post_perspective_grad_fused(
    const float* combined_gradient_a,
    unsigned long long,
    const float* combined_gradient_b,
    unsigned long long,
    const float* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* ft_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_dimension,
    unsigned int combined_offset,
    unsigned int combined_stride,
    float scale
) {
    native_ft_post_perspective_grad_fused(
        combined_gradient_a, combined_gradient_b, ft_output, bias, ft_gradient,
        bias_gradient, nullptr, batch, ft_dimension, combined_offset, combined_stride, scale, 1.0F
    );
}

extern "C" __global__ void ft_post_perspective_grad_fused_fp16(
    const float* combined_gradient_a,
    unsigned long long,
    const float* combined_gradient_b,
    unsigned long long,
    const __half* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    __half* ft_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned long long* clamp_counter,
    unsigned long long,
    unsigned int batch,
    unsigned int ft_dimension,
    unsigned int combined_offset,
    unsigned int combined_stride,
    float scale,
    float gradient_scale
) {
    native_ft_post_perspective_grad_fused(
        combined_gradient_a, combined_gradient_b, ft_output, bias, ft_gradient,
        bias_gradient, clamp_counter, batch, ft_dimension, combined_offset, combined_stride, scale,
        gradient_scale
    );
}

// FT activation backward with the L1 input gradient recomputed in place instead of read from
// materialized `dcombined` buffers. For FT column `c = combined_offset + column`:
//   dy = sum_o dl1_sorted[j][o] * l1_w[bucket][o][c]  +  sum_o dl1_sorted[j][o] * shared_w[c][o]
// Each sum runs over `o` in ascending order with one fmaf per term starting from 0, the same
// chain as `dense_mm_bwd_input_bucket_tiled_sorted_scatter` and `dense_mm_bwd_input_tiled_wide`,
// so the result is bit-identical to those kernels followed by the per-element ft_post kernel.
// Thread `x` owns one column and walks `rows_per_block` bucket-sorted rows, so its 2 * l1_out
// weights stay in registers while the bucket is unchanged. `dl1_sorted[j]` is row
// `permutation[j]` of the L1 output gradient; padding rows (`permutation[j] < 0`) are skipped and
// rows of an out-of-range bucket get no bucket term, as in the scatter kernel.
constexpr unsigned int kFtPostL1MaxOutputs = 32;
constexpr unsigned int kFtPostL1MaxRows = 64;

template <typename Activation>
__device__ __forceinline__ void native_ft_post_grad_from_l1(
    const float* l1_output_gradient_sorted,
    const int* permutation,
    const int* bucket_sorted,
    const float* l1_weights,
    const float* shared_weights,
    const Activation* ft_output,
    const float* bias,
    Activation* ft_gradient,
    float* bias_gradient,
    unsigned long long* clamp_counter,
    unsigned int padded_batch,
    unsigned int ft_dimension,
    unsigned int l1_outputs,
    unsigned int num_buckets,
    unsigned int combined_offset,
    unsigned int rows_per_block,
    float scale,
    float gradient_scale
) {
    // The block's rows, buckets and L1 output gradients are staged in shared memory so the row
    // loop below only waits on the FT activation loads.
    __shared__ float gradient_tile[kFtPostL1MaxRows][kFtPostL1MaxOutputs];
    __shared__ int row_tile[kFtPostL1MaxRows];
    __shared__ int bucket_tile[kFtPostL1MaxRows];
    const unsigned int first = blockIdx.y * rows_per_block;
    const unsigned int rows = first < padded_batch ? min(rows_per_block, padded_batch - first) : 0U;
    for (unsigned int i = threadIdx.x; i < rows * l1_outputs; i += blockDim.x) {
        const unsigned int local_row = i / l1_outputs;
        gradient_tile[local_row][i % l1_outputs] =
            l1_output_gradient_sorted[static_cast<unsigned long long>(first) * l1_outputs + i];
    }
    for (unsigned int i = threadIdx.x; i < rows; i += blockDim.x) {
        row_tile[i] = permutation[first + i];
        const int bucket = bucket_sorted[first + i];
        bucket_tile[i] = bucket >= 0 && static_cast<unsigned int>(bucket) < num_buckets ? bucket : -1;
    }
    __syncthreads();
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = ft_dimension / 2;
    if (column >= half) {
        return;
    }
    const unsigned int combined_column = combined_offset + column;
    float shared_w[kFtPostL1MaxOutputs];
    float bucket_w[kFtPostL1MaxOutputs];
#pragma unroll
    for (unsigned int o = 0; o < kFtPostL1MaxOutputs; ++o) {
        bucket_w[o] = 0.0F;
        shared_w[o] = o < l1_outputs
            ? shared_weights[static_cast<unsigned long long>(combined_column) * l1_outputs + o]
            : 0.0F;
    }
    int loaded_bucket = -1;
    float sum_a = 0.0F;
    float sum_b = 0.0F;
    for (unsigned int local_row = 0; local_row < rows; ++local_row) {
        const int row = row_tile[local_row];
        if (row < 0) {
            continue;
        }
        const int bucket = bucket_tile[local_row];
        if (bucket >= 0 && bucket != loaded_bucket) {
#pragma unroll
            for (unsigned int o = 0; o < kFtPostL1MaxOutputs; ++o) {
                bucket_w[o] = o < l1_outputs
                    ? l1_weights[(static_cast<unsigned long long>(bucket) * l1_outputs + o) *
                                     ft_dimension +
                                 combined_column]
                    : 0.0F;
            }
            loaded_bucket = bucket;
        }
        float from_bucket = 0.0F;
        float from_shared = 0.0F;
#pragma unroll
        for (unsigned int o = 0; o < kFtPostL1MaxOutputs; ++o) {
            if (o < l1_outputs) {
                const float g = gradient_tile[local_row][o];
                from_bucket = fmaf(g, bucket_w[o], from_bucket);
                from_shared = fmaf(g, shared_w[o], from_shared);
            }
        }
        if (bucket < 0) {
            from_bucket = 0.0F;
        }
        const float dy = from_bucket + from_shared;
        const unsigned long long base = static_cast<unsigned long long>(row) * ft_dimension;
        float xa;
        float xb;
        if constexpr (sizeof(Activation) == sizeof(float)) {
            xa = ft_output[base + column] + bias[column];
            xb = ft_output[base + half + column] + bias[half + column];
        } else {
            xa = __half2float(ft_output[base + column]) + bias[column];
            xb = __half2float(ft_output[base + half + column]) + bias[half + column];
        }
        const float ya = native_clamp_unit(xa);
        const float yb = native_clamp_unit(xb);
        const float grad_a = xa > 0.0F && xa < 1.0F ? dy * yb * scale : 0.0F;
        const float grad_b = xb > 0.0F && xb < 1.0F ? dy * ya * scale : 0.0F;
        if constexpr (sizeof(Activation) == sizeof(float)) {
            ft_gradient[base + column] = grad_a;
            ft_gradient[base + half + column] = grad_b;
        } else {
            const float clamped_a = native_clamp_half(grad_a * gradient_scale, clamp_counter);
            const float clamped_b = native_clamp_half(grad_b * gradient_scale, clamp_counter);
            ft_gradient[base + column] = __float2half_rn(clamped_a);
            ft_gradient[base + half + column] = __float2half_rn(clamped_b);
        }
        sum_a += grad_a;
        sum_b += grad_b;
    }
    atomicAdd(bias_gradient + column, sum_a);
    atomicAdd(bias_gradient + half + column, sum_b);
}

extern "C" __global__ void ft_post_grad_from_l1(
    const float* l1_output_gradient_sorted,
    unsigned long long,
    const int* permutation,
    unsigned long long,
    const int* bucket_sorted,
    unsigned long long,
    const float* l1_weights,
    unsigned long long,
    const float* shared_weights,
    unsigned long long,
    const float* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    float* ft_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int padded_batch,
    unsigned int ft_dimension,
    unsigned int l1_outputs,
    unsigned int num_buckets,
    unsigned int combined_offset,
    unsigned int rows_per_block,
    float scale
) {
    native_ft_post_grad_from_l1(
        l1_output_gradient_sorted, permutation, bucket_sorted, l1_weights, shared_weights,
        ft_output, bias, ft_gradient, bias_gradient, nullptr, padded_batch, ft_dimension,
        l1_outputs, num_buckets, combined_offset, rows_per_block, scale, 1.0F
    );
}

extern "C" __global__ void ft_post_grad_from_l1_fp16(
    const float* l1_output_gradient_sorted,
    unsigned long long,
    const int* permutation,
    unsigned long long,
    const int* bucket_sorted,
    unsigned long long,
    const float* l1_weights,
    unsigned long long,
    const float* shared_weights,
    unsigned long long,
    const __half* ft_output,
    unsigned long long,
    const float* bias,
    unsigned long long,
    __half* ft_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned long long* clamp_counter,
    unsigned long long,
    unsigned int padded_batch,
    unsigned int ft_dimension,
    unsigned int l1_outputs,
    unsigned int num_buckets,
    unsigned int combined_offset,
    unsigned int rows_per_block,
    float scale,
    float gradient_scale
) {
    native_ft_post_grad_from_l1(
        l1_output_gradient_sorted, permutation, bucket_sorted, l1_weights, shared_weights,
        ft_output, bias, ft_gradient, bias_gradient, clamp_counter, padded_batch, ft_dimension,
        l1_outputs, num_buckets, combined_offset, rows_per_block, scale, gradient_scale
    );
}

extern "C" __global__ void dense_mm_fwd_bucket(
    const float* input,
    unsigned long long,
    const float* weights,
    unsigned long long,
    const float* bias,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * output_dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / output_dimension);
    const unsigned int output_index = static_cast<unsigned int>(i % output_dimension);
    const int bucket = bucket_idx[row];
    if (bucket < 0 || static_cast<unsigned int>(bucket) >= num_buckets) {
        output[i] = 0.0F;
        return;
    }
    const unsigned long long weight_base =
        (static_cast<unsigned long long>(bucket) * output_dimension + output_index) * input_dimension;
    const unsigned long long input_base = static_cast<unsigned long long>(row) * input_dimension;
    float accumulator = bias[static_cast<unsigned long long>(bucket) * output_dimension + output_index];
    for (unsigned int input_index = 0; input_index < input_dimension; ++input_index) {
        accumulator += input[input_base + input_index] * weights[weight_base + input_index];
    }
    output[i] = accumulator;
}

extern "C" __global__ void dense_mm_fwd_bucket_tiled_l1_sorted(
    const float* input,
    unsigned long long,
    const float* weights,
    unsigned long long,
    const float* bias,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* output,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned int local_row = threadIdx.x >> 4;
    const unsigned int local_output = threadIdx.x & 15U;
    const unsigned int row = blockIdx.x * 16U + local_row;
    const unsigned int output_index = blockIdx.y * 16U + local_output;
    if (row >= batch || output_index >= output_dimension) {
        return;
    }
    const int bucket = bucket_idx[blockIdx.x * 16U];
    const unsigned long long output_offset =
        static_cast<unsigned long long>(row) * output_dimension + output_index;
    if (bucket < 0 || static_cast<unsigned int>(bucket) >= num_buckets) {
        output[output_offset] = 0.0F;
        return;
    }
    const unsigned long long input_base = static_cast<unsigned long long>(row) * input_dimension;
    const unsigned long long weight_base =
        (static_cast<unsigned long long>(bucket) * output_dimension + output_index) * input_dimension;
    float accumulator = bias[static_cast<unsigned long long>(bucket) * output_dimension + output_index];
    for (unsigned int input_index = 0; input_index < input_dimension; ++input_index) {
        accumulator += input[input_base + input_index] * weights[weight_base + input_index];
    }
    output[output_offset] = accumulator;
}

extern "C" __global__ void dense_mm_bwd_input_bucket(
    const float* output_gradient,
    unsigned long long,
    const float* weights,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* input_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * input_dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / input_dimension);
    const unsigned int input_index = static_cast<unsigned int>(i % input_dimension);
    const int bucket = bucket_idx[row];
    if (bucket < 0 || static_cast<unsigned int>(bucket) >= num_buckets) {
        input_gradient[i] = 0.0F;
        return;
    }
    float accumulator = 0.0F;
    for (unsigned int output_index = 0; output_index < output_dimension; ++output_index) {
        const unsigned long long weight_index =
            (static_cast<unsigned long long>(bucket) * output_dimension + output_index) *
                input_dimension +
            input_index;
        accumulator +=
            output_gradient[static_cast<unsigned long long>(row) * output_dimension + output_index] *
            weights[weight_index];
    }
    input_gradient[i] = accumulator;
}

extern "C" __global__ void dense_mm_bwd_input_tiled(
    const float* output_gradient,
    unsigned long long,
    const float* weights,
    unsigned long long,
    float* input_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension
) {
    const unsigned int input_tiles = input_dimension / 16U;
    const unsigned int batch_tile = blockIdx.x / input_tiles;
    const unsigned int input_tile = blockIdx.x % input_tiles;
    const unsigned int row = batch_tile * 16U + (threadIdx.x >> 4);
    const unsigned int input_index = input_tile * 16U + (threadIdx.x & 15U);
    if (row >= batch || input_index >= input_dimension) {
        return;
    }
    float accumulator = 0.0F;
    for (unsigned int output_index = 0; output_index < output_dimension; ++output_index) {
        accumulator +=
            output_gradient[static_cast<unsigned long long>(row) * output_dimension + output_index] *
            weights[static_cast<unsigned long long>(input_index) * output_dimension + output_index];
    }
    input_gradient[static_cast<unsigned long long>(row) * input_dimension + input_index] = accumulator;
}

// `dense_mm_bwd_input_tiled` with a 16-row x 64-input tile per block; each thread owns 4 inputs
// spaced 16 apart. The tile's output-gradient rows and weight rows pass through shared memory in
// 16-wide slices of the reduction axis, so global reads are contiguous row segments and each
// staged gradient value feeds 4 outputs. Every output accumulates `output_index` in ascending
// order like the narrow kernel, so both write identical values. `input_dimension` must be a
// multiple of 64 and the grid `(batch / 16) * (input_dimension / 64)` blocks.
extern "C" __global__ void dense_mm_bwd_input_tiled_wide(
    const float* output_gradient,
    unsigned long long,
    const float* weights,
    unsigned long long,
    float* input_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension
) {
    __shared__ float gradient_tile[16][17];
    __shared__ float weight_tile[64][17];
    const unsigned int input_tiles = input_dimension / 64U;
    const unsigned int batch_tile = blockIdx.x / input_tiles;
    const unsigned int input_tile = blockIdx.x % input_tiles;
    const unsigned int local_row = threadIdx.x >> 4;
    const unsigned int local_column = threadIdx.x & 15U;
    const unsigned int row = batch_tile * 16U + local_row;
    float accumulators[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned int base = 0; base < output_dimension; base += 16U) {
        const unsigned int output_index = base + local_column;
        const bool in_range = output_index < output_dimension;
        gradient_tile[local_row][local_column] =
            row < batch && in_range
                ? output_gradient[static_cast<unsigned long long>(row) * output_dimension + output_index]
                : 0.0F;
#pragma unroll
        for (unsigned int part = 0; part < 4U; ++part) {
            const unsigned int weight_row = input_tile * 64U + part * 16U + local_row;
            weight_tile[part * 16U + local_row][local_column] =
                weight_row < input_dimension && in_range
                    ? weights[static_cast<unsigned long long>(weight_row) * output_dimension + output_index]
                    : 0.0F;
        }
        __syncthreads();
        const unsigned int count = min(16U, output_dimension - base);
        for (unsigned int j = 0; j < count; ++j) {
            const float gradient = gradient_tile[local_row][j];
#pragma unroll
            for (unsigned int part = 0; part < 4U; ++part) {
                accumulators[part] += gradient * weight_tile[part * 16U + local_column][j];
            }
        }
        __syncthreads();
    }
    if (row < batch) {
#pragma unroll
        for (unsigned int part = 0; part < 4U; ++part) {
            const unsigned int input_index = input_tile * 64U + part * 16U + local_column;
            if (input_index < input_dimension) {
                input_gradient[static_cast<unsigned long long>(row) * input_dimension + input_index] =
                    accumulators[part];
            }
        }
    }
}

extern "C" __global__ void dense_mm_bwd_input_bucket_tiled_sorted_scatter(
    const float* output_gradient,
    unsigned long long,
    const float* weights,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    const int* permutation,
    unsigned long long,
    float* input_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned int sorted_row = blockIdx.y * 16U + (threadIdx.x >> 4);
    const unsigned int input_index = blockIdx.x * 16U + (threadIdx.x & 15U);
    if (sorted_row >= batch || input_index >= input_dimension) {
        return;
    }
    const int destination_row = permutation[sorted_row];
    if (destination_row < 0) {
        return;
    }
    const int bucket = bucket_idx[blockIdx.y * 16U];
    if (bucket < 0 || static_cast<unsigned int>(bucket) >= num_buckets) {
        input_gradient[static_cast<unsigned long long>(destination_row) * input_dimension + input_index] =
            0.0F;
        return;
    }
    float accumulator = 0.0F;
    for (unsigned int output_index = 0; output_index < output_dimension; ++output_index) {
        accumulator +=
            output_gradient[static_cast<unsigned long long>(sorted_row) * output_dimension + output_index] *
            weights[(static_cast<unsigned long long>(bucket) * output_dimension + output_index) *
                        input_dimension +
                    input_index];
    }
    input_gradient[static_cast<unsigned long long>(destination_row) * input_dimension + input_index] =
        accumulator;
}

extern "C" __global__ void dense_mm_bwd_weight_bucket_tiled_l1_sorted(
    const float* input,
    unsigned long long,
    const float* output_gradient,
    unsigned long long,
    const unsigned int* bucket_offsets,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int padded_batch,
    unsigned int input_dimension,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned int input_tiles = input_dimension / 16U;
    const unsigned int input_tile = blockIdx.x % input_tiles;
    const unsigned int output_tile = blockIdx.x / input_tiles;
    const unsigned int input_index = input_tile * 16U + (threadIdx.x >> 4);
    const unsigned int output_index = output_tile * 16U + (threadIdx.x & 15U);
    const unsigned int bucket = blockIdx.z;
    if (bucket >= num_buckets || input_index >= input_dimension || output_index >= output_dimension) {
        return;
    }
    const unsigned int bucket_start = bucket_offsets[bucket];
    const unsigned int bucket_end = min(bucket_offsets[bucket + 1U], padded_batch);
    const unsigned int bucket_tiles = (bucket_end - bucket_start) / 16U;
    const unsigned int tiles_per_split =
        (bucket_tiles + gridDim.y - 1U) / gridDim.y;
    const unsigned int first_tile = blockIdx.y * tiles_per_split;
    const unsigned int last_tile = min(first_tile + tiles_per_split, bucket_tiles);
    float accumulator = 0.0F;
    for (unsigned int tile = first_tile; tile < last_tile; ++tile) {
        const unsigned int first_row = bucket_start + tile * 16U;
        for (unsigned int lane = 0; lane < 16U; ++lane) {
            const unsigned int row = first_row + lane;
            accumulator +=
                input[static_cast<unsigned long long>(row) * input_dimension + input_index] *
                output_gradient[static_cast<unsigned long long>(row) * output_dimension + output_index];
        }
    }
    const unsigned long long index =
        (static_cast<unsigned long long>(bucket) * output_dimension + output_index) * input_dimension +
        input_index;
    atomicAdd(weight_gradient + index, accumulator);
}

extern "C" __global__ void dense_mm_bwd_weight_bucket_tiled_l2(
    const float* input,
    unsigned long long,
    const float* output_gradient,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int cells_per_bucket = input_dimension * output_dimension;
    if (cell >= cells_per_bucket) {
        return;
    }
    const unsigned int output_index = cell / input_dimension;
    const unsigned int input_index = cell % input_dimension;
    const unsigned int rows_per_split = (batch + gridDim.y - 1U) / gridDim.y;
    const unsigned int first_row = blockIdx.y * rows_per_split;
    const unsigned int last_row = min(first_row + rows_per_split, batch);
    float accumulators[kMaxSupportedNumBuckets] = {};
    const unsigned int bucket_limit = min(num_buckets, kMaxSupportedNumBuckets);
    for (unsigned int row = first_row; row < last_row; ++row) {
        const int bucket = bucket_idx[row];
        if (bucket >= 0 && static_cast<unsigned int>(bucket) < bucket_limit) {
            accumulators[bucket] +=
                input[static_cast<unsigned long long>(row) * input_dimension + input_index] *
                output_gradient[static_cast<unsigned long long>(row) * output_dimension + output_index];
        }
    }
    for (unsigned int bucket = 0; bucket < bucket_limit; ++bucket) {
        atomicAdd(weight_gradient + static_cast<unsigned long long>(bucket) * cells_per_bucket + cell,
                  accumulators[bucket]);
    }
}

extern "C" __global__ void dense_mm_bwd_weight_bucket_tiled_l3(
    const float* input,
    unsigned long long,
    const float* output_gradient,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* weight_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int input_dimension,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned int input_index = threadIdx.x % input_dimension;
    const unsigned int lane = threadIdx.x / input_dimension;
    const unsigned int lanes_per_column = blockDim.x / input_dimension;
    const unsigned int stride = gridDim.x * lanes_per_column;
    unsigned int row = blockIdx.x * lanes_per_column + lane;
    float accumulators[kMaxSupportedNumBuckets] = {};
    const unsigned int bucket_limit = min(num_buckets, kMaxSupportedNumBuckets);
    while (row < batch) {
        const int bucket = bucket_idx[row];
        if (bucket >= 0 && static_cast<unsigned int>(bucket) < bucket_limit) {
            accumulators[bucket] +=
                input[static_cast<unsigned long long>(row) * input_dimension + input_index] *
                output_gradient[static_cast<unsigned long long>(row) * output_dimension];
        }
        row += stride;
    }
    for (unsigned int bucket = 0; bucket < bucket_limit; ++bucket) {
        atomicAdd(weight_gradient + static_cast<unsigned long long>(bucket) * input_dimension + input_index,
                  accumulators[bucket]);
    }
}

// 各 (row, output) 要素を1 thread が担当し bias_gradient[output] へ直接 global atomicAdd
// する。shared-memory 縮約は行わない。結果は縮約実装と fp32 の加算順の範囲で一致し、
// native⇔cuda-oxide の数値 parity test が許容誤差内で固定する。
extern "C" __global__ void bias_grad_l1_shared(
    const float* output_gradient,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int output_dimension
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * output_dimension;
    if (i < total) {
        atomicAdd(bias_gradient + i % output_dimension, output_gradient[i]);
    }
}

// 1 thread = 1 output 列。`output_index = threadIdx.x` と `output_index >= output_dimension`
// の early return により、`blockDim.x >= output_dimension` を満たさないと
// `[blockDim.x, output_dimension)` の列が処理されず脱落する。caller は block を
// `output_dimension` 以上で launch する契約 (host 側 launch 前に assert)。
extern "C" __global__ void bias_grad_bucket_shared_sorted(
    const float* output_gradient,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int padded_batch,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned int output_index = threadIdx.x;
    const unsigned int first_row = blockIdx.x * 16U;
    if (output_index >= output_dimension || first_row >= padded_batch) {
        return;
    }
    const int bucket = bucket_idx[first_row];
    if (bucket < 0 || static_cast<unsigned int>(bucket) >= num_buckets) {
        return;
    }
    float accumulator = 0.0F;
    const unsigned int last_row = min(first_row + 16U, padded_batch);
    for (unsigned int row = first_row; row < last_row; ++row) {
        accumulator +=
            output_gradient[static_cast<unsigned long long>(row) * output_dimension + output_index];
    }
    atomicAdd(bias_gradient + static_cast<unsigned long long>(bucket) * output_dimension + output_index,
              accumulator);
}

extern "C" __global__ void bias_grad_bucket(
    const float* output_gradient,
    unsigned long long,
    const int* bucket_idx,
    unsigned long long,
    float* bias_gradient,
    unsigned long long,
    unsigned int batch,
    unsigned int output_dimension,
    unsigned int num_buckets
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = static_cast<unsigned long long>(batch) * output_dimension;
    if (i >= total) {
        return;
    }
    const unsigned int row = static_cast<unsigned int>(i / output_dimension);
    const unsigned int output_index = static_cast<unsigned int>(i % output_dimension);
    const int bucket = bucket_idx[row];
    if (bucket >= 0 && static_cast<unsigned int>(bucket) < num_buckets) {
        atomicAdd(bias_gradient + static_cast<unsigned long long>(bucket) * output_dimension + output_index,
                  output_gradient[i]);
    }
}
