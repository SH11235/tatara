#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
        const float target = lambda * wdl[i] + (1.0F - lambda) * target_wrm;

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

    const unsigned int nb = effect_bucket_factorize & 0xffffU;
    const unsigned int mode = effect_bucket_factorize >> 16;
    const unsigned long long feature = i / ft_out;
    const unsigned int column = static_cast<unsigned int>(i - feature * ft_out);
    float value = weights[i];
    if (feature < base_ft_in) {
        const unsigned long long piece = mode == 0
            ? feature % piece_inputs
            : (feature / nb) % piece_inputs;
        const unsigned long long virtual_row = mode == 2
            ? piece * nb + feature % nb
            : piece;
        const unsigned long long virtual_index =
            (static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column;
        value += weights[virtual_index];
    } else if (threat_pair_starts_len >= 2) {
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
        const unsigned long long virtual_index =
            (static_cast<unsigned long long>(ft_in) + base_virtual_rows + low) * ft_out + column;
        value += weights[virtual_index];
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

    const unsigned int nb = effect_bucket_factorize & 0xffffU;
    const unsigned int mode = effect_bucket_factorize >> 16;
    const unsigned long long feature = i / ft_out;
    const unsigned int column = static_cast<unsigned int>(i - feature * ft_out);
    float value = weights[i];
    if (feature < base_ft_in) {
        const unsigned long long piece = mode == 0
            ? feature % piece_inputs
            : (feature / nb) % piece_inputs;
        const unsigned long long virtual_row = mode == 2
            ? piece * nb + feature % nb
            : piece;
        const unsigned long long virtual_index =
            (static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column;
        value += weights[virtual_index];
    } else if (threat_pair_starts_len >= 2) {
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
        const unsigned long long virtual_index =
            (static_cast<unsigned long long>(ft_in) + base_virtual_rows + low) * ft_out + column;
        value += weights[virtual_index];
    }
    combined[i] = __float2half_rn(value);
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
            sum += gradient[feature * ft_out + column];
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
        const unsigned long long king_stride =
            static_cast<unsigned long long>(piece_inputs) * nb * ft_out;
        for (unsigned int king_bucket = 0; king_bucket < king_buckets; ++king_bucket) {
            const unsigned long long base =
                static_cast<unsigned long long>(king_bucket) * king_stride
                + static_cast<unsigned long long>(piece) * nb * ft_out + column;
            for (unsigned int effect_bucket = 0; effect_bucket < nb; ++effect_bucket) {
                sum += gradient[base + static_cast<unsigned long long>(effect_bucket) * ft_out];
            }
        }
        gradient[(static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column] = sum;
        return;
    }

    const unsigned long long row_stride = mode == 0
        ? static_cast<unsigned long long>(piece_inputs) * ft_out
        : static_cast<unsigned long long>(piece_inputs) * nb * ft_out;
    const unsigned long long base = mode == 0
        ? static_cast<unsigned long long>(piece) * ft_out + column
        : (static_cast<unsigned long long>(piece) * nb + bucket) * ft_out + column;
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;
    unsigned int king_bucket = 0;
    const unsigned int unroll_end = king_buckets > 3 ? king_buckets - 3 : 0;
    while (king_bucket < unroll_end) {
        sum0 += gradient[base + static_cast<unsigned long long>(king_bucket) * row_stride];
        sum1 += gradient[base + static_cast<unsigned long long>(king_bucket + 1) * row_stride];
        sum2 += gradient[base + static_cast<unsigned long long>(king_bucket + 2) * row_stride];
        sum3 += gradient[base + static_cast<unsigned long long>(king_bucket + 3) * row_stride];
        king_bucket += 4;
    }
    while (king_bucket < king_buckets) {
        sum0 += gradient[base + static_cast<unsigned long long>(king_bucket) * row_stride];
        ++king_bucket;
    }
    sum = (sum0 + sum1) + (sum2 + sum3);
    gradient[(static_cast<unsigned long long>(ft_in) + virtual_row) * ft_out + column] = sum;
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
        const float target = lambda * wdl[i] + (1.0F - lambda) * target_wrm;
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
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float prediction = 1.0F / (1.0F + expf(-(output[i] * scale)));
    const float score_target = 1.0F / (1.0F + expf(-(score[i] * scale)));
    const float target = lambda * wdl[i] + (1.0F - lambda) * score_target;
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

template <bool Add>
__device__ __forceinline__ void native_gather_and_sum_per_feature_fp16(
    const __half* output_gradient,
    const unsigned int* positions,
    const unsigned int* offsets,
    float* weight_gradient,
    unsigned int n_features,
    unsigned int ft_output,
    float inverse_gradient_scale
) {
    const unsigned int feature = blockIdx.x;
    const unsigned int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (feature >= n_features || row >= ft_output) {
        return;
    }
    const unsigned int start = offsets[feature];
    const unsigned int end = offsets[feature + 1];
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;
    unsigned int i = start;
    const unsigned int unroll_end = end >= start + 3 ? end - 3 : start;
    while (i < unroll_end) {
        sum0 += __half2float(output_gradient[
            static_cast<unsigned long long>(positions[i]) * ft_output + row
        ]);
        sum1 += __half2float(output_gradient[
            static_cast<unsigned long long>(positions[i + 1]) * ft_output + row
        ]);
        sum2 += __half2float(output_gradient[
            static_cast<unsigned long long>(positions[i + 2]) * ft_output + row
        ]);
        sum3 += __half2float(output_gradient[
            static_cast<unsigned long long>(positions[i + 3]) * ft_output + row
        ]);
        i += 4;
    }
    while (i < end) {
        sum0 += __half2float(output_gradient[
            static_cast<unsigned long long>(positions[i]) * ft_output + row
        ]);
        ++i;
    }
    const float sum = (sum0 + sum1) + (sum2 + sum3);
    const unsigned long long output_index =
        static_cast<unsigned long long>(feature) * ft_output + row;
    const float scaled = sum * inverse_gradient_scale;
    if (Add) {
        if (sum != 0.0F) {
            atomicAdd(weight_gradient + output_index, scaled);
        }
    } else {
        weight_gradient[output_index] = scaled;
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
        output_gradient, positions, offsets, weight_gradient, n_features, ft_output,
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
        output_gradient, positions, offsets, weight_gradient, n_features, ft_output,
        inverse_gradient_scale
    );
}

template <bool HalfState, bool Mirror>
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
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
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
    if (Mirror) {
        mirror[i] = __float2half_rn(p);
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
    unsigned int feature_count,
    unsigned int ft_columns,
    bool add
) {
    const unsigned int feature = blockIdx.x;
    const unsigned int output = blockIdx.y * blockDim.x + threadIdx.x;
    if (feature >= feature_count || output >= ft_columns) {
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
        output_gradient, positions, offsets, weight_gradient, feature_count, ft_columns, false
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
        output_gradient, positions, offsets, weight_gradient, feature_count, ft_columns, true
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
    const unsigned int base = row * max_active;
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
    const float xa = ft_output[base + column] + bias[column];
    const float xb = ft_output[base + half + column] + bias[half + column];
    const float ya = native_clamp_unit(xa);
    const float yb = native_clamp_unit(xb);
    const float grad_a = xa > 0.0F && xa < 1.0F ? dy * yb * scale : 0.0F;
    const float grad_b = xb > 0.0F && xb < 1.0F ? dy * ya * scale : 0.0F;
    ft_gradient[base + column] = grad_a;
    ft_gradient[base + half + column] = grad_b;
    atomicAdd(bias_gradient + column, grad_a);
    atomicAdd(bias_gradient + half + column, grad_b);
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
    const float xa = __half2float(ft_output[base + column]) + bias[column];
    const float xb = __half2float(ft_output[base + half + column]) + bias[half + column];
    const float ya = native_clamp_unit(xa);
    const float yb = native_clamp_unit(xb);
    const float grad_a = xa > 0.0F && xa < 1.0F ? dy * yb * scale : 0.0F;
    const float grad_b = xb > 0.0F && xb < 1.0F ? dy * ya * scale : 0.0F;
    const float scaled_a = grad_a * gradient_scale;
    const float scaled_b = grad_b * gradient_scale;
    const float clamped_a = native_clamp_half(scaled_a, clamp_counter);
    const float clamped_b = native_clamp_half(scaled_b, clamp_counter);
    ft_gradient[base + column] = __float2half_rn(clamped_a);
    ft_gradient[base + half + column] = __float2half_rn(clamped_b);
    atomicAdd(bias_gradient + column, grad_a);
    atomicAdd(bias_gradient + half + column, grad_b);
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
    float accumulators[9] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned int row = first_row; row < last_row; ++row) {
        const int bucket = bucket_idx[row];
        if (bucket >= 0 && static_cast<unsigned int>(bucket) < num_buckets) {
            accumulators[bucket] +=
                input[static_cast<unsigned long long>(row) * input_dimension + input_index] *
                output_gradient[static_cast<unsigned long long>(row) * output_dimension + output_index];
        }
    }
    for (unsigned int bucket = 0; bucket < num_buckets; ++bucket) {
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
    float accumulators[9] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    while (row < batch) {
        const int bucket = bucket_idx[row];
        if (bucket >= 0 && static_cast<unsigned int>(bucket) < num_buckets) {
            accumulators[bucket] +=
                input[static_cast<unsigned long long>(row) * input_dimension + input_index] *
                output_gradient[static_cast<unsigned long long>(row) * output_dimension];
        }
        row += stride;
    }
    for (unsigned int bucket = 0; bucket < num_buckets; ++bucket) {
        atomicAdd(weight_gradient + static_cast<unsigned long long>(bucket) * input_dimension + input_index,
                  accumulators[bucket]);
    }
}

extern "C" __global__ void bias_grad_shared_l1f(
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
