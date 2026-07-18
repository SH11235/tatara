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
            const unsigned int weight_base = static_cast<unsigned int>(column) * rows + row;
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
    float,
    float,
    float,
    float,
    const double*,
    unsigned long long,
    unsigned int extended,
    unsigned int n
) {
    __shared__ double partial[256];
    if (extended != 0) {
        // Extended WRM is rejected by the host before launch. Trap defensively so adding only its
        // prerequisite kernels cannot turn this wrapper into a silent stale-gradient path.
        asm volatile("trap;");
        return;
    }
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
