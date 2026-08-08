#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// 表示専用の旧 GPU が刺さった混載機では検出値が最小 GPU に引かれてここへ来る。
// 学習に使う GPU を TATARA_CUDA_COMPUTE で明示すれば回避できる。
#error "progress kernels require compute capability 6.0 or newer for double atomicAdd; set TATARA_CUDA_COMPUTE to the capability of the GPU used for training"
#endif

// CUDA legacy atomic functions provide device-scope relaxed read-modify-write operations.
// These reductions consume only the final values and require no ordering with other memory.

extern "C" __global__ void progress_forward(
    const int32_t* indices,
    uint64_t indices_len,
    const float* weights,
    uint64_t weights_len,
    float* preds,
    uint64_t preds_len,
    uint32_t n_pos,
    uint32_t max_inds) {
    const uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos >= n_pos || pos >= preds_len) {
        return;
    }

    float z = 0.0f;
    const uint64_t base = pos * max_inds;
    for (uint32_t j = 0; j < max_inds; ++j) {
        const uint64_t offset = base + j;
        if (offset >= indices_len) {
            return;
        }
        const int32_t index = indices[offset];
        if (index >= 0 && static_cast<uint64_t>(index) < weights_len) {
            z += weights[index];
        }
    }
    preds[pos] = 1.0f / (1.0f + expf(-z));
}

extern "C" __global__ void progress_grad(
    const int32_t* indices,
    uint64_t indices_len,
    const float* preds,
    uint64_t preds_len,
    const float* targets,
    uint64_t targets_len,
    const float* per_pos_norm,
    uint64_t per_pos_norm_len,
    float* grad,
    uint64_t grad_len,
    double* loss_acc,
    uint64_t loss_acc_len,
    uint64_t* hist,
    uint64_t hist_len,
    uint32_t n_pos,
    uint32_t max_inds) {
    const uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos >= n_pos || pos >= preds_len || pos >= targets_len || pos >= per_pos_norm_len ||
        loss_acc_len == 0 || hist_len < 8) {
        return;
    }

    const float p = preds[pos];
    const float error = p - targets[pos];
    const float scale = 2.0f * error * p * (1.0f - p) * per_pos_norm[pos];
    const uint64_t base = pos * max_inds;
    for (uint32_t j = 0; j < max_inds; ++j) {
        const uint64_t offset = base + j;
        if (offset >= indices_len) {
            return;
        }
        const int32_t index = indices[offset];
        if (index >= 0 && static_cast<uint64_t>(index) < grad_len) {
            atomicAdd(&grad[index], scale);
        }
    }
    atomicAdd(loss_acc, static_cast<double>(error) * static_cast<double>(error));

    int32_t bin = static_cast<int32_t>(p * 8.0f);
    bin = bin < 0 ? 0 : bin;
    bin = bin > 7 ? 7 : bin;
    atomicAdd(reinterpret_cast<unsigned long long*>(&hist[bin]), 1ULL);
}

extern "C" __global__ void progress_eval(
    const float* preds,
    uint64_t preds_len,
    const float* targets,
    uint64_t targets_len,
    double* loss_acc,
    uint64_t loss_acc_len,
    uint64_t* hist,
    uint64_t hist_len,
    uint32_t n_pos) {
    const uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos >= n_pos || pos >= preds_len || pos >= targets_len || loss_acc_len == 0 ||
        hist_len < 8) {
        return;
    }

    const float p = preds[pos];
    const float error = p - targets[pos];
    atomicAdd(loss_acc, static_cast<double>(error) * static_cast<double>(error));

    int32_t bin = static_cast<int32_t>(p * 8.0f);
    bin = bin < 0 ? 0 : bin;
    bin = bin > 7 ? 7 : bin;
    atomicAdd(reinterpret_cast<unsigned long long*>(&hist[bin]), 1ULL);
}

extern "C" __global__ void progress_adam_step(
    float* weights,
    uint64_t weights_len,
    float* momentum,
    uint64_t momentum_len,
    float* velocity,
    uint64_t velocity_len,
    float* grad,
    uint64_t grad_len,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    uint32_t n) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || i >= weights_len || i >= momentum_len || i >= velocity_len || i >= grad_len) {
        return;
    }

    const float g = grad[i];
    const float m = beta1 * momentum[i] + (1.0f - beta1) * g;
    const float v = beta2 * velocity[i] + (1.0f - beta2) * g * g;
    momentum[i] = m;
    velocity[i] = v;
    const float m_hat = m / fmaxf(bc1, 1.0e-30f);
    const float v_hat = v / fmaxf(bc2, 1.0e-30f);
    weights[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    grad[i] = 0.0f;
}
