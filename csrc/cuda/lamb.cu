#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "reduce.cuh"

namespace {
// blocks <min(roundup(n,1024), 1024)>,      threads<1024>
__global__ void lamb_fp32_term_reduce_1(
    int32_t n,
    const half *g,        // (n)
    const half *m,        // (n)
    const float *v,       // (n)
    const float* param,   // (n)
    const half* param_h,  // (n)
    float beta1,
    float beta2,
    float eps,
    float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float* numer, // (1024)
    float* denom // (1024)
) {
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t span = blockDim.x * gridDim.x;

    float local_numer = 0, local_denom = 0;
    for (int i = gid; i < n; i += span) {
        float local_g = __half2float(g[i]) / scale;
        float local_m = beta1 * __half2float(m[i]) + (1 - beta1) * local_g;
        float local_v = beta2 * v[i] + (1 - beta2) * local_g * local_g;
        float local_p = param[i];
        float update = local_m / bias_correction1 / (sqrtf(local_v / bias_correction2) + eps) + weight_decay * local_p;
        local_numer += local_p * local_p;
        local_denom += update * update;
    }
    local_numer = block_reduce_sum(local_numer);
    local_denom = block_reduce_sum(local_denom);

    if (threadIdx.x == 0) {
        numer[blockIdx.x] = local_numer;
        denom[blockIdx.x] = local_denom;
    }
}

// blocks <1>,      threads<1024>
__global__ void lamb_fp32_term_reduce_2(
    float* numer,       // (1024)
    float* denom,       // (1024)
    float* out_numer,   // (1)
    float* out_denom    // (1)
) {
    int tid = threadIdx.x;
    float local_numer = block_reduce_sum(numer[tid]);
    float local_denom = block_reduce_sum(denom[tid]);

    if (tid == 0) {
        out_numer[0] = local_numer;
        out_denom[0] = local_denom;
    }
}

}

void lamb_prepare_launcher(
    const torch::Tensor &param_fp32,
    const torch::Tensor &param_fp16,
    const torch::Tensor &g_fp16,
    const torch::Tensor &m_fp16,
    const torch::Tensor &v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    torch::Tensor &tmp_numer,
    torch::Tensor &tmp_denom,
    torch::Tensor &numer,
    torch::Tensor &denom
) {
    int32_t n = param_fp32.numel();
    auto g_ptr = reinterpret_cast<half*>(g_fp16.data_ptr<at::Half>());
    auto m_ptr = reinterpret_cast<half*>(m_fp16.data_ptr<at::Half>());
    auto v_ptr = v_fp32.data_ptr<float>();
    auto param_ptr = param_fp32.data_ptr<float>();
    auto param_h_ptr = reinterpret_cast<half*>(param_fp16.data_ptr<at::Half>());
    auto numer_ptr = tmp_numer.data_ptr<float>();
    auto denom_ptr = tmp_denom.data_ptr<float>();
    int32_t threads = 1024;
    dim3 block_size = dim3(threads, 1, 1);
    dim3 grid_size = dim3((n + threads - 1) / threads, 1, 1);
    dim3 clamp_grid_size = dim3(min((n + threads - 1) / threads, 1024), 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    lamb_fp32_term_reduce_1<<<clamp_grid_size, block_size, 0, stream.stream()>>>(n, g_ptr, m_ptr, v_ptr, param_ptr, param_h_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2, numer_ptr, denom_ptr);
    lamb_fp32_term_reduce_2<<<1, block_size, 0, stream.stream()>>>(numer_ptr, denom_ptr, numer.data_ptr<float>(), denom.data_ptr<float>());
}
