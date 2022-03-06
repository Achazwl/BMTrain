#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
constexpr int WARP_SZ = 32;

// blocks <block_size>,      threads<1024>
__device__ float reduce(float d_x) {
    static __shared__ float s_x[WARP_SZ];
    // int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int lid = threadIdx.x % WARP_SZ;
    int wid = threadIdx.x / WARP_SZ;

    // reduce intra warp

    float val = d_x;
    for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    
    if (lid == 0) s_x[wid] = val;
    __syncthreads();

    // reduce inter warp
    val = (tid < WARP_SZ) ? s_x[lid] : 0;
    if (wid == 0) {
        for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    return val;
}

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
    local_numer = reduce(local_numer);
    local_denom = reduce(local_denom);

    if (threadIdx.x == 0) {
        numer[blockIdx.x] = local_numer;
        denom[blockIdx.x] = local_denom;
    }
}

// blocks <1>,      threads<1024>
__global__ void lamb_fp32_term_reduce_2(
    float* numer, // (1024)
    float* denom  // (1024)
) {
    int tid = threadIdx.x;
    float local_numer = reduce(numer[tid]);
    float local_denom = reduce(denom[tid]);

    if (tid == 0) {
        numer[0] = sqrtf(local_numer / local_denom);
    }
}

// blocks <roundup(n,1024)>,      threads<1024>
__global__ void lamb_fp32_accum(
    int32_t n,
    const half *g,  // (n)
    half *m,        // (n)
    float *v,       // (n)
    float* param,   // (n)
    half* param_h,  // (n)
    float beta1,
    float beta2,
    float eps,
    float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    const float* term
) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float local_g = __half2float(g[col]) / scale;
        float local_m = beta1 * __half2float(m[col]) + (1 - beta1) * local_g;
        float local_v = beta2 * v[col] + (1 - beta2) * local_g * local_g;
        float local_p = param[col];
        local_p = local_p - lr * (*term) * ( local_m / bias_correction1 / (sqrtf(local_v / bias_correction2) + eps) + weight_decay * local_p );

        param_h[col] = __float2half(local_p);
        param[col] = local_p;
        v[col] = local_v;
        m[col] = __float2half(local_m);
    }
}
}

void lamb_launcher(
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
    torch::Tensor &tmp_denom
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
    lamb_fp32_term_reduce_2<<<1, block_size, 0, stream.stream()>>>(numer_ptr, denom_ptr);
    lamb_fp32_accum<<<grid_size, block_size, 0, stream.stream()>>>(n, g_ptr, m_ptr, v_ptr, param_ptr, param_h_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2, numer_ptr);
}