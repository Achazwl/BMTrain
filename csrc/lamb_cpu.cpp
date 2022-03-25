#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <immintrin.h>
#include <emmintrin.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#if defined(__AVX512F__)

#pragma message "Using AVX512"
#define __AVX512__ 1

#elif defined(__AVX__) and defined(__FMA__) and defined(__F16C__)

#pragma message "Using AVX256"
#define __AVX256__ 1

#endif

void lamb_cpu_prepare_launcher(
    int n,
    float* param_fp32,
    at::Half* param_fp16,
    at::Half* g_fp16,
    float* m_fp32,
    float* v_fp32,
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float* numer, float* denom
) {
    numer[0] = at::parallel_reduce(0, n, 0, 0.,
        [&](int64_t start, int64_t end, float ident) -> float {
        float partial_numer = ident;
        for (int64_t i = start; i < end; ++i) {
            float p = param_fp32[i];
            partial_numer += p * p;
        }
        return partial_numer;
    }, std::plus<float>());

    denom[0] = at::parallel_reduce(0, n, 0, 0.,
        [&](int64_t start, int64_t end, float ident) -> float {
        float partial_denom = ident;
        for (int64_t i = start; i < end; ++i) {
            float g = c10::detail::fp16_ieee_to_fp32_value(g_fp16[i].x) / scale;
            float m = m_fp32[i];
            float v = v_fp32[i];
            float p = param_fp32[i];
            m = beta1 * m + (1 - beta1) * g;
            v = beta2 * v + (1 - beta2) * g * g;
            float update = m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) + weight_decay * p;
            partial_denom += update * update;
        }
        return partial_denom;
    }, std::plus<float>());
}

void F_lamb_prepare_cpu(
    const torch::Tensor &param_fp32, 
    const torch::Tensor &param_fp16, 
    const torch::Tensor &g_fp16, 
    const torch::Tensor &m_fp32, 
    const torch::Tensor &v_fp32, 
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    int64_t step,
    torch::Tensor& numer,
    torch::Tensor& denom
) {
    CHECK_CONTIGUOUS(param_fp32);
    CHECK_CONTIGUOUS(param_fp16);
    CHECK_CONTIGUOUS(g_fp16);
    CHECK_CONTIGUOUS(m_fp32);
    CHECK_CONTIGUOUS(v_fp32);
    AT_ASSERTM(param_fp32.dtype() == torch::kFloat, "param_fp32 must be a float tensor");
    AT_ASSERTM(param_fp16.dtype() == torch::kHalf, "param_fp16 must be a half tensor");
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(m_fp32.dtype() == torch::kFloat, "m_fp32 must be a float tensor");
    AT_ASSERTM(v_fp32.dtype() == torch::kFloat, "v_fp32 must be a float tensor");
    AT_ASSERTM(param_fp32.is_cpu(), "param_fp32 must be a cpu tensor");
    AT_ASSERTM(param_fp16.is_cpu(), "param_fp16 must be a cpu tensor");
    AT_ASSERTM(g_fp16.is_cpu(), "g_fp16 must be a cpu tensor");
    AT_ASSERTM(m_fp32.is_cpu(), "m_fp32 must be a cpu tensor");
    AT_ASSERTM(v_fp32.is_cpu(), "v_fp32 must be a cpu tensor");
    AT_ASSERTM(param_fp32.numel() == param_fp16.numel(), "param_fp32 and param_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == g_fp16.numel(), "param_fp32 and g_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == m_fp32.numel(), "param_fp32 and m_fp32 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == v_fp32.numel(), "param_fp32 and v_fp32 must have the same number of elements");

    float bias_correction1 = 1 - powf(beta1, step);
    float bias_correction2 = 1 - powf(beta2, step);

    lamb_cpu_prepare_launcher(
        param_fp32.numel(),
        param_fp32.data_ptr<float>(),
        param_fp16.data_ptr<at::Half>(),
        g_fp16.data_ptr<at::Half>(),
        m_fp32.data_ptr<float>(),
        v_fp32.data_ptr<float>(),
        beta1, beta2, 
        eps, lr, 
        scale, 
        weight_decay,
        bias_correction1,
        bias_correction2,
        numer.data_ptr<float>(), denom.data_ptr<float>()
    );
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_lamb_prepare_cpu", &F_lamb_prepare_cpu, "lamb function prepare cpu");
}
