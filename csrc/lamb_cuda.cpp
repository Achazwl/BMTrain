#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void lamb_prepare_launcher(const torch::Tensor &param_fp32, const torch::Tensor &param_fp16, const torch::Tensor &g_fp16, const torch::Tensor &m_fp16, const torch::Tensor &v_fp32, float beta1, float beta2, float eps, float lr, float scale, float weight_decay, float bias_correction1, float bias_correction2, torch::Tensor &tmp_numer, torch::Tensor &tmp_denom, torch::Tensor& numer, torch::Tensor& denom);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void F_lamb_prepare(
    const torch::Tensor &param_fp32, 
    const torch::Tensor &param_fp16, 
    const torch::Tensor &g_fp16, 
    const torch::Tensor &m_fp16, 
    const torch::Tensor &v_fp32, 
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    int64_t step,
    torch::Tensor& numer,
    torch::Tensor& denom
) {
    CHECK_INPUT(param_fp32);
    CHECK_INPUT(param_fp16);
    CHECK_INPUT(g_fp16);
    CHECK_INPUT(m_fp16);
    CHECK_INPUT(v_fp32);
    AT_ASSERTM(param_fp32.dtype() == torch::kFloat, "param_fp32 must be a float tensor");
    AT_ASSERTM(param_fp16.dtype() == torch::kHalf, "param_fp16 must be a half tensor");
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(m_fp16.dtype() == torch::kHalf, "m_fp16 must be a half tensor");
    AT_ASSERTM(v_fp32.dtype() == torch::kFloat, "v_fp32 must be a float tensor");
    AT_ASSERTM(param_fp32.is_cuda(), "param_fp32 must be a CUDA tensor");
    AT_ASSERTM(param_fp16.is_cuda(), "param_fp16 must be a CUDA tensor");
    AT_ASSERTM(g_fp16.is_cuda(), "g_fp16 must be a CUDA tensor");
    AT_ASSERTM(m_fp16.is_cuda(), "m_fp16 must be a CUDA tensor");
    AT_ASSERTM(v_fp32.is_cuda(), "v_fp32 must be a CUDA tensor");
    AT_ASSERTM(param_fp32.numel() == param_fp16.numel(), "param_fp32 and param_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == g_fp16.numel(), "param_fp32 and g_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == m_fp16.numel(), "param_fp32 and m_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == v_fp32.numel(), "param_fp32 and v_fp32 must have the same number of elements");

    float bias_correction1 = 1 - powf(beta1, step);
    float bias_correction2 = 1 - powf(beta2, step);

    torch::Tensor tmp_numer = param_fp32.new_zeros({1024});
    torch::Tensor tmp_denom = param_fp32.new_zeros({1024});

    lamb_prepare_launcher(param_fp32, param_fp16, g_fp16, m_fp16, v_fp32, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2, tmp_numer, tmp_denom, numer, denom);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_lamb_prepare", &F_lamb_prepare, "lamb function prepare");
}
