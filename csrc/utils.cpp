#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void has_nan_inf_launcher(const torch::Tensor &g_fp16, torch::Tensor out);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void F_has_inf_nan(const torch::Tensor &g_fp16, torch::Tensor &out) {
    CHECK_INPUT(g_fp16);
    CHECK_INPUT(out);
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(out.dtype() == torch::kUInt8, "out must be a uint8 tensor");
    AT_ASSERTM(g_fp16.is_cuda(), "g_fp16 must be a CUDA tensor");
    AT_ASSERTM(out.is_cuda(), "out must be a CUDA tensor");

    has_nan_inf_launcher(g_fp16, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_has_inf_nan", &F_has_inf_nan, "has inf or nan");
}