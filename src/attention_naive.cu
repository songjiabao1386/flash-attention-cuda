#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include "attention_kernels.h"

torch::Tensor attention_naive_forward(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V
) {
    auto scale = 1.0 / std::sqrt(Q.size(-1));
    auto scores = torch::matmul(Q, K.transpose(-1, -2)) * scale;
    auto weights = torch::softmax(scores, -1);
    auto output = torch::matmul(weights, V);
    return output;
}

// 在这里统一注册 Pybind11 模块，确保 PyInit_custom_flash_attn_cuda 正确生成
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_naive_forward", &attention_naive_forward, "Naive Attention Forward (CUDA)");
    m.def("attention_flash_forward", &attention_flash_forward, "FlashAttention Forward (CUDA)");
}
