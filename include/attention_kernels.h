#pragma once
#include <torch/extension.h>

// 1. 朴素 Attention 前向实现（用于 Baseline 显存与速度对比）
torch::Tensor attention_naive_forward(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V
);

// 2. 自定义 FlashAttention 前向实现（Tiling + Online Softmax）
torch::Tensor attention_flash_forward(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V
);
