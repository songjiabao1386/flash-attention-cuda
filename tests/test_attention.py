import torch
import custom_flash_attn_cuda

print("=== 1. 检查 CUDA 设备与显卡型号 ===")
print("Device Name:", torch.cuda.get_device_name(0))
print("CUDA Available:", torch.cuda.is_available())

print("\n=== 2. 测试自定义算子导入与前向调用 ===")
batch_size = 2
num_heads = 4
seq_len = 1024
head_dim = 64

# 构造半精度 (fp16) 输入张量，符合大模型常用数据类型
device = torch.device("cuda")
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

# 调用我们写的 Naive Baseline
out_naive = custom_flash_attn_cuda.attention_naive_forward(Q, K, V)
print("Naive Attention Output Shape:", out_naive.shape)

# 调用 FlashAttention 前向入口（目前骨架返回全 0，后续我们将填充内核）
out_flash = custom_flash_attn_cuda.attention_flash_forward(Q, K, V)
print("FlashAttention Output Shape:", out_flash.shape)

print("\n🎉 基础验证通过！你的 CUDA 扩展环境运行完全正常。")
