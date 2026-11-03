import torch
import time
import custom_flash_attn_cuda

print("==================================================")
print("     RTX 4090 FlashAttention Benchmark Suite      ")
print("==================================================")

device = torch.device("cuda")
batch_size = 2
num_heads = 8
head_dim = 64

seq_lens = [1024, 2048, 4096, 8192]

for seq_len in seq_lens:
    print(f"\n--- Testing Sequence Length (N): {seq_len} ---")
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

    # 1. 测试 Naive Attention
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(20): # 预热并多次循环取平均
        out_naive = custom_flash_attn_cuda.attention_naive_forward(Q, K, V)
        
    torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / 20 * 1000 # ms
    naive_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB

    # 2. 测试 Custom FlashAttention
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(20):
        out_flash = custom_flash_attn_cuda.attention_flash_forward(Q, K, V)
        
    torch.cuda.synchronize()
    flash_time = (time.time() - start_time) / 20 * 1000 # ms
    flash_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB

    print(f"  [Naive Attention]  Time: {naive_time:6.2f} ms | Peak VRAM: {naive_mem:7.2f} MB")
    print(f"  [Custom FlashAttn] Time: {flash_time:6.2f} ms | Peak VRAM: {flash_mem:7.2f} MB")
    print(f"  🚀 Speedup: {naive_time / flash_time:.2f}x")

print("\n🎉 Benchmark 测试顺利完成！把这些亮眼的数据写进你的 GitHub README 吧！")
