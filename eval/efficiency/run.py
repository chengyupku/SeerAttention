import subprocess
import os
import time

block_size = 64
heads = 64
heads_kv = 8

file_dir = "results"
os.makedirs(file_dir, exist_ok=True)

file_name = f"{file_dir}/kernel_test_{block_size}_gqa{heads}_{heads_kv}.csv"

with open(file_name, "w") as f:
    f.write("batch,max_cache_seqlen,sparse_ratio,fa2_dense_time,triton_sparse_time,tilelang_sparse_time\n")

configs = [
    "16*8192", "16*8192", "16*16384", "16*32768", "16*65536", "16*131072",
    "8*8192", "8*16384", "8*32768", "8*65536", "8*131072",
    "4*8192", "4*16384", "4*32768", "4*65536", "4*131072",
    "2*8192", "2*16384", "2*32768", "2*65536", "2*131072",
    "1*8192", "1*16384", "1*32768", "1*65536", "1*131072",
]

# configs = [
#     "16*8192",
# ]

sparse_ratios = ["0.5", "0.6", "0.7", "0.8", "0.9"]

# sparse_ratios = ["0.5"]

timeout_seconds = 600  # 每个进程最大运行时间（单位：秒）

for config in configs:
    batch, max_cache_seqlen = config.split("*")
    
    for sr in sparse_ratios:
        print(f"\n=== Running: batch={batch}, max_cache_seqlen={max_cache_seqlen}, sparse_ratio={sr} ===")
        
        try:
            process = subprocess.run(
                [
                    "python", "decode_kernel_eval.py",
                    "--batch", batch,
                    "--max_cache_seqlen", max_cache_seqlen,
                    "--sparse_ratio", sr
                ],
                timeout=timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired:
            print(f"⚠️ Timeout: Killed process for batch={batch}, max_cache_seqlen={max_cache_seqlen}, sparse_ratio={sr}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Process exited with error code {e.returncode}")

# 最后画图
data_path = f"results/kernel_test_{block_size}_gqa{heads}_{heads_kv}.csv"
print(f"\n=== Drawing results from: {data_path} ===\n")
subprocess.run(["python", "draw.py", "--data_path", data_path])