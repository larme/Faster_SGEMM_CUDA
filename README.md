# Faster CUDA SGEMM from Scratch

I have added one new 30-line CUDA kernel to the [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) project by Simon Boehm, which compares TFLOPS of various kernels doing single precision matrix multiplication.

The new kernel is often faster (on this benchmark) than the standard float32 matrix multiplication from the cuBLAS library, for example:

- 43% faster when multiplying matrices of size 12288² on an A100 GPU
- 24% faster when multiplying matrices of size 12288² on a 4090 GPU
- 23% faster when multiplying matrices of size 4096² on a H100 GPU
- 11% faster when multiplying matrices of size 8192² on a H100 GPU


How to run it:

Modify CUDA_COMPUTE_CAPABILITY in CMakeLists.txt, if needed.

Modify the variable SIZE in sgemm.cu

```
mkdir build && cd build && cmake .. && cmake --build .

./sgemm 0
runs the function cublasGemmEx() 50 times.

./sgemm 13
runs my new kernel sgemmZeroRegisters() 50 times.
```


On a 4090 GPU, the average of 20 runs of SGEMM_CUDA:

```
size    tflops_cublas  tflops_my  diff
16384²  53.6           66.7       +24%
12288²  53.7           66.7       +24%  
9216²   53.5           62.6       +17%
8704²   52.7           61.0       +16%
8448²   53.5           64.1       +20%
8192²   56.3-56.5      67.1       +19%
7936²   54.3           63.6       +17%
7680²   53.4           60.0       +10%
7168²   54.7           59.3       +8%
6144²   55.3           59.8       +8%
5120²   51.9           50.3       -3%
4096²   50.8-50.9      61.8       +21%
3840²   48.3           54.8       +13%
3584²   42.2           48.9       +16%
```

On other GPUs:

```
size    tflops_cublas  tflops_my  diff      gpu
12288²  51.4           56.3       +9%       h100
8192²   50.5           56.1       +11%      h100
4096²   43.8           53.9       +23%      h100
12288²  18.9           27.0       +43%      a100
8192²   19.0           26.3       +38%      a100
4096²   17.5           19.8       +13%      a100
8192²   27.7-28.2      33.5       +19-21%   4070ts
4096²   28.7-28.8      32.5       +13%      4070ts
16384²  28.8           34.9       +21%      3090ti
12288²  28.8           34.5       +20%      3090ti
8192²   29.3           33.3       +14%      3090ti
4096²   27.9           26.7       -4%       3090ti
4096²   9.9-10.0       10.1-10.2  +1-2%     1080ti
4096²   3.8-4.3        6.7        +56-76%   T4
```

The average TFLOPS values were calculated using:
```
for i in $(seq 1 20); do ./sgemm 0; done | grep -o 'performance[^)]*' | grep -o '[^(]*$' | awk '{a=a+$1;cnt=cnt+1}END{print(a/cnt)}'
```

The matrix size is hardcoded as 4096 in sgemm.cu

Performance may vary with hardware configuration and software versions. I used CUDA 11.8.

How the new kernel works:

the code is straightforward and similar to other kernels in the original repository.
The only trick is what I call "triggering compilation to using zero registers".
The GPU has a limit of 65536 4-byte registers per block.
With 512 threads per block, that gives 65536/512 = 128 registers per thread.
Every thread calculates simultaneously 128 values, accumulating each value in one register.
That leaves 0 registers for the actual operation of the kernel.
Yet the code somehow compiles to a correctly working kernel.

Arek Paterek

```

```

The description of the original SGEMM_CUDA:

# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `309.0` | 1.3%                           |
| 2: GMEM Coalescing                  |  `1986.5` | 8.5%                           |
| 3: SMEM Caching                     |  `2980.3` | 12.8%                          |
| 4: 1D Blocktiling                   |  `8474.7` | 36.5%                          |
| 5: 2D Blocktiling                   | `15971.7` | 68.7%                          |
| 7: Avoid Bank Conflicts (Linearize) | `16213.4` | 69.7%                          |
| 8: Avoid Bank Conflicts (Offset)    | `16459.2` | 70.8%                          |
| 11: Double Buffering                | `17278.3` | 74.3%                          |
| 6: Vectorized Mem Access            | `18237.3` | 78.4%                          |
| 9: Autotuning                       | `19721.0` | 84.8%                          |
| 10: Warptiling                      | `21779.3` | 93.7%                          |
| 0: cuBLAS                           | `23249.6` | 100.0%                         |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
