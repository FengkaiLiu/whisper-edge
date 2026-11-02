// GPU device query plus copy-kernel and PCIe bandwidth test (see docs/cuda_guide.md, section 9).
#include <cstdio>
#include <cuda_runtime.h>

__global__ void copy_kernel(const float4 * __restrict__ src, float4 * __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    int clk_khz = 0, mem_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0);
    int drv = 0;
    cudaDriverGetVersion(&drv);
    printf("name                 %s\n", p.name);
    printf("compute capability   %d.%d\n", p.major, p.minor);
    printf("driver CUDA version  %d\n", drv);
    printf("SM count             %d\n", p.multiProcessorCount);
    printf("max threads / SM     %d\n", p.maxThreadsPerMultiProcessor);
    printf("max threads / block  %d\n", p.maxThreadsPerBlock);
    printf("warp size            %d\n", p.warpSize);
    printf("regs / SM            %d\n", p.regsPerMultiprocessor);
    printf("smem / SM (KiB)      %zu\n", p.sharedMemPerMultiprocessor / 1024);
    printf("smem / block opt-in  %zu\n", p.sharedMemPerBlockOptin / 1024);
    printf("L2 (MiB)             %.1f\n", p.l2CacheSize / 1048576.0);
    printf("global mem (MiB)     %zu\n", p.totalGlobalMem >> 20);
    printf("bus width (bit)      %d\n", p.memoryBusWidth);
    printf("SM clock (MHz)       %d\n", clk_khz / 1000);
    printf("mem clock (MHz)      %d\n", mem_khz / 1000);

    // device-to-device copy bandwidth: read n bytes + write n bytes
    const size_t bytes = 1ull << 30;
    const size_t n = bytes / sizeof(float4);
    float4 *a, *b;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMemset(a, 0, bytes);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int blocks = p.multiProcessorCount * 8, threads = 256, reps = 20;
    copy_kernel<<<blocks, threads>>>(a, b, n);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; r++) copy_kernel<<<blocks, threads>>>(a, b, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("copy kernel (GB/s)   %.0f  (1 GiB read + 1 GiB write, x%d)\n", 2.0 * bytes * reps / (ms * 1e6), reps);

    // host <-> device over PCIe, pinned memory
    void * h;
    cudaMallocHost(&h, 256 << 20);
    cudaEventRecord(t0);
    for (int r = 0; r < 10; r++) cudaMemcpy(a, h, 256 << 20, cudaMemcpyHostToDevice);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("H2D pinned (GB/s)    %.1f\n", 10.0 * (256 << 20) / (ms * 1e6));
    cudaEventRecord(t0);
    for (int r = 0; r < 10; r++) cudaMemcpy(h, a, 256 << 20, cudaMemcpyDeviceToHost);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("D2H pinned (GB/s)    %.1f\n", 10.0 * (256 << 20) / (ms * 1e6));
    printf("status               %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
